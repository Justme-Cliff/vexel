"""
Vexel lexer.

Converts raw source text into a flat list of tokens.

F-strings and triple-quoted strings are desugared during a preprocessing
pass before the main scan.  NEWLINE / INDENT / DEDENT tokens are
suppressed while inside open parentheses, brackets, or braces so that
multi-line expressions work naturally.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    # Literals
    INT          = auto()
    FLOAT        = auto()
    STRING       = auto()
    CHAR         = auto()   # 'a'  character literal

    # Identifiers & keywords
    IDENT        = auto()
    FN           = auto()
    LET          = auto()
    CONST        = auto()
    RETURN       = auto()
    IF           = auto()
    ELIF         = auto()
    ELSE         = auto()
    FOR          = auto()
    IN           = auto()
    WHILE        = auto()
    DO           = auto()
    BREAK        = auto()
    CONTINUE     = auto()
    STRUCT       = auto()
    TRUE         = auto()
    FALSE        = auto()
    NULL         = auto()
    NEW          = auto()
    PRINT        = auto()
    AND          = auto()
    OR           = auto()
    NOT          = auto()
    ENUM         = auto()
    MATCH        = auto()
    CASE         = auto()
    DEFAULT      = auto()
    IMPORT       = auto()
    ASSERT       = auto()
    TRY          = auto()
    CATCH        = auto()
    FINALLY      = auto()
    THROW        = auto()
    TYPE         = auto()
    AS           = auto()
    INTERFACE    = auto()
    IMPL         = auto()
    DEFER        = auto()
    ASYNC        = auto()
    AWAIT        = auto()
    YIELD        = auto()
    PUB          = auto()
    PRIV         = auto()
    TEST         = auto()
    COMPTIME     = auto()
    EXTERN       = auto()
    UNSAFE       = auto()
    WHERE        = auto()
    RAISE        = auto()

    # Arithmetic operators
    PLUS         = auto()
    MINUS        = auto()
    STAR         = auto()
    SLASH        = auto()
    PERCENT      = auto()
    STAR_STAR    = auto()   # **  exponent

    # Compound assignment
    PLUS_ASSIGN  = auto()   # +=
    MINUS_ASSIGN = auto()   # -=
    STAR_ASSIGN  = auto()   # *=
    SLASH_ASSIGN = auto()   # /=
    PERCENT_ASSIGN = auto() # %=
    AMP_ASSIGN   = auto()   # &=
    PIPE_ASSIGN  = auto()   # |=
    CARET_ASSIGN = auto()   # ^=

    # Bitwise operators
    AMPERSAND    = auto()   # &
    PIPE         = auto()   # |
    CARET        = auto()   # ^
    TILDE        = auto()   # ~
    LSHIFT       = auto()   # <<
    RSHIFT       = auto()   # >>

    # Comparison
    EQ           = auto()   # ==
    NEQ          = auto()   # !=
    LT           = auto()   # <
    GT           = auto()   # >
    LTE          = auto()   # <=
    GTE          = auto()   # >=

    # Misc operators
    ASSIGN       = auto()   # =
    ARROW        = auto()   # ->
    ELLIPSIS     = auto()   # ...
    DOTDOT       = auto()   # ..
    DOTDOT_EQ    = auto()   # ..=  inclusive range
    QUESTION     = auto()   # ?
    NULL_COALESCE = auto()  # ??
    AT           = auto()   # @   for attributes

    # Punctuation
    LPAREN       = auto()   # (
    RPAREN       = auto()   # )
    LBRACE       = auto()   # {
    RBRACE       = auto()   # }
    LBRACKET     = auto()   # [
    RBRACKET     = auto()   # ]
    COLON        = auto()   # :
    COMMA        = auto()   # ,
    DOT          = auto()   # .
    SEMICOLON    = auto()   # ;

    # Structure
    NEWLINE      = auto()
    INDENT       = auto()
    DEDENT       = auto()
    EOF          = auto()


KEYWORDS: dict[str, TT] = {
    "fn":        TT.FN,
    "let":       TT.LET,
    "const":     TT.CONST,
    "return":    TT.RETURN,
    "if":        TT.IF,
    "elif":      TT.ELIF,
    "else":      TT.ELSE,
    "for":       TT.FOR,
    "in":        TT.IN,
    "while":     TT.WHILE,
    "do":        TT.DO,
    "break":     TT.BREAK,
    "continue":  TT.CONTINUE,
    "struct":    TT.STRUCT,
    "true":      TT.TRUE,
    "false":     TT.FALSE,
    "null":      TT.NULL,
    "new":       TT.NEW,
    "print":     TT.PRINT,
    "and":       TT.AND,
    "or":        TT.OR,
    "not":       TT.NOT,
    "enum":      TT.ENUM,
    "match":     TT.MATCH,
    "case":      TT.CASE,
    "default":   TT.DEFAULT,
    "import":    TT.IMPORT,
    "assert":    TT.ASSERT,
    "try":       TT.TRY,
    "catch":     TT.CATCH,
    "finally":   TT.FINALLY,
    "throw":     TT.THROW,
    "raise":     TT.RAISE,
    "type":      TT.TYPE,
    "as":        TT.AS,
    "interface": TT.INTERFACE,
    "impl":      TT.IMPL,
    "defer":     TT.DEFER,
    "async":     TT.ASYNC,
    "await":     TT.AWAIT,
    "yield":     TT.YIELD,
    "pub":       TT.PUB,
    "priv":      TT.PRIV,
    "test":      TT.TEST,
    "comptime":  TT.COMPTIME,
    "extern":    TT.EXTERN,
    "unsafe":    TT.UNSAFE,
    "where":     TT.WHERE,
}


@dataclass
class Token:
    type:  TT
    value: object
    line:  int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.line   = 1
        self.tokens: list[Token] = []
        self._indent_stack = [0]
        self._paren_depth  = 0   # suppress structure tokens inside ( ) [ ] { }

    # ------------------------------------------------------------------ #
    #  Preprocessing (f-strings and triple-quoted strings)               #
    # ------------------------------------------------------------------ #

    def _preprocess(self, source: str) -> str:
        """
        Two passes over the raw source:
          1. Replace triple-quoted strings with single-line equivalents
             (literal newlines become the two-char sequence \\n so the
             normal escape handler turns them back into real newlines).
          2. Replace f"..." with a parenthesised concatenation expression.
        """
        result = []
        i, n = 0, len(source)
        while i < n:
            # ---- triple-quoted string: """...""" ----
            if source[i:i+3] == '"""':
                j = i + 3
                buf = []
                while j <= n - 3:
                    if source[j:j+3] == '"""':
                        j += 3
                        break
                    c = source[j]
                    if c == '\n':
                        buf.append('\\n')
                    elif c == '"':
                        buf.append('\\"')
                    elif c == '\\':
                        buf.append('\\\\')
                    else:
                        buf.append(c)
                    j += 1
                else:
                    j = n  # unterminated — will surface as lex error
                result.append('"' + ''.join(buf) + '"')
                i = j
                continue

            # ---- f-string: f"..." ----
            if source[i] == 'f' and i + 1 < n and source[i + 1] == '"':
                j = i + 2
                segments: list[tuple[str, str]] = []
                text_buf: list[str] = []

                while j < n and source[j] != '"':
                    if source[j] == '\\' and j + 1 < n:
                        esc = source[j + 1]
                        text_buf.append({'n': '\\n', 't': '\\t',
                                         '\\': '\\\\', '"': '\\"'}.get(esc, esc))
                        j += 2
                    elif source[j] == '{':
                        if text_buf:
                            segments.append(('text', ''.join(text_buf)))
                            text_buf = []
                        j += 1
                        depth = 1
                        expr_buf: list[str] = []
                        while j < n and depth > 0:
                            if source[j] == '{':
                                depth += 1
                            elif source[j] == '}':
                                depth -= 1
                            if depth > 0:
                                expr_buf.append(source[j])
                            j += 1
                        segments.append(('expr', ''.join(expr_buf).strip()))
                    else:
                        text_buf.append(source[j])
                        j += 1

                if text_buf:
                    segments.append(('text', ''.join(text_buf)))
                if j < n:
                    j += 1  # consume closing "

                parts = []
                for seg_type, seg_val in segments:
                    if seg_type == 'text':
                        parts.append('"' + seg_val + '"')
                    else:
                        parts.append('str(' + seg_val + ')')

                if not parts:
                    result.append('""')
                elif len(parts) == 1 and segments[0][0] == 'text':
                    result.append(parts[0])
                else:
                    result.append('(' + ' + '.join(parts) + ')')
                i = j
                continue

            result.append(source[i])
            i += 1
        return ''.join(result)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def tokenize(self) -> list[Token]:
        source = self._preprocess(self.source)
        for line_text in source.splitlines(keepends=True):
            self._process_line(line_text)
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self.tokens.append(Token(TT.DEDENT, None, self.line))
        self.tokens.append(Token(TT.EOF, None, self.line))
        return self.tokens

    # ------------------------------------------------------------------ #
    #  Line processing                                                     #
    # ------------------------------------------------------------------ #

    def _process_line(self, line: str):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            self.line += 1
            return

        # Indentation is only meaningful at paren_depth == 0
        if self._paren_depth == 0:
            indent  = len(line) - len(line.lstrip(" "))
            current = self._indent_stack[-1]
            if indent > current:
                self._indent_stack.append(indent)
                self.tokens.append(Token(TT.INDENT, None, self.line))
            else:
                while indent < self._indent_stack[-1]:
                    self._indent_stack.pop()
                    self.tokens.append(Token(TT.DEDENT, None, self.line))
                if indent != self._indent_stack[-1]:
                    raise LexError(f"Line {self.line}: inconsistent indentation")

        pos     = len(line) - len(line.lstrip(" "))
        content = line.rstrip("\n\r")
        while pos < len(content):
            tok, pos = self._next_token(content, pos)
            if tok is not None:
                self.tokens.append(tok)

        if self._paren_depth == 0:
            self.tokens.append(Token(TT.NEWLINE, None, self.line))
        self.line += 1

    # ------------------------------------------------------------------ #
    #  Token scanning                                                      #
    # ------------------------------------------------------------------ #

    def _next_token(self, s: str, pos: int):
        """Return (Token | None, new_pos)."""
        if s[pos] == " " or s[pos] == "\t":
            return None, pos + 1

        if s[pos] == "#":
            return None, len(s)

        ch = s[pos]

        # Char literals: 'a'  '\n'  '\t'  '\\'  '\''
        if ch == "'":
            i = pos + 1
            if i < len(s):
                if s[i] == '\\' and i + 1 < len(s):
                    esc = s[i + 1]
                    val = {'n': '\n', 't': '\t', '\\': '\\', "'": "'"}.get(esc, esc)
                    i += 2
                else:
                    val = s[i]
                    i += 1
                if i < len(s) and s[i] == "'":
                    return Token(TT.CHAR, val, self.line), i + 1
            raise LexError(f"Line {self.line}: unterminated char literal")

        # String literals (double-quoted)
        if ch == '"':
            i = pos + 1
            buf = []
            while i < len(s):
                c = s[i]
                if c == '\\' and i + 1 < len(s):
                    esc = s[i + 1]
                    buf.append({'n': '\n', 't': '\t', 'r': '\r', '\\': '\\',
                                '"': '"', '0': '\0'}.get(esc, esc))
                    i += 2
                elif c == '"':
                    break
                else:
                    buf.append(c)
                    i += 1
            return Token(TT.STRING, ''.join(buf), self.line), i + 1

        # Numeric literals — hex, binary, octal, float, decimal (with _ separators)
        if ch.isdigit() or (ch == '0' and pos + 1 < len(s) and s[pos + 1] in 'xXbBoO'):
            rest = s[pos:]
            # Hex: 0x...
            m = re.match(r"0[xX][0-9a-fA-F][0-9a-fA-F_]*", rest)
            if m:
                raw = m.group().replace('_', '')
                return Token(TT.INT, int(raw, 16), self.line), pos + m.end()
            # Binary: 0b...
            m = re.match(r"0[bB][01][01_]*", rest)
            if m:
                raw = m.group().replace('_', '')
                return Token(TT.INT, int(raw, 2), self.line), pos + m.end()
            # Octal: 0o...
            m = re.match(r"0[oO][0-7][0-7_]*", rest)
            if m:
                raw = m.group().replace('_', '')
                return Token(TT.INT, int(raw, 8), self.line), pos + m.end()
            # Float with underscore separators
            m = re.match(r"[\d_]+\.[\d_]+", rest)
            if m and re.match(r"\d", m.group()):
                raw = m.group().replace('_', '')
                return Token(TT.FLOAT, float(raw), self.line), pos + m.end()
            # Int with underscore separators
            m = re.match(r"[\d_]+", rest)
            if m and re.match(r"\d", m.group()):
                raw = m.group().replace('_', '')
                return Token(TT.INT, int(raw), self.line), pos + m.end()

        # Identifiers / keywords
        m = re.match(r"[A-Za-z_]\w*", s[pos:])
        if m:
            word = m.group()
            tt   = KEYWORDS.get(word, TT.IDENT)
            return Token(tt, word, self.line), pos + m.end()

        # Three-char operators (check before two-char)
        three = s[pos:pos + 3]
        if three == "...":
            return Token(TT.ELLIPSIS, "...", self.line), pos + 3
        if three == "..=":
            return Token(TT.DOTDOT_EQ, "..=", self.line), pos + 3
        if three == "<<=":
            return Token(TT.AMP_ASSIGN, "<<=", self.line), pos + 3   # reuse slot
        if three == ">>=":
            return Token(TT.PIPE_ASSIGN, ">>=", self.line), pos + 3  # reuse slot

        # Two-char operators
        two = s[pos:pos + 2]
        two_map = {
            "==": TT.EQ,           "!=": TT.NEQ,
            "<=": TT.LTE,          ">=": TT.GTE,
            "->": TT.ARROW,        "..": TT.DOTDOT,
            "+=": TT.PLUS_ASSIGN,  "-=": TT.MINUS_ASSIGN,
            "*=": TT.STAR_ASSIGN,  "/=": TT.SLASH_ASSIGN,
            "%=": TT.PERCENT_ASSIGN,
            "&=": TT.AMP_ASSIGN,   "|=": TT.PIPE_ASSIGN,
            "^=": TT.CARET_ASSIGN,
            "<<": TT.LSHIFT,       ">>": TT.RSHIFT,
            "**": TT.STAR_STAR,
            "??": TT.NULL_COALESCE,
        }
        if two in two_map:
            return Token(two_map[two], two, self.line), pos + 2

        # Single-char tokens
        one_map = {
            "+": TT.PLUS,     "-": TT.MINUS,    "*": TT.STAR,    "/": TT.SLASH,
            "%": TT.PERCENT,  "=": TT.ASSIGN,   "<": TT.LT,      ">": TT.GT,
            "(": TT.LPAREN,   ")": TT.RPAREN,   "{": TT.LBRACE,  "}": TT.RBRACE,
            "[": TT.LBRACKET, "]": TT.RBRACKET,
            ":": TT.COLON,    ",": TT.COMMA,    ".": TT.DOT,     "?": TT.QUESTION,
            "&": TT.AMPERSAND, "|": TT.PIPE,    "^": TT.CARET,   "~": TT.TILDE,
            "@": TT.AT,        ";": TT.SEMICOLON,
        }
        if ch in one_map:
            tt = one_map[ch]
            if tt in (TT.LPAREN, TT.LBRACKET, TT.LBRACE):
                self._paren_depth += 1
            elif tt in (TT.RPAREN, TT.RBRACKET, TT.RBRACE):
                self._paren_depth = max(0, self._paren_depth - 1)
            return Token(tt, ch, self.line), pos + 1

        raise LexError(f"Line {self.line}: unexpected character {ch!r}")
