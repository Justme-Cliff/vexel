"""
Vexel Lexer
-----------
Converts raw source code into a flat list of tokens.

New in v2:
  - ELIF, BREAK, CONTINUE, NULL, CONST keywords
  - PLUS_ASSIGN  +=   MINUS_ASSIGN  -=   STAR_ASSIGN  *=   SLASH_ASSIGN /=
  - Paren-depth tracking: NEWLINE / INDENT / DEDENT are suppressed while
    inside open parentheses, so function calls can span multiple lines.

New in v3:
  - ENUM, MATCH, CASE, DEFAULT, IMPORT, ASSERT keywords
  - QUESTION token for ?
  - { and } also count as paren depth (for multiline dict-style usage)
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    # Literals
    INT          = auto()
    FLOAT        = auto()
    STRING       = auto()

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

    # Arithmetic operators
    PLUS         = auto()
    MINUS        = auto()
    STAR         = auto()
    SLASH        = auto()
    PERCENT      = auto()

    # Compound assignment
    PLUS_ASSIGN  = auto()   # +=
    MINUS_ASSIGN = auto()   # -=
    STAR_ASSIGN  = auto()   # *=
    SLASH_ASSIGN = auto()   # /=

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
    DOTDOT       = auto()   # ..
    QUESTION     = auto()   # ?

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

    # Structure
    NEWLINE      = auto()
    INDENT       = auto()
    DEDENT       = auto()
    EOF          = auto()


KEYWORDS: dict[str, TT] = {
    "fn":       TT.FN,
    "let":      TT.LET,
    "const":    TT.CONST,
    "return":   TT.RETURN,
    "if":       TT.IF,
    "elif":     TT.ELIF,
    "else":     TT.ELSE,
    "for":      TT.FOR,
    "in":       TT.IN,
    "while":    TT.WHILE,
    "break":    TT.BREAK,
    "continue": TT.CONTINUE,
    "struct":   TT.STRUCT,
    "true":     TT.TRUE,
    "false":    TT.FALSE,
    "null":     TT.NULL,
    "new":      TT.NEW,
    "print":    TT.PRINT,
    "and":      TT.AND,
    "or":       TT.OR,
    "not":      TT.NOT,
    "enum":     TT.ENUM,
    "match":    TT.MATCH,
    "case":     TT.CASE,
    "default":  TT.DEFAULT,
    "import":   TT.IMPORT,
    "assert":   TT.ASSERT,
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
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def tokenize(self) -> list[Token]:
        for line_text in self.source.splitlines(keepends=True):
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

        # String literals (double-quoted)
        if ch == '"':
            i = pos + 1
            buf = []
            while i < len(s):
                c = s[i]
                if c == '\\' and i + 1 < len(s):
                    esc = s[i + 1]
                    buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"'}.get(esc, esc))
                    i += 2
                elif c == '"':
                    break
                else:
                    buf.append(c)
                    i += 1
            return Token(TT.STRING, ''.join(buf), self.line), i + 1

        # Float
        m = re.match(r"\d+\.\d+", s[pos:])
        if m:
            return Token(TT.FLOAT, float(m.group()), self.line), pos + m.end()

        # Int
        m = re.match(r"\d+", s[pos:])
        if m:
            return Token(TT.INT, int(m.group()), self.line), pos + m.end()

        # Identifiers / keywords
        m = re.match(r"[A-Za-z_]\w*", s[pos:])
        if m:
            word = m.group()
            tt   = KEYWORDS.get(word, TT.IDENT)
            return Token(tt, word, self.line), pos + m.end()

        # Three-char (none yet, but placeholder)

        # Two-char operators — check BEFORE single-char
        two = s[pos:pos + 2]
        two_map = {
            "==": TT.EQ,          "!=": TT.NEQ,
            "<=": TT.LTE,         ">=": TT.GTE,
            "->": TT.ARROW,       "..": TT.DOTDOT,
            "+=": TT.PLUS_ASSIGN, "-=": TT.MINUS_ASSIGN,
            "*=": TT.STAR_ASSIGN, "/=": TT.SLASH_ASSIGN,
        }
        if two in two_map:
            return Token(two_map[two], two, self.line), pos + 2

        # Single-char tokens
        one_map = {
            "+": TT.PLUS,    "-": TT.MINUS,    "*": TT.STAR,   "/": TT.SLASH,
            "%": TT.PERCENT, "=": TT.ASSIGN,   "<": TT.LT,     ">": TT.GT,
            "(": TT.LPAREN,  ")": TT.RPAREN,   "{": TT.LBRACE, "}": TT.RBRACE,
            "[": TT.LBRACKET,"]": TT.RBRACKET,
            ":": TT.COLON,   ",": TT.COMMA,    ".": TT.DOT,    "?": TT.QUESTION,
        }
        if ch in one_map:
            tt = one_map[ch]
            if tt in (TT.LPAREN, TT.LBRACKET, TT.LBRACE):
                self._paren_depth += 1
            elif tt in (TT.RPAREN, TT.RBRACKET, TT.RBRACE):
                self._paren_depth = max(0, self._paren_depth - 1)
            return Token(tt, ch, self.line), pos + 1

        raise LexError(f"Line {self.line}: unexpected character {ch!r}")
