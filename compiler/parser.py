"""
Vexel Parser  (v3)
------------------
Recursive descent parser.

New in v2:
  - elif chains
  - break / continue
  - array literals  [1, 2, 3]
  - index assignment  arr[i] = v
  - compound assignment  x += 1  x -= 1  x *= 2  x /= 2
  - for-each  for item in arr:
  - global let / const at top level
  - null literal
  - multi-arg print

New in v3:
  - import statement
  - enum declaration
  - match statement
  - assert statement
  - ternary expression  (cond ? then : else)
  - method call syntax  (obj.method(args))
"""

from compiler.lexer import Token, TT
from compiler.ast_nodes import *


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos    = 0

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, tt: TT) -> Token:
        t = self._cur()
        if t.type != tt:
            raise ParseError(
                f"Line {t.line}: expected {tt.name}, "
                f"got {t.type.name} ({t.value!r})"
            )
        return self._advance()

    def _match(self, *types: TT) -> bool:
        return self._cur().type in types

    def _skip_newlines(self):
        while self._match(TT.NEWLINE):
            self._advance()

    def _eat_newline(self):
        if self._match(TT.NEWLINE):
            self._advance()

    # ------------------------------------------------------------------ #
    #  Top-level                                                           #
    # ------------------------------------------------------------------ #

    def parse(self) -> Program:
        decls = []
        self._skip_newlines()
        while not self._match(TT.EOF):
            decls.append(self._parse_top_level())
            self._skip_newlines()
        return Program(decls)

    def _parse_top_level(self) -> Node:
        if self._match(TT.FN):
            return self._parse_fn()
        if self._match(TT.STRUCT):
            return self._parse_struct()
        if self._match(TT.LET):
            return self._parse_global_let()
        if self._match(TT.CONST):
            return self._parse_global_const()
        if self._match(TT.IMPORT):
            return self._parse_import()
        if self._match(TT.ENUM):
            return self._parse_enum()
        if self._match(TT.TYPE):
            return self._parse_type_alias()
        return self._parse_stmt()

    # ------------------------------------------------------------------ #
    #  Declarations                                                        #
    # ------------------------------------------------------------------ #

    def _parse_fn(self) -> FnDecl:
        self._expect(TT.FN)
        name = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        params = []
        while not self._match(TT.RPAREN):
            pname = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            ptype = self._parse_type()
            default = None
            if self._match(TT.ASSIGN):
                self._advance()
                default = self._parse_expr()
            params.append(Param(pname, ptype, default))
            if self._match(TT.COMMA):
                self._advance()
        self._expect(TT.RPAREN)
        ret = None
        if self._match(TT.ARROW):
            self._advance()
            ret = self._parse_type()
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        return FnDecl(name, params, ret, body)

    def _parse_type_alias(self) -> TypeAlias:
        self._expect(TT.TYPE)
        name = self._expect(TT.IDENT).value
        self._expect(TT.ASSIGN)
        target = self._parse_type()
        self._eat_newline()
        return TypeAlias(name, target)

    def _parse_type(self) -> str:
        """Parse a type annotation: int, float, str, bool, int[], Vec2, etc."""
        name = self._expect(TT.IDENT).value
        if self._match(TT.LBRACKET):
            self._advance()
            self._expect(TT.RBRACKET)
            return name + "[]"
        return name

    def _parse_struct(self) -> StructDecl:
        self._expect(TT.STRUCT)
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        fields = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            fname = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            ftype = self._parse_type()
            fields.append(StructField(fname, ftype))
            self._eat_newline()
        self._expect(TT.DEDENT)
        return StructDecl(name, fields)

    def _parse_global_let(self) -> GlobalLet:
        self._expect(TT.LET)
        name = self._expect(TT.IDENT).value
        ann  = None
        if self._match(TT.COLON):
            self._advance()
            ann = self._parse_type()
        self._expect(TT.ASSIGN)
        value = self._parse_expr()
        self._eat_newline()
        return GlobalLet(name, ann, value)

    def _parse_global_const(self) -> GlobalConst:
        self._expect(TT.CONST)
        name = self._expect(TT.IDENT).value
        ann  = None
        if self._match(TT.COLON):
            self._advance()
            ann = self._parse_type()
        self._expect(TT.ASSIGN)
        value = self._parse_expr()
        self._eat_newline()
        return GlobalConst(name, ann, value)

    def _parse_import(self) -> ImportStmt:
        self._expect(TT.IMPORT)
        path_tok = self._expect(TT.STRING)
        self._eat_newline()
        return ImportStmt(path_tok.value)

    def _parse_enum(self) -> EnumDecl:
        self._expect(TT.ENUM)
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        variants = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            variants.append(self._expect(TT.IDENT).value)
            self._eat_newline()
        self._expect(TT.DEDENT)
        return EnumDecl(name, variants)

    def _parse_block(self) -> list[Node]:
        self._expect(TT.INDENT)
        stmts = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT) or self._match(TT.EOF):
                break
            stmts.append(self._parse_stmt())
        self._expect(TT.DEDENT)
        return stmts

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _parse_stmt(self) -> Node:
        if self._match(TT.LET):
            return self._parse_let()
        if self._match(TT.RETURN):
            return self._parse_return()
        if self._match(TT.PRINT):
            return self._parse_print()
        if self._match(TT.IF):
            return self._parse_if()
        if self._match(TT.FOR):
            return self._parse_for()
        if self._match(TT.WHILE):
            return self._parse_while()
        if self._match(TT.BREAK):
            self._advance(); self._eat_newline()
            return BreakStmt()
        if self._match(TT.CONTINUE):
            self._advance(); self._eat_newline()
            return ContinueStmt()
        if self._match(TT.MATCH):
            return self._parse_match()
        if self._match(TT.ASSERT):
            return self._parse_assert()
        if self._match(TT.TRY):
            return self._parse_try()

        # Expression, assignment, compound assignment, or index assignment
        expr = self._parse_expr()

        # Compound assignment: x += 1  →  AssignStmt(x, BinOp("+", x, 1))
        compound = {
            TT.PLUS_ASSIGN:  "+",
            TT.MINUS_ASSIGN: "-",
            TT.STAR_ASSIGN:  "*",
            TT.SLASH_ASSIGN: "/",
        }
        if self._cur().type in compound:
            op = compound[self._advance().type]
            rhs = self._parse_expr()
            self._eat_newline()
            return AssignStmt(expr, BinOp(op, expr, rhs))

        # Regular assignment
        if self._match(TT.ASSIGN):
            self._advance()
            value = self._parse_expr()
            self._eat_newline()
            # Index assignment: arr[i] = v
            if isinstance(expr, IndexExpr):
                return IndexAssignStmt(expr.obj, expr.index, value)
            return AssignStmt(expr, value)

        self._eat_newline()
        return ExprStmt(expr)

    def _parse_let(self) -> LetStmt:
        self._expect(TT.LET)
        name = self._expect(TT.IDENT).value
        ann  = None
        if self._match(TT.COLON):
            self._advance()
            ann = self._parse_type()
        self._expect(TT.ASSIGN)
        value = self._parse_expr()
        self._eat_newline()
        return LetStmt(name, ann, value)

    def _parse_return(self) -> ReturnStmt:
        self._expect(TT.RETURN)
        if self._match(TT.NEWLINE) or self._match(TT.EOF) or self._match(TT.DEDENT):
            self._eat_newline()
            return ReturnStmt(None)
        value = self._parse_expr()
        self._eat_newline()
        return ReturnStmt(value)

    def _parse_print(self) -> PrintStmt:
        self._expect(TT.PRINT)
        self._expect(TT.LPAREN)
        values = []
        while not self._match(TT.RPAREN):
            values.append(self._parse_expr())
            if self._match(TT.COMMA):
                self._advance()
        self._expect(TT.RPAREN)
        self._eat_newline()
        return PrintStmt(values)

    def _parse_if(self) -> IfStmt:
        self._expect(TT.IF)
        cond = self._parse_expr()
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        then_body = self._parse_block()
        else_body = self._parse_else_chain()
        return IfStmt(cond, then_body, else_body)

    def _parse_else_chain(self):
        """Returns None, or a list of nodes for the else-body.
        Handles elif by wrapping it as a nested IfStmt."""
        if self._match(TT.ELIF):
            self._advance()
            cond = self._parse_expr()
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            body = self._parse_block()
            rest = self._parse_else_chain()
            return [IfStmt(cond, body, rest)]
        if self._match(TT.ELSE):
            self._advance()
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            return self._parse_block()
        return None

    def _parse_for(self) -> Node:
        self._expect(TT.FOR)
        var = self._expect(TT.IDENT).value

        # Enumerate: for i, v in arr
        if self._match(TT.COMMA):
            self._advance()
            val_var = self._expect(TT.IDENT).value
            self._expect(TT.IN)
            iterable = self._parse_expr()
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            body = self._parse_block()
            return ForEnumerate(var, val_var, iterable, body)

        self._expect(TT.IN)
        expr = self._parse_expr()

        if self._match(TT.DOTDOT):
            # Range loop: for i in start..end
            self._advance()
            end = self._parse_expr()
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            body = self._parse_block()
            return ForStmt(var, expr, end, body)
        else:
            # For-each: for item in array
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            body = self._parse_block()
            return ForEach(var, expr, body)

    def _parse_try(self) -> TryCatch:
        self._expect(TT.TRY)
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        try_body = self._parse_block()
        self._expect(TT.CATCH)
        catch_var = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        catch_body = self._parse_block()
        return TryCatch(try_body, catch_var, catch_body)

    def _parse_while(self) -> WhileStmt:
        self._expect(TT.WHILE)
        cond = self._parse_expr()
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        return WhileStmt(cond, body)

    def _parse_match(self) -> MatchStmt:
        self._expect(TT.MATCH)
        value = self._parse_expr()
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        cases = []
        default_body = None
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            if self._match(TT.CASE):
                self._advance()
                patterns = [self._parse_expr()]
                while self._match(TT.COMMA):
                    self._advance()
                    patterns.append(self._parse_expr())
                self._expect(TT.COLON)
                self._expect(TT.NEWLINE)
                body = self._parse_block()
                cases.append(MatchCase(patterns, body))
            elif self._match(TT.DEFAULT):
                self._advance()
                self._expect(TT.COLON)
                self._expect(TT.NEWLINE)
                default_body = self._parse_block()
            else:
                break
        self._expect(TT.DEDENT)
        return MatchStmt(value, cases, default_body)

    def _parse_assert(self) -> AssertStmt:
        self._expect(TT.ASSERT)
        cond = self._parse_expr()
        msg = None
        if self._match(TT.COMMA):
            self._advance()
            msg = self._parse_expr()
        self._eat_newline()
        return AssertStmt(cond, msg)

    # ------------------------------------------------------------------ #
    #  Expressions — precedence climbing                                  #
    # ------------------------------------------------------------------ #

    def _parse_expr(self) -> Node:
        return self._parse_ternary()

    def _parse_ternary(self) -> Node:
        left = self._parse_or()
        if self._match(TT.QUESTION):
            self._advance()
            then_val = self._parse_or()
            self._expect(TT.COLON)
            else_val = self._parse_or()
            return TernaryExpr(left, then_val, else_val)
        return left

    def _parse_or(self) -> Node:
        left = self._parse_and()
        while self._match(TT.OR):
            self._advance()
            left = BinOp("or", left, self._parse_and())
        return left

    def _parse_and(self) -> Node:
        left = self._parse_not()
        while self._match(TT.AND):
            self._advance()
            left = BinOp("and", left, self._parse_not())
        return left

    def _parse_not(self) -> Node:
        if self._match(TT.NOT):
            self._advance()
            return UnaryOp("not", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> Node:
        left = self._parse_addition()
        while self._match(TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE, TT.IN):
            tok  = self._advance()
            op   = "in" if tok.type == TT.IN else tok.value
            left = BinOp(op, left, self._parse_addition())
        return left

    def _parse_addition(self) -> Node:
        left = self._parse_multiplication()
        while self._match(TT.PLUS, TT.MINUS):
            op   = self._advance().value
            left = BinOp(op, left, self._parse_multiplication())
        return left

    def _parse_multiplication(self) -> Node:
        left = self._parse_unary()
        while self._match(TT.STAR, TT.SLASH, TT.PERCENT):
            op   = self._advance().value
            left = BinOp(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Node:
        if self._match(TT.MINUS):
            self._advance()
            return UnaryOp("-", self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self) -> Node:
        expr = self._parse_primary()
        while True:
            if self._match(TT.DOT):
                self._advance()
                field = self._expect(TT.IDENT).value
                if self._match(TT.LPAREN):
                    # method call
                    self._advance()
                    args = []
                    while not self._match(TT.RPAREN):
                        args.append(self._parse_expr())
                        if self._match(TT.COMMA):
                            self._advance()
                    self._expect(TT.RPAREN)
                    expr = MethodCall(expr, field, args)
                else:
                    expr = FieldAccess(expr, field)
            elif self._match(TT.LPAREN) and isinstance(expr, Identifier):
                self._advance()
                args = []
                while not self._match(TT.RPAREN):
                    args.append(self._parse_expr())
                    if self._match(TT.COMMA):
                        self._advance()
                self._expect(TT.RPAREN)
                expr = Call(expr.name, args)
            elif self._match(TT.LBRACKET):
                self._advance()
                idx  = self._parse_expr()
                self._expect(TT.RBRACKET)
                expr = IndexExpr(expr, idx)
            else:
                break
        return expr

    def _parse_primary(self) -> Node:
        t = self._cur()

        if t.type == TT.INT:
            self._advance(); return IntLiteral(t.value)
        if t.type == TT.FLOAT:
            self._advance(); return FloatLiteral(t.value)
        if t.type == TT.STRING:
            self._advance(); return StringLiteral(t.value)
        if t.type == TT.TRUE:
            self._advance(); return BoolLiteral(True)
        if t.type == TT.FALSE:
            self._advance(); return BoolLiteral(False)
        if t.type == TT.NULL:
            self._advance(); return NullLiteral()
        if t.type == TT.IDENT:
            self._advance(); return Identifier(t.value)
        if t.type == TT.NEW:
            self._advance()
            type_name = self._expect(TT.IDENT).value
            self._expect(TT.LPAREN)
            args = []
            while not self._match(TT.RPAREN):
                args.append(self._parse_expr())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
            return NewExpr(type_name, args)
        if t.type == TT.LBRACKET:
            self._advance()
            elems = []
            while not self._match(TT.RBRACKET):
                elems.append(self._parse_expr())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACKET)
            return ArrayLiteral(elems)
        if t.type == TT.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TT.RPAREN)
            return expr

        raise ParseError(
            f"Line {t.line}: unexpected token {t.type.name} ({t.value!r})"
        )
