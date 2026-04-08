"""
Vexel recursive-descent parser.

Converts a token stream produced by the Lexer into an AST composed of
the node types defined in ``compiler.ast_nodes``.  Line numbers are
attached to key nodes for use in error messages.
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

    def _peek(self, offset: int = 1) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

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
        # Attributes: @inline fn ...
        if self._match(TT.AT):
            return self._parse_attribute_decorated()
        # Visibility modifiers
        if self._match(TT.PUB):
            self._advance()
            inner = self._parse_top_level()
            return PubDecl(inner)
        if self._match(TT.PRIV):
            self._advance()
            inner = self._parse_top_level()
            return PrivDecl(inner)
        if self._match(TT.FN):
            return self._parse_fn()
        if self._match(TT.ASYNC):
            return self._parse_async_fn()
        if self._match(TT.STRUCT):
            return self._parse_struct()
        if self._match(TT.LET):
            return self._parse_global_let()
        if self._match(TT.CONST):
            return self._parse_global_const()
        if self._match(TT.COMPTIME):
            return self._parse_comptime()
        if self._match(TT.IMPORT):
            return self._parse_import()
        if self._match(TT.ENUM):
            return self._parse_enum_or_adt()
        if self._match(TT.TYPE):
            return self._parse_type_alias()
        if self._match(TT.INTERFACE):
            return self._parse_interface()
        if self._match(TT.IMPL):
            return self._parse_impl()
        if self._match(TT.EXTERN):
            return self._parse_extern_fn()
        if self._match(TT.TEST):
            return self._parse_test()
        return self._parse_stmt()

    # ------------------------------------------------------------------ #
    #  Declarations                                                        #
    # ------------------------------------------------------------------ #

    def _parse_fn(self) -> FnDecl:
        start_line = self._cur().line
        self._expect(TT.FN)
        name = self._expect(TT.IDENT).value

        # Generic type params: fn name[T, U](...)
        type_params = []
        if self._match(TT.LBRACKET):
            self._advance()
            while not self._match(TT.RBRACKET):
                type_params.append(self._expect(TT.IDENT).value)
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACKET)

        self._expect(TT.LPAREN)
        params = []
        while not self._match(TT.RPAREN):
            # Variadic: ...nums: int[]
            variadic = False
            if self._match(TT.ELLIPSIS):
                self._advance()
                variadic = True
            pname = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            ptype = self._parse_type()
            default = None
            if self._match(TT.ASSIGN):
                self._advance()
                default = self._parse_expr()
            params.append(Param(pname, ptype, default, variadic))
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
        node = FnDecl(name, params, ret, body, type_params)
        node.line = start_line
        return node

    def _parse_type_alias(self) -> TypeAlias:
        self._expect(TT.TYPE)
        name = self._expect(TT.IDENT).value
        self._expect(TT.ASSIGN)
        target = self._parse_type()
        self._eat_newline()
        return TypeAlias(name, target)

    def _parse_interface(self) -> InterfaceDecl:
        """interface Name:\n    fn method(self, x: int) -> str\n    ..."""
        start_line = self._cur().line
        self._expect(TT.INTERFACE)
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        methods = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            self._expect(TT.FN)
            mname = self._expect(TT.IDENT).value
            self._expect(TT.LPAREN)
            # Consume optional bare 'self' param
            if self._match(TT.IDENT) and self._cur().value == "self":
                self._advance()
                if self._match(TT.COMMA):
                    self._advance()
            params = []
            while not self._match(TT.RPAREN):
                pname = self._expect(TT.IDENT).value
                self._expect(TT.COLON)
                ptype = self._parse_type()
                params.append(Param(pname, ptype))
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
            ret = None
            if self._match(TT.ARROW):
                self._advance()
                ret = self._parse_type()
            self._eat_newline()
            methods.append(MethodSig(mname, params, ret))
        self._expect(TT.DEDENT)
        node = InterfaceDecl(name, methods)
        node.line = start_line
        return node

    def _parse_impl(self) -> ImplDecl:
        """impl InterfaceName for StructName:\n    fn method(self, ...) -> T:\n        body"""
        start_line = self._cur().line
        self._expect(TT.IMPL)
        iface_name  = self._expect(TT.IDENT).value
        self._expect(TT.FOR)   # reuses existing TT.FOR keyword
        struct_name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        methods = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            methods.append(self._parse_impl_fn(struct_name))
        self._expect(TT.DEDENT)
        node = ImplDecl(iface_name, struct_name, methods)
        node.line = start_line
        return node

    def _parse_impl_fn(self, struct_name: str) -> FnDecl:
        """Parse one method inside an impl block. 'self' gets type=struct_name."""
        start_line = self._cur().line
        self._expect(TT.FN)
        name = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        params = []
        # Consume bare 'self' as first param with type = struct_name
        if self._match(TT.IDENT) and self._cur().value == "self":
            self._advance()
            params.append(Param("self", struct_name))
            if self._match(TT.COMMA):
                self._advance()
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
        node = FnDecl(name, params, ret, body)
        node.line = start_line
        return node

    def _parse_type(self) -> str:
        """Parse a type annotation.

        Supports:
          int, float, str, bool, void
          int[], str[]           (arrays)
          dict[K, V]             (dict)
          (int, float)           (tuple)
          fn(int, float)->str    (function type)
          T?                     (nullable)
          Vec2, MyStruct         (user-defined)
          ns.TypeName            (namespaced struct - stored as ns__TypeName)
        """
        # Tuple type: (int, float, ...)
        if self._match(TT.LPAREN):
            self._advance()
            types = []
            while not self._match(TT.RPAREN):
                types.append(self._parse_type())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
            base = "(" + ",".join(types) + ")"
            if self._match(TT.QUESTION):
                self._advance()
                return base + "?"
            return base

        # Function type: fn(int, float) -> str
        if self._match(TT.FN):
            self._advance()
            self._expect(TT.LPAREN)
            param_types = []
            while not self._match(TT.RPAREN):
                param_types.append(self._parse_type())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
            ret = "void"
            if self._match(TT.ARROW):
                self._advance()
                ret = self._parse_type()
            base = "fn(" + ",".join(param_types) + ")->" + ret
            if self._match(TT.QUESTION):
                self._advance()
                return base + "?"
            return base

        name = self._expect(TT.IDENT).value

        # Namespaced type: ns.TypeName → ns__TypeName
        if self._match(TT.DOT):
            self._advance()
            tname = self._expect(TT.IDENT).value
            name = f"{name}__{tname}"

        # dict[K, V]
        if name == "dict" and self._match(TT.LBRACKET):
            self._advance()
            key_type = self._parse_type()
            self._expect(TT.COMMA)
            val_type = self._parse_type()
            self._expect(TT.RBRACKET)
            base = f"dict[{key_type},{val_type}]"
            if self._match(TT.QUESTION):
                self._advance()
                return base + "?"
            return base

        # Array: int[]
        if self._match(TT.LBRACKET):
            self._advance()
            self._expect(TT.RBRACKET)
            base = name + "[]"
            if self._match(TT.QUESTION):
                self._advance()
                return base + "?"
            return base

        # Nullable: int?
        if self._match(TT.QUESTION):
            self._advance()
            return name + "?"

        return name

    def _parse_struct(self) -> StructDecl:
        self._expect(TT.STRUCT)
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        fields = []
        methods = []
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            if self._match(TT.FN) or (self._match(TT.PUB) and self._peek(1).type == TT.FN) \
                    or (self._match(TT.PRIV) and self._peek(1).type == TT.FN):
                # Method inside struct body
                pub = False
                if self._match(TT.PUB) or self._match(TT.PRIV):
                    pub = self._advance().type == TT.PUB
                method = self._parse_impl_fn(name)
                methods.append(method)
            else:
                fname = self._expect(TT.IDENT).value
                self._expect(TT.COLON)
                ftype = self._parse_type()
                fields.append(StructField(fname, ftype))
                self._eat_newline()
        self._expect(TT.DEDENT)
        node = StructDecl(name, fields)
        node.methods = methods   # attach methods as extra attribute
        return node

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
        alias = None
        if self._match(TT.AS):
            self._advance()
            alias = self._expect(TT.IDENT).value
        self._eat_newline()
        return ImportStmt(path_tok.value, alias)

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

    def _parse_enum_or_adt(self) -> Node:
        """Parse enum — detects ADT (variants with fields) vs plain enum."""
        self._expect(TT.ENUM)
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        self._expect(TT.INDENT)
        variants = []
        is_adt = False
        while not self._match(TT.DEDENT) and not self._match(TT.EOF):
            self._skip_newlines()
            if self._match(TT.DEDENT):
                break
            vname = self._expect(TT.IDENT).value
            fields = []
            if self._match(TT.LPAREN):
                is_adt = True
                self._advance()
                while not self._match(TT.RPAREN):
                    fname = self._expect(TT.IDENT).value
                    self._expect(TT.COLON)
                    ftype = self._parse_type()
                    fields.append(StructField(fname, ftype))
                    if self._match(TT.COMMA):
                        self._advance()
                self._expect(TT.RPAREN)
            variants.append(EnumVariant(vname, fields))
            self._eat_newline()
        self._expect(TT.DEDENT)
        if is_adt or any(v.fields for v in variants):
            return EnumDeclADT(name, variants)
        # Plain enum — fall back to the old representation
        return EnumDecl(name, [v.name for v in variants])

    def _parse_async_fn(self) -> FnDecl:
        """async fn name(...) -> ret: body — parsed as a regular FnDecl for now."""
        self._expect(TT.ASYNC)
        node = self._parse_fn()
        node.name = f"__async_{node.name}"
        return node

    def _parse_extern_fn(self) -> ExternFnDecl:
        """extern fn name(params) -> ret"""
        self._expect(TT.EXTERN)
        self._expect(TT.FN)
        name = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        params = []
        while not self._match(TT.RPAREN):
            pname = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            ptype = self._parse_type()
            params.append(Param(pname, ptype))
            if self._match(TT.COMMA):
                self._advance()
        self._expect(TT.RPAREN)
        ret = None
        if self._match(TT.ARROW):
            self._advance()
            ret = self._parse_type()
        self._eat_newline()
        return ExternFnDecl(name, params, ret)

    def _parse_test(self) -> TestDecl:
        """test "name": body"""
        self._expect(TT.TEST)
        name = self._expect(TT.STRING).value
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        return TestDecl(name, body)

    def _parse_comptime(self) -> ComptimeDecl:
        """comptime let name = expr"""
        self._expect(TT.COMPTIME)
        self._expect(TT.LET)
        name = self._expect(TT.IDENT).value
        self._expect(TT.ASSIGN)
        value = self._parse_expr()
        self._eat_newline()
        return ComptimeDecl(name, value)

    def _parse_attribute_decorated(self) -> Node:
        """@attr_name(args) decl"""
        self._expect(TT.AT)
        attr_name = self._expect(TT.IDENT).value
        args = []
        if self._match(TT.LPAREN):
            self._advance()
            while not self._match(TT.RPAREN):
                args.append(self._parse_expr())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
        self._eat_newline()
        # Parse the decorated declaration and carry attribute as metadata
        decl = self._parse_top_level()
        if isinstance(decl, FnDecl):
            if not hasattr(decl, 'attributes'):
                decl.attributes = []
            decl.attributes.append(AttributeNode(attr_name, args))
        return decl

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
        if self._match(TT.DO):
            return self._parse_do_while()
        if self._match(TT.BREAK):
            self._advance()
            # labeled break: break label
            if self._match(TT.IDENT):
                label = self._advance().value
                self._eat_newline()
                return BreakLabel(label)
            self._eat_newline()
            return BreakStmt()
        if self._match(TT.CONTINUE):
            self._advance()
            # labeled continue: continue label
            if self._match(TT.IDENT):
                label = self._advance().value
                self._eat_newline()
                return ContinueLabel(label)
            self._eat_newline()
            return ContinueStmt()
        if self._match(TT.MATCH):
            return self._parse_match()
        if self._match(TT.ASSERT):
            return self._parse_assert()
        if self._match(TT.TRY):
            return self._parse_try()
        if self._match(TT.DEFER):
            return self._parse_defer()
        if self._match(TT.YIELD):
            return self._parse_yield()
        if self._match(TT.THROW) or self._match(TT.RAISE):
            return self._parse_throw()
        if self._match(TT.UNSAFE):
            return self._parse_unsafe()
        # Labeled loop: identifier followed by colon then a loop keyword (possibly after newlines)
        if self._match(TT.IDENT) and self._peek(1).type == TT.COLON:
            # Look ahead past the colon and any newlines to find a loop keyword
            look = 2
            while self._peek(look).type == TT.NEWLINE:
                look += 1
            if self._peek(look).type in (TT.FOR, TT.WHILE, TT.DO):
                return self._parse_labeled_loop()

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

    def _parse_let(self) -> Node:
        start_line = self._cur().line
        self._expect(TT.LET)

        # Struct destructure: let {x, y} = point
        if self._match(TT.LBRACE):
            self._advance()
            fields = []
            aliases = []
            while not self._match(TT.RBRACE):
                fname = self._expect(TT.IDENT).value
                alias = None
                if self._match(TT.COLON):
                    self._advance()
                    alias = self._expect(TT.IDENT).value
                fields.append(fname)
                aliases.append(alias)
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACE)
            self._expect(TT.ASSIGN)
            value = self._parse_expr()
            self._eat_newline()
            node = StructDestructure(fields, aliases, value)
            node.line = start_line
            return node

        # Array destructure: let [first, second, ...rest] = arr
        if self._match(TT.LBRACKET):
            self._advance()
            names = []
            rest_name = None
            while not self._match(TT.RBRACKET):
                if self._match(TT.ELLIPSIS):
                    self._advance()
                    rest_name = self._expect(TT.IDENT).value
                    break
                names.append(self._expect(TT.IDENT).value)
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACKET)
            self._expect(TT.ASSIGN)
            value = self._parse_expr()
            self._eat_newline()
            node = ArrayDestructure(names, rest_name, value)
            node.line = start_line
            return node

        # Tuple unpack: let (a, b): (int, float) = expr
        if self._match(TT.LPAREN):
            self._advance()
            names = []
            while not self._match(TT.RPAREN):
                names.append(self._expect(TT.IDENT).value)
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RPAREN)
            annotations = [None] * len(names)
            if self._match(TT.COLON):
                self._advance()
                ann_type = self._parse_type()  # e.g. "(int,float)"
                # Parse individual annotations from tuple type string
                if ann_type.startswith("(") and ann_type.endswith(")"):
                    inner = ann_type[1:-1]
                    # Simple split on commas (doesn't handle nested types with commas)
                    parts = self._split_type_list(inner)
                    annotations = parts + [None] * (len(names) - len(parts))
            self._expect(TT.ASSIGN)
            value = self._parse_expr()
            self._eat_newline()
            node = TupleUnpack(names, annotations, value)
            node.line = start_line
            return node

        name = self._expect(TT.IDENT).value
        ann  = None
        if self._match(TT.COLON):
            self._advance()
            ann = self._parse_type()
        self._expect(TT.ASSIGN)
        value = self._parse_expr()
        self._eat_newline()
        node = LetStmt(name, ann, value)
        node.line = start_line
        return node

    def _split_type_list(self, s: str) -> list[str]:
        """Split 'int,float,str' respecting nested parens/brackets."""
        parts = []
        depth = 0
        cur = []
        for c in s:
            if c in ('(', '[', '<'):
                depth += 1
            elif c in (')', ']', '>'):
                depth -= 1
            if c == ',' and depth == 0:
                parts.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(c)
        if cur:
            parts.append(''.join(cur).strip())
        return parts

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

    def _parse_try(self) -> Node:
        """try/catch[es]/finally — supports typed and multi-catch."""
        self._expect(TT.TRY)
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        try_body = self._parse_block()

        catches = []
        finally_body = None

        # Parse one or more catch clauses
        while self._match(TT.CATCH):
            self._advance()
            # Typed catch: catch ErrorType as e: OR plain catch e:
            error_type = None
            if self._match(TT.IDENT):
                # peek ahead — if next is AS, this is a type
                if self._peek(1).type == TT.AS:
                    error_type = self._advance().value
                    self._advance()  # consume 'as'
                var = self._expect(TT.IDENT).value
            else:
                var = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            catch_body = self._parse_block()
            catches.append(CatchClause(error_type, var, catch_body))

        if self._match(TT.FINALLY):
            self._advance()
            self._expect(TT.COLON)
            self._expect(TT.NEWLINE)
            finally_body = self._parse_block()

        if not catches:
            # backward-compat: bare try without catch — treat as try/catch all
            return TryCatchFinally(try_body, [], finally_body)
        return TryCatchFinally(try_body, catches, finally_body)

    def _parse_do_while(self) -> DoWhileStmt:
        """do: body while condition:"""
        self._expect(TT.DO)
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        self._expect(TT.WHILE)
        cond = self._parse_expr()
        # Optionally consume trailing colon (while cond:)
        if self._match(TT.COLON):
            self._advance()
        self._eat_newline()
        return DoWhileStmt(body, cond)

    def _parse_defer(self) -> DeferStmt:
        """defer expr  — or  defer print(...) / defer call()"""
        self._expect(TT.DEFER)
        # If next token is print keyword, parse as print statement stored in defer
        if self._match(TT.PRINT):
            stmt = self._parse_print()
            return DeferStmt(stmt)
        expr = self._parse_expr()
        self._eat_newline()
        return DeferStmt(expr)

    def _parse_yield(self) -> YieldStmt:
        """yield [expr]"""
        self._expect(TT.YIELD)
        if self._match(TT.NEWLINE) or self._match(TT.EOF):
            self._eat_newline()
            return YieldStmt(None)
        value = self._parse_expr()
        self._eat_newline()
        return YieldStmt(value)

    def _parse_throw(self) -> ThrowStmt:
        """throw expr  or  raise expr"""
        self._advance()   # consume throw/raise
        value = self._parse_expr()
        self._eat_newline()
        return ThrowStmt(value)

    def _parse_unsafe(self) -> UnsafeBlock:
        """unsafe: body"""
        self._expect(TT.UNSAFE)
        self._expect(TT.COLON)
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        return UnsafeBlock(body)

    def _parse_labeled_loop(self) -> LabeledStmt:
        """label: for/while/do ..."""
        label = self._advance().value   # consume label name
        self._expect(TT.COLON)
        # eat optional newlines between label: and the loop keyword
        while self._match(TT.NEWLINE):
            self._advance()
        if self._match(TT.FOR):
            stmt = self._parse_for()
        elif self._match(TT.WHILE):
            stmt = self._parse_while()
        else:
            stmt = self._parse_do_while()
        return LabeledStmt(label, stmt)

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
                patterns = [self._parse_case_pattern()]
                while self._match(TT.COMMA):
                    self._advance()
                    patterns.append(self._parse_case_pattern())
                # Guard condition: case x if x > 0:
                guard = None
                if self._match(TT.IF):
                    self._advance()
                    guard = self._parse_expr()
                self._expect(TT.COLON)
                self._expect(TT.NEWLINE)
                body = self._parse_block()
                if guard is not None:
                    cases.append(MatchCaseGuard(patterns, guard, body))
                else:
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

    def _parse_case_pattern(self) -> Node:
        """Parse a single match case pattern.
        If it looks like  NAME(ident, ident, ...)  where all args are bare
        identifiers, produce a TypePattern; otherwise fall back to a regular
        expression.
        """
        if self._match(TT.IDENT) and self._peek(1).type == TT.LPAREN:
            saved = self.pos
            type_name = self._advance().value
            self._advance()   # LPAREN
            bindings = []
            ok = True
            while not self._match(TT.RPAREN):
                if self._match(TT.IDENT):
                    bindings.append(self._advance().value)
                else:
                    ok = False
                    break
                if self._match(TT.COMMA):
                    self._advance()
            if ok and self._match(TT.RPAREN):
                self._advance()   # consume RPAREN
                return TypePattern(type_name, bindings)
            # Backtrack
            self.pos = saved
        return self._parse_expr()

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
        return self._parse_null_coalesce()

    def _parse_null_coalesce(self) -> Node:
        left = self._parse_ternary()
        while self._match(TT.NULL_COALESCE):
            self._advance()
            right = self._parse_ternary()
            left = NullCoalesceExpr(left, right)
        return left

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
        left = self._parse_bitwise_or()
        _CMP = (TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE, TT.IN)
        while self._match(*_CMP):
            tok  = self._advance()
            op   = "in" if tok.type == TT.IN else tok.value
            right = self._parse_bitwise_or()
            cmp1  = BinOp(op, left, right)
            # Chained comparison: 0 < x < 10  →  (0 < x) and (x < 10)
            if self._match(*_CMP):
                tok2 = self._advance()
                op2  = "in" if tok2.type == TT.IN else tok2.value
                right2 = self._parse_bitwise_or()
                cmp2   = BinOp(op2, right, right2)
                left   = BinOp("and", cmp1, cmp2)
            else:
                left = cmp1
        return left

    def _parse_bitwise_or(self) -> Node:
        left = self._parse_bitwise_xor()
        while self._match(TT.PIPE):
            self._advance()
            left = BinOp("|", left, self._parse_bitwise_xor())
        return left

    def _parse_bitwise_xor(self) -> Node:
        left = self._parse_bitwise_and()
        while self._match(TT.CARET):
            self._advance()
            left = BinOp("^", left, self._parse_bitwise_and())
        return left

    def _parse_bitwise_and(self) -> Node:
        left = self._parse_shift()
        while self._match(TT.AMPERSAND):
            self._advance()
            left = BinOp("&", left, self._parse_shift())
        return left

    def _parse_shift(self) -> Node:
        left = self._parse_addition()
        while self._match(TT.LSHIFT, TT.RSHIFT):
            op = "<<" if self._cur().type == TT.LSHIFT else ">>"
            self._advance()
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
        while self._match(TT.STAR, TT.SLASH, TT.PERCENT, TT.STAR_STAR):
            tok = self._advance()
            op = "**" if tok.type == TT.STAR_STAR else tok.value
            left = BinOp(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Node:
        if self._match(TT.MINUS):
            self._advance()
            return UnaryOp("-", self._parse_unary())
        if self._match(TT.TILDE):
            self._advance()
            return UnaryOp("~", self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self) -> Node:
        expr = self._parse_primary()
        while True:
            # Optional chaining: expr?.field or expr?.method(...)
            if self._match(TT.QUESTION) and self._peek(1).type == TT.DOT:
                self._advance()  # consume ?
                self._advance()  # consume .
                field = self._expect(TT.IDENT).value
                if self._match(TT.LPAREN):
                    self._advance()
                    args = []
                    while not self._match(TT.RPAREN):
                        args.append(self._parse_expr())
                        if self._match(TT.COMMA):
                            self._advance()
                    self._expect(TT.RPAREN)
                    # optional method call — wrap as OptionalChain then call
                    chain = OptionalChainExpr(expr, field)
                    expr = MethodCall(chain, "__opt_call__", args)
                else:
                    expr = OptionalChainExpr(expr, field)
            elif self._match(TT.DOT):
                call_line = self._cur().line
                self._advance()
                field = self._expect(TT.IDENT).value
                if self._match(TT.LPAREN):
                    # method call  (or namespace call: ns.func(...))
                    self._advance()
                    args = []
                    while not self._match(TT.RPAREN):
                        args.append(self._parse_expr())
                        if self._match(TT.COMMA):
                            self._advance()
                    self._expect(TT.RPAREN)
                    mc = MethodCall(expr, field, args)
                    mc.line = call_line
                    expr = mc
                else:
                    expr = FieldAccess(expr, field)
            elif self._match(TT.LPAREN) and isinstance(expr, Identifier):
                call_line = getattr(expr, 'line', 0) or self._cur().line
                self._advance()
                args = []
                while not self._match(TT.RPAREN):
                    # Named argument: name = expr
                    if (self._match(TT.IDENT) and self._peek(1).type == TT.ASSIGN):
                        aname = self._advance().value
                        self._advance()  # consume =
                        aval = self._parse_expr()
                        args.append(NamedArg(aname, aval))
                    else:
                        args.append(self._parse_expr())
                    if self._match(TT.COMMA):
                        self._advance()
                self._expect(TT.RPAREN)
                c = Call(expr.name, args)
                c.line = call_line
                expr = c
            elif self._match(TT.LBRACKET):
                self._advance()
                # Slice: expr[start..end] or expr[start..=end] or expr[..end] or expr[start..]
                if self._match(TT.DOTDOT) or self._match(TT.DOTDOT_EQ):
                    inclusive = self._cur().type == TT.DOTDOT_EQ
                    self._advance()
                    end = None if self._match(TT.RBRACKET) else self._parse_expr()
                    self._expect(TT.RBRACKET)
                    expr = SliceExpr(expr, None, end, inclusive)
                else:
                    start = self._parse_expr()
                    if self._match(TT.DOTDOT) or self._match(TT.DOTDOT_EQ):
                        inclusive = self._cur().type == TT.DOTDOT_EQ
                        self._advance()
                        end = None if self._match(TT.RBRACKET) else self._parse_expr()
                        self._expect(TT.RBRACKET)
                        expr = SliceExpr(expr, start, end, inclusive)
                    else:
                        self._expect(TT.RBRACKET)
                        expr = IndexExpr(expr, start)
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
        if t.type == TT.CHAR:
            self._advance(); return CharLiteral(t.value)
        if t.type == TT.TRUE:
            self._advance(); return BoolLiteral(True)
        if t.type == TT.FALSE:
            self._advance(); return BoolLiteral(False)
        if t.type == TT.NULL:
            self._advance(); return NullLiteral()
        if t.type == TT.IDENT:
            self._advance()
            n = Identifier(t.value); n.line = t.line; return n
        if t.type == TT.AWAIT:
            self._advance()
            expr = self._parse_expr()
            return AwaitExpr(expr)
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
            if self._match(TT.RBRACKET):
                self._advance()
                return ArrayLiteral([])
            first = self._parse_expr()
            # List comprehension: [expr for var in iterable [if cond]]
            if self._match(TT.FOR):
                self._advance()
                var = self._expect(TT.IDENT).value
                self._expect(TT.IN)
                iterable = self._parse_expr()
                condition = None
                if self._match(TT.IF):
                    self._advance()
                    condition = self._parse_expr()
                self._expect(TT.RBRACKET)
                return ListComp(first, var, iterable, condition)
            # Normal array literal
            elems = [first]
            if self._match(TT.COMMA):
                self._advance()
            while not self._match(TT.RBRACKET):
                elems.append(self._parse_expr())
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACKET)
            return ArrayLiteral(elems)

        if t.type == TT.LPAREN:
            self._advance()
            first = self._parse_expr()
            if self._match(TT.COMMA):
                # Tuple literal: (a, b, ...)
                elems = [first]
                while self._match(TT.COMMA):
                    self._advance()
                    if self._match(TT.RPAREN):
                        break
                    elems.append(self._parse_expr())
                self._expect(TT.RPAREN)
                return TupleLiteral(elems)
            self._expect(TT.RPAREN)
            return first

        if t.type == TT.LBRACE:
            self._advance()
            pairs = []
            while not self._match(TT.RBRACE):
                key = self._parse_expr()
                self._expect(TT.COLON)
                val = self._parse_expr()
                pairs.append((key, val))
                if self._match(TT.COMMA):
                    self._advance()
            self._expect(TT.RBRACE)
            return DictLiteral(pairs)

        # Lambda expression: fn(x: int) -> int: return x * 2
        if t.type == TT.FN:
            return self._parse_lambda()

        raise ParseError(
            f"Line {t.line}: unexpected token {t.type.name} ({t.value!r})"
        )

    def _parse_lambda(self) -> LambdaExpr:
        """Parse an anonymous function expression.
        fn(x: int, y: int) -> int:
            return x + y
        Or single-line (body on same line after colon when used as expression context)
        """
        self._expect(TT.FN)
        self._expect(TT.LPAREN)
        params = []
        while not self._match(TT.RPAREN):
            pname = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            ptype = self._parse_type()
            params.append(Param(pname, ptype))
            if self._match(TT.COMMA):
                self._advance()
        self._expect(TT.RPAREN)
        ret = None
        if self._match(TT.ARROW):
            self._advance()
            ret = self._parse_type()
        self._expect(TT.COLON)
        # Single-line lambda: fn(...) -> T: return expr
        if not self._match(TT.NEWLINE):
            stmt = self._parse_stmt()
            return LambdaExpr(params, ret, [stmt])
        self._expect(TT.NEWLINE)
        body = self._parse_block()
        return LambdaExpr(params, ret, body)
