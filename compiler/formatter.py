"""
Vexel source code formatter (#72).

Walks the AST produced by the parser and emits canonically formatted
Vexel source.  The output is deterministic: same AST → same text.
"""

from __future__ import annotations
from compiler.ast_nodes import *


class Formatter:
    """Emit canonical Vexel source from an AST."""

    INDENT = "    "

    def format(self, program: Program) -> str:
        lines: list[str] = []
        for i, decl in enumerate(program.declarations):
            text = self._fmt_node(decl, 0)
            if text:
                lines.append(text)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    #  Dispatch                                                            #
    # ------------------------------------------------------------------ #

    def _fmt_node(self, node: Node, depth: int) -> str:
        ind = self.INDENT * depth
        t   = type(node)

        if t == Program:       return self.format(node)
        if t == FnDecl:        return self._fmt_fn(node, depth)
        if t == StructDecl:    return self._fmt_struct(node, depth)
        if t == EnumDecl:      return self._fmt_enum(node, depth)
        if t == InterfaceDecl: return self._fmt_interface(node, depth)
        if t == ImplDecl:      return self._fmt_impl(node, depth)
        if t == GlobalLet:     return f"{ind}let {node.name}{self._type_ann(node.type_annotation)} = {self._fmt_expr(node.value)}"
        if t == GlobalConst:   return f"{ind}const {node.name}{self._type_ann(node.type_annotation)} = {self._fmt_expr(node.value)}"
        if t == TypeAlias:     return f"{ind}type {node.name} = {node.target}"
        if t == ImportStmt:
            alias = f" as {node.alias}" if node.alias else ""
            return f"{ind}import \"{node.path}\"{alias}"
        if t == NamespaceHint: return ""
        if t == PubDecl:       return f"{ind}pub {self._fmt_node(node.inner, 0).lstrip()}"
        if t == PrivDecl:      return f"{ind}priv {self._fmt_node(node.inner, 0).lstrip()}"
        if t == ComptimeDecl:  return f"{ind}comptime let {node.name} = {self._fmt_expr(node.value)}"
        if t == ExternFnDecl:  return self._fmt_extern_fn(node, depth)
        if t == TestDecl:      return self._fmt_test(node, depth)
        # Statements
        return self._fmt_stmt(node, depth)

    # ------------------------------------------------------------------ #
    #  Declarations                                                        #
    # ------------------------------------------------------------------ #

    def _fmt_fn(self, node: FnDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        params = ", ".join(self._fmt_param(p) for p in node.params)
        tp   = f"[{', '.join(node.type_params)}]" if node.type_params else ""
        if node.named_returns:
            ret = " -> (" + ", ".join(f"{n}: {t}" for n, t in node.named_returns) + ")"
        elif node.return_type:
            ret = f" -> {node.return_type}"
        else:
            ret = ""
        head = f"{ind}fn {node.name}{tp}({params}){ret}:"
        body = self._fmt_body(node.body, depth + 1)
        return head + "\n" + body

    def _fmt_param(self, p: Param) -> str:
        vari = "..." if p.variadic else ""
        s    = f"{vari}{p.name}: {p.type_name}"
        if p.default is not None:
            s += f" = {self._fmt_expr(p.default)}"
        return s

    def _fmt_struct(self, node: StructDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        ind1 = self.INDENT * (depth + 1)
        head = f"{ind}struct {node.name}:"
        fields = []
        for f in node.fields:
            line = f"{ind1}{f.name}: {f.type_name}"
            if f.default is not None:
                line += f" = {self._fmt_expr(f.default)}"
            fields.append(line)
        return head + "\n" + "\n".join(fields)

    def _fmt_enum(self, node: EnumDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        ind1 = self.INDENT * (depth + 1)
        head = f"{ind}enum {node.name}:"
        return head + "\n" + "\n".join(f"{ind1}{v}" for v in node.variants)

    def _fmt_interface(self, node: InterfaceDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        ind1 = self.INDENT * (depth + 1)
        head = f"{ind}interface {node.name}:"
        sigs = []
        for m in node.methods:
            params = ", ".join(self._fmt_param(p) for p in m.params)
            ret    = f" -> {m.return_type}" if m.return_type else ""
            sigs.append(f"{ind1}fn {m.name}({params}){ret}")
        return head + "\n" + "\n".join(sigs)

    def _fmt_impl(self, node: ImplDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        head = f"{ind}impl {node.interface_name} for {node.struct_name}:"
        methods = [self._fmt_fn(m, depth + 1) for m in node.methods]
        return head + "\n" + "\n\n".join(methods)

    def _fmt_extern_fn(self, node: ExternFnDecl, depth: int) -> str:
        ind    = self.INDENT * depth
        params = ", ".join(self._fmt_param(p) for p in node.params)
        ret    = f" -> {node.return_type}" if node.return_type else ""
        return f"{ind}extern fn {node.name}({params}){ret}"

    def _fmt_test(self, node: TestDecl, depth: int) -> str:
        ind  = self.INDENT * depth
        head = f"{ind}test \"{node.name}\":"
        body = self._fmt_body(node.body, depth + 1)
        return head + "\n" + body

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _fmt_body(self, stmts: list, depth: int) -> str:
        lines = []
        for s in stmts:
            t = self._fmt_stmt(s, depth)
            if t:
                lines.append(t)
        return "\n".join(lines) if lines else self.INDENT * depth + "pass"

    def _fmt_stmt(self, node: Node, depth: int) -> str:
        ind = self.INDENT * depth
        t   = type(node)

        if t == LetStmt:
            return f"{ind}let {node.name}{self._type_ann(node.type_annotation)} = {self._fmt_expr(node.value)}"
        if t == TupleUnpack:
            names = ", ".join(node.names)
            anns  = [a or "" for a in node.annotations]
            ann_s = (": (" + ", ".join(anns) + ")") if any(anns) else ""
            return f"{ind}let ({names}){ann_s} = {self._fmt_expr(node.value)}"
        if t == AssignStmt:
            return f"{ind}{self._fmt_expr(node.target)} = {self._fmt_expr(node.value)}"
        if t == IndexAssignStmt:
            return f"{ind}{self._fmt_expr(node.obj)}[{self._fmt_expr(node.index)}] = {self._fmt_expr(node.value)}"
        if t == ReturnStmt:
            if node.value is None:
                return f"{ind}return"
            return f"{ind}return {self._fmt_expr(node.value)}"
        if t == PrintStmt:
            args = ", ".join(self._fmt_expr(v) for v in node.values)
            return f"{ind}print({args})"
        if t == IfStmt:
            return self._fmt_if(node, depth)
        if t == ForStmt:
            return self._fmt_for(node, depth)
        if t == ForEach:
            return self._fmt_foreach(node, depth)
        if t == ForEnumerate:
            body = self._fmt_body(node.body, depth + 1)
            return f"{ind}for {node.idx_var}, {node.val_var} in {self._fmt_expr(node.iterable)}:\n{body}"
        if t == WhileStmt:
            body = self._fmt_body(node.body, depth + 1)
            return f"{ind}while {self._fmt_expr(node.condition)}:\n{body}"
        if t == DoWhileStmt:
            body = self._fmt_body(node.body, depth + 1)
            return f"{ind}do:\n{body}\n{ind}while {self._fmt_expr(node.condition)}:"
        if t == BreakStmt:    return f"{ind}break"
        if t == ContinueStmt: return f"{ind}continue"
        if t == BreakLabel:   return f"{ind}break {node.label}"
        if t == ContinueLabel:return f"{ind}continue {node.label}"
        if t == LabeledStmt:
            return f"{ind}{node.label}:\n{self._fmt_stmt(node.stmt, depth)}"
        if t == ExprStmt:     return f"{ind}{self._fmt_expr(node.expr)}"
        if t == AssertStmt:
            msg = f", {self._fmt_expr(node.message)}" if node.message else ""
            return f"{ind}assert {self._fmt_expr(node.condition)}{msg}"
        if t == DeferStmt:
            kw = "defer_on_error" if node.on_error_only else "defer"
            return f"{ind}{kw} {self._fmt_expr(node.expr)}"
        if t == YieldStmt:
            val = f" {self._fmt_expr(node.value)}" if node.value else ""
            return f"{ind}yield{val}"
        if t == ThrowStmt:    return f"{ind}throw {self._fmt_expr(node.value)}"
        if t == RaiseStmt:    return f"{ind}raise {self._fmt_expr(node.value)}"
        if t == TryCatch:
            tb   = self._fmt_body(node.try_body, depth + 1)
            cb   = self._fmt_body(node.catch_body, depth + 1)
            return f"{ind}try:\n{tb}\n{ind}catch {node.catch_var}:\n{cb}"
        if t == TryCatchFinally:
            return self._fmt_try_catch_finally(node, depth)
        if t == MatchStmt:
            return self._fmt_match(node, depth)
        if t == UnsafeBlock:
            body = self._fmt_body(node.body, depth + 1)
            return f"{ind}unsafe:\n{body}"
        if t == StructDestructure:
            fields = ", ".join(
                (f"{f}: {a}" if a else f) for f, a in zip(node.fields, node.aliases))
            return f"{ind}let {{{fields}}} = {self._fmt_expr(node.value)}"
        if t == ArrayDestructure:
            parts = list(node.names)
            if node.rest_name:
                parts.append(f"...{node.rest_name}")
            return f"{ind}let [{', '.join(parts)}] = {self._fmt_expr(node.value)}"
        # Declarations that can appear inside fn bodies
        if t == FnDecl:    return self._fmt_fn(node, depth)
        return f"{ind}# <unformatted: {type(node).__name__}>"

    def _fmt_if(self, node: IfStmt, depth: int) -> str:
        ind  = self.INDENT * depth
        then = self._fmt_body(node.then_body, depth + 1)
        s    = f"{ind}if {self._fmt_expr(node.condition)}:\n{then}"
        if node.else_body:
            # Nested IfStmt → elif
            if len(node.else_body) == 1 and isinstance(node.else_body[0], IfStmt):
                elif_text = self._fmt_if(node.else_body[0], depth)
                s += "\n" + ind + "el" + elif_text.lstrip()
            else:
                els = self._fmt_body(node.else_body, depth + 1)
                s  += f"\n{ind}else:\n{els}"
        return s

    def _fmt_for(self, node: ForStmt, depth: int) -> str:
        ind  = self.INDENT * depth
        step = f" step {self._fmt_expr(node.step)}" if node.step else ""
        op   = "..=" if node.inclusive else ".."
        body = self._fmt_body(node.body, depth + 1)
        return f"{ind}for {node.var} in {self._fmt_expr(node.start)}{op}{self._fmt_expr(node.end)}{step}:\n{body}"

    def _fmt_foreach(self, node: ForEach, depth: int) -> str:
        ind  = self.INDENT * depth
        body = self._fmt_body(node.body, depth + 1)
        return f"{ind}for {node.var} in {self._fmt_expr(node.iterable)}:\n{body}"

    def _fmt_try_catch_finally(self, node: TryCatchFinally, depth: int) -> str:
        ind  = self.INDENT * depth
        tb   = self._fmt_body(node.try_body, depth + 1)
        s    = f"{ind}try:\n{tb}"
        for cl in node.catches:
            etype = f" {cl.error_type}" if cl.error_type else ""
            cb    = self._fmt_body(cl.body, depth + 1)
            s    += f"\n{ind}catch{etype} as {cl.var}:\n{cb}"
        if node.finally_body:
            fb = self._fmt_body(node.finally_body, depth + 1)
            s += f"\n{ind}finally:\n{fb}"
        return s

    def _fmt_match(self, node: MatchStmt, depth: int) -> str:
        ind  = self.INDENT * depth
        ind1 = self.INDENT * (depth + 1)
        s    = f"{ind}match {self._fmt_expr(node.value)}:\n"
        for case in node.cases:
            pats = ", ".join(self._fmt_expr(p) for p in case.patterns)
            body = self._fmt_body(case.body, depth + 2)
            s   += f"{ind1}case {pats}:\n{body}\n"
        if node.default_body:
            body = self._fmt_body(node.default_body, depth + 2)
            s   += f"{ind1}default:\n{body}\n"
        return s.rstrip()

    # ------------------------------------------------------------------ #
    #  Expressions                                                         #
    # ------------------------------------------------------------------ #

    def _fmt_expr(self, node: Node) -> str:
        t = type(node)
        if t == IntLiteral:    return str(node.value)
        if t == FloatLiteral:  return repr(node.value)
        if t == BoolLiteral:   return "true" if node.value else "false"
        if t == StringLiteral: return '"' + node.value.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if t == CharLiteral:   return f"'{node.value}'"
        if t == NullLiteral:   return "null"
        if t == Identifier:    return node.name
        if t == BinOp:
            l, r = self._fmt_expr(node.left), self._fmt_expr(node.right)
            return f"({l} {node.op} {r})"
        if t == UnaryOp:       return f"{node.op}{self._fmt_expr(node.operand)}"
        if t == TernaryExpr:
            return f"{self._fmt_expr(node.condition)} ? {self._fmt_expr(node.then_val)} : {self._fmt_expr(node.else_val)}"
        if t == Call:
            args = ", ".join(self._fmt_expr(a) for a in node.args)
            return f"{node.func}({args})"
        if t == MethodCall:
            args = ", ".join(self._fmt_expr(a) for a in node.args)
            return f"{self._fmt_expr(node.obj)}.{node.method}({args})"
        if t == FieldAccess:   return f"{self._fmt_expr(node.obj)}.{node.field}"
        if t == IndexExpr:     return f"{self._fmt_expr(node.obj)}[{self._fmt_expr(node.index)}]"
        if t == SliceExpr:
            s   = self._fmt_expr(node.start) if node.start else ""
            e   = self._fmt_expr(node.end)   if node.end   else ""
            op  = "..=" if node.inclusive else ".."
            return f"{self._fmt_expr(node.obj)}[{s}{op}{e}]"
        if t == NewExpr:
            args = ", ".join(self._fmt_expr(a) for a in node.args)
            return f"new {node.type_name}({args})"
        if t == ArrayLiteral:
            return "[" + ", ".join(self._fmt_expr(e) for e in node.elements) + "]"
        if t == DictLiteral:
            pairs = ", ".join(f"{self._fmt_expr(k)}: {self._fmt_expr(v)}" for k, v in node.pairs)
            return "{" + pairs + "}"
        if t == TupleLiteral:
            return "(" + ", ".join(self._fmt_expr(e) for e in node.elements) + ")"
        if t == SpreadExpr:    return f"...{self._fmt_expr(node.value)}"
        if t == LambdaExpr:
            params = ", ".join(self._fmt_param(p) for p in node.params)
            ret    = f" -> {node.ret_type}" if node.ret_type else ""
            body   = self._fmt_body(node.body, 1)
            return f"fn({params}){ret}: {body.strip()}"
        if t == NullCoalesceExpr:
            return f"{self._fmt_expr(node.left)} ?? {self._fmt_expr(node.right)}"
        if t == OptionalChainExpr:
            return f"{self._fmt_expr(node.obj)}?.{node.field}"
        if t == AwaitExpr:     return f"await {self._fmt_expr(node.expr)}"
        if t == PipeExpr:      return f"{self._fmt_expr(node.value)} |> {self._fmt_expr(node.func)}"
        if t == RangeExpr:
            op = "..=" if node.inclusive else ".."
            step = f" step {self._fmt_expr(node.step)}" if node.step else ""
            return f"{self._fmt_expr(node.start)}{op}{self._fmt_expr(node.end)}{step}"
        if t == ListComp:
            cond = f" if {self._fmt_expr(node.condition)}" if node.condition else ""
            return f"[{self._fmt_expr(node.expr)} for {node.var} in {self._fmt_expr(node.iterable)}{cond}]"
        if t == StructUpdateExpr:
            fields = ", ".join(f"{n}: {self._fmt_expr(v)}" for n, v in node.fields)
            return "{" + f"...{self._fmt_expr(node.base)}, {fields}" + "}"
        if t == NamedArg:      return f"{node.name} = {self._fmt_expr(node.value)}"
        if t == ChainedCompare:
            parts = [self._fmt_expr(node.operands[0])]
            for op, operand in zip(node.ops, node.operands[1:]):
                parts.append(op)
                parts.append(self._fmt_expr(operand))
            return " ".join(parts)
        return f"<{type(node).__name__}>"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _type_ann(self, ann: str | None) -> str:
        return f": {ann}" if ann else ""
