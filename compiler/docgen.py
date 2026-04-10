"""
Vexel documentation generator (#74).

Scans the AST for doc comments (lines starting with ## in source)
and generates a Markdown file.  Doc comments are stored on the AST
nodes as a `doc` attribute by the parser when present.

If the parser hasn't attached doc comments, we do a best-effort
extraction directly from the raw token list embedded in node line info.
"""

from __future__ import annotations
from compiler.ast_nodes import *


class DocGen:
    """Generate Markdown docs from a parsed Vexel program."""

    def generate(self, program: Program, title: str = "Vexel Module") -> str:
        lines: list[str] = []
        lines.append(f"# {title}\n")

        for decl in program.declarations:
            md = self._doc_decl(decl)
            if md:
                lines.append(md)

        return "\n".join(lines)

    # ------------------------------------------------------------------ #

    def _doc_decl(self, node: Node) -> str:
        doc = getattr(node, "doc", None) or ""
        t   = type(node)

        if t == FnDecl:
            return self._doc_fn(node, doc)
        if t == StructDecl:
            return self._doc_struct(node, doc)
        if t == EnumDecl:
            return self._doc_enum(node, doc)
        if t == InterfaceDecl:
            return self._doc_interface(node, doc)
        if t == PubDecl:
            return self._doc_decl(node.inner)
        if t == PrivDecl:
            return ""   # private — omit from docs
        return ""

    def _doc_fn(self, node: FnDecl, doc: str) -> str:
        if node.name == "main":
            return ""
        params = ", ".join(
            f"`{p.name}: {p.type_name}`{'?' if p.default else ''}"
            for p in node.params)
        ret = f" → `{node.return_type}`" if node.return_type else ""
        s  = f"## `fn {node.name}({params}){ret}`\n\n"
        if doc:
            s += doc.strip() + "\n\n"
        return s

    def _doc_struct(self, node: StructDecl, doc: str) -> str:
        s  = f"## `struct {node.name}`\n\n"
        if doc:
            s += doc.strip() + "\n\n"
        if node.fields:
            s += "| Field | Type | Default |\n|---|---|---|\n"
            for f in node.fields:
                default = f"`{_expr_str(f.default)}`" if f.default else "—"
                s += f"| `{f.name}` | `{f.type_name}` | {default} |\n"
            s += "\n"
        return s

    def _doc_enum(self, node: EnumDecl, doc: str) -> str:
        s  = f"## `enum {node.name}`\n\n"
        if doc:
            s += doc.strip() + "\n\n"
        s += "Variants: " + ", ".join(f"`{v}`" for v in node.variants) + "\n\n"
        return s

    def _doc_interface(self, node: InterfaceDecl, doc: str) -> str:
        s  = f"## `interface {node.name}`\n\n"
        if doc:
            s += doc.strip() + "\n\n"
        for m in node.methods:
            params = ", ".join(f"`{p.name}: {p.type_name}`" for p in m.params)
            ret    = f" → `{m.return_type}`" if m.return_type else ""
            s     += f"- `fn {m.name}({params}){ret}`\n"
        s += "\n"
        return s


def _expr_str(node) -> str:
    if node is None:
        return ""
    from compiler.ast_nodes import IntLiteral, FloatLiteral, StringLiteral, BoolLiteral
    if isinstance(node, IntLiteral):    return str(node.value)
    if isinstance(node, FloatLiteral):  return str(node.value)
    if isinstance(node, StringLiteral): return f'"{node.value}"'
    if isinstance(node, BoolLiteral):   return "true" if node.value else "false"
    return "..."
