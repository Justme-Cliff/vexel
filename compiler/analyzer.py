"""
Vexel Semantic Analyzer  (v3)
------------------------------
Two-pass analysis:
  Pass 1 — register all function / struct / global / enum signatures.
  Pass 2 — walk bodies, check undefined names, infer types.

New in v2:
  - Array type support ("int[]", "float[]", …)
  - Break / continue validation (must be inside a loop)
  - Built-in functions: len, sqrt, abs, min, max, pow, floor, ceil,
                        int, float, str
  - Global let / const
  - ForEach
  - Multi-arg print

New in v3:
  - Enum declarations
  - Match statement
  - Assert statement
  - Import statement (no-op; handled before analysis)
  - MethodCall type inference
  - TernaryExpr type inference
  - PI / E constants
  - New builtins: exit, read_file, write_file, append_file, file_exists,
                  rand, rand_int, sin, cos, tan, log, log2
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from compiler.ast_nodes import *


VX_INT   = "int"
VX_FLOAT = "float"
VX_BOOL  = "bool"
VX_STR   = "str"
VX_VOID  = "void"
VX_NULL  = "null"

# Built-in functions that the codegen handles without a FnDecl.
BUILTINS: dict[str, tuple[list[str], str]] = {
    # name → ([param_types...], return_type)   (types are just hints, not enforced)
    "len":         (["any"],                    VX_INT),
    "sqrt":        (["float"],                  VX_FLOAT),
    "abs":         (["any"],                    VX_INT),    # overloaded
    "min":         (["any", "any"],             VX_INT),    # overloaded
    "max":         (["any", "any"],             VX_INT),    # overloaded
    "pow":         (["float", "float"],         VX_FLOAT),
    "floor":       (["float"],                  VX_FLOAT),
    "ceil":        (["float"],                  VX_FLOAT),
    "int":         (["any"],                    VX_INT),
    "float":       (["any"],                    VX_FLOAT),
    "str":         (["any"],                    VX_STR),
    "bool":        (["any"],                    VX_BOOL),
    # v3 additions
    "exit":        (["int"],                    VX_VOID),
    "read_file":   (["str"],                    VX_STR),
    "write_file":  (["str", "str"],             VX_VOID),
    "append_file": (["str", "str"],             VX_VOID),
    "file_exists": (["str"],                    VX_BOOL),
    "rand":        ([],                         VX_FLOAT),
    "rand_int":    (["int", "int"],             VX_INT),
    "sin":         (["float"],                  VX_FLOAT),
    "cos":         (["float"],                  VX_FLOAT),
    "tan":         (["float"],                  VX_FLOAT),
    "log":         (["float"],                  VX_FLOAT),
    "log2":        (["float"],                  VX_FLOAT),
}


@dataclass
class FnSig:
    params:      list[tuple[str, str]]
    return_type: str


@dataclass
class AnalysisResult:
    errors:        list[str]
    fn_sigs:       dict[str, FnSig]
    struct_fields: dict[str, list[tuple[str, str]]]
    global_types:  dict[str, str]


class Analyzer:
    def __init__(self):
        self._errors:        list[str]                   = []
        self._fn_sigs:       dict[str, FnSig]            = {}
        self._struct_fields: dict[str, list[tuple[str, str]]] = {}
        self._global_types:  dict[str, str]              = {}
        self._scopes:        list[dict[str, str]]        = [{}]
        self._loop_depth:    int                         = 0

    # ------------------------------------------------------------------ #

    def analyze(self, program: Program) -> AnalysisResult:
        # Inject built-ins
        for name, (params, ret) in BUILTINS.items():
            self._fn_sigs[name] = FnSig([(f"a{i}", t) for i, t in enumerate(params)], ret)

        # Pass 1
        for decl in program.declarations:
            if isinstance(decl, FnDecl):       self._register_fn(decl)
            elif isinstance(decl, StructDecl): self._register_struct(decl)
            elif isinstance(decl, (GlobalLet, GlobalConst)):
                self._register_global(decl)
            elif isinstance(decl, EnumDecl):
                for i, variant in enumerate(decl.variants):
                    key = f"{decl.name}.{variant}"
                    self._global_types[key] = VX_INT
                    self._scopes[0][key]    = VX_INT

        # Pass 2
        for decl in program.declarations:
            if isinstance(decl, FnDecl):
                self._analyze_fn(decl)

        return AnalysisResult(
            self._errors, self._fn_sigs,
            self._struct_fields, self._global_types,
        )

    # ------------------------------------------------------------------ #
    #  Registration                                                        #
    # ------------------------------------------------------------------ #

    def _register_fn(self, d: FnDecl):
        params = [(p.name, p.type_name) for p in d.params]
        self._fn_sigs[d.name] = FnSig(params, d.return_type or VX_VOID)

    def _register_struct(self, d: StructDecl):
        self._struct_fields[d.name] = [(f.name, f.type_name) for f in d.fields]

    def _register_global(self, d):
        t = d.type_annotation or self._infer_expr(d.value)
        self._global_types[d.name] = t
        self._scopes[0][d.name]    = t

    # ------------------------------------------------------------------ #
    #  Scopes                                                              #
    # ------------------------------------------------------------------ #

    def _push(self): self._scopes.append({})
    def _pop(self):  self._scopes.pop()

    def _declare(self, name: str, t: str):
        self._scopes[-1][name] = t

    def _lookup(self, name: str) -> Optional[str]:
        for s in reversed(self._scopes):
            if name in s: return s[name]
        return None

    def _err(self, msg: str): self._errors.append(msg)

    # ------------------------------------------------------------------ #
    #  Functions                                                           #
    # ------------------------------------------------------------------ #

    def _analyze_fn(self, d: FnDecl):
        self._push()
        for p in d.params:
            self._declare(p.name, p.type_name)
        for stmt in d.body:
            self._analyze_stmt(stmt, d.return_type or VX_VOID)
        self._pop()

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _analyze_stmt(self, node: Node, ret_type: str):
        if isinstance(node, LetStmt):
            t = self._infer_expr(node.value)
            self._declare(node.name, node.type_annotation or t)

        elif isinstance(node, AssignStmt):
            if isinstance(node.target, Identifier):
                if self._lookup(node.target.name) is None:
                    self._err(f"Undefined variable '{node.target.name}'")
            self._infer_expr(node.value)

        elif isinstance(node, IndexAssignStmt):
            self._infer_expr(node.obj)
            self._infer_expr(node.index)
            self._infer_expr(node.value)

        elif isinstance(node, ReturnStmt):
            if node.value: self._infer_expr(node.value)

        elif isinstance(node, PrintStmt):
            for v in node.values: self._infer_expr(v)

        elif isinstance(node, IfStmt):
            self._infer_expr(node.condition)
            self._push()
            for s in node.then_body: self._analyze_stmt(s, ret_type)
            self._pop()
            if node.else_body:
                self._push()
                for s in node.else_body: self._analyze_stmt(s, ret_type)
                self._pop()

        elif isinstance(node, ForStmt):
            self._infer_expr(node.start); self._infer_expr(node.end)
            self._push(); self._loop_depth += 1
            self._declare(node.var, VX_INT)
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, ForEach):
            arr_type = self._infer_expr(node.iterable)
            elem_type = arr_type[:-2] if arr_type.endswith("[]") else VX_INT
            self._push(); self._loop_depth += 1
            self._declare(node.var, elem_type)
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, WhileStmt):
            self._infer_expr(node.condition)
            self._push(); self._loop_depth += 1
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, BreakStmt):
            if self._loop_depth == 0:
                self._err("'break' outside loop")

        elif isinstance(node, ContinueStmt):
            if self._loop_depth == 0:
                self._err("'continue' outside loop")

        elif isinstance(node, ExprStmt):
            self._infer_expr(node.expr)

        elif isinstance(node, MatchStmt):
            self._infer_expr(node.value)
            for case in node.cases:
                for pat in case.patterns:
                    self._infer_expr(pat)
                self._push()
                for s in case.body:
                    self._analyze_stmt(s, ret_type)
                self._pop()
            if node.default_body:
                self._push()
                for s in node.default_body:
                    self._analyze_stmt(s, ret_type)
                self._pop()

        elif isinstance(node, AssertStmt):
            self._infer_expr(node.condition)
            if node.message:
                self._infer_expr(node.message)

        elif isinstance(node, (EnumDecl, ImportStmt)):
            pass  # handled in pass 1 or before analysis

    # ------------------------------------------------------------------ #
    #  Expression type inference                                           #
    # ------------------------------------------------------------------ #

    def _infer_expr(self, node: Node) -> str:
        if isinstance(node, IntLiteral):    return VX_INT
        if isinstance(node, FloatLiteral):  return VX_FLOAT
        if isinstance(node, BoolLiteral):   return VX_BOOL
        if isinstance(node, StringLiteral): return VX_STR
        if isinstance(node, NullLiteral):   return VX_NULL

        if isinstance(node, ArrayLiteral):
            if not node.elements: return "int[]"
            et = self._infer_expr(node.elements[0])
            return f"{et}[]"

        if isinstance(node, Identifier):
            # Built-in constants
            if node.name in ("PI", "E"):
                return VX_FLOAT
            t = self._lookup(node.name)
            if t is None:
                self._err(f"Undefined variable '{node.name}'")
                return VX_INT
            return t

        if isinstance(node, BinOp):
            lt = self._infer_expr(node.left)
            rt = self._infer_expr(node.right)
            if node.op in ("==","!=","<",">","<=",">=","and","or"):
                return VX_BOOL
            # str + str → str
            if node.op == "+" and lt == VX_STR: return VX_STR
            if lt == VX_FLOAT or rt == VX_FLOAT: return VX_FLOAT
            return lt

        if isinstance(node, UnaryOp):
            t = self._infer_expr(node.operand)
            return VX_BOOL if node.op == "not" else t

        if isinstance(node, Call):
            sig = self._fn_sigs.get(node.func)
            if sig is None:
                self._err(f"Undefined function '{node.func}'")
                return VX_VOID
            # For overloaded builtins, infer return type from args
            if node.func in ("abs", "min", "max"):
                if node.args:
                    at = self._infer_expr(node.args[0])
                    return VX_FLOAT if at == VX_FLOAT else VX_INT
            for a in node.args: self._infer_expr(a)
            return sig.return_type

        if isinstance(node, MethodCall):
            obj_t = self._infer_expr(node.obj)
            for a in node.args: self._infer_expr(a)
            if obj_t == VX_STR:
                str_methods = {
                    "len":         VX_INT,
                    "upper":       VX_STR,
                    "lower":       VX_STR,
                    "trim":        VX_STR,
                    "contains":    VX_BOOL,
                    "starts_with": VX_BOOL,
                    "ends_with":   VX_BOOL,
                    "replace":     VX_STR,
                    "split":       "str[]",
                }
                return str_methods.get(node.method, VX_STR)
            if obj_t.endswith("[]"):
                elem_t = obj_t[:-2]
                arr_methods = {
                    "len":      VX_INT,
                    "push":     VX_VOID,
                    "pop":      elem_t,
                    "contains": VX_BOOL,
                    "reverse":  VX_VOID,
                }
                return arr_methods.get(node.method, VX_VOID)
            return VX_VOID

        if isinstance(node, TernaryExpr):
            self._infer_expr(node.condition)
            then_t = self._infer_expr(node.then_val)
            self._infer_expr(node.else_val)
            return then_t

        if isinstance(node, FieldAccess):
            # Check if this is an enum access: Color.Red -> look up "Color.Red"
            if isinstance(node.obj, Identifier):
                dotted = f"{node.obj.name}.{node.field}"
                t = self._lookup(dotted)
                if t is not None:
                    return t
            ot = self._infer_expr(node.obj)
            for fn, ft in self._struct_fields.get(ot, []):
                if fn == node.field: return ft
            self._err(f"Type '{ot}' has no field '{node.field}'")
            return VX_INT

        if isinstance(node, NewExpr):
            if node.type_name not in self._struct_fields:
                self._err(f"Unknown struct '{node.type_name}'")
            return node.type_name

        if isinstance(node, IndexExpr):
            ot = self._infer_expr(node.obj)
            self._infer_expr(node.index)
            if ot.endswith("[]"): return ot[:-2]
            if ot == VX_STR: return VX_STR   # string indexing returns single char as str
            return VX_INT

        return VX_INT
