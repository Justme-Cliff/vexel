"""
Vexel AST Node Definitions  (v5 / v6)
--------------------------------------
All node types the parser can produce.
New in v2: ArrayLiteral, ForEach, BreakStmt, ContinueStmt,
           GlobalLet, GlobalConst, NullLiteral.
New in v3: MethodCall, EnumDecl, MatchCase, MatchStmt, AssertStmt,
           ImportStmt, TernaryExpr.
New in v4: TryCatch, ForEnumerate, TypeAlias.
           Param gains optional default value.
New in v5: LambdaExpr, TupleLiteral, TupleUnpack, NamespaceHint.
           ImportStmt gains optional alias.
           Param gains variadic flag.
           FnDecl gains type_params for generics.
New in v6: InterfaceDecl, MethodSig, ImplDecl, TypePattern.
           Node gains optional `line` attribute (set by parser).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


class Node:
    """Base AST node.  The parser may attach a `line` attribute after construction."""
    line: int = 0   # source line; 0 = unknown


# ============================================================
# Literals
# ============================================================

@dataclass
class IntLiteral(Node):
    value: int

@dataclass
class FloatLiteral(Node):
    value: float

@dataclass
class BoolLiteral(Node):
    value: bool

@dataclass
class StringLiteral(Node):
    value: str

@dataclass
class NullLiteral(Node):
    pass

@dataclass
class ArrayLiteral(Node):
    elements: List[Node]

@dataclass
class DictLiteral(Node):
    pairs: List[tuple]   # list of (key_node, val_node)

@dataclass
class TupleLiteral(Node):
    elements: List[Node]


# ============================================================
# Expressions
# ============================================================

@dataclass
class Identifier(Node):
    name: str

@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node

@dataclass
class UnaryOp(Node):
    op: str
    operand: Node

@dataclass
class Call(Node):
    func: str
    args: List[Node]

@dataclass
class MethodCall(Node):
    obj: Node
    method: str
    args: List[Node]

@dataclass
class FieldAccess(Node):
    obj: Node
    field: str

@dataclass
class NewExpr(Node):
    type_name: str
    args: List[Node]

@dataclass
class IndexExpr(Node):
    obj: Node
    index: Node

@dataclass
class TernaryExpr(Node):
    condition: Node
    then_val: Node
    else_val: Node

@dataclass
class LambdaExpr(Node):
    """Anonymous function: fn(x: int) -> int: return x * 2"""
    params: List['Param']
    ret_type: Optional[str]
    body: List[Node]


# ============================================================
# Statements
# ============================================================

@dataclass
class LetStmt(Node):
    name: str
    type_annotation: Optional[str]
    value: Node

@dataclass
class TupleUnpack(Node):
    """let (a, b): (int, float) = expr"""
    names: List[str]
    annotations: List[Optional[str]]
    value: Node

@dataclass
class AssignStmt(Node):
    target: Node
    value: Node

@dataclass
class IndexAssignStmt(Node):
    obj: Node
    index: Node
    value: Node

@dataclass
class ReturnStmt(Node):
    value: Optional[Node]

@dataclass
class PrintStmt(Node):
    values: List[Node]          # multi-arg: print("x =", x)

@dataclass
class IfStmt(Node):
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]]   # elif desugars into nested IfStmt here

@dataclass
class ForStmt(Node):
    var:   str
    start: Node
    end:   Node
    body:  List[Node]

@dataclass
class ForEach(Node):             # for item in array:
    var:      str
    iterable: Node
    body:     List[Node]

@dataclass
class WhileStmt(Node):
    condition: Node
    body: List[Node]

@dataclass
class BreakStmt(Node):
    pass

@dataclass
class ContinueStmt(Node):
    pass

@dataclass
class ExprStmt(Node):
    expr: Node

@dataclass
class AssertStmt(Node):
    condition: Node
    message: Optional[Node]


# ============================================================
# Declarations
# ============================================================

@dataclass
class Param(Node):
    name:      str
    type_name: str
    default:   Optional[Node] = None
    variadic:  bool = False          # ...param: T[]

@dataclass
class FnDecl(Node):
    name:        str
    params:      List[Param]
    return_type: Optional[str]
    body:        List[Node]
    type_params: List[str] = field(default_factory=list)  # [T, U, ...]

@dataclass
class StructField(Node):
    name: str
    type_name: str

@dataclass
class StructDecl(Node):
    name: str
    fields: List[StructField]

@dataclass
class GlobalLet(Node):
    name: str
    type_annotation: Optional[str]
    value: Node

@dataclass
class GlobalConst(Node):
    name: str
    type_annotation: Optional[str]
    value: Node

@dataclass
class EnumDecl(Node):
    name: str
    variants: List[str]

@dataclass
class MatchCase(Node):
    patterns: List[Node]
    body: List[Node]

@dataclass
class MatchStmt(Node):
    value: Node
    cases: List[MatchCase]
    default_body: Optional[List[Node]]

@dataclass
class ImportStmt(Node):
    path:  str
    alias: Optional[str] = None   # import "math.vx" as math

@dataclass
class NamespaceHint(Node):
    """Injected by resolve_imports() to tell the compiler about namespace aliases."""
    alias: str

@dataclass
class TryCatch(Node):
    try_body:   List[Node]
    catch_var:  str
    catch_body: List[Node]

@dataclass
class ForEnumerate(Node):          # for i, v in arr:
    idx_var:  str
    val_var:  str
    iterable: Node
    body:     List[Node]

@dataclass
class TypeAlias(Node):
    name:   str
    target: str

@dataclass
class Program(Node):
    declarations: List[Node]


# ============================================================
# v6 — Interfaces and implementation blocks
# ============================================================

@dataclass
class MethodSig(Node):
    """Interface method signature — no body.  Does NOT include the 'self' parameter."""
    name:        str
    params:      List[Param]
    return_type: Optional[str]

@dataclass
class InterfaceDecl(Node):
    name:    str
    methods: List[MethodSig]

@dataclass
class ImplDecl(Node):
    interface_name: str
    struct_name:    str
    methods:        List[FnDecl]   # each has 'self' as first Param

@dataclass
class TypePattern(Node):
    """In a match case: Circle(r, c) — checks concrete type and binds fields positionally."""
    type_name: str
    bindings:  List[str]   # variable names for positional fields
