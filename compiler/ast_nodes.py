"""
Vexel AST Node Definitions  (v3)
---------------------------------
All node types the parser can produce.
New in v2: ArrayLiteral, ForEach, BreakStmt, ContinueStmt,
           GlobalLet, GlobalConst, NullLiteral.
New in v3: MethodCall, EnumDecl, MatchCase, MatchStmt, AssertStmt,
           ImportStmt, TernaryExpr.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


class Node:
    pass


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


# ============================================================
# Statements
# ============================================================

@dataclass
class LetStmt(Node):
    name: str
    type_annotation: Optional[str]
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
    name: str
    type_name: str

@dataclass
class FnDecl(Node):
    name: str
    params: List[Param]
    return_type: Optional[str]
    body: List[Node]

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
    path: str

@dataclass
class Program(Node):
    declarations: List[Node]
