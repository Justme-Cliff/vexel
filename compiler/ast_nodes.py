"""
Vexel AST node definitions.

All node types produced by the parser.  Every node carries an optional
``line`` attribute (set by the parser) used for error reporting.
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


# ============================================================
# v7 — New language features
# ============================================================

@dataclass
class CharLiteral(Node):
    """Single character literal: 'a'  Stored as the character string."""
    value: str

@dataclass
class DoWhileStmt(Node):
    """do: body while condition:"""
    body:      List[Node]
    condition: Node

@dataclass
class LabeledStmt(Node):
    """label: loop_stmt  — attaches a name to any loop for break/continue."""
    label: str
    stmt:  Node   # must be a loop node

@dataclass
class BreakLabel(Node):
    """break label — break out of the named loop."""
    label: str

@dataclass
class ContinueLabel(Node):
    """continue label — continue the named loop."""
    label: str

@dataclass
class DeferStmt(Node):
    """defer expr — run expr when the enclosing function returns."""
    expr: Node

@dataclass
class YieldStmt(Node):
    """yield value — produce a value from a generator (future)."""
    value: Optional[Node]

@dataclass
class NullCoalesceExpr(Node):
    """left ?? right — return left if not null, else right."""
    left:  Node
    right: Node

@dataclass
class OptionalChainExpr(Node):
    """obj?.field — return null if obj is null, else obj.field."""
    obj:   Node
    field: str

@dataclass
class ListComp(Node):
    """[expr for var in iterable if condition]"""
    expr:      Node
    var:       str
    iterable:  Node
    condition: Optional[Node]

@dataclass
class StructDestructure(Node):
    """let {x, y} = point  — bind named fields."""
    fields:  List[str]
    aliases: List[Optional[str]]   # renamed bindings: {x: a, y: b}
    value:   Node

@dataclass
class ArrayDestructure(Node):
    """let [first, second, ...rest] = arr"""
    names:    List[str]            # positional names; None = skip
    rest_name: Optional[str]       # name for the rest slice
    value:    Node

@dataclass
class MatchCaseGuard(Node):
    """match case with guard: case x if x > 0: body"""
    patterns: List[Node]
    guard:    Optional[Node]
    body:     List[Node]

@dataclass
class TestDecl(Node):
    """test "name": body  — a test block."""
    name: str
    body: List[Node]

@dataclass
class ExternFnDecl(Node):
    """extern fn name(params) -> ret  — declare a C function."""
    name:        str
    params:      List['Param']
    return_type: Optional[str]

@dataclass
class AttributeNode(Node):
    """@attribute_name(args)  — metadata on the next declaration."""
    name: str
    args: List[Node]

@dataclass
class CatchClause(Node):
    """catch ErrorType as var: body  — typed catch clause."""
    error_type: Optional[str]   # None = catch all
    var:        str
    body:       List[Node]

@dataclass
class TryCatchFinally(Node):
    """try/catch(es)/finally with typed catch clauses."""
    try_body:     List[Node]
    catches:      List[CatchClause]
    finally_body: Optional[List[Node]]

@dataclass
class ThrowStmt(Node):
    """throw expr  — raise an error."""
    value: Node

@dataclass
class RaiseStmt(Node):
    """raise expr  — alias for throw."""
    value: Node

@dataclass
class NamedArg(Node):
    """name = expr  — named argument at a call site."""
    name:  str
    value: Node

@dataclass
class AwaitExpr(Node):
    """await expr  — wait for an async result."""
    expr: Node

@dataclass
class EnumVariant(Node):
    """Single variant in an algebraic enum: Circle(radius: float)"""
    name:   str
    fields: List['StructField']   # may be empty for unit variants

@dataclass
class EnumDeclADT(Node):
    """Algebraic Data Type enum with per-variant payload fields."""
    name:     str
    variants: List[EnumVariant]

@dataclass
class MatchCaseADT(Node):
    """ADT match case: case Circle(r): ..."""
    variant_name: str
    bindings:     List[str]
    guard:        Optional[Node]
    body:         List[Node]

@dataclass
class PubDecl(Node):
    """pub decl  — marks a declaration as public."""
    inner: Node

@dataclass
class PrivDecl(Node):
    """priv decl  — marks a declaration as private."""
    inner: Node

@dataclass
class ComptimeDecl(Node):
    """comptime let name = expr  — compile-time constant."""
    name:  str
    value: Node

@dataclass
class UnsafeBlock(Node):
    """unsafe: body  — unsafe code block."""
    body: List[Node]

@dataclass
class SliceExpr(Node):
    """obj[start..end] or obj[start..=end]  — slice/range index."""
    obj:       Node
    start:     Optional[Node]
    end:       Optional[Node]
    inclusive: bool = False   # ..= vs ..
