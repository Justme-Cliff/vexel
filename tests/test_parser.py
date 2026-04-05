"""Tests for the Vexel parser."""
import sys
sys.path.insert(0, ".")

from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.ast_nodes import *


def parse(source):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def test_hello_world():
    prog = parse('fn main():\n    print("Hello")\n')
    assert len(prog.declarations) == 1
    fn = prog.declarations[0]
    assert isinstance(fn, FnDecl)
    assert fn.name == "main"
    assert len(fn.body) == 1
    assert isinstance(fn.body[0], PrintStmt)
    assert isinstance(fn.body[0].values[0], StringLiteral)
    assert fn.body[0].values[0].value == "Hello"
    print("PASS: test_hello_world")


def test_let_and_return():
    src = "fn add(a: int, b: int) -> int:\n    let r: int = a + b\n    return r\n"
    prog = parse(src)
    fn = prog.declarations[0]
    assert fn.return_type == "int"
    assert len(fn.params) == 2
    assert fn.params[0].name == "a"
    let_stmt = fn.body[0]
    assert isinstance(let_stmt, LetStmt)
    assert isinstance(let_stmt.value, BinOp)
    assert let_stmt.value.op == "+"
    print("PASS: test_let_and_return")


def test_if_else():
    src = "fn check(x: int):\n    if x > 0:\n        print(x)\n    else:\n        print(x)\n"
    prog = parse(src)
    fn = prog.declarations[0]
    stmt = fn.body[0]
    assert isinstance(stmt, IfStmt)
    assert isinstance(stmt.condition, BinOp)
    assert stmt.condition.op == ">"
    assert stmt.else_body is not None
    print("PASS: test_if_else")


def test_for_loop():
    src = "fn count():\n    for i in 0..10:\n        print(i)\n"
    prog = parse(src)
    fn = prog.declarations[0]
    for_stmt = fn.body[0]
    assert isinstance(for_stmt, ForStmt)
    assert for_stmt.var == "i"
    assert isinstance(for_stmt.start, IntLiteral)
    assert for_stmt.start.value == 0
    assert isinstance(for_stmt.end, IntLiteral)
    assert for_stmt.end.value == 10
    print("PASS: test_for_loop")


def test_struct():
    src = "struct Vec2:\n    x: float\n    y: float\n"
    prog = parse(src)
    struct = prog.declarations[0]
    assert isinstance(struct, StructDecl)
    assert struct.name == "Vec2"
    assert len(struct.fields) == 2
    assert struct.fields[0].name == "x"
    assert struct.fields[0].type_name == "float"
    print("PASS: test_struct")


def test_new_expr():
    src = "fn main():\n    let v: Vec2 = new Vec2(1.0, 2.0)\n"
    prog = parse(src)
    fn = prog.declarations[0]
    let_stmt = fn.body[0]
    assert isinstance(let_stmt.value, NewExpr)
    assert let_stmt.value.type_name == "Vec2"
    assert len(let_stmt.value.args) == 2
    print("PASS: test_new_expr")


def test_field_access():
    src = "fn main():\n    let vx: float = v.x\n"
    prog = parse(src)
    fn = prog.declarations[0]
    let_stmt = fn.body[0]
    assert isinstance(let_stmt.value, FieldAccess)
    assert let_stmt.value.field == "x"
    print("PASS: test_field_access")


def test_while_loop():
    src = "fn main():\n    let x: int = 0\n    while x < 10:\n        x = x + 1\n"
    prog = parse(src)
    fn = prog.declarations[0]
    while_stmt = fn.body[1]
    assert isinstance(while_stmt, WhileStmt)
    assert isinstance(while_stmt.body[0], AssignStmt)
    print("PASS: test_while_loop")


def test_recursion():
    src = "fn fib(n: int) -> int:\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    prog = parse(src)
    fn = prog.declarations[0]
    ret = fn.body[1]
    assert isinstance(ret, ReturnStmt)
    assert isinstance(ret.value, BinOp)
    assert isinstance(ret.value.left, Call)
    assert ret.value.left.func == "fib"
    print("PASS: test_recursion")


if __name__ == "__main__":
    test_hello_world()
    test_let_and_return()
    test_if_else()
    test_for_loop()
    test_struct()
    test_new_expr()
    test_field_access()
    test_while_loop()
    test_recursion()
    print("\nAll parser tests passed.")
