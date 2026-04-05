"""Tests for the Vexel lexer."""
import sys
sys.path.insert(0, ".")

from compiler.lexer import Lexer, TT


def tokens(source):
    return [(t.type, t.value) for t in Lexer(source).tokenize()]


def test_hello_world():
    src = 'fn main():\n    print("Hello, World!")\n'
    toks = tokens(src)
    types = [t for t, _ in toks]
    assert TT.FN in types
    assert TT.IDENT in types
    assert TT.PRINT in types
    assert TT.STRING in types
    assert TT.INDENT in types
    assert TT.DEDENT in types
    assert TT.EOF in types
    print("PASS: test_hello_world")


def test_let_statement():
    toks = tokens("let x: int = 42\n")
    types = [t for t, _ in toks]
    assert TT.LET in types
    assert TT.COLON in types
    assert TT.ASSIGN in types
    assert (TT.INT, 42) in toks
    print("PASS: test_let_statement")


def test_function_with_return():
    src = "fn add(a: int, b: int) -> int:\n    return a\n"
    toks = tokens(src)
    types = [t for t, _ in toks]
    assert TT.ARROW in types
    assert TT.RETURN in types
    print("PASS: test_function_with_return")


def test_operators():
    toks = tokens("x == 1 and y != 2\n")
    types = [t for t, _ in toks]
    assert TT.EQ in types
    assert TT.AND in types
    assert TT.NEQ in types
    print("PASS: test_operators")


def test_float():
    toks = tokens("let pi: float = 3.14\n")
    assert (TT.FLOAT, 3.14) in toks
    print("PASS: test_float")


if __name__ == "__main__":
    test_hello_world()
    test_let_statement()
    test_function_with_return()
    test_operators()
    test_float()
    print("\nAll lexer tests passed.")
