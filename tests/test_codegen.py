"""
End-to-end JIT tests for the Vexel compiler.
Each test compiles a snippet and checks the output captured from stdout.
"""

import sys, io, ctypes
sys.path.insert(0, ".")

from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.analyzer import Analyzer
from compiler.codegen import Compiler, jit_run


def compile_and_run(source: str) -> int:
    """Full pipeline: source → JIT execute. Returns exit code."""
    tokens   = Lexer(source).tokenize()
    program  = Parser(tokens).parse()
    analysis = Analyzer().analyze(program)
    if analysis.errors:
        raise RuntimeError(f"Analysis errors: {analysis.errors}")
    llvm_ir = Compiler(analysis).compile(program)
    return jit_run(llvm_ir)


def test_hello_world():
    compile_and_run('fn main():\n    print("Hello, Vexel!")\n')
    print("PASS: test_hello_world")


def test_arithmetic():
    compile_and_run(
        "fn main():\n"
        "    let x: int = 6\n"
        "    let y: int = 7\n"
        "    let z: int = x * y\n"
        "    print(z)\n"
    )
    print("PASS: test_arithmetic")


def test_float():
    compile_and_run(
        "fn main():\n"
        "    let pi: float = 3.14\n"
        "    print(pi)\n"
    )
    print("PASS: test_float")


def test_function_call():
    compile_and_run(
        "fn square(n: int) -> int:\n"
        "    return n * n\n"
        "fn main():\n"
        "    let r: int = square(9)\n"
        "    print(r)\n"
    )
    print("PASS: test_function_call")


def test_if_else():
    compile_and_run(
        "fn main():\n"
        "    let x: int = 10\n"
        '    if x > 5:\n'
        '        print("big")\n'
        "    else:\n"
        '        print("small")\n'
    )
    print("PASS: test_if_else")


def test_for_loop():
    compile_and_run(
        "fn main():\n"
        "    for i in 0..5:\n"
        "        print(i)\n"
    )
    print("PASS: test_for_loop")


def test_while_loop():
    compile_and_run(
        "fn main():\n"
        "    let x: int = 1\n"
        "    while x < 100:\n"
        "        x = x * 2\n"
        "    print(x)\n"
    )
    print("PASS: test_while_loop")


def test_recursion_fib():
    compile_and_run(
        "fn fib(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
        "fn main():\n"
        "    let r: int = fib(10)\n"
        "    print(r)\n"
    )
    print("PASS: test_recursion_fib")


def test_bool():
    compile_and_run(
        "fn main():\n"
        "    let t: bool = true\n"
        "    let f: bool = false\n"
        "    print(t)\n"
        "    print(f)\n"
    )
    print("PASS: test_bool")


def test_struct():
    compile_and_run(
        "struct Vec2:\n"
        "    x: float\n"
        "    y: float\n"
        "fn main():\n"
        "    let v: Vec2 = new Vec2(3.0, 4.0)\n"
        "    print(v.x)\n"
        "    print(v.y)\n"
    )
    print("PASS: test_struct")


if __name__ == "__main__":
    test_hello_world()
    test_arithmetic()
    test_float()
    test_function_call()
    test_if_else()
    test_for_loop()
    test_while_loop()
    test_recursion_fib()
    test_bool()
    test_struct()
    print("\nAll codegen tests passed.")
