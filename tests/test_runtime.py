"""Integration tests for the Vexel runtime (arrays, strings, control flow)."""
import sys
sys.path.insert(0, ".")

from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.analyzer import Analyzer
from compiler.codegen import Compiler, jit_run


def run(source: str) -> int:
    tokens   = Lexer(source).tokenize()
    program  = Parser(tokens).parse()
    analysis = Analyzer().analyze(program)
    if analysis.errors:
        raise RuntimeError(f"Analysis errors: {analysis.errors}")
    ir = Compiler(analysis).compile(program)
    return jit_run(ir)


def test_arrays_basic():
    run("""
fn main():
    let nums: int[] = [10, 20, 30, 40, 50]
    print(nums[0])
    print(nums[4])
    print(len(nums))
""")
    print("PASS: test_arrays_basic")


def test_array_mutation():
    run("""
fn main():
    let a: int[] = [1, 2, 3]
    a[1] = 99
    print(a[1])
""")
    print("PASS: test_array_mutation")


def test_array_foreach():
    run("""
fn main():
    let nums: int[] = [5, 10, 15]
    for n in nums:
        print(n)
""")
    print("PASS: test_array_foreach")


def test_float_array():
    run("""
fn main():
    let vals: float[] = [1.5, 2.5, 3.5]
    print(vals[0])
    print(vals[2])
    print(len(vals))
""")
    print("PASS: test_float_array")


def test_string_concat():
    run("""
fn main():
    let a: str = "Hello"
    let b: str = " World"
    let c: str = a + b
    print(c)
""")
    print("PASS: test_string_concat")


def test_string_len():
    run("""
fn main():
    let s: str = "Vexel"
    print(len(s))
""")
    print("PASS: test_string_len")


def test_string_eq():
    run("""
fn main():
    let a: str = "hello"
    let b: str = "hello"
    if a == b:
        print("equal")
    else:
        print("not equal")
""")
    print("PASS: test_string_eq")


def test_math_builtins():
    run("""
fn main():
    print(abs(-42))
    print(min(3, 7))
    print(max(3, 7))
    print(sqrt(16.0))
    print(floor(3.9))
    print(ceil(3.1))
""")
    print("PASS: test_math_builtins")


def test_type_casts():
    run("""
fn main():
    let f: float = 3.9
    let i: int = int(f)
    print(i)
    let x: int = 5
    let y: float = float(x)
    print(y)
    let s: str = str(42)
    print(s)
""")
    print("PASS: test_type_casts")


def test_break():
    run("""
fn main():
    let i: int = 0
    while i < 100:
        if i == 5:
            break
        i += 1
    print(i)
""")
    print("PASS: test_break")


def test_continue():
    run("""
fn main():
    let sum: int = 0
    for i in 0..10:
        if i == 5:
            continue
        sum += i
    print(sum)
""")
    print("PASS: test_continue")


def test_compound_assign():
    run("""
fn main():
    let x: int = 10
    x += 5
    print(x)
    x -= 3
    print(x)
    x *= 2
    print(x)
    x /= 4
    print(x)
""")
    print("PASS: test_compound_assign")


def test_elif():
    run("""
fn main():
    let x: int = 0
    if x > 0:
        print("positive")
    elif x < 0:
        print("negative")
    else:
        print("zero")
""")
    print("PASS: test_elif")


def test_multiline_call():
    run("""
fn add(a: int, b: int, c: int) -> int:
    return a + b + c

fn main():
    let r: int = add(
        10,
        20,
        30
    )
    print(r)
""")
    print("PASS: test_multiline_call")


def test_multi_print():
    run("""
fn main():
    let x: int = 42
    print("answer:", x)
""")
    print("PASS: test_multi_print")


def test_global_let():
    run("""
let MAX: int = 1000

fn main():
    print(MAX)
""")
    print("PASS: test_global_let")


def test_short_circuit():
    run("""
fn main():
    let t: bool = true
    let f: bool = false
    print(t and f)
    print(t or f)
    print(not t)
""")
    print("PASS: test_short_circuit")


if __name__ == "__main__":
    test_arrays_basic()
    test_array_mutation()
    test_array_foreach()
    test_float_array()
    test_string_concat()
    test_string_len()
    test_string_eq()
    test_math_builtins()
    test_type_casts()
    test_break()
    test_continue()
    test_compound_assign()
    test_elif()
    test_multiline_call()
    test_multi_print()
    test_global_let()
    test_short_circuit()
    print("\nAll v2 tests passed.")
