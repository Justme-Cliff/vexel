# Vexel

A compiled programming language with Python-like syntax that compiles to native machine code via LLVM.

Write code that reads like English. Run it at native speed.

```vexel
fn main():
    let name: str = "World"
    print("Hello, " + name + "!")

    let nums: int[] = [1, 2, 3, 4, 5]
    nums.push(6)

    for n in nums:
        print(n * n)
```

## Features

- **Clean, readable syntax** — indentation-based blocks, no braces or semicolons
- **Static types** — `int`, `float`, `bool`, `str`, arrays, structs, enums
- **Compiles to native binaries** — via LLVM + MinGW gcc
- **Fast JIT mode** — `vexel run` executes instantly without a separate compile step
- **Cross-platform** — Windows, Linux, macOS

## Installation

**Requirements:**
- Python 3.10+
- `pip install llvmlite`
- MinGW gcc (Windows: [Strawberry Perl](https://strawberryperl.com/) includes it)

**Setup:**
```
git clone https://github.com/Justme-Cliff/vexel
cd vexel
```

Add the `vexel` folder to your PATH, or run directly with `python main.py`.

On Windows a `vexel.bat` is included — add the project folder to PATH and you can run `vexel` from anywhere.

## Usage

```bash
vexel run   hello.vx           # JIT compile and run immediately
vexel compile hello.vx         # compile to hello.exe
vexel compile hello.vx --sdl2  # compile with SDL2 for graphics
vexel ir    hello.vx           # print LLVM IR (for debugging)
vexel parse hello.vx           # print AST (for debugging)
vexel lex   hello.vx           # print tokens (for debugging)
```

## Language reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | 64-bit integer | `42` |
| `float` | 64-bit double | `3.14` |
| `bool` | boolean | `true` / `false` |
| `str` | UTF-8 string | `"hello"` |
| `T[]` | array of T | `int[]`, `str[]` |
| `MyStruct` | custom struct | user-defined |

### Variables

```vexel
let x: int = 10
let name: str = "Vexel"
let nums: int[] = [1, 2, 3]
const MAX: int = 100        # top-level constant
```

### Functions

```vexel
fn add(a: int, b: int) -> int:
    return a + b

fn greet(name: str):
    print("Hello, " + name)
```

### Control flow

```vexel
# if / elif / else
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")

# range loop
for i in 0..10:
    print(i)

# for-each
for item in my_array:
    print(item)

# while
while x > 0:
    x -= 1

# match
match status:
    case 0:
        print("ok")
    case 1:
        print("error")
    default:
        print("unknown")

# ternary
let label: str = x > 0 ? "positive" : "non-positive"

# break / continue
for i in 0..100:
    if i == 5:
        break
```

### Structs

```vexel
struct Vec2:
    x: float
    y: float

let v: Vec2 = new Vec2(3.0, 4.0)
print(v.x)
```

### Enums

```vexel
enum Direction:
    North
    South
    East
    West

let dir: int = Direction.North
```

### Arrays

```vexel
let nums: int[] = [10, 20, 30]
nums.push(40)           # append
let last: int = nums.pop()   # remove and return last
print(nums.len())       # length
print(nums.contains(20))     # bool
nums.reverse()          # in-place reverse
print(nums[0])          # index access
nums[0] = 99            # index assignment

for n in nums:          # for-each
    print(n)
```

### String methods

```vexel
let s: str = "Hello, World"
print(s.upper())            # "HELLO, WORLD"
print(s.lower())            # "hello, world"
print(s.len())              # 12
print(s.contains("World"))  # true
print(s.starts_with("Hello"))  # true
print(s.ends_with("World"))    # true
print(s.replace("World", "Vexel"))  # "Hello, Vexel"
print(s.trim())             # strips whitespace
print(s[0])                 # "H"

let parts: str[] = s.split(", ")
for p in parts:
    print(p)
```

### Math builtins

```vexel
sqrt(x)     floor(x)    ceil(x)
abs(x)      pow(a, b)
min(a, b)   max(a, b)
sin(x)      cos(x)      tan(x)
log(x)      log2(x)
rand()              # float 0-1
rand_int(a, b)      # int in [a, b]
PI                  # 3.14159...
E                   # 2.71828...
```

### Type casts

```vexel
int(3.9)     # 3
float(5)     # 5.0
str(42)      # "42"
bool(1)      # true
```

### File I/O

```vexel
write_file("out.txt", "hello")
append_file("out.txt", " world")
let content: str = read_file("out.txt")
print(file_exists("out.txt"))    # true
```

### Import

```vexel
import "utils.vx"

# All functions and structs from utils.vx are now available
```

### Assert and exit

```vexel
assert x > 0, "x must be positive"
exit(0)
```

### Operators

```
+  -  *  /  %          arithmetic
==  !=  <  >  <=  >=   comparison
and  or  not            logical
+=  -=  *=  /=         compound assignment
?   :                   ternary  (cond ? a : b)
```

## Examples

### Fibonacci

```vexel
fn fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main():
    for i in 0..10:
        print(fib(i))
```

### Game entity system

```vexel
struct Entity:
    x: float
    y: float
    speed: float

fn move(e: Entity, dx: float, dy: float) -> Entity:
    return new Entity(e.x + dx, e.y + dy, e.speed)

fn main():
    let player: Entity = new Entity(0.0, 0.0, 5.0)
    let moved: Entity = move(player, 10.0, 3.0)
    print(moved.x)
    print(moved.y)
```

### Working with strings

```vexel
fn main():
    let words: str[] = "the quick brown fox".split(" ")
    for w in words:
        print(w.upper())
```

## Project structure

```
compiler/
├── compiler/
│   ├── lexer.py        — tokenizer
│   ├── parser.py       — recursive descent parser
│   ├── ast_nodes.py    — AST node types
│   ├── analyzer.py     — type checking and semantic analysis
│   └── codegen.py      — LLVM IR code generation
├── stdlib/
│   ├── runtime.c       — C runtime (GC, string/array/file helpers)
│   └── vx_sdl2.c       — SDL2 graphics wrapper
├── examples/           — example Vexel programs
├── tests/              — test suite
├── main.py             — CLI
└── vexel.bat           — Windows launcher
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
