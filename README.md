# Vexel

A compiled programming language with Python-like syntax that compiles to native machine code via LLVM.

Write code that reads like English. Run it at native speed.

```vexel
fn main():
    let name: str = "World"
    print(f"Hello, {name}!")

    let nums: int[] = [1, 2, 3, 4, 5]
    nums.push(6)

    for i, n in nums:
        print(f"{i}: {n * n}")
```

## Features

- **Clean, readable syntax** — indentation-based blocks, no braces or semicolons
- **Static types** — `int`, `float`, `bool`, `str`, arrays, dicts, tuples, structs, enums
- **Interfaces** — vtable-based polymorphism with `interface` / `impl`
- **Generics** — monomorphized generic functions `fn first[T](arr: T[]) -> T`
- **Closures** — first-class functions and lambdas `fn(x: int) -> int`
- **Null safety** — nullable types `T?`, null checks
- **Modules** — namespace imports `import "lib.vx" as math`
- **Compiles to native binaries** — via LLVM + gcc
- **Fast JIT mode** — `vexel run` executes instantly without a separate compile step
- **VS Code syntax highlighting** — included in `vexel-vscode/`

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

On Windows a `vexel.bat` is included — add the project folder to PATH and run `vexel` from anywhere.

**VS Code syntax highlighting:**

1. Download the file `vexel-vscode/vexel-1.0.0.vsix` from this repo
2. Open VS Code
3. Press `Ctrl+Shift+P` and type `Extensions: Install from VSIX...`
4. Select the downloaded `vexel-1.0.0.vsix` file
5. Reload VS Code when prompted

Your `.vx` files will now have syntax highlighting.

## Usage

```bash
vexel run   hello.vx           # JIT compile and run immediately
vexel compile hello.vx         # compile to hello.exe / hello
vexel compile hello.vx --sdl2  # compile with SDL2 for graphics
vexel ir    hello.vx           # print LLVM IR
vexel parse hello.vx           # print AST
vexel lex   hello.vx           # print tokens
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
| `dict[K, V]` | hash map | `dict[str, int]` |
| `(T1, T2)` | tuple | `(int, float)` |
| `T?` | nullable T | `int?`, `str?` |
| `fn(T) -> R` | function type | `fn(int) -> bool` |
| `MyStruct` | user-defined struct | — |
| `MyInterface` | interface type | — |

### Variables

```vexel
let x: int = 10
let name: str = "Vexel"
let nums: int[] = [1, 2, 3]
let ratio: float = 3.14
const MAX: int = 100            # top-level constant
type Speed = float              # type alias
```

### String interpolation and multi-line strings

```vexel
let x: int = 42
print(f"The answer is {x}!")    # f-strings

let text: str = """
Line one
Line two
Line three
"""
```

### Functions

```vexel
fn add(a: int, b: int) -> int:
    return a + b

fn greet(name: str = "World"):  # default parameters
    print(f"Hello, {name}!")

fn sum(...nums: int[]) -> int:  # variadic
    let total: int = 0
    for n in nums:
        total += n
    return total

fn min_max(arr: int[]) -> (int, int):  # multi-return via tuple
    return (arr[0], arr[arr.len() - 1])

let (lo, hi): (int, int) = min_max([3, 1, 4, 1, 5])
```

### Generics

```vexel
fn first[T](arr: T[]) -> T:
    return arr[0]

fn last[T](arr: T[]) -> T:
    return arr[arr.len() - 1]

let x: int = first([10, 20, 30])      # 10
let s: str = last(["a", "b", "c"])    # "c"
```

### Closures and first-class functions

```vexel
let double: fn(int) -> int = fn(x: int) -> int:
    return x * 2

fn apply(arr: int[], f: fn(int) -> int) -> int[]:
    let result: int[] = []
    for x in arr:
        result.push(f(x))
    return result

let doubled: int[] = apply([1, 2, 3, 4, 5], double)
```

### Interfaces

```vexel
interface Shape:
    fn area(self) -> float
    fn name(self) -> str

struct Circle:
    radius: float

struct Rect:
    width: float
    height: float

impl Shape for Circle:
    fn area(self) -> float:
        return 3.14159 * self.radius * self.radius
    fn name(self) -> str:
        return "Circle"

impl Shape for Rect:
    fn area(self) -> float:
        return self.width * self.height
    fn name(self) -> str:
        return "Rect"

fn describe(s: Shape):
    print(s.name() + " has area " + str(s.area()))

fn main():
    let c: Shape = new Circle(5.0)
    let r: Shape = new Rect(4.0, 3.0)
    describe(c)    # Circle has area 78.5397
    describe(r)    # Rect has area 12
```

### Pattern matching on types

Works on interface values — checks the concrete type and binds fields:

```vexel
fn classify(s: Shape) -> str:
    match s:
        case Circle(radius):
            return "circle with radius " + str(radius)
        case Rect(w, h):
            return "rect " + str(w) + "x" + str(h)
    return "unknown"
```

### Null safety

```vexel
let maybe: int? = null
if maybe != null:
    print(maybe)

let val: str? = "hello"
print(val)   # prints: hello
```

### Modules and namespaces

```vexel
import "mathlib.vx" as math

let x: int = math.square(5)    # 25
let y: float = math.average(3.0, 7.0)  # 5.0
```

### Dictionaries

```vexel
let scores: dict[str, int] = {}
scores["Alice"] = 95
scores["Bob"]   = 87
print(scores["Alice"])          # 95
print(scores.has("Bob"))        # true
let keys: str[] = scores.keys()
scores.remove("Bob")
print(scores.len())             # 1
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

# for-each with index
for i, item in my_array:
    print(f"{i}: {item}")

# while
while x > 0:
    x -= 1

# chained comparisons
if 0 < x < 10:
    print("single digit")

# ternary
let label: str = x > 0 ? "positive" : "non-positive"

# break / continue
for i in 0..100:
    if i == 5:
        break
```

### Match statement

```vexel
# value matching
match status:
    case 0:
        print("ok")
    case 1, 2:
        print("warning")
    default:
        print("unknown")

# type pattern matching (on interface values)
match shape:
    case Circle(r):
        print("radius=" + str(r))
    case Rect(w, h):
        print("width=" + str(w))
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
match dir:
    case Direction.North: print("going north")
    case Direction.South: print("going south")
```

### Arrays

```vexel
let nums: int[] = [10, 20, 30]
nums.push(40)
let last: int = nums.pop()
print(nums.len())
print(nums.contains(20))    # true
nums.reverse()
print(nums[0])
nums[0] = 99

for n in nums:
    print(n)
```

### String methods

```vexel
let s: str = "Hello, World"
print(s.upper())                     # "HELLO, WORLD"
print(s.lower())                     # "hello, world"
print(s.len())                       # 12
print(s.contains("World"))           # true
print(s.starts_with("Hello"))        # true
print(s.ends_with("World"))          # true
print(s.replace("World", "Vexel"))  # "Hello, Vexel"
print(s.trim())                      # strips whitespace
print(s[0])                          # "H"

let parts: str[] = s.split(", ")
```

### Error handling

```vexel
try:
    let content: str = read_file("missing.txt")
    if content == "":
        throw("File empty")
catch e:
    print("Error: " + e)
```

### Math builtins

```vexel
sqrt(x)    floor(x)    ceil(x)    round(x)
abs(x)     pow(a, b)
min(a, b)  max(a, b)   clamp(x, lo, hi)
lerp(a, b, t)          atan2(y, x)
sin(x)     cos(x)      tan(x)
log(x)     log2(x)
rand()                 # float 0.0–1.0
rand_int(a, b)         # int in [a, b]
PI                     # 3.14159...
E                      # 2.71828...
```

### Conversion builtins

```vexel
int(3.9)          # 3
float(5)          # 5.0
str(42)           # "42"
bool(1)           # true
parse_int("42")   # 42
parse_float("3.14")  # 3.14
```

### OS and file builtins

```vexel
read_file("data.txt")
write_file("out.txt", "hello")
append_file("log.txt", "line\n")
file_exists("data.txt")         # bool
os_cwd()                        # current directory as str
os_mkdir("new_dir")             # bool
os_delete("file.txt")           # bool
os_list_dir(".")                # str[]
```

### Time and input

```vexel
let ts: int = time_now()        # Unix timestamp
let line: str = input("Name: ") # read a line from stdin
```

### Operators

```
+  -  *  /  %           arithmetic
==  !=  <  >  <=  >=    comparison (chainable: 0 < x < 10)
and  or  not             logical
+=  -=  *=  /=          compound assignment
->                       return type / function arrow
?  :                     ternary  (cond ? a : b)
..                       range  (0..10)
...                      variadic param prefix
?                        nullable type suffix  (int?)
```

### Assert and exit

```vexel
assert x > 0, "x must be positive"
exit(0)
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

### Higher-order functions

```vexel
fn apply(arr: int[], f: fn(int) -> int) -> int[]:
    let result: int[] = []
    for x in arr:
        result.push(f(x))
    return result

fn main():
    let nums: int[] = [1, 2, 3, 4, 5]
    let doubled: int[] = apply(nums, fn(x: int) -> int:
        return x * 2
    )
    for n in doubled:
        print(n)
```

### Interface polymorphism

```vexel
interface Animal:
    fn speak(self) -> str

struct Dog:
    name: str

struct Cat:
    name: str

impl Animal for Dog:
    fn speak(self) -> str:
        return self.name + " says: woof!"

impl Animal for Cat:
    fn speak(self) -> str:
        return self.name + " says: meow!"

fn make_noise(a: Animal):
    print(a.speak())

fn main():
    let d: Animal = new Dog("Rex")
    let c: Animal = new Cat("Whiskers")
    make_noise(d)
    make_noise(c)
```

### Namespace imports

```vexel
# mathlib.vx
fn square(n: int) -> int: return n * n
fn cube(n: int) -> int:   return n * n * n

# main.vx
import "mathlib.vx" as math

fn main():
    print(math.square(4))   # 16
    print(math.cube(3))     # 27
```

## Project structure

```
compiler/
├── compiler/
│   ├── lexer.py        tokenizer
│   ├── parser.py       recursive descent parser
│   ├── ast_nodes.py    AST node types
│   ├── analyzer.py     type checking and semantic analysis
│   └── codegen.py      LLVM IR code generation
├── stdlib/
│   ├── runtime.c       C runtime helpers
│   └── vx_sdl2.c       SDL2 graphics wrapper
├── vexel-vscode/       VS Code syntax highlighting extension
├── examples/           example Vexel programs
├── tests/              test suite
├── main.py             CLI entry point
└── vexel.bat           Windows launcher
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
