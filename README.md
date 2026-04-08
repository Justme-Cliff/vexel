# Vexel

A compiled programming language with Python-like syntax that compiles to native machine code via LLVM.

Write code that reads like Python. Run it at native speed.

```vexel
fn main():
    let name: str = "World"
    print("Hello, " + name + "!")

    let nums: int[] = [1, 2, 3, 4, 5]
    let doubled: int[] = [x * 2 for x in nums if x > 2]
    print(doubled)

    let uid: str = uuid_v4()
    print(uid)
```

## Features

- **Clean syntax** — indentation-based blocks, no braces or semicolons
- **Static types** — `int`, `float`, `bool`, `str`, `char`, arrays, dicts, tuples, structs, enums
- **Integer variants** — `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`
- **Interfaces** — vtable-based polymorphism with `interface` / `impl`
- **Generics** — monomorphized generic functions `fn first[T](arr: T[]) -> T`
- **Closures with capture** — lambdas close over outer variables
- **Algebraic data types** — ADT enums with per-variant payload fields
- **Match guards** — `case n if n > 0:`
- **Null safety** — nullable types `T?`, null coalescing `??`, optional chaining `?.`
- **Defer** — guaranteed cleanup on scope exit (like Go)
- **Labeled loops** — `break outer` / `continue outer` for nested loops
- **do-while** — body runs at least once
- **Destructuring** — `let {x, y} = point` and `let [a, b, ...rest] = arr`
- **List comprehensions** — `[x * 2 for x in arr if x > 0]`
- **Comptime constants** — `comptime let N = 1024 * 1024`
- **Extern fn** — declare C functions directly
- **pub / priv** — visibility modifiers
- **Unsafe blocks** — escape hatch for low-level code
- **Test framework** — `test "name": body` blocks with `assert_eq`, `assert_true`, etc.
- **Threads** — `thread_spawn` / `thread_join` / `thread_sleep`
- **Mutexes** — `mutex_new` / `mutex_lock` / `mutex_unlock`
- **Atomic operations** — `atomic_new`, `atomic_add`, `atomic_load`, `atomic_compare_swap`
- **Rich stdlib** — base64, UUID, SHA-256, CSV, JSON, HTTP, regex-ready
- **Compiles to native binaries** — via LLVM + gcc
- **Fast JIT mode** — `vexel run` executes instantly
- **VS Code syntax highlighting** — included in `vexel-vscode/`

## Installation

**Requirements:**
- Python 3.10+
- `pip install llvmlite`
- MinGW gcc (Windows: [Strawberry Perl](https://strawberryperl.com/) includes it)

**Setup:**
```
git clone https://github.com/Justme-Cliff/vexel-lang
cd vexel-lang
```

Add the project folder to your PATH, or run directly with `python main.py`.

On Windows a `vexel.bat` is included — add the project folder to PATH and run `vexel` from anywhere.

**VS Code syntax highlighting:**

1. Download `vexel-vscode/vexel-1.0.0.vsix` from this repo
2. Open VS Code → `Ctrl+Shift+P` → `Extensions: Install from VSIX...`
3. Select the `.vsix` file and reload VS Code

## Usage

```bash
vexel run     hello.vx        # JIT compile and run immediately
vexel compile hello.vx        # compile to hello.exe / hello
vexel ir      hello.vx        # print LLVM IR
vexel parse   hello.vx        # print AST
vexel lex     hello.vx        # print tokens
```

## Language Reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | 64-bit integer | `42` |
| `i8` `u8` `i32` `u32` | sized integers | `let b: u8 = 255` |
| `float` | 64-bit double | `3.14` |
| `bool` | boolean | `true` / `false` |
| `char` | 8-bit character | `let c: char = 65` |
| `str` | null-terminated string | `"hello"` |
| `T[]` | array of T | `int[]`, `str[]` |
| `dict[K, V]` | hash map | `dict[str, int]` |
| `(T1, T2)` | tuple | `(int, float)` |
| `T?` | nullable T | `int?`, `str?` |
| `fn(T) -> R` | function type | `fn(int) -> bool` |

### Variables

```vexel
let x: int = 10
let name: str = "Vexel"
let big: int = 1_000_000        # numeric separator
let hex: int = 0xFF             # hex literal
let bin: int = 0b1010_1010      # binary literal
let oct: int = 0o755            # octal literal
const MAX: int = 100            # top-level constant
comptime let BUF = 256 * 4      # compile-time constant
type Speed = float              # type alias
```

### Functions

```vexel
fn add(a: int, b: int) -> int:
    return a + b

fn greet(name: str = "World"):  # default parameters
    print("Hello, " + name + "!")

fn sum(...nums: int[]) -> int:  # variadic
    let total: int = 0
    for n in nums:
        total += n
    return total

fn min_max(arr: int[]) -> (int, int):
    return (arr[0], arr[arr.len() - 1])

pub fn public_api() -> int:     # visibility
    return 42

extern fn strlen(s: str) -> int # C FFI
```

### Closures with Capture

```vexel
fn main():
    let multiplier: int = 3
    let triple: fn(int) -> int = fn(x: int) -> int: return x * multiplier
    print(triple(7))    # 21 — multiplier is captured from outer scope
```

### Generics

```vexel
fn first[T](arr: T[]) -> T:
    return arr[0]

fn identity[T](x: T) -> T:
    return x

let x: int = first([10, 20, 30])   # 10
let s: str = identity("hello")     # "hello"
```

### Interfaces

```vexel
interface Shape:
    fn area() -> float
    fn name() -> str

struct Circle:
    radius: float

impl Shape for Circle:
    fn area(self) -> float:
        return 3.14159 * self.radius * self.radius
    fn name(self) -> str:
        return "Circle"

fn describe(s: Shape):
    print(s.name() + " area=" + str(s.area()))

fn main():
    let c: Shape = new Circle(5.0)
    describe(c)
```

### Enums

```vexel
# Simple enum
enum Direction:
    North
    South
    East
    West

let dir: int = Direction.North
match dir:
    case Direction.North: print("going north")
    default:              print("other")
```

### Match with Guards

```vexel
let score: int = 85
match score:
    case s if s >= 90: print("A")
    case s if s >= 80: print("B")
    case s if s >= 70: print("C")
    default:           print("F")
```

### Null Safety

```vexel
let maybe: int? = null
let val: int = maybe ?? 42      # null coalescing — val = 42

let user: User? = find_user(id)
let name: str? = user?.name     # optional chaining — null if user is null
```

### Destructuring

```vexel
# Struct destructuring
let {x, y} = point

# Array destructuring
let [first, second] = arr

# Tuple unpacking
let (q, r): (int, int) = divmod(17, 5)
```

### List Comprehensions

```vexel
let evens: int[] = [x for x in nums if x % 2 == 0]
let squares: int[] = [x * x for x in 0..10]
```

### Defer

```vexel
fn load_file(path: str) -> str:
    let f = open(path)
    defer f.close()         # runs when function exits, no matter what
    return f.read()
```

### Labeled Loops

```vexel
outer:
for i in 0..10:
    for j in 0..10:
        if i == j:
            break outer     # breaks the outer loop
```

### do-while

```vexel
let i: int = 0
do:
    print(i)
    i = i + 1
while i < 5:
```

### Error Handling

```vexel
try:
    let data: str = read_file("missing.txt")
    if data == "":
        throw("File is empty")
catch IOError as e:
    print("IO error: " + e)
catch e:
    print("Error: " + e)
finally:
    print("cleanup")
```

### Comptime

```vexel
comptime let MAX_SIZE = 1024 * 64
comptime let FLAG = 1 << 8
comptime let MASK = 0xFF & FLAG
```

### Test Framework

```vexel
test "addition":
    assert_eq(1 + 1, 2)
    assert_true(10 > 5)
    assert_false(3 > 10)

test "strings":
    let s: str = str_upper("hello")
    assert_eq(s, "HELLO")
```

### Threads and Synchronization

```vexel
let mu: int? = mutex_new()
let counter: int? = atomic_new(0)

fn worker():
    mutex_lock(mu)
    atomic_add(counter, 1)
    mutex_unlock(mu)

fn main():
    let t1: int = thread_spawn(worker)
    let t2: int = thread_spawn(worker)
    thread_join(t1)
    thread_join(t2)
    print(atomic_load(counter))   # 2
```

### Atomic Operations

```vexel
let val: int? = atomic_new(0)
atomic_add(val, 10)
atomic_sub(val, 3)
let n: int = atomic_load(val)           # 7
let ok: bool = atomic_compare_swap(val, 7, 100)  # true
```

### Bitwise Operators

```vexel
let flags: int = 0b0101 & 0b0011   # AND  → 1
let mask:  int = 1 << 4             # shift → 16
let inv:   int = ~flags             # NOT
let merged = flags | 0b1000         # OR
let diff   = flags ^ 0b0111         # XOR
```

### String Builtins

```vexel
str_upper(s)                    # "HELLO"
str_lower(s)                    # "hello"
str_trim(s)                     # strip whitespace
str_find(s, "world")            # index or -1
str_slice(s, 0, 5)              # substring
str_repeat("ab", 3)             # "ababab"
str_replace(s, "old", "new")
str_split(s, ",")               # str[]
str_contains(s, "lo")           # bool
str_starts_with(s, "He")        # bool
str_ends_with(s, "ld")          # bool
char_at(s, 0)                   # single-char str
char_to_int("A")                # 65
int_to_char(65)                 # "A"
str_char_len(s)                 # UTF-8 codepoint count
str_format("%s is %d", name, age)
```

### Array Builtins

```vexel
array_sort(arr)
array_reverse(arr)
array_index_of(arr, val)        # int or -1
array_contains(arr, val)        # bool
array_slice(arr, 1, 4)          # new sub-array
array_join(words, ", ")         # str
```

### Math Builtins

```vexel
sqrt(x)   floor(x)   ceil(x)   round(x)
abs(x)    pow(a, b)
min(a, b) max(a, b)  clamp(x, lo, hi)
lerp(a, b, t)        atan2(y, x)
sin(x)    cos(x)     tan(x)
log(x)    log2(x)    log10(x)
exp(x)    hypot(a, b)
rand()               rand_int(a, b)
```

### Stdlib Builtins

```vexel
# Encoding
base64_encode(s)        # str
base64_decode(s)        # str

# Crypto / IDs
sha256(s)               # hex string
uuid_v4()               # "xxxxxxxx-xxxx-4xxx-..."

# Data
csv_parse(s, ",")       # str[][]
json_stringify_int(n)   # "42"
json_stringify_float(f) # "3.14"
json_stringify_str(s)   # "\"hello\""

# OS
argv()                  # str[]
env_get("HOME")         # str
env_set("KEY", "val")
shell("ls -la")         # str (stdout)
http_get(url)           # str (via curl)

# File I/O
read_file("data.txt")
write_file("out.txt", content)
append_file("log.txt", line)
file_exists("data.txt")   # bool
os_cwd()
os_mkdir("dir")
os_delete("file")
os_list_dir(".")          # str[]

# Time and input
time_now()              # Unix timestamp int
time_format(ts)         # human-readable str
input("Name: ")         # read stdin line

# Type helpers
type_of(val)            # str
is_null(val)            # bool
to_int(s)               # str/float → int
to_float(s)             # str/int → float
```

### Conversion

```vexel
int(3.9)          # 3
float(5)          # 5.0
str(42)           # "42"
bool(1)           # true
parse_int("42")   # 42
parse_float("3.14")
```

### Operators

```
+  -  *  /  %  **        arithmetic (** = power)
&  |  ^  ~  <<  >>       bitwise
==  !=  <  >  <=  >=     comparison
and  or  not              logical
+=  -=  *=  /=           compound assignment
??                        null coalesce
?.                        optional chain
..   ..=                  range (exclusive / inclusive)
->                        return type arrow
?   :                     ternary  (cond ? a : b)
...                       variadic param prefix
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

### Generic Stack

```vexel
fn push[T](stack: T[], val: T) -> T[]:
    stack.push(val)
    return stack

fn pop[T](stack: T[]) -> T:
    return stack.pop()

fn main():
    let s: int[] = []
    push(s, 10)
    push(s, 20)
    print(pop(s))   # 20
```

### Interface Polymorphism

```vexel
interface Animal:
    fn speak() -> str

struct Dog:
    name: str

struct Cat:
    name: str

impl Animal for Dog:
    fn speak(self) -> str: return self.name + " says: woof!"

impl Animal for Cat:
    fn speak(self) -> str: return self.name + " says: meow!"

fn main():
    let animals: Animal[] = [new Dog("Rex"), new Cat("Whiskers")]
    for a in animals:
        print(a.speak())
```

### Thread-Safe Counter

```vexel
let mu: int? = mutex_new()
let count: int? = atomic_new(0)

fn increment():
    mutex_lock(mu)
    atomic_add(count, 1)
    mutex_unlock(mu)

fn main():
    let threads: int[] = []
    for i in 0..10:
        threads.push(thread_spawn(increment))
    for t in threads:
        thread_join(t)
    print(atomic_load(count))   # 10
```

### CSV Processing

```vexel
fn main():
    let data: str = read_file("sales.csv")
    let rows: str[][] = csv_parse(data, ",")
    for row in rows:
        print(row[0] + ": " + row[1])
```

## Project Structure

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
