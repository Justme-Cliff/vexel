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
```

## Features

### Language Core
- **Clean syntax** — indentation-based blocks, no braces or semicolons
- **Static types** — `int`, `float`, `bool`, `str`, `char`, arrays, dicts, tuples
- **Null safety** — nullable `T?`, null coalescing `??`, optional chaining `?.`
- **Ranges** — `0..10`, `0..=n`, slice expressions `arr[1..4]`
- **Pipe operator** — `value |> func`
- **Ternary** — `cond ? a : b`
- **Spread** — `...arr` in call sites

### Control Flow
- `if / elif / else`, `while`, `for i in 0..n`, `for item in arr`, `for i, v in arr`
- `do-while` — body runs at least once
- `match / case` with guards (`case x if x > 0:`)
- Labeled loops — `break outer` / `continue outer`
- List comprehensions — `[x * 2 for x in arr if x > 0]`

### Functions
- Default parameters, variadic (`...args`), named arguments at call sites
- Named return values
- Lambdas / closures with mutable capture
- `defer` — guaranteed cleanup on scope exit (like Go)
- `defer_on_error` — only runs on exception path

### Types & Type System
- **Structs** — typed fields, default values, struct update syntax `{ ...base, field: val }`
- **Generic structs** — `struct Stack[T]:`
- **Generic functions** — `fn map[T, U](...)`
- **Generic bounds** — `fn foo[T: Printable](...)` enforced by analyzer
- **Interfaces** — vtable polymorphism with `interface` / `impl`
- **Operator overloading** — `impl Add for Vec2`
- **Simple enums** and **ADT enums** with per-variant payload fields and methods
- **Error types** — `error IOError(path: str)`
- **Result / Option** — `Ok / Err / Some / None`, `.unwrap()`, `.is_ok()`
- **Error propagation** — `expr?` (returns early on Err/null)
- **`pub` / `priv`** visibility across multi-file projects

### Destructuring
- Struct — `let {x, y} = point`
- Array — `let [a, b, ...rest] = arr`
- Tuple — `let (a, b) = expr`

### Safety & Diagnostics
- Bounds checks, overflow detection, null dereference checks
- `assert`, `assert_eq`, `assert_true`, `assert_false`
- Stack traces on panic + crash log written to `vexel_crash.log`
- `try / catch / finally` with typed catch clauses
- `throw` / `raise`
- Typo suggestions in error messages ("did you mean?")

### Concurrency
- `thread_spawn` / `thread_join` / `thread_sleep`
- `mutex_new` / `mutex_lock` / `mutex_unlock`
- Atomic operations — `atomic_new`, `atomic_add`, `atomic_load`, `atomic_compare_swap`
- Channels — `channel_new` / `channel_send` / `channel_recv` / `channel_close`, `chan_select`
- RWLock, CondVar
- Thread pool
- Signals — `signal_handle`
- Process spawn — `process_spawn`

### Standard Library
- **Strings** — `str_upper`, `str_lower`, `str_trim`, `str_find`, `str_split`, `str_replace`, `str_contains`, `str_starts_with`, `str_ends_with`, `str_repeat`, `str_pad`, `char_at`, `str_format`, and more
- **Arrays** — `push`, `pop`, `map`, `filter`, `reduce`, `sort`, `reverse`, `slice`, `flat_map`, `zip`, `chunk`, `sum`, `min`, `max`, `any`, `all`, `array_index_of`, `array_contains`, `array_join`
- **Dict** — `get`, `set`, `has_key`, `keys`, `values`, `items`, `remove`, `merge`, `get_or`
- **Set** — `set_new`, `set_add`, `set_contains`, `set_remove`, `set_union`, `set_intersect`, `set_diff`
- **Math** — `sqrt`, `abs`, `floor`, `ceil`, `round`, `pow`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `atan2`, `exp`, `hypot`, `lerp`, `clamp`, `rand`, `rand_int`, `PI`, `E`
- **DateTime** — `datetime_now`, `datetime_format`, `datetime_parse`, `datetime_add_days`, `datetime_diff_days`, `datetime_timestamp`
- **JSON** — `json_parse`, `json_stringify`
- **UUID** — `uuid_v1`
- **Environment** — `env_load`, `env_get`, `env_set`
- **Benchmarking** — `benchmark()`, `bench_ns()`
- **Progress bar** — `progress_new`, `progress_update`, `progress_done`
- **Terminal UI** — `tui_init`, `tui_draw`, `tui_clear`, `tui_color`

### Networking & I/O
- TCP sockets, UDP sockets, named pipes
- HTTP server
- WebSocket
- TLS

### Cryptography & Compression
- `bcrypt_hash` / `bcrypt_verify`
- `argon2_hash` / `argon2_verify`
- `zlib_compress` / `zlib_decompress`

### Database
- SQLite — `sqlite_open`, `sqlite_exec`, `sqlite_query`, `sqlite_close`

### Data Formats
- XML — `xml_parse`, `xml_stringify`
- YAML — `yaml_parse`, `yaml_stringify`
- `@serialize` — serialize structs to JSON / binary

### Metaprogramming & Attributes
- `@serialize` on structs
- `@repr(C)` for C-compatible struct layout
- `comptime let` — compile-time constants
- `unsafe` blocks
- `extern fn` — declare C functions (FFI)

### Build & Targets
- `vexel run` — JIT execution via LLVM MCJIT
- `vexel compile` — AOT to native binary
- Cross-compilation: `x86_64-linux`, `aarch64-linux`
- Dead code elimination (internal linkage for non-`pub` functions)

### Multi-file / Modules
- `import "file.vx"`
- `import "math.vx" as math` — aliased namespace import
- `pub` / `priv` visibility

---

## Installation

### Option 1 — Download the pre-built exe (Windows)

1. Download `vexel.exe` from [Releases](https://github.com/Justme-Cliff/vexel-lang/releases)
2. Add it to your PATH
3. On first run, the VS Code extension is installed automatically (if VS Code is detected)

### Option 2 — Run from source

**Requirements:**
- Python 3.10+
- `pip install llvmlite`
- MinGW gcc (Windows: [Strawberry Perl](https://strawberryperl.com/) includes it)

```
git clone https://github.com/Justme-Cliff/vexel-lang
cd vexel-lang
```

Add the project folder to your PATH, or run with `python main.py`.  
On Windows a `vexel.bat` is included.

### VS Code Extension

**Automatic (exe users):** The extension is installed automatically on first run.

**Manual:**
1. Download `vexel-vscode/vexel-1.1.0.vsix`
2. VS Code → `Ctrl+Shift+P` → `Extensions: Install from VSIX...`
3. Select the `.vsix` file and reload VS Code

---

## CLI Commands

```bash
vexel run     hello.vx            # JIT compile and run immediately
vexel run     hello.vx --watch    # re-run on every file save
vexel compile hello.vx            # compile to hello.exe / hello
vexel compile hello.vx -o out     # specify output name
vexel compile hello.vx --target x86_64-linux-gnu  # cross-compile
vexel build                       # build from vexel.toml project file
vexel test    hello.vx            # run all test blocks
vexel fmt     hello.vx            # auto-format in place
vexel doc     hello.vx            # generate Markdown docs
vexel ir      hello.vx            # print LLVM IR
vexel parse   hello.vx            # print AST
vexel lex     hello.vx            # print tokens
vexel repl                        # interactive REPL
```

---

## Language Reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | 64-bit integer | `42` |
| `float` | 64-bit double | `3.14` |
| `bool` | boolean | `true` / `false` |
| `char` | 8-bit character | `'A'` |
| `str` | string | `"hello"` |
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

fn greet(name: str = "World"):  # default parameter
    print("Hello, " + name + "!")

fn sum(...nums: int[]) -> int:  # variadic
    let total: int = 0
    for n in nums:
        total += n
    return total

pub fn public_api() -> int:     # visibility
    return 42

extern fn strlen(s: str) -> int # C FFI
```

### Closures

```vexel
fn main():
    let multiplier: int = 3
    let triple: fn(int) -> int = fn(x: int) -> int: return x * multiplier
    print(triple(7))   # 21
```

### Generics

```vexel
fn first[T](arr: T[]) -> T:
    return arr[0]

struct Stack[T]:
    items: T[]

fn foo[T: Printable](val: T):  # bounded generic
    val.print()
```

### Interfaces

```vexel
interface Shape:
    fn area() -> float

struct Circle:
    radius: float

impl Shape for Circle:
    fn area(self) -> float:
        return 3.14159 * self.radius * self.radius

fn describe(s: Shape):
    print(s.area())
```

### ADT Enums

```vexel
enum Shape:
    Circle(radius: float)
    Rect(w: float, h: float)
    Point

fn area(s: Shape) -> float:
    match s:
        case Circle(r): return 3.14 * r * r
        case Rect(w, h): return w * h
        case Point: return 0.0
```

### Error Handling

```vexel
error IOError(path: str)
error NetworkError(code: int)

fn read_data(path: str) -> Result[str]:
    if not file_exists(path):
        return Err(IOError(path))
    return Ok(read_file(path))

fn process(path: str):
    let data: str = read_data(path)?   # propagate error with ?
    print(data)

try:
    process("data.txt")
catch IOError as e:
    print("IO error: " + e.path)
catch e:
    print("Error: " + e)
finally:
    print("done")
```

### Result / Option

```vexel
let r: Result[int] = Ok(42)
if r.is_ok():
    print(r.unwrap())

let maybe: Option[str] = Some("hello")
let val: str = maybe.unwrap_or("default")
```

### Channels

```vexel
let ch: int? = channel_new(10)

fn producer():
    channel_send(ch, 42)
    channel_close(ch)

fn main():
    thread_spawn(producer)
    let val: int = channel_recv(ch)
    print(val)   # 42
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
let val: int = maybe ?? 42      # null coalescing
let name: str? = user?.name     # optional chaining
```

### Destructuring

```vexel
let {x, y} = point              # struct
let [first, second, ...rest] = arr  # array
let (q, r): (int, int) = divmod(17, 5)  # tuple
```

### Defer

```vexel
fn load_file(path: str) -> str:
    let f = open(path)
    defer f.close()
    defer_on_error log_error("load failed")
    return f.read()
```

### Operator Overloading

```vexel
struct Vec2:
    x: float
    y: float

impl Add for Vec2:
    fn add(self, other: Vec2) -> Vec2:
        return new Vec2(self.x + other.x, self.y + other.y)

fn main():
    let a: Vec2 = new Vec2(1.0, 2.0)
    let b: Vec2 = new Vec2(3.0, 4.0)
    let c: Vec2 = a + b   # calls Vec2.add
```

### Serialization

```vexel
@serialize
struct User:
    name: str
    age:  int

fn main():
    let u: User = new User("Alice", 30)
    let json: str = u.to_json()
    print(json)   # {"name":"Alice","age":30}
```

### SQLite

```vexel
fn main():
    let db: int? = sqlite_open("data.db")
    sqlite_exec(db, "CREATE TABLE IF NOT EXISTS users (id INT, name TEXT)")
    sqlite_exec(db, "INSERT INTO users VALUES (1, 'Alice')")
    let rows: str[][] = sqlite_query(db, "SELECT * FROM users")
    for row in rows:
        print(row[0] + ": " + row[1])
    sqlite_close(db)
```

### Labeled Loops

```vexel
outer:
for i in 0..10:
    for j in 0..10:
        if i == j:
            break outer
```

### Test Framework

```vexel
test "addition":
    assert_eq(1 + 1, 2)
    assert_true(10 > 5)

test "strings":
    assert_eq(str_upper("hello"), "HELLO")
```

Run with: `vexel test myfile.vx`

### Project File (vexel.toml)

```toml
[project]
name = "myapp"
version = "0.1.0"
main = "src/main.vx"
```

Run with: `vexel build`

---

## Project Structure

```
compiler/
├── compiler/
│   ├── lexer.py          tokenizer
│   ├── parser.py         recursive descent parser
│   ├── ast_nodes.py      AST node types
│   ├── analyzer.py       type checking / semantic analysis
│   └── codegen.py        LLVM IR code generation
├── stdlib/
│   ├── runtime.c         C runtime helpers
│   └── vx_sdl2.c         SDL2 graphics wrapper
├── vexel-vscode/         VS Code extension (syntax highlighting + icon)
├── examples/             example Vexel programs
├── tests/                test suite
├── main.py               CLI entry point
├── vexel.bat             Windows launcher
└── vexel.toml            (optional) project config
```

## License

MIT
