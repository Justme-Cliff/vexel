# Vexel — Future Improvements

Things left to build. Tell me to implement any of these when you're back.

---

## 1. Memory Management (GC or Reference Counting)
**Priority: High — the biggest real-world issue right now.**
Everything is malloc'd and never freed, so long-running programs leak memory.
Options:
- **Bump allocator + arena reset** — simple, fast, suits scripts
- **Reference counting** — add a ref-count header to every heap object; decrement on scope exit
- **Mark-and-sweep GC** — full GC; most correct but most complex

---

## 2. Capturing Closures
Right now lambdas can't close over variables from the surrounding scope.
```vexel
let multiplier: int = 3
let triple: fn(int) -> int = fn(x: int) -> int:
    return x * multiplier   # currently broken — multiplier not captured
```
Fix: heap-allocate a "closure environment" struct containing the captured variables, pass it as a hidden extra arg to the lambda's LLVM function.

---

## 3. Generic Structs
Generic functions work (`fn first[T]`) but generic structs don't.
```vexel
struct Stack[T]:
    items: T[]
    fn push(self, val: T):
        self.items.push(val)
    fn pop(self) -> T:
        return self.items.pop()
```
Fix: monomorphize structs the same way functions are — generate a concrete `Stack__int`, `Stack__str`, etc. on demand.

---

## 4. Standard Library Expansion
The current stdlib is thin. Useful additions:
- **`json`** — `json_parse(s: str) -> dict[str, str]`, `json_stringify(d: dict) -> str`
- **`http`** — basic `http_get(url: str) -> str`, `http_post(url: str, body: str) -> str`
- **`datetime`** — `date_format(ts: int, fmt: str) -> str`, `date_parse(s: str) -> int`
- **`math`** — `pi`, `e` as constants; `hypot`, `log10`, `exp`
- **`str`** — `str_format(template: str, args: str[]) -> str` (printf-style)
- **`collections`** — `Set[T]` type with `.add()`, `.contains()`, `.remove()`

---

## 5. Runtime Error Locations (Debug Info)
Crashes and runtime errors don't say which line of .vx code caused them.
```
vexel error: null pointer dereference at line 42 in main.vx
```
Fix: embed LLVM DWARF debug info or insert line-number tracking globals that get printed on crash/assert.
