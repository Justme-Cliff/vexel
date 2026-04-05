# Contributing to Vexel

Thanks for your interest in contributing to the Vexel programming language.

## Project structure

```
compiler/
├── compiler/
│   ├── lexer.py        — tokenizer (source → tokens)
│   ├── parser.py       — parser (tokens → AST)
│   ├── ast_nodes.py    — AST node dataclasses
│   ├── analyzer.py     — semantic analysis + type inference
│   └── codegen.py      — LLVM IR code generation
├── stdlib/
│   ├── runtime.c       — C runtime (GC, string helpers, array ops, file I/O)
│   └── vx_sdl2.c       — SDL2 wrapper for graphics
├── examples/           — example .vx programs
├── tests/              — test suite
└── main.py             — CLI entry point (vexel command)
```

## How the compiler works

Source code flows through five stages:

1. **Lexer** — converts raw text into a flat list of tokens. Handles indentation by emitting `INDENT`/`DEDENT` tokens (like Python). Paren-depth tracking suppresses structural tokens inside `(` or `[` so multi-line calls work.

2. **Parser** — recursive descent parser that builds an AST from tokens. Each grammar rule is a `_parse_*` method.

3. **Analyzer** — walks the AST, checks types, resolves variable scopes, and builds a `FnSig` table that codegen uses to look up function signatures.

4. **Codegen** — walks the AST and emits LLVM IR using the `llvmlite` Python bindings. Complex operations (string methods, array push/pop, file I/O) are implemented as private LLVM functions defined inline in the module so they work in both JIT and compiled modes.

5. **Execution** — either JIT via MCJIT (`vexel run`) or native binary via gcc link (`vexel compile`).

## Adding a new builtin function

1. Add its name and signature to `BUILTINS` in `compiler/analyzer.py`
2. Handle the name in `_compile_call` in `compiler/codegen.py`
3. If it needs a new libc extern, declare it in `_declare_externs`
4. If it's complex (needs a loop), define a private helper via `_get_helper` / `_build_helper`

## Adding a new statement type

1. Add a dataclass to `compiler/ast_nodes.py`
2. Add a keyword to `TT` enum and `KEYWORDS` dict in `compiler/lexer.py`
3. Add a `_parse_*` method in `compiler/parser.py` and call it from `_parse_stmt` or `_parse_top_level`
4. Handle the new node in `compiler/analyzer.py` (`_analyze_stmt`)
5. Handle it in `compiler/codegen.py` (`_compile_stmt`)
6. Add a test in `tests/`

## Running tests

```
python tests/test_lexer.py
python tests/test_parser.py
python tests/test_codegen.py
python tests/test_v2.py
```

## Code style

- Python 3.10+
- 4-space indentation
- Type hints where they add clarity
- Keep each compiler stage self-contained — lexer knows nothing about codegen, etc.

## Requirements

- Python 3.10+
- `llvmlite` 0.47+
- MinGW gcc (for `vexel compile` and `vexel run` with runtime)
- SDL2 (optional, for graphics programs)
