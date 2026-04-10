"""
vexel — the Vexel compiler CLI

Commands:
  vexel lex      <file.vx>            Print token stream
  vexel parse    <file.vx>            Print AST
  vexel analyze  <file.vx>            Run semantic analysis
  vexel ir       <file.vx>            Print LLVM IR
  vexel run      <file.vx>            JIT-compile and run
  vexel compile  <file.vx> [-o out]   Compile to native binary
                            [--sdl2]  Link with SDL2

Flags:
  -o <output>   Output binary name (default: stem of the input file)
  --sdl2        Enable SDL2 built-ins and link SDL2
"""

from __future__ import annotations

import copy
import os
import pprint
import subprocess
import sys
import tempfile
import argparse

from compiler.lexer import Lexer, LexError
from compiler.parser import Parser, ParseError
from compiler.analyzer import Analyzer, FnSig
from compiler.ast_nodes import (
    Program, ImportStmt, FnDecl, StructDecl,
    GlobalLet, GlobalConst, EnumDecl, NamespaceHint,
)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def die(msg: str) -> None:
    print(f"vexel error: {msg}", file=sys.stderr)
    sys.exit(1)


def read_source(path: str) -> str:
    if not os.path.exists(path):
        die(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _prefix_decl(decl, prefix: str):
    """Return a deep copy of *decl* with its name prepended by *prefix*__."""
    d = copy.deepcopy(decl)
    if isinstance(d, (FnDecl, StructDecl, GlobalLet, GlobalConst, EnumDecl)):
        d.name = f"{prefix}__{d.name}"
    return d


# ------------------------------------------------------------------ #
#  Import resolver                                                     #
# ------------------------------------------------------------------ #

def resolve_imports(program: Program, base_dir: str) -> Program:
    """
    Walk *program*, load any ImportStmt files, recursively resolve their
    imports, and return a new Program with all declarations merged.

    Imported ``main`` functions are dropped.  If an import has an alias
    (``import "foo.vx" as foo``), all declarations from that file are
    renamed with the alias prefix (``foo__name``) and a NamespaceHint is
    inserted so the analyzer / codegen know ``foo`` is a namespace.
    """
    merged: list = []
    visited: set = set()       # (abs_path, alias) pairs already processed
    inserted_ns: set = set()   # namespace aliases already inserted

    base_dir = os.path.abspath(base_dir)

    def process(prog: Program, cur_dir: str, alias: str | None = None) -> None:
        cur_dir = os.path.abspath(cur_dir)
        for decl in prog.declarations:
            if isinstance(decl, ImportStmt):
                path = decl.path if decl.path.endswith(".vx") else decl.path + ".vx"
                abs_path = os.path.normpath(os.path.join(cur_dir, path))
                sub_alias = decl.alias
                key = (abs_path, sub_alias)
                if key in visited:
                    continue
                visited.add(key)
                try:
                    src = open(abs_path, encoding="utf-8").read()
                except OSError as e:
                    die(f"Cannot open import '{abs_path}': {e}")
                try:
                    tokens = Lexer(src).tokenize()
                except LexError as e:
                    die(f"Lex error in '{abs_path}': {e}")
                try:
                    sub_prog = Parser(tokens).parse()
                except ParseError as e:
                    die(f"Parse error in '{abs_path}': {e}")
                if sub_alias and sub_alias not in inserted_ns:
                    inserted_ns.add(sub_alias)
                    merged.append(NamespaceHint(sub_alias))
                process(sub_prog, os.path.dirname(abs_path), sub_alias)
            else:
                if isinstance(decl, FnDecl) and decl.name == "main" and cur_dir != base_dir:
                    continue
                merged.append(_prefix_decl(decl, alias) if alias else decl)

    process(program, base_dir)
    return Program(merged)


# ------------------------------------------------------------------ #
#  Pipeline stages                                                     #
# ------------------------------------------------------------------ #

def pipeline_lex(source: str):
    try:
        return Lexer(source).tokenize()
    except LexError as e:
        die(str(e))


def pipeline_parse(tokens):
    try:
        return Parser(tokens).parse()
    except ParseError as e:
        die(str(e))


def pipeline_analyze(program: Program, sdl2: bool = False):
    from compiler.sdl2_builtins import SDL2_BUILTINS

    analyzer = Analyzer()
    if sdl2:
        for vx_name, (param_types, ret_type) in SDL2_BUILTINS.items():
            sig = FnSig([(f"a{i}", t) for i, t in enumerate(param_types)], ret_type)
            analyzer._fn_sigs[vx_name] = sig

    result = analyzer.analyze(program)
    if result.errors:
        for err in result.errors:
            print(f"  error: {err}", file=sys.stderr)
        die("Analysis failed")
    return result


def pipeline_codegen(program: Program, analysis, sdl2: bool = False,
                     target_triple: str | None = None,
                     debug_mode: bool = False) -> str:
    from compiler.codegen import Compiler
    from compiler.sdl2_builtins import SDL2_BUILTINS, SDL2_C_NAMES
    from llvmlite import ir

    compiler = Compiler(analysis, target_triple=target_triple, debug_mode=debug_mode)
    if sdl2:
        for vx_name, (param_types, ret_type) in SDL2_BUILTINS.items():
            c_name     = SDL2_C_NAMES[vx_name]
            llvm_params = [compiler._vx_to_llvm(t) for t in param_types]
            llvm_ret    = compiler._vx_to_llvm(ret_type)
            fn          = ir.Function(compiler.module, ir.FunctionType(llvm_ret, llvm_params), name=c_name)
            sig = analysis.fn_sigs.get(vx_name) or FnSig(
                [(f"a{i}", t) for i, t in enumerate(param_types)], ret_type
            )
            compiler._functions[vx_name] = {"fn": fn, "sig": sig}

    return compiler.compile(program)


def pipeline_full(args, sdl2: bool = False):
    """Run lex → parse → import resolution → analyze and return (program, analysis)."""
    base_dir = os.path.dirname(os.path.abspath(args.file))
    source   = read_source(args.file)
    tokens   = pipeline_lex(source)
    program  = pipeline_parse(tokens)
    program  = resolve_imports(program, base_dir)
    analysis = pipeline_analyze(program, sdl2=sdl2)
    return program, analysis


# ------------------------------------------------------------------ #
#  Commands                                                            #
# ------------------------------------------------------------------ #

def cmd_lex(args) -> None:
    source = read_source(args.file)
    for tok in pipeline_lex(source):
        print(tok)


def cmd_parse(args) -> None:
    source  = read_source(args.file)
    tokens  = pipeline_lex(source)
    program = pipeline_parse(tokens)
    pprint.pprint(program, indent=2)


def cmd_analyze(args) -> None:
    program, analysis = pipeline_full(args, sdl2=args.sdl2)
    print("Analysis OK")
    print(f"  Functions : {list(analysis.fn_sigs.keys())}")
    print(f"  Structs   : {list(analysis.struct_fields.keys())}")


def cmd_ir(args) -> None:
    program, analysis = pipeline_full(args, sdl2=args.sdl2)
    print(pipeline_codegen(program, analysis, sdl2=args.sdl2))


def cmd_run(args) -> None:
    from compiler.codegen import jit_run, CodegenError

    debug = getattr(args, "debug", False)
    program, analysis = pipeline_full(args)
    llvm_ir = pipeline_codegen(program, analysis, debug_mode=debug)
    try:
        sys.exit(jit_run(llvm_ir))
    except CodegenError as e:
        die(str(e))


def cmd_compile(args) -> None:
    from compiler.codegen import _init_llvm, CodegenError
    from llvmlite import binding

    GCC = r"C:\Strawberry\c\bin\gcc.exe" if os.path.exists(r"C:\Strawberry\c\bin\gcc.exe") else "gcc"

    program, analysis = pipeline_full(args, sdl2=args.sdl2)
    llvm_ir = pipeline_codegen(program, analysis, sdl2=args.sdl2)

    stem   = os.path.splitext(os.path.basename(args.file))[0]
    output = args.output or stem
    if sys.platform == "win32" and not output.endswith(".exe"):
        output += ".exe"

    runtime_c   = os.path.join(os.path.dirname(__file__), "stdlib", "runtime.c")
    runtime_obj = _compile_c_object(GCC, runtime_c) if os.path.exists(runtime_c) else None

    sdl2_obj = None
    if args.sdl2:
        sdl2_c   = os.path.join(os.path.dirname(__file__), "stdlib", "vx_sdl2.c")
        sdl2_inc = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "include")
        sdl2_obj = _compile_c_object(GCC, sdl2_c, extra_flags=["-DVX_SDL2_ENABLED", f"-I{sdl2_inc}"])

    obj_path = None
    try:
        _init_llvm()
        triple = binding.get_default_triple()
        tm     = binding.Target.from_triple(triple).create_target_machine(reloc="pic", codemodel="default")
        mod    = binding.parse_assembly(llvm_ir)
        mod.verify()

        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
            f.write(tm.emit_object(mod))
            obj_path = f.name

        link_cmd = [GCC, obj_path]
        if runtime_obj: link_cmd.append(runtime_obj)
        if sdl2_obj:    link_cmd.append(sdl2_obj)
        link_cmd += ["-o", output, "-lm"]

        if args.sdl2:
            sdl2_lib = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "lib")
            link_cmd += [f"-L{sdl2_lib}", "-lSDL2", "-lSDL2main"]

        result = subprocess.run(link_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            die(f"Link error:\n{result.stderr}")
        print(f"Compiled: {output}")

    finally:
        for path in filter(None, [obj_path, runtime_obj, sdl2_obj]):
            try:
                os.unlink(path)
            except OSError:
                pass


def _compile_c_object(gcc: str, src: str, extra_flags: list[str] | None = None) -> str:
    """Compile a C source file to a temp object file; return the object path."""
    obj = tempfile.mktemp(suffix=".o")
    cmd = [gcc, "-O2", "-c", src, "-o", obj] + (extra_flags or [])
    r   = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"Failed to compile {os.path.basename(src)}:\n{r.stderr}")
    return obj


# ------------------------------------------------------------------ #
#  REPL (#71)                                                          #
# ------------------------------------------------------------------ #

def cmd_repl(args) -> None:
    """Interactive read-eval-print loop."""
    from compiler.lexer  import Lexer, LexError
    from compiler.parser import Parser, ParseError
    from compiler.codegen import jit_run, CodegenError

    _DECL_STARTERS = ("fn ", "struct ", "enum ", "interface ", "impl ",
                      "type ", "extern ", "pub ", "priv ", "comptime ",
                      "const ", "let ")

    print("Vexel REPL  (type 'exit' or Ctrl-C/Ctrl-D to quit)")
    decl_lines: list[str] = []   # accumulated top-level declarations
    stmt_lines: list[str] = []   # accumulated statements (inside main)

    while True:
        try:
            line = input(">>> ").rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.strip() in ("exit", "quit", ""):
            if line.strip() in ("exit", "quit"):
                break
            continue

        stripped = line.strip()
        is_decl = any(stripped.startswith(s) for s in _DECL_STARTERS)

        if is_decl:
            # Just register the declaration — don't run
            decl_lines.append(line)
            print("  (registered)")
            continue

        # Build a full program: declarations + main(){prev stmts + new stmt}
        indent = "    "
        all_stmts = stmt_lines + [line]
        source = "\n".join(decl_lines)
        source += "\nfn main():\n"
        source += "\n".join(indent + s for s in all_stmts) + "\n"

        try:
            tokens  = Lexer(source).tokenize()
            program = pipeline_parse(tokens)
            base_dir = os.getcwd()
            program  = resolve_imports(program, base_dir)
            analysis = pipeline_analyze(program)
            llvm_ir  = pipeline_codegen(program, analysis)
            jit_run(llvm_ir)
            stmt_lines.append(line)       # only keep if it compiled & ran
        except SystemExit:
            stmt_lines.append(line)
        except (LexError, ParseError) as e:
            print(f"  error: {e}")
        except CodegenError as e:
            print(f"  codegen error: {e}")
        except Exception as e:
            print(f"  runtime error: {e}")


# ------------------------------------------------------------------ #
#  Formatter (#72)                                                     #
# ------------------------------------------------------------------ #

def cmd_fmt(args) -> None:
    """Reformat a Vexel source file in-place (or print to stdout)."""
    from compiler.formatter import Formatter
    source  = read_source(args.file)
    tokens  = pipeline_lex(source)
    program = pipeline_parse(tokens)
    fmt = Formatter()
    out = fmt.format(program)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Formatted: {args.output}")
    else:
        # In-place
        with open(args.file, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Formatted: {args.file}")


# ------------------------------------------------------------------ #
#  Doc generator (#74)                                                 #
# ------------------------------------------------------------------ #

def cmd_doc(args) -> None:
    """Extract doc comments and generate Markdown documentation."""
    from compiler.docgen import DocGen
    source  = read_source(args.file)
    tokens  = pipeline_lex(source)
    program = pipeline_parse(tokens)
    dg  = DocGen()
    md  = dg.generate(program, title=os.path.splitext(os.path.basename(args.file))[0])
    out = args.output or (os.path.splitext(args.file)[0] + ".md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Docs written to: {out}")


# ------------------------------------------------------------------ #
#  Watch mode (#78)                                                    #
# ------------------------------------------------------------------ #

def _watch_run(args) -> None:
    """Poll the source file; recompile + rerun on change."""
    import time
    from compiler.codegen import jit_run, CodegenError

    path      = os.path.abspath(args.file)
    last_mtime = 0.0
    print(f"[watch] Watching {os.path.basename(path)}  (Ctrl-C to stop)")
    while True:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            time.sleep(0.5)
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            if last_mtime != 0.0:          # skip the very first "change"
                print(f"\n[watch] Change detected — rebuilding...")
            try:
                program, analysis = pipeline_full(args)
                llvm_ir = pipeline_codegen(program, analysis)
                jit_run(llvm_ir)
            except SystemExit:
                pass
            except Exception as e:
                print(f"[watch] error: {e}")
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[watch] Stopped.")
            break


# ------------------------------------------------------------------ #
#  Command wrappers with extra flags                                   #
# ------------------------------------------------------------------ #

def cmd_run_ex(args) -> None:
    """run with optional --watch flag."""
    if args.watch:
        _watch_run(args)
    else:
        cmd_run(args)


def cmd_compile_ex(args) -> None:
    """compile with optional --target flag for cross-compilation (#86)."""
    from compiler.codegen import _init_llvm, CodegenError
    from llvmlite import binding

    GCC = r"C:\Strawberry\c\bin\gcc.exe" if os.path.exists(r"C:\Strawberry\c\bin\gcc.exe") else "gcc"

    target_triple = args.target or None

    program, analysis = pipeline_full(args, sdl2=args.sdl2)
    llvm_ir = pipeline_codegen(program, analysis, sdl2=args.sdl2,
                                target_triple=target_triple)

    stem   = os.path.splitext(os.path.basename(args.file))[0]
    output = args.output or stem
    if sys.platform == "win32" and not target_triple and not output.endswith(".exe"):
        output += ".exe"

    runtime_c   = os.path.join(os.path.dirname(__file__), "stdlib", "runtime.c")
    runtime_obj = _compile_c_object(GCC, runtime_c) if os.path.exists(runtime_c) else None

    sdl2_obj = None
    if args.sdl2:
        sdl2_c   = os.path.join(os.path.dirname(__file__), "stdlib", "vx_sdl2.c")
        sdl2_inc = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "include")
        sdl2_obj = _compile_c_object(GCC, sdl2_c, extra_flags=["-DVX_SDL2_ENABLED", f"-I{sdl2_inc}"])

    obj_path = None
    try:
        _init_llvm()
        triple = target_triple or binding.get_default_triple()
        target = binding.Target.from_triple(triple)
        tm     = target.create_target_machine(reloc="pic", codemodel="default")
        mod    = binding.parse_assembly(llvm_ir)
        mod.verify()

        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
            f.write(tm.emit_object(mod))
            obj_path = f.name

        link_cmd = [GCC, obj_path]
        if runtime_obj: link_cmd.append(runtime_obj)
        if sdl2_obj:    link_cmd.append(sdl2_obj)
        link_cmd += ["-o", output, "-lm"]

        if args.sdl2:
            sdl2_lib = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "lib")
            link_cmd += [f"-L{sdl2_lib}", "-lSDL2", "-lSDL2main"]

        if target_triple:
            # Cross-compiling: user may need to supply their own sysroot/gcc
            print(f"[cross] Target: {triple}")

        result = subprocess.run(link_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            die(f"Link error:\n{result.stderr}")
        print(f"Compiled: {output}")

    finally:
        for path in filter(None, [obj_path, runtime_obj, sdl2_obj]):
            try:
                os.unlink(path)
            except OSError:
                pass


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def cmd_build(args) -> None:
    """Read vexel.toml in CWD and compile the project (#34)."""
    toml_path = os.path.join(os.getcwd(), "vexel.toml")
    if not os.path.exists(toml_path):
        die("No vexel.toml found in current directory")
    # Minimal TOML parser (key = "value" pairs, sections [name])
    cfg: dict = {}
    section = ""
    with open(toml_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                cfg.setdefault(section, {})
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if section:
                    cfg[section][k] = v
                else:
                    cfg[k] = v
    project = cfg.get("project", {})
    main_file = project.get("main", "main.vx")
    output    = project.get("name", os.path.splitext(main_file)[0])
    print(f"Building {project.get('name', output)} v{project.get('version', '?')}")

    class _FakeArgs:
        file   = main_file
        output = output
        sdl2   = False
        target = None
        debug  = False

    cmd_compile_ex(_FakeArgs())


def cmd_test(args) -> None:
    """Run all test blocks in a Vexel source file."""
    from compiler.codegen import jit_run, CodegenError
    from compiler.ast_nodes import TestDecl, FnDecl, ReturnStmt, IntLiteral, Program

    source  = read_source(args.file)
    tokens  = pipeline_lex(source)
    program = pipeline_parse(tokens)
    base_dir = os.path.dirname(os.path.abspath(args.file))
    program  = resolve_imports(program, base_dir)

    # Find all test blocks
    tests = [d for d in program.declarations if isinstance(d, TestDecl)]
    if not tests:
        print("No tests found.")
        return

    passed = failed = 0
    for test in tests:
        # Wrap each test body in a standalone main() and run it
        test_prog = Program(
            [d for d in program.declarations if not isinstance(d, TestDecl)] +
            [FnDecl("main", [], None, test.body)]
        )
        try:
            analysis = pipeline_analyze(test_prog)
            llvm_ir  = pipeline_codegen(test_prog, analysis)
            jit_run(llvm_ir)
            print(f"  PASS  {test.name}")
            passed += 1
        except SystemExit as e:
            if e.code == 0:
                print(f"  PASS  {test.name}")
                passed += 1
            else:
                print(f"  FAIL  {test.name}  (exit {e.code})")
                failed += 1
        except Exception as e:
            print(f"  FAIL  {test.name}  ({e})")
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} tests passed")
    if failed:
        sys.exit(1)


_COMMANDS = {
    "lex":     cmd_lex,
    "parse":   cmd_parse,
    "analyze": cmd_analyze,
    "ir":      cmd_ir,
    "run":     cmd_run_ex,
    "compile": cmd_compile_ex,
    "repl":    cmd_repl,
    "fmt":     cmd_fmt,
    "doc":     cmd_doc,
    "build":   cmd_build,
    "test":    cmd_test,
}

# Commands that don't take a file argument
_NO_FILE_CMDS = {"repl", "build"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="vexel", description="The Vexel compiler")
    parser.add_argument("command", choices=list(_COMMANDS), help="Action to perform")
    parser.add_argument("file", nargs="?", default=None, help="Vexel source file (.vx)")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--sdl2",   action="store_true", help="Enable SDL2 built-ins")
    parser.add_argument("--watch",  action="store_true", help="Recompile on file change (run only)")
    parser.add_argument("--target", default=None,        help="Cross-compile target triple (compile only)")
    parser.add_argument("--debug",  action="store_true", help="Enable debug mode (bounds checks etc.)")
    args = parser.parse_args()

    if args.command not in _NO_FILE_CMDS and args.file is None:
        parser.error(f"command '{args.command}' requires a file argument")

    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
