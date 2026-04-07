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
                     target_triple: str | None = None) -> str:
    from compiler.codegen import Compiler
    from compiler.sdl2_builtins import SDL2_BUILTINS, SDL2_C_NAMES
    from llvmlite import ir

    compiler = Compiler(analysis, target_triple=target_triple)
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

    program, analysis = pipeline_full(args)
    llvm_ir = pipeline_codegen(program, analysis)
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
#  Entry point                                                         #
# ------------------------------------------------------------------ #

_COMMANDS = {
    "lex":     cmd_lex,
    "parse":   cmd_parse,
    "analyze": cmd_analyze,
    "ir":      cmd_ir,
    "run":     cmd_run,
    "compile": cmd_compile,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="vexel", description="The Vexel compiler")
    parser.add_argument("command", choices=_COMMANDS, help="Action to perform")
    parser.add_argument("file", help="Vexel source file (.vx)")
    parser.add_argument("-o", "--output", default=None, help="Output binary path (compile only)")
    parser.add_argument("--sdl2", action="store_true", help="Enable SDL2 built-ins")
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
