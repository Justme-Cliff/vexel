"""
vexel — the Vexel compiler

Commands:
  python main.py lex      <file.vx>            Print token stream
  python main.py parse    <file.vx>            Print AST
  python main.py analyze  <file.vx>            Run semantic analysis
  python main.py ir       <file.vx>            Print LLVM IR
  python main.py run      <file.vx>            JIT-compile and run
  python main.py compile  <file.vx> [-o out]   Compile to native binary
                          [--sdl2]             Link with SDL2

Flags:
  -o <output>   Output binary name (default: same name as input, no extension)
  --sdl2        Enable SDL2 built-ins and link SDL2 (requires setup_sdl2.py)
"""

import sys
import os
import argparse

# ------------------------------------------------------------------ #
#  Import resolver (v3)                                                #
# ------------------------------------------------------------------ #

def resolve_imports(program, base_dir: str):
    """
    Walk program declarations, load any ImportStmt files, recursively resolve
    their imports, and return a new Program with all declarations merged.
    Imported 'main' functions are dropped.
    """
    import os as _os
    from compiler.lexer import Lexer, LexError
    from compiler.parser import Parser, ParseError
    from compiler.ast_nodes import ImportStmt, FnDecl, Program

    merged = []
    visited = set()

    # Normalise base_dir to an absolute path for comparison
    base_dir = _os.path.abspath(base_dir)

    def process(prog, cur_dir: str):
        cur_dir = _os.path.abspath(cur_dir)
        for decl in prog.declarations:
            if isinstance(decl, ImportStmt):
                path = decl.path
                if not path.endswith(".vx"):
                    path = path + ".vx"
                abs_path = _os.path.normpath(_os.path.join(cur_dir, path))
                if abs_path in visited:
                    continue
                visited.add(abs_path)
                try:
                    with open(abs_path, encoding="utf-8") as f:
                        src = f.read()
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
                process(sub_prog, _os.path.dirname(abs_path))
            else:
                # Drop 'main' functions that come from imported files
                if isinstance(decl, FnDecl) and decl.name == "main" \
                        and cur_dir != base_dir:
                    continue
                merged.append(decl)

    process(program, base_dir)
    from compiler.ast_nodes import Program
    return Program(merged)


# ------------------------------------------------------------------ #
#  Shared pipeline helpers                                             #
# ------------------------------------------------------------------ #

def read_source(path: str) -> str:
    if not os.path.exists(path):
        die(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def die(msg: str):
    print(f"vexel error: {msg}", file=sys.stderr)
    sys.exit(1)


def pipeline_lex(source: str):
    from compiler.lexer import Lexer, LexError
    try:
        return Lexer(source).tokenize()
    except LexError as e:
        die(str(e))


def pipeline_parse(tokens):
    from compiler.parser import Parser, ParseError
    try:
        return Parser(tokens).parse()
    except ParseError as e:
        die(str(e))


def pipeline_analyze(program, sdl2: bool = False):
    from compiler.analyzer import Analyzer
    from compiler.sdl2_builtins import SDL2_BUILTINS, SDL2_C_NAMES
    from compiler.analyzer import FnSig

    analyzer = Analyzer()

    # Inject SDL2 built-in signatures before analysis
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


def pipeline_codegen(program, analysis, sdl2: bool = False, target_triple: str | None = None):
    from compiler.codegen import Compiler
    from compiler.sdl2_builtins import SDL2_BUILTINS, SDL2_C_NAMES
    from llvmlite import ir
    from compiler.codegen import I64_TY, F64_TY, I1_TY, I8PTR, VOID_TY, I32_TY

    compiler = Compiler(analysis, target_triple=target_triple)

    # Inject SDL2 extern declarations before compiling
    if sdl2:
        for vx_name, (param_types, ret_type) in SDL2_BUILTINS.items():
            c_name = SDL2_C_NAMES[vx_name]
            llvm_params = [compiler._vx_to_llvm(t) for t in param_types]
            llvm_ret    = compiler._vx_to_llvm(ret_type)
            fn_type     = ir.FunctionType(llvm_ret, llvm_params)
            fn          = ir.Function(compiler.module, fn_type, name=c_name)

            # Also register under the Vexel name so calls resolve
            from compiler.analyzer import FnSig
            sig = analysis.fn_sigs.get(vx_name) or FnSig(
                [(f"a{i}", t) for i, t in enumerate(param_types)], ret_type
            )
            compiler._functions[vx_name] = {"fn": fn, "sig": sig}

    return compiler.compile(program)


# ------------------------------------------------------------------ #
#  Commands                                                            #
# ------------------------------------------------------------------ #

def cmd_lex(args):
    source = read_source(args.file)
    tokens = pipeline_lex(source)
    for tok in tokens:
        print(tok)


def cmd_parse(args):
    import pprint
    source  = read_source(args.file)
    tokens  = pipeline_lex(source)
    program = pipeline_parse(tokens)
    pprint.pprint(program, indent=2)


def cmd_analyze(args):
    base_dir = os.path.dirname(os.path.abspath(args.file))
    source   = read_source(args.file)
    tokens   = pipeline_lex(source)
    program  = pipeline_parse(tokens)
    program  = resolve_imports(program, base_dir)
    analysis = pipeline_analyze(program, sdl2=args.sdl2)
    print("Analysis OK")
    print(f"  Functions : {list(analysis.fn_sigs.keys())}")
    print(f"  Structs   : {list(analysis.struct_fields.keys())}")


def cmd_ir(args):
    base_dir = os.path.dirname(os.path.abspath(args.file))
    source   = read_source(args.file)
    tokens   = pipeline_lex(source)
    program  = pipeline_parse(tokens)
    program  = resolve_imports(program, base_dir)
    analysis = pipeline_analyze(program, sdl2=args.sdl2)
    llvm_ir  = pipeline_codegen(program, analysis, sdl2=args.sdl2)
    print(llvm_ir)


def cmd_run(args):
    from compiler.codegen import jit_run, CodegenError
    base_dir = os.path.dirname(os.path.abspath(args.file))
    source   = read_source(args.file)
    tokens   = pipeline_lex(source)
    program  = pipeline_parse(tokens)
    program  = resolve_imports(program, base_dir)
    analysis = pipeline_analyze(program)
    llvm_ir  = pipeline_codegen(program, analysis)
    try:
        exit_code = jit_run(llvm_ir)
        sys.exit(exit_code)
    except CodegenError as e:
        die(str(e))


def cmd_compile(args):
    from compiler.codegen import compile_to_binary, CodegenError, MINGW_TRIPLE
    import subprocess, tempfile, os

    GCC = r"C:\Strawberry\c\bin\gcc.exe"
    if not os.path.exists(GCC):
        GCC = "gcc"  # hope it's on PATH

    # Use default MSVC triple on Windows; runtime.c provides __chkstk stub
    # for MinGW compatibility.
    target_triple = None

    base_dir = os.path.dirname(os.path.abspath(args.file))
    source   = read_source(args.file)
    tokens   = pipeline_lex(source)
    program  = pipeline_parse(tokens)
    program  = resolve_imports(program, base_dir)
    analysis = pipeline_analyze(program, sdl2=args.sdl2)
    llvm_ir  = pipeline_codegen(program, analysis, sdl2=args.sdl2,
                                 target_triple=target_triple)

    stem   = os.path.splitext(os.path.basename(args.file))[0]
    output = args.output or stem
    if sys.platform == "win32" and not output.endswith(".exe"):
        output += ".exe"

    # Compile runtime.c to a temp object
    runtime_c = os.path.join(os.path.dirname(__file__), "stdlib", "runtime.c")
    runtime_obj = None
    if os.path.exists(runtime_c):
        runtime_obj = tempfile.mktemp(suffix=".o")
        r = subprocess.run(
            [GCC, "-O2", "-c", runtime_c, "-o", runtime_obj],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            die(f"Failed to compile runtime.c:\n{r.stderr}")

    # Compile SDL2 bindings if requested
    sdl2_obj = None
    if args.sdl2:
        sdl2_c   = os.path.join(os.path.dirname(__file__), "stdlib", "vx_sdl2.c")
        sdl2_inc = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "include")
        sdl2_obj = tempfile.mktemp(suffix=".o")
        r = subprocess.run(
            [GCC, "-O2", "-DVX_SDL2_ENABLED", "-c", sdl2_c,
             f"-I{sdl2_inc}", "-o", sdl2_obj],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            die(f"Failed to compile vx_sdl2.c (is SDL2 installed?):\n{r.stderr}")

    try:
        # Build the link command
        from compiler.codegen import _init_llvm
        from llvmlite import binding
        import tempfile

        _init_llvm()
        triple = target_triple or binding.get_default_triple()
        target = binding.Target.from_triple(triple)
        tm     = target.create_target_machine(reloc="pic", codemodel="default")
        mod    = binding.parse_assembly(llvm_ir)
        mod.verify()
        obj_bytes = tm.emit_object(mod)

        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as obj_f:
            obj_f.write(obj_bytes)
            obj_path = obj_f.name

        cmd = [GCC, obj_path]
        if runtime_obj:  cmd.append(runtime_obj)
        if sdl2_obj:     cmd.append(sdl2_obj)
        cmd += ["-o", output, "-lm"]

        if args.sdl2:
            sdl2_lib = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2", "lib")
            cmd += [f"-L{sdl2_lib}", "-lSDL2", "-lSDL2main"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            die(f"Link error:\n{result.stderr}")

        print(f"Compiled: {output}")

    finally:
        try: os.unlink(obj_path)
        except: pass
        if runtime_obj:
            try: os.unlink(runtime_obj)
            except: pass
        if sdl2_obj:
            try: os.unlink(sdl2_obj)
            except: pass


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(prog="vexel", description="The Vexel compiler")
    parser.add_argument("command",
                        choices=["lex", "parse", "analyze", "ir", "run", "compile"],
                        help="What to do")
    parser.add_argument("file", help="Vexel source file (.vx)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output binary path (compile only)")
    parser.add_argument("--sdl2", action="store_true",
                        help="Enable SDL2 built-ins")
    args = parser.parse_args()

    dispatch = {
        "lex":     cmd_lex,
        "parse":   cmd_parse,
        "analyze": cmd_analyze,
        "ir":      cmd_ir,
        "run":     cmd_run,
        "compile": cmd_compile,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
