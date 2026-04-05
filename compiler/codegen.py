"""
Vexel LLVM Code Generator  (v3)
---------------------------------
New in v2:
  - Arrays (int[], float[], bool[], str[]) — heap-allocated {i8*, i64} header
  - String concatenation (+) and equality (==)
  - Math built-ins: sqrt abs min max pow floor ceil
  - Type casts: int() float() str() bool()
  - len() for arrays and strings
  - break / continue
  - Compound assignment (handled by parser, transparent here)
  - for-each over arrays
  - Global variables (constants)
  - Multi-arg print  →  space-separated on one line
  - Null literal
  - Float printing via %g (no trailing zeros)
  - Index assignment

New in v3:
  - Dynamic arrays: {i8*, i64, i64} (data, len, cap)
  - String methods: len, upper, lower, trim, contains, starts_with, ends_with,
                    replace, split, and string indexing
  - Array methods: len, push, pop, contains, reverse
  - File I/O: read_file, write_file, append_file, file_exists
  - Enums (int constants)
  - Match statement
  - Assert statement
  - Ternary expression
  - More math: sin, cos, tan, log, log2, rand, rand_int
  - PI / E constants
  - exit() builtin
  - Import statement (no-op; resolved before codegen)
  - Private LLVM helper functions
"""

from __future__ import annotations
import ctypes, sys
from llvmlite import ir, binding
from compiler.ast_nodes import *
from compiler.analyzer import AnalysisResult

# ------------------------------------------------------------------ #
#  Primitive LLVM types                                               #
# ------------------------------------------------------------------ #
I1_TY   = ir.IntType(1)
I8_TY   = ir.IntType(8)
I32_TY  = ir.IntType(32)
I64_TY  = ir.IntType(64)
F64_TY  = ir.DoubleType()
VOID_TY = ir.VoidType()
I8PTR   = ir.PointerType(I8_TY)

MINGW_TRIPLE = "x86_64-w64-mingw32"

# Hardcoded element sizes (bytes)
_ELEM_SIZES = {"int": 8, "float": 8, "bool": 1, "str": 8}
_ARRAY_HEADER_SIZE = 24   # {i8*(8), i64(8), i64(8)} — data, len, cap


def _elem_size(vx_type: str) -> int:
    return _ELEM_SIZES.get(vx_type, 8)


class CodegenError(Exception):
    pass


class Compiler:
    def __init__(self, analysis: AnalysisResult,
                 target_triple: str | None = None):
        self.analysis  = analysis
        self.module    = ir.Module(name="vexel_module")
        self.module.triple = target_triple or binding.get_default_triple()
        self.builder:    ir.IRBuilder | None = None
        self.current_fn: ir.Function  | None = None

        # Scope stack: [{name: {"ptr": alloca, "vx_type": str}}]
        self._scope_stack: list[dict] = [{}]

        # Compiled function registry
        self._functions:  dict[str, dict] = {}
        self._structs:    dict[str, dict] = {}
        self._globals:    dict[str, dict] = {}   # global vars

        # String constant cache
        self._gstr_cache:   dict[str, ir.GlobalVariable] = {}
        self._gstr_counter: int = 0

        # Loop break/continue stacks
        self._break_targets:    list[ir.Block] = []
        self._continue_targets: list[ir.Block] = []

        # Private helper function cache
        self._helper_fns: dict[str, ir.Function] = {}

        self._define_array_type()
        self._declare_externs()

    # ------------------------------------------------------------------ #
    #  Array struct type                                                   #
    # ------------------------------------------------------------------ #

    def _define_array_type(self):
        self.arr_type = self.module.context.get_identified_type("vx_array")
        if self.arr_type.is_opaque:
            # {i8* data, i64 len, i64 cap}
            self.arr_type.set_body(I8PTR, I64_TY, I64_TY)
        self.arr_ptr_type = ir.PointerType(self.arr_type)

    # ------------------------------------------------------------------ #
    #  External declarations                                               #
    # ------------------------------------------------------------------ #

    def _declare_externs(self):
        def _fn(ret, *params, name, vararg=False):
            ft = ir.FunctionType(ret, list(params), var_arg=vararg)
            return ir.Function(self.module, ft, name=name)

        self.printf      = _fn(I32_TY, I8PTR,          name="printf",   vararg=True)
        self.malloc_fn   = _fn(I8PTR,  I64_TY,          name="malloc")
        self.free_fn     = _fn(VOID_TY, I8PTR,          name="free")
        self.strlen_fn   = _fn(I64_TY,  I8PTR,          name="strlen")
        self.memcpy_fn   = _fn(I8PTR,   I8PTR, I8PTR, I64_TY, name="memcpy")
        self.sprintf_fn  = _fn(I32_TY,  I8PTR,          name="sprintf", vararg=True)

        # Math (libm)
        self.sqrt_fn  = _fn(F64_TY, F64_TY,         name="sqrt")
        self.fabs_fn  = _fn(F64_TY, F64_TY,         name="fabs")
        self.llabs_fn = _fn(I64_TY, I64_TY,         name="llabs")
        self.pow_fn   = _fn(F64_TY, F64_TY, F64_TY, name="pow")
        self.floor_fn = _fn(F64_TY, F64_TY,         name="floor")
        self.ceil_fn  = _fn(F64_TY, F64_TY,         name="ceil")

        # strcmp for string equality
        self.strcmp_fn  = _fn(I32_TY, I8PTR, I8PTR,         name="strcmp")

        # v3 additional libc
        self.realloc_fn = _fn(I8PTR,  I8PTR, I64_TY,        name="realloc")
        self.strncmp_fn = _fn(I32_TY, I8PTR, I8PTR, I64_TY, name="strncmp")
        self.strstr_fn  = _fn(I8PTR,  I8PTR, I8PTR,         name="strstr")
        self.toupper_fn = _fn(I32_TY, I32_TY,               name="toupper")
        self.tolower_fn = _fn(I32_TY, I32_TY,               name="tolower")
        self.exit_fn    = _fn(VOID_TY, I32_TY,              name="exit")
        self.rand_fn    = _fn(I32_TY,                        name="rand")
        self.srand_fn   = _fn(VOID_TY, I32_TY,              name="srand")
        self.time_fn    = _fn(I64_TY,  I8PTR,               name="time")
        self.sin_fn     = _fn(F64_TY,  F64_TY,              name="sin")
        self.cos_fn     = _fn(F64_TY,  F64_TY,              name="cos")
        self.tan_fn     = _fn(F64_TY,  F64_TY,              name="tan")
        self.log_fn     = _fn(F64_TY,  F64_TY,              name="log")
        self.log2_fn    = _fn(F64_TY,  F64_TY,              name="log2")

        # File I/O
        self.fopen_fn   = _fn(I8PTR,  I8PTR, I8PTR,                  name="fopen")
        self.fclose_fn  = _fn(I32_TY, I8PTR,                          name="fclose")
        self.fread_fn   = _fn(I64_TY, I8PTR, I64_TY, I64_TY, I8PTR,  name="fread")
        self.fwrite_fn  = _fn(I64_TY, I8PTR, I64_TY, I64_TY, I8PTR,  name="fwrite")
        self.fseek_fn   = _fn(I32_TY, I8PTR, I64_TY, I32_TY,         name="fseek")
        self.ftell_fn   = _fn(I64_TY, I8PTR,                          name="ftell")

    # ------------------------------------------------------------------ #
    #  Global string helpers                                               #
    # ------------------------------------------------------------------ #

    def _global_str(self, content: str) -> ir.GlobalVariable:
        if content in self._gstr_cache:
            return self._gstr_cache[content]
        raw    = content.encode("utf8") + b"\0"
        arr_ty = ir.ArrayType(I8_TY, len(raw))
        gv     = ir.GlobalVariable(self.module, arr_ty,
                                   name=f".str.{self._gstr_counter}")
        self._gstr_counter += 1
        gv.linkage        = "private"
        gv.global_constant = True
        gv.initializer    = ir.Constant(arr_ty, bytearray(raw))
        self._gstr_cache[content] = gv
        return gv

    def _gstr_ptr(self, gv: ir.GlobalVariable) -> ir.Value:
        z = ir.Constant(I32_TY, 0)
        return self.builder.gep(gv, [z, z], inbounds=True)

    def _gstr_ptr_const(self, gv: ir.GlobalVariable) -> ir.Value:
        """Return a constant GEP (for use outside a builder context)."""
        z = ir.Constant(I32_TY, 0)
        return gv.gep([z, z])

    # ------------------------------------------------------------------ #
    #  Type helpers                                                        #
    # ------------------------------------------------------------------ #

    def _vx_to_llvm(self, vx: str) -> ir.Type:
        if vx == "int":    return I64_TY
        if vx == "float":  return F64_TY
        if vx == "bool":   return I1_TY
        if vx == "str":    return I8PTR
        if vx == "void":   return VOID_TY
        if vx == "null":   return I8PTR
        if vx.endswith("[]"):  return self.arr_ptr_type
        if vx in self._structs:
            return ir.PointerType(self._structs[vx]["llvm_type"])
        raise CodegenError(f"Unknown type: {vx!r}")

    def _infer_type(self, node: Node) -> str:
        if isinstance(node, IntLiteral):    return "int"
        if isinstance(node, FloatLiteral):  return "float"
        if isinstance(node, BoolLiteral):   return "bool"
        if isinstance(node, StringLiteral): return "str"
        if isinstance(node, NullLiteral):   return "null"
        if isinstance(node, ArrayLiteral):
            if not node.elements: return "int[]"
            return self._infer_type(node.elements[0]) + "[]"
        if isinstance(node, Identifier):
            if node.name in ("PI", "E"): return "float"
            info = self._lookup(node.name)
            return info["vx_type"] if info else "int"
        if isinstance(node, BinOp):
            if node.op in ("==","!=","<",">","<=",">=","and","or"): return "bool"
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if node.op == "+" and lt == "str": return "str"
            return "float" if lt == "float" or rt == "float" else lt
        if isinstance(node, UnaryOp):
            return "bool" if node.op == "not" else self._infer_type(node.operand)
        if isinstance(node, Call):
            # Overloaded builtins
            if node.func in ("abs","min","max") and node.args:
                at = self._infer_type(node.args[0])
                return "float" if at == "float" else "int"
            sig = self.analysis.fn_sigs.get(node.func)
            return sig.return_type if sig else "void"
        if isinstance(node, MethodCall):
            obj_t = self._infer_type(node.obj)
            if obj_t == "str":
                return {"len": "int", "upper": "str", "lower": "str", "trim": "str",
                        "contains": "bool", "starts_with": "bool", "ends_with": "bool",
                        "replace": "str", "split": "str[]"}.get(node.method, "str")
            if obj_t.endswith("[]"):
                elem_t = obj_t[:-2]
                return {"len": "int", "push": "void", "pop": elem_t,
                        "contains": "bool", "reverse": "void"}.get(node.method, "void")
            return "void"
        if isinstance(node, TernaryExpr):
            return self._infer_type(node.then_val)
        if isinstance(node, FieldAccess):
            # Check if this is an enum access: Color.Red
            if isinstance(node.obj, Identifier):
                dotted = f"{node.obj.name}.{node.field}"
                info = self._lookup(dotted)
                if info is not None:
                    return info["vx_type"]
            ot = self._infer_type(node.obj)
            s  = self._structs.get(ot)
            if s:
                for fn, ft in s["fields"]:
                    if fn == node.field: return ft
            return "int"
        if isinstance(node, NewExpr):   return node.type_name
        if isinstance(node, IndexExpr):
            ot = self._infer_type(node.obj)
            if ot == "str": return "str"
            return ot[:-2] if ot.endswith("[]") else "int"
        return "int"

    # ------------------------------------------------------------------ #
    #  Scope                                                               #
    # ------------------------------------------------------------------ #

    def _push_scope(self): self._scope_stack.append({})
    def _pop_scope(self):  self._scope_stack.pop()

    def _declare(self, name: str, ptr: ir.Value, vx_type: str):
        self._scope_stack[-1][name] = {"ptr": ptr, "vx_type": vx_type}

    def _lookup(self, name: str) -> dict | None:
        for s in reversed(self._scope_stack):
            if name in s: return s[name]
        return self._globals.get(name)

    # ------------------------------------------------------------------ #
    #  Main compile entry                                                  #
    # ------------------------------------------------------------------ #

    def compile(self, program: Program) -> str:
        # 1. Struct definitions
        for d in program.declarations:
            if isinstance(d, StructDecl): self._define_struct(d)

        # 2. Enum definitions (global i64 constants)
        for d in program.declarations:
            if isinstance(d, EnumDecl):
                for i, variant in enumerate(d.variants):
                    gname = f"{d.name}.{variant}"
                    gv = ir.GlobalVariable(self.module, I64_TY, name=gname)
                    gv.linkage = "internal"
                    gv.global_constant = True
                    gv.initializer = ir.Constant(I64_TY, i)
                    self._globals[gname] = {"ptr": gv, "vx_type": "int"}

        # 3. Global variables
        for d in program.declarations:
            if isinstance(d, (GlobalLet, GlobalConst)):
                self._compile_global(d)

        # 4. Forward-declare all functions
        for d in program.declarations:
            if isinstance(d, FnDecl): self._declare_fn(d)

        # 5. Compile function bodies
        for d in program.declarations:
            if isinstance(d, FnDecl): self._compile_fn(d)

        return str(self.module)

    # ------------------------------------------------------------------ #
    #  Globals                                                             #
    # ------------------------------------------------------------------ #

    def _compile_global(self, d):
        name    = d.name
        vx_type = d.type_annotation or self._infer_type(d.value)
        ll_type = self._vx_to_llvm(vx_type)

        # Only constant initializers supported
        if isinstance(d.value, IntLiteral):
            init = ir.Constant(ll_type, d.value.value)
        elif isinstance(d.value, FloatLiteral):
            init = ir.Constant(ll_type, d.value.value)
        elif isinstance(d.value, BoolLiteral):
            init = ir.Constant(ll_type, int(d.value.value))
        elif isinstance(d.value, StringLiteral):
            init = ir.Constant(I8PTR, None)
            ll_type = I8PTR
        else:
            init = ir.Constant(ll_type, 0)

        gv          = ir.GlobalVariable(self.module, ll_type, name=name)
        gv.linkage  = "internal"
        gv.initializer = init
        self._globals[name] = {"ptr": gv, "vx_type": vx_type}

    # ------------------------------------------------------------------ #
    #  Structs                                                             #
    # ------------------------------------------------------------------ #

    def _define_struct(self, d: StructDecl):
        lt = self.module.context.get_identified_type(d.name)
        lt.set_body(*[self._vx_to_llvm(f.type_name) for f in d.fields])
        self._structs[d.name] = {
            "llvm_type": lt,
            "fields": [(f.name, f.type_name) for f in d.fields],
        }

    # ------------------------------------------------------------------ #
    #  Functions                                                           #
    # ------------------------------------------------------------------ #

    def _declare_fn(self, d: FnDecl):
        sig        = self.analysis.fn_sigs[d.name]
        param_tys  = [self._vx_to_llvm(t) for _, t in sig.params]
        ret_ty     = I32_TY if d.name == "main" and sig.return_type == "void" \
                     else self._vx_to_llvm(sig.return_type)
        fn_ty      = ir.FunctionType(ret_ty, param_tys)
        fn         = ir.Function(self.module, fn_ty, name=d.name)
        for i, (pname, _) in enumerate(sig.params):
            fn.args[i].name = pname
        self._functions[d.name] = {"fn": fn, "sig": sig}

    def _compile_fn(self, d: FnDecl):
        info = self._functions[d.name]
        fn   = info["fn"]
        self.current_fn = fn

        entry = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._push_scope()

        for arg, param in zip(fn.args, d.params):
            al = self.builder.alloca(self._vx_to_llvm(param.type_name), name=param.name)
            self.builder.store(arg, al)
            self._declare(param.name, al, param.type_name)

        for stmt in d.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)

        if not self.builder.block.is_terminated:
            if fn.ftype.return_type == VOID_TY:
                self.builder.ret_void()
            else:
                self.builder.ret(ir.Constant(fn.ftype.return_type, 0))

        self._pop_scope()

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _compile_stmt(self, node: Node):
        if self.builder.block.is_terminated:
            return   # dead code — skip

        if   isinstance(node, LetStmt):         self._compile_let(node)
        elif isinstance(node, AssignStmt):       self._compile_assign(node)
        elif isinstance(node, IndexAssignStmt):  self._compile_index_assign(node)
        elif isinstance(node, ReturnStmt):       self._compile_return(node)
        elif isinstance(node, PrintStmt):        self._compile_print(node)
        elif isinstance(node, IfStmt):           self._compile_if(node)
        elif isinstance(node, ForStmt):          self._compile_for(node)
        elif isinstance(node, ForEach):          self._compile_foreach(node)
        elif isinstance(node, WhileStmt):        self._compile_while(node)
        elif isinstance(node, BreakStmt):        self._compile_break()
        elif isinstance(node, ContinueStmt):     self._compile_continue()
        elif isinstance(node, ExprStmt):         self._compile_expr(node.expr)
        elif isinstance(node, MatchStmt):        self._compile_match(node)
        elif isinstance(node, AssertStmt):       self._compile_assert(node)
        elif isinstance(node, (EnumDecl, ImportStmt)):
            pass  # handled in compile() pass or before codegen
        else:
            raise CodegenError(f"Unknown stmt: {type(node).__name__}")

    def _compile_let(self, node: LetStmt):
        val, vt  = self._compile_expr(node.value)
        declared = node.type_annotation or vt
        ll_ty    = self._vx_to_llvm(declared)

        if declared == "float" and vt == "int":
            val = self.builder.sitofp(val, F64_TY); vt = "float"

        al = self.builder.alloca(ll_ty, name=node.name)
        self.builder.store(val, al)
        self._declare(node.name, al, declared)

    def _compile_assign(self, node: AssignStmt):
        val, vt = self._compile_expr(node.value)
        if isinstance(node.target, Identifier):
            info = self._lookup(node.target.name)
            if info is None:
                raise CodegenError(f"Undefined variable '{node.target.name}'")
            # Auto-promote int→float
            if info["vx_type"] == "float" and vt == "int":
                val = self.builder.sitofp(val, F64_TY)
            self.builder.store(val, info["ptr"])
        elif isinstance(node.target, FieldAccess):
            ptr = self._field_ptr(node.target)
            self.builder.store(val, ptr)

    def _compile_index_assign(self, node: IndexAssignStmt):
        val, vt      = self._compile_expr(node.value)
        arr_val, avt = self._compile_expr(node.obj)
        idx_val, _   = self._compile_expr(node.index)
        elem_vt      = avt[:-2] if avt.endswith("[]") else "int"
        elem_lt      = self._vx_to_llvm(elem_vt)

        data_ptr = self._arr_data_ptr(arr_val, elem_lt)
        ep       = self.builder.gep(data_ptr, [idx_val], inbounds=True)
        if elem_vt == "float" and vt == "int":
            val = self.builder.sitofp(val, F64_TY)
        self.builder.store(val, ep)

    def _compile_return(self, node: ReturnStmt):
        ret_ty = self.current_fn.ftype.return_type
        if node.value is None:
            if ret_ty == VOID_TY: self.builder.ret_void()
            else:                 self.builder.ret(ir.Constant(ret_ty, 0))
        else:
            val, vt = self._compile_expr(node.value)
            if ret_ty == F64_TY and vt == "int":
                val = self.builder.sitofp(val, F64_TY)
            self.builder.ret(val)

    # ------------------------------------------------------------------ #
    #  Print (multi-arg, space-separated)                                 #
    # ------------------------------------------------------------------ #

    def _compile_print(self, node: PrintStmt):
        for i, v in enumerate(node.values):
            val, vt = self._compile_expr(v)
            self._print_value(val, vt)
            if i < len(node.values) - 1:
                sp = self._global_str(" ")
                self.builder.call(self.printf, [self._gstr_ptr(sp)])
        nl = self._global_str("\n")
        self.builder.call(self.printf, [self._gstr_ptr(nl)])

    def _print_value(self, val: ir.Value, vt: str):
        if vt == "int":
            fmt = self._gstr_ptr(self._global_str("%lld"))
            self.builder.call(self.printf, [fmt, val])
        elif vt == "float":
            fmt = self._gstr_ptr(self._global_str("%g"))
            self.builder.call(self.printf, [fmt, val])
        elif vt == "str":
            fmt = self._gstr_ptr(self._global_str("%s"))
            self.builder.call(self.printf, [fmt, val])
        elif vt == "bool":
            t = self._gstr_ptr(self._global_str("true"))
            f = self._gstr_ptr(self._global_str("false"))
            s = self.builder.select(val, t, f)
            fmt = self._gstr_ptr(self._global_str("%s"))
            self.builder.call(self.printf, [fmt, s])
        elif vt.endswith("[]"):
            fmt = self._gstr_ptr(self._global_str(f"<{vt} len="))
            self.builder.call(self.printf, [fmt])
            lp  = self.builder.gep(val, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
            ln  = self.builder.load(lp)
            fmtd = self._gstr_ptr(self._global_str("%lld>"))
            self.builder.call(self.printf, [fmtd, ln])
        else:
            # Struct / unknown — print type
            fmt = self._gstr_ptr(self._global_str(f"<{vt}>"))
            self.builder.call(self.printf, [fmt])

    # ------------------------------------------------------------------ #
    #  If / elif / else                                                    #
    # ------------------------------------------------------------------ #

    def _compile_if(self, node: IfStmt):
        cond, _ = self._compile_expr(node.condition)
        fn      = self.current_fn

        then_b  = fn.append_basic_block("if.then")
        else_b  = fn.append_basic_block("if.else")
        merge_b = fn.append_basic_block("if.merge")

        self.builder.cbranch(cond, then_b, else_b)

        # then
        self.builder.position_at_end(then_b)
        self._push_scope()
        for s in node.then_body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(s)
        self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_b)

        # else
        self.builder.position_at_end(else_b)
        if node.else_body:
            self._push_scope()
            for s in node.else_body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)

    # ------------------------------------------------------------------ #
    #  Match statement                                                     #
    # ------------------------------------------------------------------ #

    def _compile_match(self, node: MatchStmt):
        fn = self.current_fn
        val, vt = self._compile_expr(node.value)

        # Alloca to hold the match value (so we can reload in each case)
        if vt == "float":
            val_al = self.builder.alloca(F64_TY, name="match_val")
        elif vt == "str":
            val_al = self.builder.alloca(I8PTR, name="match_val")
        else:
            val_al = self.builder.alloca(I64_TY, name="match_val")
        self.builder.store(val, val_al)

        merge_b = fn.append_basic_block("match.merge")

        for case in node.cases:
            # Build an "any pattern matches" check
            # For each pattern, compare and OR together
            case_body_b = fn.append_basic_block("match.case.body")
            next_b = fn.append_basic_block("match.case.next")

            # Evaluate all patterns and OR them
            # Load val once per case
            loaded = self.builder.load(val_al)

            # Build chain: if pat1 or pat2 or ... -> case_body_b else next_b
            combined_cond = None
            for pat in case.patterns:
                pv, pt = self._compile_expr(pat)
                if vt == "str" or pt == "str":
                    r = self.builder.call(self.strcmp_fn, [loaded, pv])
                    c = self.builder.icmp_signed("==", r, ir.Constant(I32_TY, 0))
                elif vt == "float" or pt == "float":
                    if pt == "int": pv = self.builder.sitofp(pv, F64_TY)
                    if vt == "int": loaded_f = self.builder.sitofp(loaded, F64_TY)
                    else: loaded_f = loaded
                    c = self.builder.fcmp_ordered("==", loaded_f, pv)
                else:
                    c = self.builder.icmp_signed("==", loaded, pv)
                if combined_cond is None:
                    combined_cond = c
                else:
                    combined_cond = self.builder.or_(combined_cond, c)

            if combined_cond is None:
                combined_cond = ir.Constant(I1_TY, 0)

            self.builder.cbranch(combined_cond, case_body_b, next_b)

            self.builder.position_at_end(case_body_b)
            self._push_scope()
            for s in case.body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_b)

            self.builder.position_at_end(next_b)

        # Default body (currently positioned at last next_b)
        if node.default_body:
            self._push_scope()
            for s in node.default_body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)

    # ------------------------------------------------------------------ #
    #  Assert statement                                                    #
    # ------------------------------------------------------------------ #

    def _compile_assert(self, node: AssertStmt):
        fn = self.current_fn
        cond, _ = self._compile_expr(node.condition)

        ok_b   = fn.append_basic_block("assert.ok")
        fail_b = fn.append_basic_block("assert.fail")
        self.builder.cbranch(cond, ok_b, fail_b)

        self.builder.position_at_end(fail_b)
        if node.message:
            msg_v, _ = self._compile_expr(node.message)
            fmt = self._gstr_ptr(self._global_str("Assertion failed: %s\n"))
            self.builder.call(self.printf, [fmt, msg_v])
        else:
            fmt = self._gstr_ptr(self._global_str("Assertion failed\n"))
            self.builder.call(self.printf, [fmt])
        self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
        self.builder.unreachable()

        self.builder.position_at_end(ok_b)

    # ------------------------------------------------------------------ #
    #  Loops                                                               #
    # ------------------------------------------------------------------ #

    def _compile_for(self, node: ForStmt):
        fn          = self.current_fn
        start_v, _  = self._compile_expr(node.start)
        end_v,   _  = self._compile_expr(node.end)
        if start_v.type != I64_TY: start_v = self.builder.fptosi(start_v, I64_TY)
        if end_v.type   != I64_TY: end_v   = self.builder.fptosi(end_v,   I64_TY)

        i_al = self.builder.alloca(I64_TY, name=node.var)
        self.builder.store(start_v, i_al)

        chk = fn.append_basic_block("for.check")
        bdy = fn.append_basic_block("for.body")
        ext = fn.append_basic_block("for.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv   = self.builder.load(i_al)
        cond = self.builder.icmp_signed("<", iv, end_v)
        self.builder.cbranch(cond, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        self._declare(node.var, i_al, "int")
        self._break_targets.append(ext)
        self._continue_targets.append(chk)
        for s in node.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(s)
        self._break_targets.pop()
        self._continue_targets.pop()
        self._pop_scope()
        if not self.builder.block.is_terminated:
            ic   = self.builder.load(i_al)
            inxt = self.builder.add(ic, ir.Constant(I64_TY, 1))
            self.builder.store(inxt, i_al)
            self.builder.branch(chk)

        self.builder.position_at_end(ext)

    def _compile_foreach(self, node: ForEach):
        fn         = self.current_fn
        arr_v, avt = self._compile_expr(node.iterable)
        elem_vt    = avt[:-2] if avt.endswith("[]") else "int"
        elem_lt    = self._vx_to_llvm(elem_vt)

        # Load length (field index 1)
        lp     = self.builder.gep(arr_v, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        arr_ln = self.builder.load(lp)

        # Counter
        i_al = self.builder.alloca(I64_TY, name="_fe_i")
        self.builder.store(ir.Constant(I64_TY, 0), i_al)

        # Item alloca
        item_al = self.builder.alloca(elem_lt, name=node.var)

        chk = fn.append_basic_block("fe.check")
        bdy = fn.append_basic_block("fe.body")
        ext = fn.append_basic_block("fe.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv   = self.builder.load(i_al)
        cond = self.builder.icmp_signed("<", iv, arr_ln)
        self.builder.cbranch(cond, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        self._declare(node.var, item_al, elem_vt)
        dp = self._arr_data_ptr(arr_v, elem_lt)
        ep = self.builder.gep(dp, [iv], inbounds=True)
        ev = self.builder.load(ep)
        self.builder.store(ev, item_al)

        self._break_targets.append(ext)
        self._continue_targets.append(chk)
        for s in node.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(s)
        self._break_targets.pop()
        self._continue_targets.pop()
        self._pop_scope()

        if not self.builder.block.is_terminated:
            ic   = self.builder.load(i_al)
            inxt = self.builder.add(ic, ir.Constant(I64_TY, 1))
            self.builder.store(inxt, i_al)
            self.builder.branch(chk)

        self.builder.position_at_end(ext)

    def _compile_while(self, node: WhileStmt):
        fn  = self.current_fn
        chk = fn.append_basic_block("while.check")
        bdy = fn.append_basic_block("while.body")
        ext = fn.append_basic_block("while.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        cv, _ = self._compile_expr(node.condition)
        self.builder.cbranch(cv, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        self._break_targets.append(ext)
        self._continue_targets.append(chk)
        for s in node.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(s)
        self._break_targets.pop()
        self._continue_targets.pop()
        self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(chk)

        self.builder.position_at_end(ext)

    def _compile_break(self):
        if not self._break_targets:
            raise CodegenError("break outside loop")
        self.builder.branch(self._break_targets[-1])

    def _compile_continue(self):
        if not self._continue_targets:
            raise CodegenError("continue outside loop")
        self.builder.branch(self._continue_targets[-1])

    # ------------------------------------------------------------------ #
    #  Expressions                                                         #
    # ------------------------------------------------------------------ #

    def _compile_expr(self, node: Node) -> tuple[ir.Value, str]:
        if isinstance(node, IntLiteral):
            return ir.Constant(I64_TY, node.value), "int"
        if isinstance(node, FloatLiteral):
            return ir.Constant(F64_TY, node.value), "float"
        if isinstance(node, BoolLiteral):
            return ir.Constant(I1_TY, int(node.value)), "bool"
        if isinstance(node, StringLiteral):
            return self._gstr_ptr(self._global_str(node.value)), "str"
        if isinstance(node, NullLiteral):
            return ir.Constant(I8PTR, None), "null"

        if isinstance(node, ArrayLiteral):
            return self._compile_array_literal(node)

        if isinstance(node, Identifier):
            # Built-in constants
            if node.name == "PI":
                return ir.Constant(F64_TY, 3.141592653589793), "float"
            if node.name == "E":
                return ir.Constant(F64_TY, 2.718281828459045), "float"
            info = self._lookup(node.name)
            if info is None:
                raise CodegenError(f"Undefined variable '{node.name}'")
            return self.builder.load(info["ptr"], name=node.name), info["vx_type"]

        if isinstance(node, BinOp):    return self._compile_binop(node)
        if isinstance(node, UnaryOp):  return self._compile_unary(node)
        if isinstance(node, Call):     return self._compile_call(node)
        if isinstance(node, MethodCall): return self._compile_method_call(node)

        if isinstance(node, TernaryExpr):
            return self._compile_ternary(node)

        if isinstance(node, FieldAccess):
            # Enum access: Color.Red
            if isinstance(node.obj, Identifier):
                dotted = f"{node.obj.name}.{node.field}"
                info = self._lookup(dotted)
                if info is not None:
                    return self.builder.load(info["ptr"], name=dotted), info["vx_type"]
            ptr = self._field_ptr(node)
            vt  = self._infer_type(node)
            return self.builder.load(ptr, name=node.field), vt

        if isinstance(node, NewExpr):  return self._compile_new(node)

        if isinstance(node, IndexExpr):
            obj_v, obj_t = self._compile_expr(node.obj)
            idx_v, _     = self._compile_expr(node.index)
            # String indexing: return 1-char string
            if obj_t == "str":
                cp  = self.builder.gep(obj_v, [idx_v], inbounds=False)
                ch  = self.builder.load(cp)
                buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 2)])
                self.builder.store(ch, buf)
                np  = self.builder.gep(buf, [ir.Constant(I64_TY, 1)], inbounds=False)
                self.builder.store(ir.Constant(I8_TY, 0), np)
                return buf, "str"
            # Array indexing
            elem_vt = obj_t[:-2] if obj_t.endswith("[]") else "int"
            elem_lt = self._vx_to_llvm(elem_vt)
            dp  = self._arr_data_ptr(obj_v, elem_lt)
            ep  = self.builder.gep(dp, [idx_v], inbounds=True)
            return self.builder.load(ep), elem_vt

        raise CodegenError(f"Unknown expr: {type(node).__name__}")

    # ------------------------------------------------------------------ #
    #  Ternary expression                                                  #
    # ------------------------------------------------------------------ #

    def _compile_ternary(self, node: TernaryExpr) -> tuple[ir.Value, str]:
        fn = self.current_fn
        cond_v, _ = self._compile_expr(node.condition)

        then_b  = fn.append_basic_block("tern.then")
        else_b  = fn.append_basic_block("tern.else")
        merge_b = fn.append_basic_block("tern.merge")

        self.builder.cbranch(cond_v, then_b, else_b)

        self.builder.position_at_end(then_b)
        then_v, then_t = self._compile_expr(node.then_val)
        then_block = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(else_b)
        else_v, else_t = self._compile_expr(node.else_val)
        # Promote int->float if types differ
        if then_t == "float" and else_t == "int":
            else_v = self.builder.sitofp(else_v, F64_TY)
        elif then_t == "int" and else_t == "float":
            then_v = None  # will be recomputed with promotion — just use else type
        else_block = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)
        result_t = then_t
        ll_ty = self._vx_to_llvm(result_t)
        phi = self.builder.phi(ll_ty)
        phi.add_incoming(then_v, then_block)
        phi.add_incoming(else_v, else_block)
        return phi, result_t

    # ------------------------------------------------------------------ #
    #  Method calls                                                        #
    # ------------------------------------------------------------------ #

    def _compile_method_call(self, node: MethodCall) -> tuple[ir.Value, str]:
        obj_v, obj_t = self._compile_expr(node.obj)
        method = node.method

        if obj_t == "str":
            return self._compile_str_method(obj_v, method, node.args)

        if obj_t.endswith("[]"):
            elem_vt = obj_t[:-2]
            return self._compile_arr_method(obj_v, obj_t, elem_vt, method, node.args)

        raise CodegenError(f"No methods on type '{obj_t}'")

    def _compile_str_method(self, s: ir.Value, method: str, args) -> tuple[ir.Value, str]:
        if method == "len":
            return self.builder.call(self.strlen_fn, [s]), "int"

        if method == "upper":
            fn = self._get_helper("__vx_str_upper")
            return self.builder.call(fn, [s]), "str"

        if method == "lower":
            fn = self._get_helper("__vx_str_lower")
            return self.builder.call(fn, [s]), "str"

        if method == "trim":
            fn = self._get_helper("__vx_str_trim")
            return self.builder.call(fn, [s]), "str"

        if method == "contains":
            sub_v, _ = self._compile_expr(args[0])
            result = self.builder.call(self.strstr_fn, [s, sub_v])
            null_ptr = ir.Constant(I8PTR, None)
            null_int = self.builder.ptrtoint(null_ptr, I64_TY)
            res_int  = self.builder.ptrtoint(result, I64_TY)
            return self.builder.icmp_unsigned("!=", res_int, null_int), "bool"

        if method == "starts_with":
            prefix_v, _ = self._compile_expr(args[0])
            plen = self.builder.call(self.strlen_fn, [prefix_v])
            r = self.builder.call(self.strncmp_fn, [s, prefix_v, plen])
            return self.builder.icmp_signed("==", r, ir.Constant(I32_TY, 0)), "bool"

        if method == "ends_with":
            suffix_v, _ = self._compile_expr(args[0])
            fn_h = self._get_helper("__vx_str_ends_with")
            result = self.builder.call(fn_h, [s, suffix_v])
            return result, "bool"

        if method == "replace":
            old_v, _ = self._compile_expr(args[0])
            new_v, _ = self._compile_expr(args[1])
            fn_h = self._get_helper("__vx_str_replace")
            return self.builder.call(fn_h, [s, old_v, new_v]), "str"

        if method == "split":
            delim_v, _ = self._compile_expr(args[0])
            fn_h = self._get_helper("__vx_str_split")
            raw = self.builder.call(fn_h, [s, delim_v])
            arr_ptr = self.builder.bitcast(raw, self.arr_ptr_type)
            return arr_ptr, "str[]"

        raise CodegenError(f"Unknown string method '{method}'")

    def _compile_arr_method(self, arr_v: ir.Value, arr_t: str, elem_vt: str,
                            method: str, args) -> tuple[ir.Value, str]:
        elem_lt = self._vx_to_llvm(elem_vt)
        esz = ir.Constant(I64_TY, _elem_size(elem_vt))

        if method == "len":
            lp = self.builder.gep(arr_v, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
            return self.builder.load(lp), "int"

        if method == "push":
            elem_v, ev_t = self._compile_expr(args[0])
            if elem_vt == "float" and ev_t == "int":
                elem_v = self.builder.sitofp(elem_v, F64_TY)
            elem_al = self.builder.alloca(elem_lt)
            self.builder.store(elem_v, elem_al)
            elem_raw = self.builder.bitcast(elem_al, I8PTR)
            arr_raw  = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_push")
            self.builder.call(fn_h, [arr_raw, elem_raw, esz])
            return ir.Constant(I64_TY, 0), "void"

        if method == "pop":
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_pop")
            raw_result = self.builder.call(fn_h, [arr_raw, esz])
            typed_ptr = self.builder.bitcast(raw_result, ir.PointerType(elem_lt))
            return self.builder.load(typed_ptr), elem_vt

        if method == "contains":
            elem_v, ev_t = self._compile_expr(args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            if elem_vt == "int":
                fn_h = self._get_helper("__vx_array_contains_i64")
                return self.builder.call(fn_h, [arr_raw, elem_v]), "bool"
            elif elem_vt == "float":
                if ev_t == "int": elem_v = self.builder.sitofp(elem_v, F64_TY)
                fn_h = self._get_helper("__vx_array_contains_f64")
                return self.builder.call(fn_h, [arr_raw, elem_v]), "bool"
            else:  # str
                fn_h = self._get_helper("__vx_array_contains_str")
                return self.builder.call(fn_h, [arr_raw, elem_v]), "bool"

        if method == "reverse":
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_reverse")
            self.builder.call(fn_h, [arr_raw, esz])
            return ir.Constant(I64_TY, 0), "void"

        raise CodegenError(f"Unknown array method '{method}'")

    # ------------------------------------------------------------------ #
    #  Private helper functions                                            #
    # ------------------------------------------------------------------ #

    def _get_helper(self, name: str) -> ir.Function:
        if name in self._helper_fns:
            return self._helper_fns[name]
        fn = self._build_helper(name)
        self._helper_fns[name] = fn
        return fn

    def _build_helper(self, name: str) -> ir.Function:
        builders = {
            "__vx_str_upper":            self._build_str_upper,
            "__vx_str_lower":            self._build_str_lower,
            "__vx_str_trim":             self._build_str_trim,
            "__vx_str_ends_with":        self._build_str_ends_with,
            "__vx_str_replace":          self._build_str_replace,
            "__vx_str_split":            self._build_str_split,
            "__vx_array_push":           self._build_array_push,
            "__vx_array_pop":            self._build_array_pop,
            "__vx_array_contains_i64":   self._build_array_contains_i64,
            "__vx_array_contains_f64":   self._build_array_contains_f64,
            "__vx_array_contains_str":   self._build_array_contains_str,
            "__vx_array_reverse":        self._build_array_reverse,
            "__vx_file_read":            self._build_file_read,
            "__vx_file_write":           self._build_file_write,
            "__vx_file_append":          self._build_file_append,
            "__vx_file_exists":          self._build_file_exists,
        }
        if name not in builders:
            raise CodegenError(f"Unknown helper function: {name}")
        return builders[name]()

    def _save_builder(self):
        """Save current builder state."""
        return self.builder, self.current_fn

    def _restore_builder(self, state):
        """Restore builder state."""
        self.builder, self.current_fn = state

    def _build_str_upper(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_str_upper")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        ln = b.call(self.strlen_fn, [s])
        buf = b.call(self.malloc_fn, [b.add(ln, ir.Constant(I64_TY, 1))])
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ext = fn.append_basic_block("ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        sp = b.gep(s, [iv], inbounds=False)
        ch = b.load(sp)
        ch_up = b.call(self.toupper_fn, [b.zext(ch, I32_TY)])
        ch_t  = b.trunc(ch_up, I8_TY)
        b.store(ch_t, b.gep(buf, [iv], inbounds=False))
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [ln], inbounds=False))
        b.ret(buf)
        return fn

    def _build_str_lower(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_str_lower")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        ln = b.call(self.strlen_fn, [s])
        buf = b.call(self.malloc_fn, [b.add(ln, ir.Constant(I64_TY, 1))])
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ext = fn.append_basic_block("ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        sp = b.gep(s, [iv], inbounds=False)
        ch = b.load(sp)
        ch_lo = b.call(self.tolower_fn, [b.zext(ch, I32_TY)])
        ch_t  = b.trunc(ch_lo, I8_TY)
        b.store(ch_t, b.gep(buf, [iv], inbounds=False))
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [ln], inbounds=False))
        b.ret(buf)
        return fn

    def _build_str_trim(self) -> ir.Function:
        """Trim leading and trailing spaces from a string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_str_trim")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]

        # Find start (skip leading spaces)
        start_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), start_al)

        chk_s = fn.append_basic_block("trim.start.chk")
        bdy_s = fn.append_basic_block("trim.start.bdy")
        done_s = fn.append_basic_block("trim.start.done")
        b.branch(chk_s)

        b.position_at_end(chk_s)
        si = b.load(start_al)
        sp = b.gep(s, [si], inbounds=False)
        ch = b.load(sp)
        is_space = b.icmp_signed("==", b.zext(ch, I32_TY), ir.Constant(I32_TY, 32))
        is_nonzero = b.icmp_signed("!=", b.zext(ch, I32_TY), ir.Constant(I32_TY, 0))
        go_on = b.and_(is_space, is_nonzero)
        b.cbranch(go_on, bdy_s, done_s)

        b.position_at_end(bdy_s)
        b.store(b.add(si, ir.Constant(I64_TY, 1)), start_al)
        b.branch(chk_s)

        b.position_at_end(done_s)
        start_idx = b.load(start_al)
        start_ptr = b.gep(s, [start_idx], inbounds=False)

        # Get length of trimmed start
        ln = b.call(self.strlen_fn, [start_ptr])

        # Find end (skip trailing spaces)
        end_al = b.alloca(I64_TY)
        b.store(ln, end_al)

        chk_e = fn.append_basic_block("trim.end.chk")
        bdy_e = fn.append_basic_block("trim.end.bdy")
        done_e = fn.append_basic_block("trim.end.done")
        b.branch(chk_e)

        b.position_at_end(chk_e)
        ei = b.load(end_al)
        has_chars = b.icmp_signed(">", ei, ir.Constant(I64_TY, 0))
        b.cbranch(has_chars, bdy_e, done_e)

        b.position_at_end(bdy_e)
        prev = b.sub(ei, ir.Constant(I64_TY, 1))
        ep = b.gep(start_ptr, [prev], inbounds=False)
        ec = b.load(ep)
        is_trail_space = b.icmp_signed("==", b.zext(ec, I32_TY), ir.Constant(I32_TY, 32))
        # only trim if space
        new_end = b.select(is_trail_space, prev, ei)
        b.store(new_end, end_al)
        # if not a space, done
        b.cbranch(is_trail_space, chk_e, done_e)

        b.position_at_end(done_e)
        final_len = b.load(end_al)
        buf = b.call(self.malloc_fn, [b.add(final_len, ir.Constant(I64_TY, 1))])
        b.call(self.memcpy_fn, [buf, start_ptr, final_len])
        null_p = b.gep(buf, [final_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _build_str_ends_with(self) -> ir.Function:
        """Check if s ends with suffix."""
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I8PTR]),
                         name="__vx_str_ends_with")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, suffix = fn.args[0], fn.args[1]
        slen   = b.call(self.strlen_fn, [s])
        suflen = b.call(self.strlen_fn, [suffix])

        # if suflen > slen: return false
        too_long = b.icmp_signed(">", suflen, slen)
        ret_false_b = fn.append_basic_block("ret_false")
        cmp_b = fn.append_basic_block("cmp")
        b.cbranch(too_long, ret_false_b, cmp_b)

        b.position_at_end(ret_false_b)
        b.ret(ir.Constant(I1_TY, 0))

        b.position_at_end(cmp_b)
        offset = b.sub(slen, suflen)
        tail_ptr = b.gep(s, [offset], inbounds=False)
        r = b.call(self.strncmp_fn, [tail_ptr, suffix, suflen])
        result = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.ret(result)
        return fn

    def _build_str_replace(self) -> ir.Function:
        """Replace all occurrences of old in s with new_str."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR, I8PTR]),
                         name="__vx_str_replace")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, old, new_s = fn.args[0], fn.args[1], fn.args[2]

        slen   = b.call(self.strlen_fn, [s])
        oldlen = b.call(self.strlen_fn, [old])
        newlen = b.call(self.strlen_fn, [new_s])

        # Count occurrences to compute result buffer size
        # Then build the result string
        # Allocate a generous buffer: slen + count * (newlen - oldlen)
        # Simple approach: allocate slen * (newlen+1) + newlen as upper bound
        # Actually: just use slen + max_count * max(newlen, 1) * 2
        # We'll use a simpler approach: alloc slen * (newlen + 1) + 1
        # to guarantee enough space (worst case every char is replaced)
        buf_sz = b.add(b.mul(slen, b.add(newlen, ir.Constant(I64_TY, 1))),
                       ir.Constant(I64_TY, 1))
        buf = b.call(self.malloc_fn, [buf_sz])

        # Write position
        out_al  = b.alloca(I64_TY)
        cur_al  = b.alloca(I8PTR)
        b.store(ir.Constant(I64_TY, 0), out_al)
        b.store(s, cur_al)

        # oldlen == 0 edge case: just return copy
        old_zero = b.icmp_signed("==", oldlen, ir.Constant(I64_TY, 0))
        do_copy_b = fn.append_basic_block("do_copy")
        loop_b = fn.append_basic_block("rep.loop")
        b.cbranch(old_zero, do_copy_b, loop_b)

        b.position_at_end(do_copy_b)
        b.call(self.memcpy_fn, [buf, s, b.add(slen, ir.Constant(I64_TY, 1))])
        b.ret(buf)

        b.position_at_end(loop_b)
        cur = b.load(cur_al)
        found = b.call(self.strstr_fn, [cur, old])
        found_int = b.ptrtoint(found, I64_TY)
        is_null = b.icmp_unsigned("==", found_int, ir.Constant(I64_TY, 0))

        found_b  = fn.append_basic_block("rep.found")
        no_found_b = fn.append_basic_block("rep.nofound")
        b.cbranch(is_null, no_found_b, found_b)

        b.position_at_end(found_b)
        out_i = b.load(out_al)
        # Copy bytes before the match
        cur2 = b.load(cur_al)
        found2 = b.call(self.strstr_fn, [cur2, old])
        cur_int  = b.ptrtoint(cur2, I64_TY)
        found_int2 = b.ptrtoint(found2, I64_TY)
        before_len = b.sub(found_int2, cur_int)
        out_ptr = b.gep(buf, [out_i], inbounds=False)
        b.call(self.memcpy_fn, [out_ptr, cur2, before_len])
        new_out_i = b.add(out_i, before_len)
        # Copy replacement
        out_ptr2 = b.gep(buf, [new_out_i], inbounds=False)
        b.call(self.memcpy_fn, [out_ptr2, new_s, newlen])
        new_out_i2 = b.add(new_out_i, newlen)
        b.store(new_out_i2, out_al)
        # Advance cur past old
        new_cur = b.gep(found2, [oldlen], inbounds=False)
        b.store(new_cur, cur_al)
        b.branch(loop_b)

        b.position_at_end(no_found_b)
        # Copy remainder
        out_i3 = b.load(out_al)
        cur3   = b.load(cur_al)
        rem_len = b.call(self.strlen_fn, [cur3])
        out_ptr3 = b.gep(buf, [out_i3], inbounds=False)
        b.call(self.memcpy_fn, [out_ptr3, cur3, rem_len])
        final_out = b.add(out_i3, rem_len)
        null_p = b.gep(buf, [final_out], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _build_str_split(self) -> ir.Function:
        """Split s by delim, return vx_array* (as i8*) of str."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_str_split")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, delim = fn.args[0], fn.args[1]

        # Allocate result vx_array header
        hsz = ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)
        raw_hdr = b.call(self.malloc_fn, [hsz])
        hdr = b.bitcast(raw_hdr, self.arr_ptr_type)

        # Initial capacity = 8
        init_cap = ir.Constant(I64_TY, 8)
        # Allocate data: cap * sizeof(i8*) = cap * 8
        data_sz = b.mul(init_cap, ir.Constant(I64_TY, 8))
        raw_data = b.call(self.malloc_fn, [data_sz])

        dp_field  = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        len_field = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        cap_field = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,2)], inbounds=True)
        b.store(raw_data, dp_field)
        b.store(ir.Constant(I64_TY, 0), len_field)
        b.store(init_cap, cap_field)

        dlen = b.call(self.strlen_fn, [delim])
        cur_al = b.alloca(I8PTR)
        b.store(s, cur_al)

        loop_b  = fn.append_basic_block("split.loop")
        found_b = fn.append_basic_block("split.found")
        done_b  = fn.append_basic_block("split.done")
        b.branch(loop_b)

        b.position_at_end(loop_b)
        cur = b.load(cur_al)
        found = b.call(self.strstr_fn, [cur, delim])
        found_int = b.ptrtoint(found, I64_TY)
        is_null = b.icmp_unsigned("==", found_int, ir.Constant(I64_TY, 0))
        b.cbranch(is_null, done_b, found_b)

        b.position_at_end(found_b)
        cur2 = b.load(cur_al)
        found2 = b.call(self.strstr_fn, [cur2, delim])
        cur_int    = b.ptrtoint(cur2, I64_TY)
        found_int2 = b.ptrtoint(found2, I64_TY)
        tok_len = b.sub(found_int2, cur_int)
        # Allocate token string
        tok_buf = b.call(self.malloc_fn, [b.add(tok_len, ir.Constant(I64_TY, 1))])
        b.call(self.memcpy_fn, [tok_buf, cur2, tok_len])
        null_p = b.gep(tok_buf, [tok_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        # Push token into array: grow if needed
        cur_len = b.load(len_field)
        cur_cap = b.load(cap_field)
        need_grow = b.icmp_signed(">=", cur_len, cur_cap)
        grow_b2   = fn.append_basic_block("split.grow")
        push_b    = fn.append_basic_block("split.push")
        b.cbranch(need_grow, grow_b2, push_b)

        b.position_at_end(grow_b2)
        new_cap = b.mul(cur_cap, ir.Constant(I64_TY, 2))
        new_sz  = b.mul(new_cap, ir.Constant(I64_TY, 8))
        old_d   = b.load(dp_field)
        new_d   = b.call(self.realloc_fn, [old_d, new_sz])
        b.store(new_d, dp_field)
        b.store(new_cap, cap_field)
        b.branch(push_b)

        b.position_at_end(push_b)
        cur_len2 = b.load(len_field)
        data_ptr = b.load(dp_field)
        # Cast data_ptr to i8** and store tok_buf at index cur_len2
        pp = b.bitcast(data_ptr, ir.PointerType(I8PTR))
        ep = b.gep(pp, [cur_len2], inbounds=False)
        b.store(tok_buf, ep)
        b.store(b.add(cur_len2, ir.Constant(I64_TY, 1)), len_field)
        # Advance cur past delim
        new_cur = b.gep(found2, [dlen], inbounds=False)
        b.store(new_cur, cur_al)
        b.branch(loop_b)

        b.position_at_end(done_b)
        # Add last token (remainder)
        cur_last = b.load(cur_al)
        last_len = b.call(self.strlen_fn, [cur_last])
        last_buf = b.call(self.malloc_fn, [b.add(last_len, ir.Constant(I64_TY, 1))])
        b.call(self.memcpy_fn, [last_buf, cur_last, b.add(last_len, ir.Constant(I64_TY, 1))])
        # Push last token
        cur_len3 = b.load(len_field)
        cur_cap3 = b.load(cap_field)
        ng3 = b.icmp_signed(">=", cur_len3, cur_cap3)
        grow3 = fn.append_basic_block("split.grow3")
        push3 = fn.append_basic_block("split.push3")
        b.cbranch(ng3, grow3, push3)

        b.position_at_end(grow3)
        nc3 = b.mul(cur_cap3, ir.Constant(I64_TY, 2))
        ns3 = b.mul(nc3, ir.Constant(I64_TY, 8))
        od3 = b.load(dp_field)
        nd3 = b.call(self.realloc_fn, [od3, ns3])
        b.store(nd3, dp_field)
        b.store(nc3, cap_field)
        b.branch(push3)

        b.position_at_end(push3)
        cl3 = b.load(len_field)
        dp3 = b.load(dp_field)
        pp3 = b.bitcast(dp3, ir.PointerType(I8PTR))
        ep3 = b.gep(pp3, [cl3], inbounds=False)
        b.store(last_buf, ep3)
        b.store(b.add(cl3, ir.Constant(I64_TY, 1)), len_field)
        b.ret(raw_hdr)
        return fn

    def _build_array_push(self) -> ir.Function:
        """Push an element into a dynamic array. arr is i8* (vx_array*), elem is i8*, esz is i64."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR, I64_TY]),
                         name="__vx_array_push")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, elem_raw, esz = fn.args[0], fn.args[1], fn.args[2]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr  = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        cap_ptr  = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,2)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)

        ln  = b.load(len_ptr)
        cap = b.load(cap_ptr)

        need_grow = b.icmp_signed(">=", ln, cap)
        grow_b   = fn.append_basic_block("push.grow")
        no_grow_b = fn.append_basic_block("push.store")
        b.cbranch(need_grow, grow_b, no_grow_b)

        b.position_at_end(grow_b)
        cap2 = b.mul(cap, ir.Constant(I64_TY, 2))
        is_lt4 = b.icmp_signed("<", cap2, ir.Constant(I64_TY, 4))
        new_cap = b.select(is_lt4, ir.Constant(I64_TY, 4), cap2)
        new_sz  = b.mul(new_cap, esz)
        old_data = b.load(data_ptr_field)
        new_data = b.call(self.realloc_fn, [old_data, new_sz])
        b.store(new_data, data_ptr_field)
        b.store(new_cap, cap_ptr)
        b.branch(no_grow_b)

        b.position_at_end(no_grow_b)
        data   = b.load(data_ptr_field)
        offset = b.mul(ln, esz)
        dst    = b.gep(data, [offset], inbounds=False)
        b.call(self.memcpy_fn, [dst, elem_raw, esz])
        new_ln = b.add(ln, ir.Constant(I64_TY, 1))
        b.store(new_ln, len_ptr)
        b.ret_void()
        return fn

    def _build_array_pop(self) -> ir.Function:
        """Pop last element from array. Returns i8* pointing to element data (caller loads)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY]),
                         name="__vx_array_pop")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, esz = fn.args[0], fn.args[1]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)

        ln = b.load(len_ptr)
        # Decrement len
        new_ln = b.sub(ln, ir.Constant(I64_TY, 1))
        b.store(new_ln, len_ptr)
        # Return pointer to last element
        data   = b.load(data_ptr_field)
        offset = b.mul(new_ln, esz)
        elem_ptr = b.gep(data, [offset], inbounds=False)
        b.ret(elem_ptr)
        return fn

    def _build_array_contains_i64(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I64_TY]),
                         name="__vx_array_contains_i64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, val = fn.args[0], fn.args[1]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln   = b.load(len_ptr)
        data = b.load(data_ptr_field)
        typed = b.bitcast(data, ir.PointerType(I64_TY))

        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)

        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ret_true  = fn.append_basic_block("ret_true")
        ret_false = fn.append_basic_block("ret_false")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ret_false)

        b.position_at_end(bdy)
        ep = b.gep(typed, [iv], inbounds=False)
        ev = b.load(ep)
        is_eq = b.icmp_signed("==", ev, val)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, ret_true, chk)

        b.position_at_end(ret_true)
        b.ret(ir.Constant(I1_TY, 1))
        b.position_at_end(ret_false)
        b.ret(ir.Constant(I1_TY, 0))
        return fn

    def _build_array_contains_f64(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, F64_TY]),
                         name="__vx_array_contains_f64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, val = fn.args[0], fn.args[1]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln   = b.load(len_ptr)
        data = b.load(data_ptr_field)
        typed = b.bitcast(data, ir.PointerType(F64_TY))

        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)

        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ret_true  = fn.append_basic_block("ret_true")
        ret_false = fn.append_basic_block("ret_false")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ret_false)

        b.position_at_end(bdy)
        ep = b.gep(typed, [iv], inbounds=False)
        ev = b.load(ep)
        is_eq = b.fcmp_ordered("==", ev, val)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, ret_true, chk)

        b.position_at_end(ret_true)
        b.ret(ir.Constant(I1_TY, 1))
        b.position_at_end(ret_false)
        b.ret(ir.Constant(I1_TY, 0))
        return fn

    def _build_array_contains_str(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I8PTR]),
                         name="__vx_array_contains_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, val = fn.args[0], fn.args[1]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln   = b.load(len_ptr)
        data = b.load(data_ptr_field)
        typed = b.bitcast(data, ir.PointerType(I8PTR))

        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)

        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ret_true  = fn.append_basic_block("ret_true")
        ret_false = fn.append_basic_block("ret_false")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ret_false)

        b.position_at_end(bdy)
        ep = b.gep(typed, [iv], inbounds=False)
        ev = b.load(ep)
        r  = b.call(self.strcmp_fn, [ev, val])
        is_eq = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, ret_true, chk)

        b.position_at_end(ret_true)
        b.ret(ir.Constant(I1_TY, 1))
        b.position_at_end(ret_false)
        b.ret(ir.Constant(I1_TY, 0))
        return fn

    def _build_array_reverse(self) -> ir.Function:
        """Reverse an array in-place. arr is i8* (vx_array*), esz is element size."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I64_TY]),
                         name="__vx_array_reverse")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, esz = fn.args[0], fn.args[1]

        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        len_ptr = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        data_ptr_field = b.gep(arr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln   = b.load(len_ptr)
        data = b.load(data_ptr_field)

        # left = 0, right = ln - 1
        left_al  = b.alloca(I64_TY)
        right_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), left_al)
        b.store(b.sub(ln, ir.Constant(I64_TY, 1)), right_al)

        # tmp buffer for swap
        tmp = b.call(self.malloc_fn, [esz])

        chk = fn.append_basic_block("rev.chk")
        bdy = fn.append_basic_block("rev.bdy")
        ext = fn.append_basic_block("rev.ext")
        b.branch(chk)

        b.position_at_end(chk)
        lv = b.load(left_al)
        rv = b.load(right_al)
        b.cbranch(b.icmp_signed("<", lv, rv), bdy, ext)

        b.position_at_end(bdy)
        l2 = b.load(left_al)
        r2 = b.load(right_al)
        loff = b.mul(l2, esz)
        roff = b.mul(r2, esz)
        lptr = b.gep(data, [loff], inbounds=False)
        rptr = b.gep(data, [roff], inbounds=False)
        # tmp = *lptr
        b.call(self.memcpy_fn, [tmp, lptr, esz])
        # *lptr = *rptr
        b.call(self.memcpy_fn, [lptr, rptr, esz])
        # *rptr = tmp
        b.call(self.memcpy_fn, [rptr, tmp, esz])
        b.store(b.add(l2, ir.Constant(I64_TY, 1)), left_al)
        b.store(b.sub(r2, ir.Constant(I64_TY, 1)), right_al)
        b.branch(chk)

        b.position_at_end(ext)
        b.ret_void()
        return fn

    def _build_file_read(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_file_read")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path = fn.args[0]

        mode_r_gv = self._global_str("rb")
        mode_r = mode_r_gv.gep([ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)])
        f = b.call(self.fopen_fn, [path, mode_r])

        f_int = b.ptrtoint(f, I64_TY)
        is_null = b.icmp_unsigned("==", f_int, ir.Constant(I64_TY, 0))
        null_b = fn.append_basic_block("null_ret")
        ok_b   = fn.append_basic_block("ok")
        b.cbranch(is_null, null_b, ok_b)

        b.position_at_end(null_b)
        empty_gv = self._global_str("")
        empty_ptr = empty_gv.gep([ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)])
        b.ret(empty_ptr)

        b.position_at_end(ok_b)
        SEEK_END = ir.Constant(I32_TY, 2)
        SEEK_SET = ir.Constant(I32_TY, 0)
        b.call(self.fseek_fn, [f, ir.Constant(I64_TY, 0), SEEK_END])
        sz = b.call(self.ftell_fn, [f])
        b.call(self.fseek_fn, [f, ir.Constant(I64_TY, 0), SEEK_SET])
        buf = b.call(self.malloc_fn, [b.add(sz, ir.Constant(I64_TY, 1))])
        b.call(self.fread_fn, [buf, ir.Constant(I64_TY, 1), sz, f])
        b.call(self.fclose_fn, [f])
        np = b.gep(buf, [sz], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), np)
        b.ret(buf)
        return fn

    def _build_file_write(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_file_write")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path, content = fn.args[0], fn.args[1]

        mode_w_gv = self._global_str("w")
        mode_w = mode_w_gv.gep([ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)])
        f = b.call(self.fopen_fn, [path, mode_w])

        f_int   = b.ptrtoint(f, I64_TY)
        is_null = b.icmp_unsigned("==", f_int, ir.Constant(I64_TY, 0))
        null_b  = fn.append_basic_block("null_ret")
        ok_b    = fn.append_basic_block("ok")
        b.cbranch(is_null, null_b, ok_b)

        b.position_at_end(null_b)
        b.ret_void()

        b.position_at_end(ok_b)
        clen = b.call(self.strlen_fn, [content])
        b.call(self.fwrite_fn, [content, ir.Constant(I64_TY, 1), clen, f])
        b.call(self.fclose_fn, [f])
        b.ret_void()
        return fn

    def _build_file_append(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_file_append")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path, content = fn.args[0], fn.args[1]

        mode_a_gv = self._global_str("a")
        mode_a = mode_a_gv.gep([ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)])
        f = b.call(self.fopen_fn, [path, mode_a])

        f_int   = b.ptrtoint(f, I64_TY)
        is_null = b.icmp_unsigned("==", f_int, ir.Constant(I64_TY, 0))
        null_b  = fn.append_basic_block("null_ret")
        ok_b    = fn.append_basic_block("ok")
        b.cbranch(is_null, null_b, ok_b)

        b.position_at_end(null_b)
        b.ret_void()

        b.position_at_end(ok_b)
        clen = b.call(self.strlen_fn, [content])
        b.call(self.fwrite_fn, [content, ir.Constant(I64_TY, 1), clen, f])
        b.call(self.fclose_fn, [f])
        b.ret_void()
        return fn

    def _build_file_exists(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR]),
                         name="__vx_file_exists")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path = fn.args[0]

        mode_r_gv = self._global_str("r")
        mode_r = mode_r_gv.gep([ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)])
        f = b.call(self.fopen_fn, [path, mode_r])

        f_int   = b.ptrtoint(f, I64_TY)
        is_null = b.icmp_unsigned("==", f_int, ir.Constant(I64_TY, 0))
        null_b  = fn.append_basic_block("null_ret")
        ok_b    = fn.append_basic_block("ok")
        b.cbranch(is_null, null_b, ok_b)

        b.position_at_end(null_b)
        b.ret(ir.Constant(I1_TY, 0))

        b.position_at_end(ok_b)
        b.call(self.fclose_fn, [f])
        b.ret(ir.Constant(I1_TY, 1))
        return fn

    # ------------------------------------------------------------------ #
    #  Binary ops                                                          #
    # ------------------------------------------------------------------ #

    def _compile_binop(self, node: BinOp) -> tuple[ir.Value, str]:
        op = node.op

        # Short-circuit AND/OR
        if op == "and":
            return self._compile_and(node)
        if op == "or":
            return self._compile_or(node)

        lv, lt = self._compile_expr(node.left)
        rv, rt = self._compile_expr(node.right)

        # String + concatenation (inline — no external runtime needed)
        if op == "+" and lt == "str":
            return self._str_concat_inline(lv, rv), "str"

        # String equality
        if op == "==" and lt == "str":
            r = self.builder.call(self.strcmp_fn, [lv, rv])
            return self.builder.icmp_signed("==", r, ir.Constant(I32_TY, 0)), "bool"
        if op == "!=" and lt == "str":
            r = self.builder.call(self.strcmp_fn, [lv, rv])
            return self.builder.icmp_signed("!=", r, ir.Constant(I32_TY, 0)), "bool"

        # Promote int → float
        if lt == "float" and rt == "int":
            rv = self.builder.sitofp(rv, F64_TY); rt = "float"
        elif lt == "int" and rt == "float":
            lv = self.builder.sitofp(lv, F64_TY); lt = "float"

        is_float = (lt == "float")

        cmp_ops = {"==","!=","<",">","<=",">="}
        if op in cmp_ops:
            if is_float:
                return self.builder.fcmp_ordered(op, lv, rv), "bool"
            return self.builder.icmp_signed(op, lv, rv), "bool"

        arith = {
            "+": (self.builder.fadd, self.builder.add),
            "-": (self.builder.fsub, self.builder.sub),
            "*": (self.builder.fmul, self.builder.mul),
            "/": (self.builder.fdiv, self.builder.sdiv),
            "%": (self.builder.frem, self.builder.srem),
        }
        if op in arith:
            f_op, i_op = arith[op]
            return (f_op(lv, rv) if is_float else i_op(lv, rv)), lt

        raise CodegenError(f"Unknown binary op: {op!r}")

    def _compile_and(self, node: BinOp):
        fn      = self.current_fn
        rhs_b   = fn.append_basic_block("and.rhs")
        merge_b = fn.append_basic_block("and.merge")

        lv, _ = self._compile_expr(node.left)
        lblock = self.builder.block
        self.builder.cbranch(lv, rhs_b, merge_b)

        self.builder.position_at_end(rhs_b)
        rv, _  = self._compile_expr(node.right)
        rblock = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)
        phi = self.builder.phi(I1_TY)
        phi.add_incoming(ir.Constant(I1_TY, 0), lblock)
        phi.add_incoming(rv, rblock)
        return phi, "bool"

    def _compile_or(self, node: BinOp):
        fn      = self.current_fn
        rhs_b   = fn.append_basic_block("or.rhs")
        merge_b = fn.append_basic_block("or.merge")

        lv, _ = self._compile_expr(node.left)
        lblock = self.builder.block
        self.builder.cbranch(lv, merge_b, rhs_b)

        self.builder.position_at_end(rhs_b)
        rv, _  = self._compile_expr(node.right)
        rblock = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)
        phi = self.builder.phi(I1_TY)
        phi.add_incoming(ir.Constant(I1_TY, 1), lblock)
        phi.add_incoming(rv, rblock)
        return phi, "bool"

    def _compile_unary(self, node: UnaryOp) -> tuple[ir.Value, str]:
        val, vt = self._compile_expr(node.operand)
        if node.op == "-":
            if vt == "float": return self.builder.fsub(ir.Constant(F64_TY, 0.0), val), "float"
            return self.builder.neg(val), "int"
        if node.op == "not":
            bv = self.builder.trunc(val, I1_TY) if val.type != I1_TY else val
            return self.builder.not_(bv), "bool"
        raise CodegenError(f"Unknown unary op: {node.op!r}")

    # ------------------------------------------------------------------ #
    #  Function calls  (includes built-ins)                               #
    # ------------------------------------------------------------------ #

    def _compile_call(self, node: Call) -> tuple[ir.Value, str]:
        name = node.func

        # --- Type casts ---
        if name == "int":
            val, vt = self._compile_expr(node.args[0])
            if vt == "float":  return self.builder.fptosi(val, I64_TY), "int"
            if vt == "bool":   return self.builder.zext(val, I64_TY),   "int"
            return val, "int"

        if name == "float":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int":    return self.builder.sitofp(val, F64_TY), "float"
            if vt == "bool":
                iv = self.builder.zext(val, I64_TY)
                return self.builder.sitofp(iv, F64_TY), "float"
            return val, "float"

        if name == "bool":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int":   return self.builder.trunc(val, I1_TY), "bool"
            if vt == "float":
                z = ir.Constant(F64_TY, 0.0)
                return self.builder.fcmp_ordered("!=", val, z), "bool"
            if val.type != I1_TY: return self.builder.trunc(val, I1_TY), "bool"
            return val, "bool"

        if name == "str":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int":   return self._int_to_str_inline(val),   "str"
            if vt == "float": return self._float_to_str_inline(val), "str"
            if vt == "bool":
                t = self._gstr_ptr(self._global_str("true"))
                f = self._gstr_ptr(self._global_str("false"))
                return self.builder.select(val, t, f), "str"
            return val, "str"

        # --- len ---
        if name == "len":
            val, vt = self._compile_expr(node.args[0])
            if vt == "str":
                return self.builder.call(self.strlen_fn, [val]), "int"
            if vt.endswith("[]"):
                lp = self.builder.gep(val, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
                return self.builder.load(lp), "int"
            raise CodegenError(f"len() not supported for type {vt!r}")

        # --- Math ---
        if name == "sqrt":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.sqrt_fn, [val]), "float"

        if name == "abs":
            val, vt = self._compile_expr(node.args[0])
            if vt == "float": return self.builder.call(self.fabs_fn,  [val]), "float"
            return self.builder.call(self.llabs_fn, [val]), "int"

        if name == "min":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            if at == "float" or bt == "float":
                if at == "int": av = self.builder.sitofp(av, F64_TY)
                if bt == "int": bv = self.builder.sitofp(bv, F64_TY)
                c = self.builder.fcmp_ordered("<", av, bv)
                return self.builder.select(c, av, bv), "float"
            c = self.builder.icmp_signed("<", av, bv)
            return self.builder.select(c, av, bv), "int"

        if name == "max":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            if at == "float" or bt == "float":
                if at == "int": av = self.builder.sitofp(av, F64_TY)
                if bt == "int": bv = self.builder.sitofp(bv, F64_TY)
                c = self.builder.fcmp_ordered(">", av, bv)
                return self.builder.select(c, av, bv), "float"
            c = self.builder.icmp_signed(">", av, bv)
            return self.builder.select(c, av, bv), "int"

        if name == "pow":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            if at == "int": av = self.builder.sitofp(av, F64_TY)
            if bt == "int": bv = self.builder.sitofp(bv, F64_TY)
            return self.builder.call(self.pow_fn, [av, bv]), "float"

        if name == "floor":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.floor_fn, [val]), "float"

        if name == "ceil":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.ceil_fn, [val]), "float"

        # --- v3 Math ---
        if name == "sin":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.sin_fn, [val]), "float"

        if name == "cos":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.cos_fn, [val]), "float"

        if name == "tan":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.tan_fn, [val]), "float"

        if name == "log":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.log_fn, [val]), "float"

        if name == "log2":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.log2_fn, [val]), "float"

        if name == "rand":
            r  = self.builder.call(self.rand_fn, [])
            rf = self.builder.sitofp(r, F64_TY)
            max_f = ir.Constant(F64_TY, 2147483647.0)
            return self.builder.fdiv(rf, max_f), "float"

        if name == "rand_int":
            av, _ = self._compile_expr(node.args[0])
            bv, _ = self._compile_expr(node.args[1])
            range_ = self.builder.add(self.builder.sub(bv, av), ir.Constant(I64_TY, 1))
            r   = self.builder.call(self.rand_fn, [])
            r64 = self.builder.sext(r, I64_TY)
            r64 = self.builder.srem(r64, range_)
            # Ensure non-negative result
            neg  = self.builder.icmp_signed("<", r64, ir.Constant(I64_TY, 0))
            r64  = self.builder.select(neg, self.builder.add(r64, range_), r64)
            return self.builder.add(r64, av), "int"

        # --- exit ---
        if name == "exit":
            val, vt = self._compile_expr(node.args[0])
            if vt != "int": val = self.builder.fptosi(val, I32_TY)
            else:           val = self.builder.trunc(val, I32_TY)
            self.builder.call(self.exit_fn, [val])
            return ir.Constant(I64_TY, 0), "void"

        # --- File I/O ---
        if name == "read_file":
            path_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_file_read")
            return self.builder.call(fn_h, [path_v]), "str"

        if name == "write_file":
            path_v, _    = self._compile_expr(node.args[0])
            content_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_file_write")
            self.builder.call(fn_h, [path_v, content_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "append_file":
            path_v, _    = self._compile_expr(node.args[0])
            content_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_file_append")
            self.builder.call(fn_h, [path_v, content_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "file_exists":
            path_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_file_exists")
            return self.builder.call(fn_h, [path_v]), "bool"

        # --- User-defined functions ---
        fi = self._functions.get(name)
        if fi is None:
            raise CodegenError(f"Undefined function '{name}'")
        fn  = fi["fn"]
        sig = fi["sig"]
        compiled_args = []
        for arg_node, (_, pt) in zip(node.args, sig.params):
            av, at = self._compile_expr(arg_node)
            if pt == "float" and at == "int":
                av = self.builder.sitofp(av, F64_TY)
            compiled_args.append(av)
        result = self.builder.call(fn, compiled_args)
        return result, sig.return_type

    # ------------------------------------------------------------------ #
    #  Arrays                                                              #
    # ------------------------------------------------------------------ #

    def _arr_data_ptr(self, arr_v: ir.Value, elem_lt: ir.Type) -> ir.Value:
        """Load data pointer from array header and cast to elem_lt*."""
        dp_ptr = self.builder.gep(arr_v,
                                  [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                  inbounds=True)
        raw    = self.builder.load(dp_ptr)
        return self.builder.bitcast(raw, ir.PointerType(elem_lt))

    def _compile_array_literal(self, node: ArrayLiteral) -> tuple[ir.Value, str]:
        n = len(node.elements)
        elem_vt = self._infer_type(node.elements[0]) if n > 0 else "int"
        elem_lt = self._vx_to_llvm(elem_vt)
        arr_vt  = elem_vt + "[]"

        esz  = ir.Constant(I64_TY, _elem_size(elem_vt))
        cnt  = ir.Constant(I64_TY, n)
        tsz  = self.builder.mul(cnt, esz) if n > 0 else ir.Constant(I64_TY, 8)

        # Allocate data
        raw_data = self.builder.call(self.malloc_fn, [tsz])
        data_ptr = self.builder.bitcast(raw_data, ir.PointerType(elem_lt))

        # Fill elements
        for i, elem in enumerate(node.elements):
            val, vt = self._compile_expr(elem)
            if elem_vt == "float" and vt == "int":
                val = self.builder.sitofp(val, F64_TY)
            ep = self.builder.gep(data_ptr, [ir.Constant(I64_TY, i)], inbounds=True)
            self.builder.store(val, ep)

        # Allocate header {i8*, i64, i64}
        hsz        = ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)
        raw_header = self.builder.call(self.malloc_fn, [hsz])
        hdr_ptr    = self.builder.bitcast(raw_header, self.arr_ptr_type)

        # Store data ptr, length, and capacity (cap = len initially)
        dp_field = self.builder.gep(hdr_ptr,
                                    [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)],
                                    inbounds=True)
        self.builder.store(raw_data, dp_field)
        ln_field = self.builder.gep(hdr_ptr,
                                    [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)],
                                    inbounds=True)
        self.builder.store(cnt, ln_field)
        cap_field = self.builder.gep(hdr_ptr,
                                     [ir.Constant(I32_TY,0), ir.Constant(I32_TY,2)],
                                     inbounds=True)
        # Initial capacity = max(len, 4) so push() has room
        initial_cap = ir.Constant(I64_TY, max(n, 4))
        self.builder.store(initial_cap, cap_field)

        return hdr_ptr, arr_vt

    # ------------------------------------------------------------------ #
    #  Structs                                                             #
    # ------------------------------------------------------------------ #

    def _compile_new(self, node: NewExpr) -> tuple[ir.Value, str]:
        si = self._structs.get(node.type_name)
        if si is None:
            raise CodegenError(f"Unknown struct '{node.type_name}'")
        lt     = si["llvm_type"]
        fields = si["fields"]

        # sizeof via GEP trick
        null_ptr = ir.Constant(ir.PointerType(lt), None)
        sp       = self.builder.gep(null_ptr, [ir.Constant(I32_TY, 1)], inbounds=False)
        sz       = self.builder.ptrtoint(sp, I64_TY)
        raw      = self.builder.call(self.malloc_fn, [sz])
        sptr     = self.builder.bitcast(raw, ir.PointerType(lt))

        for idx, (fname, ftype) in enumerate(fields):
            fv = ir.Constant(self._vx_to_llvm(ftype), 0)
            if idx < len(node.args):
                fv, fvt = self._compile_expr(node.args[idx])
                if ftype == "float" and fvt == "int":
                    fv = self.builder.sitofp(fv, F64_TY)
            fp = self.builder.gep(sptr,
                                  [ir.Constant(I32_TY,0), ir.Constant(I32_TY,idx)],
                                  inbounds=True)
            self.builder.store(fv, fp)
        return sptr, node.type_name

    # ------------------------------------------------------------------ #
    #  Inline string helpers (no external runtime needed)                 #
    # ------------------------------------------------------------------ #

    def _str_concat_inline(self, a: ir.Value, b: ir.Value) -> ir.Value:
        la    = self.builder.call(self.strlen_fn, [a])
        lb    = self.builder.call(self.strlen_fn, [b])
        total = self.builder.add(la, lb)
        buf   = self.builder.call(self.malloc_fn,
                                  [self.builder.add(total, ir.Constant(I64_TY, 1))])
        self.builder.call(self.memcpy_fn, [buf, a, la])
        tail  = self.builder.gep(buf, [la], inbounds=False)
        self.builder.call(self.memcpy_fn, [tail, b, lb])
        null_pos = self.builder.gep(buf, [total], inbounds=False)
        self.builder.store(ir.Constant(I8_TY, 0), null_pos)
        return buf

    def _int_to_str_inline(self, val: ir.Value) -> ir.Value:
        buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])
        fmt = self._gstr_ptr(self._global_str("%lld"))
        self.builder.call(self.sprintf_fn, [buf, fmt, val])
        return buf

    def _float_to_str_inline(self, val: ir.Value) -> ir.Value:
        buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])
        fmt = self._gstr_ptr(self._global_str("%g"))
        self.builder.call(self.sprintf_fn, [buf, fmt, val])
        return buf

    def _field_ptr(self, node: FieldAccess) -> ir.Value:
        ov, ot = self._compile_expr(node.obj)
        si     = self._structs.get(ot)
        if si is None:
            raise CodegenError(f"'{ot}' is not a struct")
        for idx, (fn, _) in enumerate(si["fields"]):
            if fn == node.field:
                return self.builder.gep(ov,
                                        [ir.Constant(I32_TY,0), ir.Constant(I32_TY,idx)],
                                        inbounds=True)
        raise CodegenError(f"Struct '{ot}' has no field '{node.field}'")


# ================================================================== #
#  JIT runner                                                          #
# ================================================================== #

def _init_llvm():
    for fn in (binding.initialize,
               binding.initialize_native_target,
               binding.initialize_native_asmprinter):
        try: fn()
        except RuntimeError: pass


def jit_run(llvm_ir: str) -> int:
    _init_llvm()
    target = binding.Target.from_default_triple()
    tm     = target.create_target_machine()
    mod    = binding.parse_assembly(llvm_ir)
    mod.verify()
    engine = binding.create_mcjit_compiler(mod, tm)
    engine.finalize_object()
    engine.run_static_constructors()
    addr   = engine.get_function_address("main")
    if not addr:
        raise CodegenError("No 'main' function found")
    ctypes.CFUNCTYPE(None)(addr)()
    return 0


# ================================================================== #
#  Native binary                                                       #
# ================================================================== #

def compile_to_binary(llvm_ir: str, output_path: str,
                      runtime_obj: str | None = None,
                      gcc_path: str = "gcc",
                      target_triple: str | None = None):
    import subprocess, tempfile, os
    _init_llvm()
    triple = target_triple or MINGW_TRIPLE
    target = binding.Target.from_triple(triple)
    tm     = target.create_target_machine(reloc="pic", codemodel="default")
    mod    = binding.parse_assembly(llvm_ir)
    mod.verify()
    obj    = tm.emit_object(mod)
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        f.write(obj); obj_path = f.name
    try:
        cmd = [gcc_path, obj_path, "-o", output_path, "-lm"]
        if runtime_obj: cmd.insert(1, runtime_obj)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise CodegenError(f"Linker error:\n{r.stderr}")
    finally:
        os.unlink(obj_path)
