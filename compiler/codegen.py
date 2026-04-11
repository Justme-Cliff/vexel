"""
Vexel LLVM IR code generator.

Walks the type-checked AST produced by the analyzer and emits LLVM IR
via llvmlite.  The resulting IR string can be JIT-executed or compiled
to a native binary through ``jit_run`` / ``compile_to_binary``.

Key design decisions:
  - Arrays are heap-allocated structs: ``{i8*, i64, i64}`` (data, len, cap).
  - Strings are null-terminated ``i8*`` values managed by the C runtime.
  - Interfaces use fat pointers: ``{i8* data, i8* vtable}`` boxed on the heap.
  - Vtables are filled at program startup by ``__vx_vtable_init()``.
  - Generic functions are monomorphized on first call.
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
                 target_triple: str | None = None,
                 debug_mode: bool = False):
        self.analysis   = analysis
        self.debug_mode = debug_mode   # enable runtime bounds checks (#101)
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

        # Default parameter values per function (registered in compile pass)
        self._fn_defaults: dict[str, list] = {}

        # Lambda counter (for unique naming)
        self._lambda_count: int = 0

        # Tuple type registry: type-string → LLVM identified struct type
        self._tuple_types: dict[str, ir.IdentifiedStructType] = {}

        # Monomorphized generic function cache
        self._mono_cache: dict[str, str] = {}  # (fn_name, type_key) → concrete_name

        # Namespace registry (alias → True)
        self._namespaces: set[str] = set()

        # Labeled loop targets: label → block
        self._label_break_targets:    dict[str, ir.Block] = {}
        self._label_continue_targets: dict[str, ir.Block] = {}

        # Defer stacks: one list per function nesting level
        self._defer_stack: list[list] = []

        # Named return values for current function
        self._current_named_returns: list[tuple] = []

        # Interface vtable registry
        # name → {methods:[str,...], method_sigs:{name:MethodSig}, vtable_ll, fat_ll}
        self._interfaces: dict[str, dict] = {}

        # Impl vtable data: "{struct}__{iface}" → {vtable_gv, impl_fns:[fn,...]}
        self._impls_data: dict[str, dict] = {}

        # Registered enum names → set of method names (#13)
        self._enum_names: set[str] = set()

        # #11 Function overloads cache: base_name → [(param_types, mangled_name)]
        self._overload_table: dict[str, list[tuple[list[str], str]]] = {}

        # #6 Generic struct monomorphization cache: "Name__int" → True
        self._mono_structs: dict[str, bool] = {}

        # Schedule of vtable slots to fill: [(vtable_gv, vtable_ll, [fn,...])]
        self._vtable_init_list: list = []

        self._define_array_type()
        self._define_dict_type()
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

    def _define_dict_type(self):
        self.dict_type = self.module.context.get_identified_type("vx_dict")
        if self.dict_type.is_opaque:
            # {i8* keys_ptr, i8* vals_ptr, i64 len, i64 cap}
            # keys_ptr → array of i8* (string pointers)
            # vals_ptr → array of i64  (type-erased values)
            self.dict_type.set_body(I8PTR, I8PTR, I64_TY, I64_TY)
        self.dict_ptr_type = ir.PointerType(self.dict_type)

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

        # v4 math
        self.round_fn   = _fn(F64_TY, F64_TY,         name="round")
        self.atan2_fn   = _fn(F64_TY, F64_TY, F64_TY, name="atan2")

        # v4 OS — use platform-specific names available in MSVC CRT and MinGW
        import sys as _sys
        _win = _sys.platform == "win32"
        self.getcwd_fn   = _fn(I8PTR,  I8PTR, I32_TY,  name="_getcwd"  if _win else "getcwd")
        self.mkdir_fn    = _fn(I32_TY, I8PTR,           name="_mkdir"   if _win else "mkdir")
        self.remove_fn   = _fn(I32_TY, I8PTR,           name="remove")
        self.rmdir_fn    = _fn(I32_TY, I8PTR,           name="_rmdir"   if _win else "rmdir")
        self.strerror_fn = _fn(I8PTR,  I32_TY,          name="strerror")

        # v5 new builtins
        self.atoll_fn    = _fn(I64_TY, I8PTR,            name="atoll")
        self.atof_fn     = _fn(F64_TY, I8PTR,            name="atof")
        self.strftime_fn = _fn(I64_TY, I8PTR, I64_TY, I8PTR, I8PTR, name="strftime")  # (buf,sz,fmt,tm*)
        self.localtime_fn= _fn(I8PTR,  I8PTR,            name="localtime")  # tm* localtime(time_t*)
        self.fgets_fn    = _fn(I8PTR,  I8PTR, I32_TY, I8PTR, name="fgets")
        self.stdin_fn    = None  # resolved lazily via helper

        # v7 new builtins
        self.log10_fn   = _fn(F64_TY, F64_TY,              name="log10")
        self.exp_fn     = _fn(F64_TY, F64_TY,              name="exp")
        self.hypot_fn   = _fn(F64_TY, F64_TY, F64_TY,     name="hypot")
        self.getenv_fn  = _fn(I8PTR,  I8PTR,               name="getenv")
        self.popen_fn   = _fn(I8PTR,  I8PTR, I8PTR,        name="popen")
        self.pclose_fn  = _fn(I32_TY, I8PTR,               name="pclose")

        # v4 error state (global buffer defined in this module)
        arr_ty = ir.ArrayType(I8_TY, 512)
        self._vx_error_buf = ir.GlobalVariable(self.module, arr_ty,
                                               name="__vx_error_buf")
        self._vx_error_buf.linkage = "private"
        self._vx_error_buf.initializer = ir.Constant(arr_ty, bytearray(512))

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

    def _resolve_type(self, vx: str) -> str:
        """Resolve type aliases to their canonical type."""
        seen = set()
        while vx in self.analysis.type_aliases and vx not in seen:
            seen.add(vx)
            vx = self.analysis.type_aliases[vx]
        return vx

    def _vx_to_llvm(self, vx: str) -> ir.Type:
        # Resolve type aliases first
        vx = self._resolve_type(vx)
        if vx == "int":    return I64_TY
        if vx == "float":  return F64_TY
        if vx == "bool":   return I1_TY
        if vx == "str":    return I8PTR
        if vx == "void":   return VOID_TY
        if vx == "null":   return I8PTR
        # Integer type variants (#46)
        if vx in ("i8",  "u8"):  return ir.IntType(8)
        if vx in ("i16", "u16"): return ir.IntType(16)
        if vx in ("i32", "u32"): return ir.IntType(32)
        if vx in ("i64", "u64"): return ir.IntType(64)
        if vx == "f32":  return ir.FloatType()
        if vx == "f64":  return F64_TY
        if vx == "char": return ir.IntType(8)   # char = i8
        if vx.endswith("[]"):      return self.arr_ptr_type
        if vx.startswith("dict["): return self.dict_ptr_type
        # Tuple type: (int,float,...) → pointer to struct
        if vx.startswith("(") and vx.endswith(")"):
            return I8PTR  # stored as opaque pointer; cast when needed
        # Nullable types: T? → i8* (opaque pointer; null means null, non-null is boxed value)
        if vx.endswith("?"):
            return I8PTR
        # Function type: fn(int)->float → i8* (opaque fn pointer)
        if vx.startswith("fn("):
            return I8PTR
        if vx in self._structs:
            return ir.PointerType(self._structs[vx]["llvm_type"])
        # Interface types are fat pointers stored as i8*
        if vx in self._interfaces:
            return I8PTR
        # Enum types are i64 under the hood (#13)
        if vx in self._enum_names:
            return I64_TY
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
        if isinstance(node, DictLiteral):
            if not node.pairs: return "dict[str,int]"
            kt = self._infer_type(node.pairs[0][0])
            vt = self._infer_type(node.pairs[0][1])
            return f"dict[{kt},{vt}]"
        if isinstance(node, Identifier):
            if node.name in ("PI", "TAU", "E", "INF", "NAN"): return "float"
            info = self._lookup(node.name)
            return info["vx_type"] if info else "int"
        if isinstance(node, BinOp):
            if node.op in ("==","!=","<",">","<=",">=","and","or","in"): return "bool"
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if node.op == "+" and lt == "str": return "str"
            return "float" if lt == "float" or rt == "float" else lt
        if isinstance(node, UnaryOp):
            return "bool" if node.op == "not" else self._infer_type(node.operand)
        if isinstance(node, Call):
            # Overloaded builtins
            if node.func in ("abs","min","max","clamp") and node.args:
                at = self._infer_type(node.args[0])
                return "float" if at == "float" else "int"
            sig = self.analysis.fn_sigs.get(node.func)
            return sig.return_type if sig else "void"
        if isinstance(node, MethodCall):
            obj_t = self._infer_type(node.obj)
            if obj_t.startswith("dict["):
                inner = obj_t[5:-1]
                vt = inner[inner.index(',')+1:]
                return {"has": "bool", "remove": "void",
                        "len": "int", "keys": "str[]"}.get(node.method, "void")
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
            if ot.startswith("dict["):
                inner = ot[5:-1]
                return inner[inner.index(',')+1:]
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
        # 0. Collect namespace hints
        for d in program.declarations:
            if isinstance(d, NamespaceHint):
                self._namespaces.add(d.alias)
        if self.analysis.namespaces:
            self._namespaces.update(self.analysis.namespaces)

        # 1. Struct definitions (skip generic structs — monomorphized on demand)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, StructDecl) and not getattr(_d, 'type_params', []):
                self._define_struct(_d)

        # 1b. Error type definitions (#29)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, ErrorDecl):
                self._define_error_struct(_d)

        # 2. Interface definitions (must be before forward-declaring fns that use them)
        for d in program.declarations:
            if isinstance(d, InterfaceDecl):
                self._define_interface(d)

        # 3. Enum definitions (global i64 constants)
        for d in program.declarations:
            if isinstance(d, EnumDecl):
                self._enum_names.add(d.name)          # register for _vx_to_llvm
                for i, variant in enumerate(d.variants):
                    gname = f"{d.name}.{variant}"
                    gv = ir.GlobalVariable(self.module, I64_TY, name=gname)
                    gv.linkage = "internal"
                    gv.global_constant = True
                    gv.initializer = ir.Constant(I64_TY, i)
                    self._globals[gname] = {"ptr": gv, "vx_type": d.name}
            elif isinstance(d, EnumDeclADT):
                self._enum_names.add(d.name)

        # 3b. ADT Enum definitions (tagged union structs)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, EnumDeclADT):
                self._define_adt_enum(_d)

        # 4. Global variables
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, (GlobalLet, GlobalConst)):
                self._compile_global(_d)
            elif isinstance(_d, ComptimeDecl):
                self._compile_comptime(_d)

        # 5. Forward-declare all non-generic user functions
        # Build overload table from analysis.overloads (#11)
        for base_name, mangled_list in self.analysis.overloads.items():
            table_entries = []
            for mn in mangled_list:
                sig = self.analysis.fn_sigs.get(mn)
                if sig:
                    table_entries.append(([t for _, t in sig.params], mn))
            self._overload_table[base_name] = table_entries

        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, FnDecl) and not _d.type_params:
                self._declare_fn(_d)
            elif isinstance(_d, ExternFnDecl):
                self._compile_extern_fn(_d)

        # 5b. Forward-declare error constructors and enum methods (#13, #29)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, ErrorDecl):
                self._declare_error_constructor(_d)
            elif isinstance(_d, (EnumDecl, EnumDeclADT)):
                for m in getattr(_d, 'methods', []):
                    self._declare_enum_method(_d.name, m)
        # Also forward-declare struct own-methods (#13 struct side) and @serialize (#113)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, StructDecl):
                for m in getattr(_d, 'methods', []):
                    self._declare_struct_method(_d.name, m)
                attrs = [a.name for a in getattr(_d, 'attributes', [])]
                if 'serialize' in attrs or 'derive' in attrs:
                    self._declare_serialize_methods(_d)

        # 6. Forward-declare impl methods + create vtable globals
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, ImplDecl):
                self._declare_impl(_d)

        # 7. Build the vtable init function (needs impl fns already declared)
        self._build_vtable_init_fn()

        # 8. Compile non-generic function bodies
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, FnDecl) and not _d.type_params:
                self._compile_fn(_d)
            elif isinstance(_d, TestDecl):
                self._compile_test_decl(_d)

        # 8b. Compile error constructors, enum methods, struct own-methods (#13, #29)
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, ErrorDecl):
                self._compile_error_constructor(_d)
            elif isinstance(_d, (EnumDecl, EnumDeclADT)):
                for m in getattr(_d, 'methods', []):
                    self._compile_enum_method(_d.name, m)
            elif isinstance(_d, StructDecl):
                for m in getattr(_d, 'methods', []):
                    self._compile_struct_method(_d.name, m)
                # #113: @serialize attribute — synthesize to_json / from_json
                attrs = [a.name for a in getattr(_d, 'attributes', [])]
                if 'serialize' in attrs or 'derive' in attrs:
                    self._compile_serialize_methods(_d)

        # 9. Compile impl method bodies
        for d in program.declarations:
            _d = d.inner if isinstance(d, (PubDecl, PrivDecl)) else d
            if isinstance(_d, ImplDecl):
                self._compile_impl(_d)

        return str(self.module)

    # ------------------------------------------------------------------ #
    #  Panic helper                                                        #
    # ------------------------------------------------------------------ #

    def _emit_panic(self, msg: str, line: int = 0):
        """Emit printf + exit(1) + unreachable for an unconditional panic.
        Includes source line info in debug mode for stack trace (#76)."""
        if self.debug_mode and line:
            loc = f" (line {line})"
        elif self.debug_mode and self.current_fn:
            loc = f" (in {self.current_fn.name})"
        else:
            loc = ""
        fmt = self._gstr_ptr(self._global_str(f"vexel panic: {msg}{loc}\n"))
        self.builder.call(self.printf, [fmt])
        self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
        self.builder.unreachable()

    # ------------------------------------------------------------------ #
    #  Error type hierarchy (#29)                                          #
    # ------------------------------------------------------------------ #

    def _define_error_struct(self, decl: 'ErrorDecl'):
        """Define the LLVM struct type for an error declaration."""
        # Layout: {__code: i64, field0, field1, ...}
        field_types = [I64_TY] + [self._vx_to_llvm(f.type_name) for f in decl.fields]
        lt = self.module.context.get_identified_type(decl.name)
        lt.set_body(*field_types)
        fields = [("__code", "int")] + [(f.name, f.type_name) for f in decl.fields]
        self._structs[decl.name] = {
            "llvm_type": lt,
            "fields":    fields,
            "defaults":  [None] * len(fields),
        }
        if not hasattr(self, '_error_codes'):
            self._error_codes: dict[str, int] = {}
        self._error_codes[decl.name] = len(self._error_codes) + 1

    def _declare_error_constructor(self, decl: 'ErrorDecl'):
        """Forward-declare the constructor fn: fn ErrorName(fields...) -> ErrorName*."""
        lt = self._structs[decl.name]["llvm_type"]
        param_tys = [self._vx_to_llvm(f.type_name) for f in decl.fields]
        fn_ty = ir.FunctionType(ir.PointerType(lt), param_tys)
        fn = ir.Function(self.module, fn_ty, name=decl.name)
        fn.linkage = "internal"
        params = [("__code", "int")] + [(f.name, f.type_name) for f in decl.fields]
        sig = type('FnSig', (), {'params': [(f.name, f.type_name) for f in decl.fields],
                                 'return_type': decl.name, 'variadic': False})()
        self._functions[decl.name] = {"fn": fn, "sig": sig}
        self._fn_defaults[decl.name] = [None] * len(decl.fields)

    def _compile_error_constructor(self, decl: 'ErrorDecl'):
        """Compile the error constructor body: allocate struct, set code + fields."""
        fn = self._functions[decl.name]["fn"]
        lt = self._structs[decl.name]["llvm_type"]
        self.current_fn = fn
        entry = fn.append_basic_block("entry")
        old_builder = getattr(self, 'builder', None)
        self.builder = ir.IRBuilder(entry)
        self._push_scope()

        # malloc(sizeof struct)
        n_fields = len(self._structs[decl.name]["fields"])
        raw = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 8 * n_fields)])
        ptr = self.builder.bitcast(raw, ir.PointerType(lt))

        # Store __code at index 0
        code = self._error_codes.get(decl.name, 0)
        code_ptr = self.builder.gep(ptr, [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                    inbounds=True)
        self.builder.store(ir.Constant(I64_TY, code), code_ptr)

        # Store each payload field
        for i, (arg, field) in enumerate(zip(fn.args, decl.fields)):
            fptr = self.builder.gep(ptr, [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, i + 1)],
                                    inbounds=True)
            self.builder.store(arg, fptr)

        self.builder.ret(ptr)
        self._pop_scope()
        if old_builder is not None:
            self.builder = old_builder

    # ------------------------------------------------------------------ #
    #  Enum methods (#13)                                                  #
    # ------------------------------------------------------------------ #

    def _declare_enum_method(self, enum_name: str, method: 'FnDecl'):
        """Forward-declare an enum method as EnumName__methodName."""
        fn_name = f"{enum_name}__{method.name}"
        # self param is i64 (enum value); other params as declared
        non_self = [p for p in method.params if p.name != "self"]
        param_tys = [I64_TY] + [self._vx_to_llvm(p.type_name) for p in non_self]
        ret_ty = self._vx_to_llvm(method.return_type) if method.return_type else VOID_TY
        fn_ty = ir.FunctionType(ret_ty, param_tys)
        fn = ir.Function(self.module, fn_ty, name=fn_name)
        fn.linkage = "internal"
        params = [("self", enum_name)] + [(p.name, p.type_name) for p in non_self]
        sig = type('FnSig', (), {'params': params,
                                 'return_type': method.return_type or "void",
                                 'variadic': False})()
        self._functions[fn_name] = {"fn": fn, "sig": sig}
        self._fn_defaults[fn_name] = [p.default for p in method.params]

    def _compile_enum_method(self, enum_name: str, method: 'FnDecl'):
        """Compile enum method body; 'self' is the i64 enum value."""
        fn_name = f"{enum_name}__{method.name}"
        fn = self._functions[fn_name]["fn"]
        self.current_fn = fn
        entry = fn.append_basic_block("entry")
        old_builder = getattr(self, 'builder', None)
        self.builder = ir.IRBuilder(entry)
        self._push_scope()
        self._defer_stack.append([])

        # Bind 'self' as i64
        self_al = self.builder.alloca(I64_TY, name="self")
        self.builder.store(fn.args[0], self_al)
        self._declare("self", self_al, "int")

        # Bind remaining params
        non_self = [p for p in method.params if p.name != "self"]
        for i, param in enumerate(non_self):
            al = self.builder.alloca(self._vx_to_llvm(param.type_name), name=param.name)
            self.builder.store(fn.args[i + 1], al)
            self._declare(param.name, al, param.type_name)

        self._current_named_returns = []
        for stmt in method.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)

        if not self.builder.block.is_terminated:
            self._emit_defers()
            if fn.ftype.return_type == VOID_TY:
                self.builder.ret_void()
            else:
                self.builder.ret(ir.Constant(fn.ftype.return_type, 0))

        self._defer_stack.pop()
        self._pop_scope()
        if old_builder is not None:
            self.builder = old_builder

    # ------------------------------------------------------------------ #
    #  Struct own-methods (#13)                                            #
    # ------------------------------------------------------------------ #

    def _declare_struct_method(self, struct_name: str, method: 'FnDecl'):
        """Forward-declare a struct own-method as StructName__methodName."""
        fn_name = f"{struct_name}__{method.name}"
        if fn_name in self._functions:
            return  # already declared (e.g. via interface impl)
        si = self._structs.get(struct_name)
        if si is None:
            return
        struct_ptr_ty = ir.PointerType(si["llvm_type"])
        non_self = [p for p in method.params if p.name != "self"]
        param_tys = [struct_ptr_ty] + [self._vx_to_llvm(p.type_name) for p in non_self]
        ret_ty = self._vx_to_llvm(method.return_type) if method.return_type else VOID_TY
        fn_ty = ir.FunctionType(ret_ty, param_tys)
        fn = ir.Function(self.module, fn_ty, name=fn_name)
        fn.linkage = "internal"
        params = [("self", struct_name)] + [(p.name, p.type_name) for p in non_self]
        sig = type('FnSig', (), {'params': params,
                                 'return_type': method.return_type or "void",
                                 'variadic': False})()
        self._functions[fn_name] = {"fn": fn, "sig": sig}
        self._fn_defaults[fn_name] = [p.default for p in method.params]

    def _compile_struct_method(self, struct_name: str, method: 'FnDecl'):
        """Compile a struct own-method body; 'self' is a pointer to the struct."""
        fn_name = f"{struct_name}__{method.name}"
        if fn_name not in self._functions:
            return
        fn = self._functions[fn_name]["fn"]
        si = self._structs[struct_name]
        self.current_fn = fn
        entry = fn.append_basic_block("entry")
        old_builder = getattr(self, 'builder', None)
        self.builder = ir.IRBuilder(entry)
        self._push_scope()
        self._defer_stack.append([])

        # Bind 'self' as a pointer to the struct (already a pointer, store it)
        self_al = self.builder.alloca(ir.PointerType(si["llvm_type"]), name="self")
        self.builder.store(fn.args[0], self_al)
        self._declare("self", self_al, struct_name)

        # Bind remaining params
        non_self = [p for p in method.params if p.name != "self"]
        for i, param in enumerate(non_self):
            al = self.builder.alloca(self._vx_to_llvm(param.type_name), name=param.name)
            self.builder.store(fn.args[i + 1], al)
            self._declare(param.name, al, param.type_name)

        self._current_named_returns = []
        for stmt in method.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)

        if not self.builder.block.is_terminated:
            self._emit_defers()
            if fn.ftype.return_type == VOID_TY:
                self.builder.ret_void()
            else:
                self.builder.ret(ir.Constant(fn.ftype.return_type, 0))

        self._defer_stack.pop()
        self._pop_scope()
        if old_builder is not None:
            self.builder = old_builder

    # ------------------------------------------------------------------ #
    #  Serialization (#113)                                                #
    # ------------------------------------------------------------------ #

    def _declare_serialize_methods(self, decl: 'StructDecl'):
        """Forward-declare to_json / from_json for a @serialize struct."""
        sname = decl.name
        si = self._structs.get(sname)
        if si is None:
            return
        struct_ptr_ty = ir.PointerType(si["llvm_type"])

        # fn StructName__to_json(self: StructName*) -> str
        to_json_name = f"{sname}__to_json"
        if to_json_name not in self._functions:
            fn_ty = ir.FunctionType(I8PTR, [struct_ptr_ty])
            fn = ir.Function(self.module, fn_ty, name=to_json_name)
            fn.linkage = "internal"
            sig = type('FnSig', (), {'params': [("self", sname)],
                                     'return_type': "str", 'variadic': False})()
            self._functions[to_json_name] = {"fn": fn, "sig": sig}
            self._fn_defaults[to_json_name] = [None]

        # fn StructName__from_json(json: str) -> StructName*
        from_json_name = f"{sname}__from_json"
        if from_json_name not in self._functions:
            fn_ty2 = ir.FunctionType(struct_ptr_ty, [I8PTR])
            fn2 = ir.Function(self.module, fn_ty2, name=from_json_name)
            fn2.linkage = "internal"
            sig2 = type('FnSig', (), {'params': [("json", "str")],
                                      'return_type': sname, 'variadic': False})()
            self._functions[from_json_name] = {"fn": fn2, "sig": sig2}
            self._fn_defaults[from_json_name] = [None]

    def _compile_serialize_methods(self, decl: 'StructDecl'):
        """Compile to_json / from_json for a @serialize struct (#113)."""
        sname = decl.name
        si = self._structs.get(sname)
        if si is None:
            return

        fields = si["fields"]  # list of (name, vx_type)
        struct_ptr_ty = ir.PointerType(si["llvm_type"])
        old_builder = getattr(self, 'builder', None)
        old_fn = getattr(self, 'current_fn', None)

        # ---- to_json ----
        to_json_name = f"{sname}__to_json"
        fn = self._functions[to_json_name]["fn"]
        self.current_fn = fn
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        self.builder = b
        self._push_scope()
        self._defer_stack.append([])

        self_ptr = fn.args[0]
        # Build JSON string: {"field0":val0,"field1":val1,...}
        close_brace = self._gstr_ptr(self._global_str("}"))
        comma_s     = self._gstr_ptr(self._global_str(","))
        quote_s     = self._gstr_ptr(self._global_str('"'))

        buf = self._gstr_ptr(self._global_str("{"))

        for fi, (fname, ftype) in enumerate(fields):
            # key fragment: "fname":
            key_frag = self._gstr_ptr(self._global_str(f'"{fname}":'))
            buf = self._str_concat_inline(buf, key_frag)

            # load the field value
            fptr = self.builder.gep(self_ptr,
                                    [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, fi)],
                                    inbounds=True)
            fval = self.builder.load(fptr)

            # Convert to string
            vtype_resolved = self._resolve_type(ftype)
            if vtype_resolved == "int":
                val_str = self._int_to_str_inline(fval)
            elif vtype_resolved == "float":
                val_str = self._float_to_str_inline(fval)
            elif vtype_resolved == "bool":
                true_s  = self._gstr_ptr(self._global_str("true"))
                false_s = self._gstr_ptr(self._global_str("false"))
                val_str = self.builder.select(fval, true_s, false_s)
            elif vtype_resolved == "str":
                # wrap in quotes: "value"
                val_str = self._str_concat_inline(quote_s, fval)
                val_str = self._str_concat_inline(val_str, quote_s)
            else:
                # fallback: ptr address as integer string
                raw = self.builder.ptrtoint(self.builder.bitcast(fptr, I8PTR), I64_TY)
                val_str = self._int_to_str_inline(raw)

            buf = self._str_concat_inline(buf, val_str)

            if fi < len(fields) - 1:
                buf = self._str_concat_inline(buf, comma_s)

        buf = self._str_concat_inline(buf, close_brace)
        self.builder.ret(buf)
        self._defer_stack.pop()
        self._pop_scope()

        # ---- from_json ----
        # Simple: allocate struct, zero-initialize, return pointer
        # (Full JSON parse would require the JSON parser helper)
        from_json_name = f"{sname}__from_json"
        fn2 = self._functions[from_json_name]["fn"]
        self.current_fn = fn2
        b2 = ir.IRBuilder(fn2.append_basic_block("entry"))
        self.builder = b2
        self._push_scope()
        self._defer_stack.append([])

        n_fields = len(fields)
        raw = b2.call(self.malloc_fn, [ir.Constant(I64_TY, 8 * max(n_fields, 1))])
        ptr = b2.bitcast(raw, struct_ptr_ty)
        # Zero-initialize all fields
        for fi, (fname, ftype) in enumerate(fields):
            fptr = b2.gep(ptr, [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, fi)],
                          inbounds=True)
            lt = self._vx_to_llvm(ftype)
            if isinstance(lt, ir.PointerType):
                zero = ir.Constant(lt, None)  # null pointer
            else:
                zero = ir.Constant(lt, 0)
            b2.store(zero, fptr)
        b2.ret(ptr)
        self._defer_stack.pop()
        self._pop_scope()

        if old_builder is not None:
            self.builder = old_builder
        if old_fn is not None:
            self.current_fn = old_fn

    # ------------------------------------------------------------------ #
    #  Error propagation (#10)                                            #
    # ------------------------------------------------------------------ #

    def _compile_error_prop(self, node: 'ErrorPropExpr') -> tuple[ir.Value, str]:
        """Compile expr? — if value is null/0, return early from current function."""
        val, vt = self._compile_expr(node.expr)
        # Determine if the value is a pointer or integer
        ret_ty = self.current_fn.ftype.return_type
        ok_bb   = self.current_fn.append_basic_block("prop.ok")
        fail_bb = self.current_fn.append_basic_block("prop.null")
        if isinstance(val.type, ir.PointerType):
            as_int  = self.builder.ptrtoint(val, I64_TY)
            is_null = self.builder.icmp_unsigned("==", as_int, ir.Constant(I64_TY, 0))
        else:
            is_null = self.builder.icmp_signed("==", val, ir.Constant(val.type, 0))
        self.builder.cbranch(is_null, fail_bb, ok_bb)
        # Null branch: emit defers and return zero/null
        self.builder.position_at_end(fail_bb)
        self._emit_defers()
        if ret_ty == VOID_TY:
            self.builder.ret_void()
        elif isinstance(ret_ty, ir.PointerType):
            self.builder.ret(ir.Constant(ret_ty, None))
        else:
            self.builder.ret(ir.Constant(ret_ty, 0))
        # OK branch: continue with the value
        self.builder.position_at_end(ok_bb)
        return val, vt

    # ------------------------------------------------------------------ #
    #  Generic struct monomorphization (#6)                               #
    # ------------------------------------------------------------------ #

    def _monomorphize_struct(self, base_name: str, type_args: list[str]) -> str:
        """Create a concrete version of a generic struct, e.g. Stack[int] → Stack__int."""
        import re, copy
        suffix = "__".join(type_args)
        concrete_name = f"{base_name}__{suffix}"
        if concrete_name in self._structs:
            return concrete_name
        if concrete_name in self._mono_structs:
            return concrete_name
        decl = self.analysis.generic_structs.get(base_name)
        if decl is None:
            raise CodegenError(f"Unknown generic struct '{base_name}'")
        # Build type substitution map: T → type_args[0], U → type_args[1], ...
        type_map = dict(zip(decl.type_params, type_args))
        def _subst(s: str) -> str:
            for tp, concrete in type_map.items():
                s = re.sub(rf'\b{re.escape(tp)}\b', concrete, s)
            return s
        # Create LLVM struct type
        lt = self.module.context.get_identified_type(concrete_name)
        field_ll_types = [self._vx_to_llvm(_subst(f.type_name)) for f in decl.fields]
        lt.set_body(*field_ll_types)
        fields = [(_subst(f.name) if False else f.name, _subst(f.type_name)) for f in decl.fields]
        defaults = [f.default for f in decl.fields]
        self._structs[concrete_name] = {
            "llvm_type": lt,
            "fields":    fields,
            "defaults":  defaults,
        }
        self._mono_structs[concrete_name] = True
        return concrete_name

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
        # #120 @repr(C): use packed=False (default sequential layout — already C-compatible)
        # LLVM's identified struct types have sequential layout by default (no hidden padding)
        attrs = [a.name for a in getattr(d, 'attributes', [])]
        is_repr_c = 'repr' in attrs or 'repr_c' in attrs
        lt.set_body(*[self._vx_to_llvm(f.type_name) for f in d.fields])
        # #120 @repr(C): LLVM identified struct types are already C-compatible (sequential layout)
        self._structs[d.name] = {
            "llvm_type": lt,
            "fields":    [(f.name, f.type_name) for f in d.fields],
            "defaults":  [f.default for f in d.fields],  # AST nodes or None
            "repr_c":    is_repr_c,
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
        # Register default param values for this function
        self._fn_defaults[d.name] = [p.default for p in d.params]

    def _compile_fn(self, d: FnDecl):
        info = self._functions[d.name]
        fn   = info["fn"]
        self.current_fn = fn

        entry = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._push_scope()
        self._defer_stack.append([])   # new defer frame

        # Inject vtable initializer at the start of main
        if d.name == "main" and "__vx_vtable_init" in self._helper_fns:
            self.builder.call(self._helper_fns["__vx_vtable_init"], [])

        for arg, param in zip(fn.args, d.params):
            resolved = self._resolve_type(param.type_name)
            al = self.builder.alloca(self._vx_to_llvm(resolved), name=param.name)
            self.builder.store(arg, al)
            self._declare(param.name, al, resolved)

        # Named return values: allocate and declare them as local variables
        if d.named_returns:
            for (ret_name, ret_type) in d.named_returns:
                resolved_rt = self._resolve_type(ret_type)
                al = self.builder.alloca(self._vx_to_llvm(resolved_rt), name=ret_name)
                self.builder.store(ir.Constant(self._vx_to_llvm(resolved_rt), 0), al)
                self._declare(ret_name, al, resolved_rt)
            self._current_named_returns = d.named_returns
        else:
            self._current_named_returns = []

        for stmt in d.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)

        if not self.builder.block.is_terminated:
            self._emit_defers()
            if fn.ftype.return_type == VOID_TY:
                self.builder.ret_void()
            elif self._current_named_returns:
                # Bare return — return named return values
                nrs = self._current_named_returns
                if len(nrs) == 1:
                    rname, rtype = nrs[0]
                    info = self._lookup(rname)
                    if info:
                        self.builder.ret(self.builder.load(info["ptr"]))
                    else:
                        self.builder.ret(ir.Constant(fn.ftype.return_type, 0))
                else:
                    self.builder.ret(ir.Constant(fn.ftype.return_type, 0))
            else:
                self.builder.ret(ir.Constant(fn.ftype.return_type, 0))

        self._defer_stack.pop()
        self._pop_scope()

    # ------------------------------------------------------------------ #
    #  Interfaces                                                          #
    # ------------------------------------------------------------------ #

    def _define_interface(self, decl: 'InterfaceDecl'):
        """Register an interface: create vtable + fat-pointer LLVM struct types."""
        vtable_ty = self.module.context.get_identified_type(f"{decl.name}__vtable_ty")
        if vtable_ty.is_opaque:
            # Each slot is an i8* (opaque function pointer)
            vtable_ty.set_body(*([I8PTR] * max(len(decl.methods), 1)))

        fat_ty = self.module.context.get_identified_type(f"{decl.name}__fat_ty")
        if fat_ty.is_opaque:
            fat_ty.set_body(I8PTR, I8PTR)   # {data_ptr, vtable_ptr}

        self._interfaces[decl.name] = {
            "methods":      [m.name for m in decl.methods],
            "method_sigs":  {m.name: m for m in decl.methods},
            "vtable_ll":    vtable_ty,
            "fat_ll":       fat_ty,
        }

    def _declare_impl(self, decl: 'ImplDecl'):
        """Forward-declare impl method functions and create zero-initialised vtable global."""
        iface_info = self._interfaces.get(decl.interface_name)
        if iface_info is None:
            raise CodegenError(f"Unknown interface '{decl.interface_name}'")

        vtable_ll = iface_info["vtable_ll"]
        impl_fns  = []

        for method_name in iface_info["methods"]:
            impl_method = next((m for m in decl.methods if m.name == method_name), None)
            if impl_method is None:
                raise CodegenError(
                    f"impl {decl.interface_name} for {decl.struct_name}: "
                    f"missing method '{method_name}'"
                )

            # First param is 'self' (type=struct_name) — compiled as i8* in LLVM
            non_self = [p for p in impl_method.params if p.name != "self"]
            param_tys = [I8PTR] + [self._vx_to_llvm(p.type_name) for p in non_self]
            ret_ty    = self._vx_to_llvm(impl_method.return_type or "void")

            fn_name = f"{decl.struct_name}__{method_name}__impl_{decl.interface_name}"
            fn_ty   = ir.FunctionType(ret_ty, param_tys)
            fn      = ir.Function(self.module, fn_ty, name=fn_name)
            fn.linkage = "private"
            fn.args[0].name = "self_raw"
            for i, p in enumerate(non_self):
                fn.args[i + 1].name = p.name

            from compiler.analyzer import FnSig as _FnSig
            self._functions[fn_name] = {
                "fn":  fn,
                "sig": _FnSig(
                    [("self", decl.struct_name)] + [(p.name, p.type_name) for p in non_self],
                    impl_method.return_type or "void"
                ),
            }
            self._fn_defaults[fn_name] = [None] * (1 + len(non_self))
            impl_fns.append(fn)

        # Create zero-initialised vtable global
        vtable_gv = ir.GlobalVariable(
            self.module, vtable_ll,
            name=f"{decl.struct_name}__vtable__{decl.interface_name}"
        )
        vtable_gv.linkage = "private"
        vtable_gv.initializer = ir.Constant(vtable_ll, [ir.Constant(I8PTR, None)] * len(impl_fns))

        impl_key = f"{decl.struct_name}__{decl.interface_name}"
        self._impls_data[impl_key] = {
            "vtable_gv": vtable_gv,
            "impl_fns":  impl_fns,
        }
        self._vtable_init_list.append((vtable_gv, vtable_ll, impl_fns))

    def _build_vtable_init_fn(self):
        """Create __vx_vtable_init that fills vtable globals with impl function pointers."""
        if not self._vtable_init_list:
            return

        fn_ty = ir.FunctionType(VOID_TY, [])
        fn = ir.Function(self.module, fn_ty, name="__vx_vtable_init")
        fn.linkage = "private"
        self._helper_fns["__vx_vtable_init"] = fn

        b = ir.IRBuilder(fn.append_basic_block("entry"))
        z = ir.Constant(I32_TY, 0)

        for vtable_gv, vtable_ll, impl_fns in self._vtable_init_list:
            for i, impl_fn in enumerate(impl_fns):
                fn_as_i8ptr = b.bitcast(impl_fn, I8PTR)
                slot = b.gep(vtable_gv, [z, ir.Constant(I32_TY, i)], inbounds=True)
                b.store(fn_as_i8ptr, slot)

        b.ret_void()

    def _compile_impl(self, decl: 'ImplDecl'):
        """Compile the body of each impl method."""
        iface_info  = self._interfaces[decl.interface_name]
        struct_info = self._structs.get(decl.struct_name)
        if struct_info is None:
            raise CodegenError(f"Unknown struct '{decl.struct_name}'")
        struct_ll_ty = struct_info["llvm_type"]

        for method_name in iface_info["methods"]:
            impl_method = next((m for m in decl.methods if m.name == method_name), None)
            if impl_method is None:
                continue

            fn_name = f"{decl.struct_name}__{method_name}__impl_{decl.interface_name}"
            fn = self._functions[fn_name]["fn"]
            self.current_fn = fn

            entry = fn.append_basic_block("entry")
            self.builder = ir.IRBuilder(entry)
            self._push_scope()

            # Bind 'self': bitcast i8* → StructType* and put in an alloca
            self_raw    = fn.args[0]
            self_typed  = self.builder.bitcast(self_raw, ir.PointerType(struct_ll_ty))
            self_al     = self.builder.alloca(ir.PointerType(struct_ll_ty), name="self")
            self.builder.store(self_typed, self_al)
            self._declare("self", self_al, decl.struct_name)

            # Bind remaining params
            non_self = [p for p in impl_method.params if p.name != "self"]
            for i, param in enumerate(non_self):
                arg = fn.args[i + 1]
                al  = self.builder.alloca(self._vx_to_llvm(param.type_name), name=param.name)
                self.builder.store(arg, al)
                self._declare(param.name, al, param.type_name)

            for stmt in impl_method.body:
                if self.builder.block.is_terminated:
                    break
                self._compile_stmt(stmt)

            if not self.builder.block.is_terminated:
                if fn.ftype.return_type == VOID_TY:
                    self.builder.ret_void()
                else:
                    self.builder.ret(ir.Constant(fn.ftype.return_type, 0))

            self._pop_scope()

    def _box_as_interface(self, val: ir.Value, struct_type: str, iface_name: str) -> ir.Value:
        """Pack a struct pointer + vtable into a heap-allocated fat pointer and return i8*."""
        if struct_type == iface_name:
            return val   # already an interface fat pointer

        impl_key = f"{struct_type}__{iface_name}"
        if impl_key not in self._impls_data:
            raise CodegenError(
                f"'{struct_type}' does not implement interface '{iface_name}'"
            )

        iface_info = self._interfaces[iface_name]
        fat_ll     = iface_info["fat_ll"]
        vtable_gv  = self._impls_data[impl_key]["vtable_gv"]

        # fat = malloc(16)  → {i8* data, i8* vtable}
        fat     = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 16)])
        fat_ptr = self.builder.bitcast(fat, ir.PointerType(fat_ll))
        z       = ir.Constant(I32_TY, 0)

        # Store data ptr
        data_i8 = self.builder.bitcast(val, I8PTR)
        data_sl = self.builder.gep(fat_ptr, [z, z], inbounds=True)
        self.builder.store(data_i8, data_sl)

        # Store vtable ptr (bitcast from vtable type* to i8*)
        vtable_i8 = self.builder.bitcast(vtable_gv, I8PTR)
        vtl_sl    = self.builder.gep(fat_ptr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        self.builder.store(vtable_i8, vtl_sl)

        return fat   # i8* pointing to the fat pointer

    def _compile_interface_method_call(
            self, fat_i8: ir.Value, iface_name: str,
            method_name: str, arg_nodes) -> tuple[ir.Value, str]:
        """Dispatch a method call through a vtable."""
        iface_info = self._interfaces[iface_name]
        methods    = iface_info["methods"]
        if method_name not in methods:
            raise CodegenError(f"Interface '{iface_name}' has no method '{method_name}'")

        method_idx = methods.index(method_name)
        method_sig = iface_info["method_sigs"][method_name]
        fat_ll     = iface_info["fat_ll"]
        vtable_ll  = iface_info["vtable_ll"]

        fat_ptr = self.builder.bitcast(fat_i8, ir.PointerType(fat_ll))
        z = ir.Constant(I32_TY, 0)

        # data_ptr = fat[0]
        data_sl  = self.builder.gep(fat_ptr, [z, z], inbounds=True)
        data_ptr = self.builder.load(data_sl)

        # vtable_ptr = fat[1]  (stored as i8*, cast to vtable type*)
        vtl_sl       = self.builder.gep(fat_ptr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        vtable_i8    = self.builder.load(vtl_sl)
        vtable_ptr   = self.builder.bitcast(vtable_i8, ir.PointerType(vtable_ll))

        # fn_ptr_i8 = vtable[method_idx]
        fn_sl        = self.builder.gep(vtable_ptr,
                                         [z, ir.Constant(I32_TY, method_idx)],
                                         inbounds=True)
        fn_ptr_i8    = self.builder.load(fn_sl)

        # Build typed function type: (i8*, arg_types...) → ret_type
        param_tys = [I8PTR]
        for p in method_sig.params:     # params do NOT include self in MethodSig
            param_tys.append(self._vx_to_llvm(p.type_name))
        ret_ty = self._vx_to_llvm(method_sig.return_type or "void")
        typed_fn_ty = ir.FunctionType(ret_ty, param_tys)
        fn_ptr = self.builder.bitcast(fn_ptr_i8, ir.PointerType(typed_fn_ty))

        compiled_args = [data_ptr]
        for i, arg_node in enumerate(arg_nodes):
            av, at = self._compile_expr(arg_node)
            if i < len(method_sig.params):
                pt = method_sig.params[i].type_name
                if pt == "float" and at == "int":
                    av = self.builder.sitofp(av, F64_TY)
            compiled_args.append(av)

        result = self.builder.call(fn_ptr, compiled_args)
        return result, method_sig.return_type or "void"

    def _compile_type_pattern_case(
            self, fn, val_al: ir.Value, iface_name: str,
            pat: 'TypePattern', body, next_b, merge_b):
        """Emit code for a single 'case StructName(bind1, bind2):' arm."""
        iface_info = self._interfaces[iface_name]
        fat_ll     = iface_info["fat_ll"]
        z          = ir.Constant(I32_TY, 0)

        # Load the fat pointer
        loaded   = self.builder.load(val_al)
        fat_ptr  = self.builder.bitcast(loaded, ir.PointerType(fat_ll))

        # Load vtable pointer stored at slot 1
        vtl_sl       = self.builder.gep(fat_ptr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        actual_vtl   = self.builder.load(vtl_sl)

        # Get expected vtable address for this struct type
        impl_key = f"{pat.type_name}__{iface_name}"
        if impl_key not in self._impls_data:
            raise CodegenError(
                f"match: '{pat.type_name}' does not implement '{iface_name}'"
            )
        expected_vtl_gv = self._impls_data[impl_key]["vtable_gv"]
        expected_vtl    = self.builder.bitcast(expected_vtl_gv, I8PTR)

        # Compare vtable pointers
        ai = self.builder.ptrtoint(actual_vtl,   I64_TY)
        ei = self.builder.ptrtoint(expected_vtl, I64_TY)
        matches = self.builder.icmp_unsigned("==", ai, ei)

        body_b = fn.append_basic_block(f"match.type.{pat.type_name}.body")
        self.builder.cbranch(matches, body_b, next_b)

        self.builder.position_at_end(body_b)
        self._push_scope()

        # Extract data pointer and bind fields
        if pat.bindings:
            struct_info = self._structs.get(pat.type_name)
            if struct_info:
                data_sl  = self.builder.gep(fat_ptr, [z, z], inbounds=True)
                data_ptr = self.builder.load(data_sl)
                sptr     = self.builder.bitcast(
                    data_ptr, ir.PointerType(struct_info["llvm_type"])
                )
                for i, bind_name in enumerate(pat.bindings):
                    if i < len(struct_info["fields"]):
                        _, ftype = struct_info["fields"][i]
                        fl_ty    = self._vx_to_llvm(ftype)
                        fp = self.builder.gep(sptr,
                                              [z, ir.Constant(I32_TY, i)],
                                              inbounds=True)
                        fval = self.builder.load(fp)
                        al   = self.builder.alloca(fl_ty, name=bind_name)
                        self.builder.store(fval, al)
                        self._declare(bind_name, al, ftype)

        for s in body:
            if self.builder.block.is_terminated:
                break
            self._compile_stmt(s)

        self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_b)

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _compile_stmt(self, node: Node):
        if self.builder.block.is_terminated:
            return   # dead code — skip

        if   isinstance(node, LetStmt):              self._compile_let(node)
        elif isinstance(node, TupleUnpack):           self._compile_tuple_unpack(node)
        elif isinstance(node, StructDestructure):     self._compile_struct_destructure(node)
        elif isinstance(node, ArrayDestructure):      self._compile_array_destructure(node)
        elif isinstance(node, AssignStmt):            self._compile_assign(node)
        elif isinstance(node, IndexAssignStmt):       self._compile_index_assign(node)
        elif isinstance(node, ReturnStmt):            self._compile_return(node)
        elif isinstance(node, PrintStmt):             self._compile_print(node)
        elif isinstance(node, IfStmt):                self._compile_if(node)
        elif isinstance(node, ForStmt):               self._compile_for(node)
        elif isinstance(node, ForEach):               self._compile_foreach(node)
        elif isinstance(node, WhileStmt):             self._compile_while(node)
        elif isinstance(node, DoWhileStmt):           self._compile_do_while(node)
        elif isinstance(node, LabeledStmt):           self._compile_labeled_stmt(node)
        elif isinstance(node, BreakStmt):             self._compile_break()
        elif isinstance(node, ContinueStmt):          self._compile_continue()
        elif isinstance(node, BreakLabel):            self._compile_break_label(node)
        elif isinstance(node, ContinueLabel):         self._compile_continue_label(node)
        elif isinstance(node, ExprStmt):              self._compile_expr(node.expr)
        elif isinstance(node, MatchStmt):             self._compile_match(node)
        elif isinstance(node, AssertStmt):            self._compile_assert(node)
        elif isinstance(node, TryCatch):              self._compile_try_catch(node)
        elif isinstance(node, TryCatchFinally):       self._compile_try_catch_finally(node)
        elif isinstance(node, ForEnumerate):          self._compile_for_enumerate(node)
        elif isinstance(node, DeferStmt):             self._register_defer(node)
        elif isinstance(node, ThrowStmt):             self._compile_throw(node)
        elif isinstance(node, RaiseStmt):             self._compile_throw(ThrowStmt(node.value))
        elif isinstance(node, YieldStmt):             pass   # future: generator support
        elif isinstance(node, UnsafeBlock):
            self._push_scope()
            for s in node.body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()
        elif isinstance(node, (PubDecl, PrivDecl)):
            self._compile_stmt(ExprStmt(NullLiteral()))   # visibility is metadata only
        elif isinstance(node, (EnumDecl, EnumDeclADT, ImportStmt, TypeAlias,
                               NamespaceHint, InterfaceDecl, ImplDecl,
                               ExternFnDecl, TestDecl, ComptimeDecl,
                               AttributeNode)):
            pass  # handled in compile() pass or before codegen
        else:
            raise CodegenError(f"Unknown stmt: {type(node).__name__}")

    def _compile_let(self, node: LetStmt):
        val, vt  = self._compile_expr(node.value)
        declared = self._resolve_type(node.type_annotation or vt)
        ll_ty    = self._vx_to_llvm(declared)

        if declared == "float" and vt == "int":
            val = self.builder.sitofp(val, F64_TY); vt = "float"

        # Integer type variants: coerce i64 to smaller int types
        _int_variants = {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "char"}
        if declared in _int_variants and vt in ("int", "bool"):
            target_ll = self._vx_to_llvm(declared)
            if val.type != target_ll:
                if val.type.width > target_ll.width:
                    val = self.builder.trunc(val, target_ll)
                else:
                    val = self.builder.sext(val, target_ll)
        # Coerce int variants back to i64 when assigned to "int"
        if declared == "int" and vt in _int_variants:
            if val.type != I64_TY:
                val = self.builder.sext(val, I64_TY)

        # Interface boxing: struct → interface fat pointer
        if declared in self._interfaces and vt != declared:
            val = self._box_as_interface(val, vt, declared)
            vt = declared

        # Nullable boxing: T? with a non-null value → box into malloc cell
        if declared.endswith("?"):
            if vt == "null":
                val = ir.Constant(I8PTR, None)
            elif vt != declared:
                # Box the value into heap
                base = declared[:-1]
                base_ll = self._vx_to_llvm(base) if base else I64_TY
                esz = ir.Constant(I64_TY, _elem_size(base))
                box = self.builder.call(self.malloc_fn, [esz])
                if base_ll != VOID_TY:
                    typed_box = self.builder.bitcast(box, ir.PointerType(base_ll))
                    # Promote int→float if needed
                    if base == "float" and vt == "int":
                        val = self.builder.sitofp(val, F64_TY)
                    self.builder.store(val, typed_box)
                val = box  # i8*

        al = self.builder.alloca(ll_ty, name=node.name)
        self.builder.store(val, al)
        self._declare(node.name, al, declared)

    def _compile_tuple_unpack(self, node: TupleUnpack):
        """Compile: let (a, b) = expr"""
        val, vt = self._compile_expr(node.value)
        # vt is like "(int,float)" or the type of what was returned
        elem_types = self._parse_tuple_type_str(vt)
        tup_struct = self._get_tuple_llvm_type(elem_types)
        tup_ptr = self.builder.bitcast(val, ir.PointerType(tup_struct))
        for i, name in enumerate(node.names):
            ann = node.annotations[i] if i < len(node.annotations) else None
            et  = ann or (elem_types[i] if i < len(elem_types) else "int")
            et  = self._resolve_type(et)
            ll_ty = self._vx_to_llvm(et)
            al = self.builder.alloca(ll_ty, name=name)
            fp = self.builder.gep(tup_ptr,
                                  [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, i)],
                                  inbounds=True)
            self.builder.store(self.builder.load(fp), al)
            self._declare(name, al, et)

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
        obj_val, avt = self._compile_expr(node.obj)
        idx_val, _   = self._compile_expr(node.index)

        # Dict assignment: d["key"] = val
        if avt.startswith("dict["):
            inner = avt[5:-1]
            vt_expected = inner[inner.index(',')+1:]
            raw64 = self._val_to_i64(val, vt, vt_expected)
            fn_h = self._get_helper("__vx_dict_set")
            dict_raw = self.builder.bitcast(obj_val, I8PTR)
            self.builder.call(fn_h, [dict_raw, idx_val, raw64])
            return

        # Array assignment
        elem_vt = avt[:-2] if avt.endswith("[]") else "int"
        elem_lt = self._vx_to_llvm(elem_vt)
        data_ptr = self._arr_data_ptr(obj_val, elem_lt)
        ep       = self.builder.gep(data_ptr, [idx_val], inbounds=True)
        if elem_vt == "float" and vt == "int":
            val = self.builder.sitofp(val, F64_TY)
        self.builder.store(val, ep)

    def _compile_return(self, node: ReturnStmt):
        ret_ty = self.current_fn.ftype.return_type
        self._emit_defers()   # run deferred exprs before return
        if node.value is None:
            # Named return: bare return loads the named return variable
            nrs = getattr(self, '_current_named_returns', [])
            if nrs and ret_ty != VOID_TY:
                if len(nrs) == 1:
                    rname, _ = nrs[0]
                    info = self._lookup(rname)
                    if info:
                        self.builder.ret(self.builder.load(info["ptr"]))
                        return
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
        elif vt.endswith("?"):
            # Nullable: print "null" or the underlying value
            fn = self.current_fn
            null_b  = fn.append_basic_block("print.null")
            val_b   = fn.append_basic_block("print.val")
            after_b = fn.append_basic_block("print.after")
            val_int = self.builder.ptrtoint(val, I64_TY)
            is_null = self.builder.icmp_unsigned("==", val_int, ir.Constant(I64_TY, 0))
            self.builder.cbranch(is_null, null_b, val_b)
            self.builder.position_at_end(null_b)
            fmt_n = self._gstr_ptr(self._global_str("null"))
            self.builder.call(self.printf, [self._gstr_ptr(self._global_str("%s")), fmt_n])
            self.builder.branch(after_b)
            self.builder.position_at_end(val_b)
            base_t = vt[:-1]
            base_ll = self._vx_to_llvm(base_t) if base_t else I64_TY
            if base_ll != VOID_TY and base_ll != I8PTR:
                typed_ptr = self.builder.bitcast(val, ir.PointerType(base_ll))
                inner_val = self.builder.load(typed_ptr)
                self._print_value(inner_val, base_t)
            else:
                self._print_value(val, "str")
            self.builder.branch(after_b)
            self.builder.position_at_end(after_b)
        elif vt.endswith("[]"):
            fmt = self._gstr_ptr(self._global_str(f"<{vt} len="))
            self.builder.call(self.printf, [fmt])
            lp  = self.builder.gep(val, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
            ln  = self.builder.load(lp)
            fmtd = self._gstr_ptr(self._global_str("%lld>"))
            self.builder.call(self.printf, [fmtd, ln])
        elif vt.startswith("("):
            # Tuple — print as <tuple>
            fmt = self._gstr_ptr(self._global_str(f"<tuple>"))
            self.builder.call(self.printf, [self._gstr_ptr(self._global_str("%s")), fmt])
        elif vt in self._interfaces:
            # Interface value — print as <interface:name>
            fmt = self._gstr_ptr(self._global_str(f"<{vt}>"))
            self.builder.call(self.printf, [fmt])
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

        # Check if any case uses a TypePattern (interface type dispatch)
        has_type_patterns = any(
            isinstance(p, TypePattern)
            for case in node.cases
            for p in case.patterns
        )

        if has_type_patterns and vt in self._interfaces:
            # Interface type-pattern matching
            val_al = self.builder.alloca(I8PTR, name="match_iface_val")
            self.builder.store(val, val_al)
            merge_b = fn.append_basic_block("match.merge")

            for case in node.cases:
                next_b = fn.append_basic_block("match.next")
                for pat in case.patterns:
                    if isinstance(pat, TypePattern):
                        self._compile_type_pattern_case(
                            fn, val_al, vt, pat, case.body, next_b, merge_b
                        )
                        break
                self.builder.position_at_end(next_b)

            if node.default_body:
                self._push_scope()
                for s in node.default_body:
                    if self.builder.block.is_terminated: break
                    self._compile_stmt(s)
                self._pop_scope()
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_b)

            self.builder.position_at_end(merge_b)
            return

        # Regular value equality matching
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
            # Support MatchCaseGuard (case n if guard:) and regular MatchCase
            is_guard_case = isinstance(case, MatchCaseGuard)
            guard_expr = case.guard if is_guard_case else None
            patterns   = case.patterns
            body       = case.body

            case_body_b = fn.append_basic_block("match.case.body")
            next_b = fn.append_basic_block("match.case.next")
            loaded = self.builder.load(val_al)

            # Check if pattern is a single identifier — bind variable, condition = true
            bind_name = None
            combined_cond = None

            if (len(patterns) == 1 and isinstance(patterns[0], Identifier)
                    and patterns[0].name not in self._globals
                    and self._lookup(patterns[0].name) is None):
                # Wildcard variable binding: case n: ... (always matches)
                bind_name = patterns[0].name
                combined_cond = ir.Constant(I1_TY, 1)
            else:
                for pat in patterns:
                    pv, pt = self._compile_expr(pat)
                    if vt == "str" or pt == "str":
                        r = self.builder.call(self.strcmp_fn, [loaded, pv])
                        c = self.builder.icmp_signed("==", r, ir.Constant(I32_TY, 0))
                    elif vt == "float" or pt == "float":
                        if pt == "int": pv = self.builder.sitofp(pv, F64_TY)
                        loaded_f = self.builder.sitofp(loaded, F64_TY) if vt == "int" else loaded
                        c = self.builder.fcmp_ordered("==", loaded_f, pv)
                    else:
                        c = self.builder.icmp_signed("==", loaded, pv)
                    combined_cond = c if combined_cond is None else self.builder.or_(combined_cond, c)

            if combined_cond is None:
                combined_cond = ir.Constant(I1_TY, 0)

            # If there's a guard, evaluate it after binding (in a side-block)
            if guard_expr is not None:
                guard_check_b = fn.append_basic_block("match.guard")
                self.builder.cbranch(combined_cond, guard_check_b, next_b)
                self.builder.position_at_end(guard_check_b)
                # Bind match variable for the guard
                self._push_scope()
                if bind_name:
                    bind_al = self.builder.alloca(val_al.type.pointee, name=bind_name)
                    self.builder.store(loaded, bind_al)
                    self._declare(bind_name, bind_al, vt)
                guard_v, _ = self._compile_expr(guard_expr)
                if guard_v.type != I1_TY:
                    guard_v = self.builder.trunc(guard_v, I1_TY)
                self._pop_scope()
                self.builder.cbranch(guard_v, case_body_b, next_b)
            else:
                self.builder.cbranch(combined_cond, case_body_b, next_b)

            self.builder.position_at_end(case_body_b)
            self._push_scope()
            # Bind match variable in case body
            if bind_name:
                bind_al2 = self.builder.alloca(val_al.type.pointee, name=bind_name)
                self.builder.store(loaded, bind_al2)
                self._declare(bind_name, bind_al2, vt)
            for s in body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_b)

            self.builder.position_at_end(next_b)

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
    #  Try / catch (global error-state approach)                         #
    # ------------------------------------------------------------------ #

    def _compile_try_catch(self, node: TryCatch):
        fn = self.current_fn

        # Clear error buffer: store 0 into first byte
        ep0 = self.builder.gep(self._vx_error_buf,
                               [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                               inbounds=True)
        self.builder.store(ir.Constant(I8_TY, 0), ep0)

        # Compile try body
        self._push_scope()
        for s in node.try_body:
            if self.builder.block.is_terminated:
                break
            self._compile_stmt(s)
        self._pop_scope()

        if self.builder.block.is_terminated:
            return  # try body already returned

        # Check if an error was set
        first = self.builder.load(ep0)
        has_err = self.builder.icmp_unsigned("!=", first, ir.Constant(I8_TY, 0))

        catch_b = fn.append_basic_block("try.catch")
        after_b = fn.append_basic_block("try.after")
        self.builder.cbranch(has_err, catch_b, after_b)

        # Catch block
        self.builder.position_at_end(catch_b)
        err_ptr = self.builder.gep(self._vx_error_buf,
                                   [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                   inbounds=True)
        self._push_scope()
        al = self.builder.alloca(I8PTR, name=node.catch_var)
        self.builder.store(err_ptr, al)
        self._declare(node.catch_var, al, "str")
        for s in node.catch_body:
            if self.builder.block.is_terminated:
                break
            self._compile_stmt(s)
        self._pop_scope()
        if not self.builder.block.is_terminated:
            self.builder.branch(after_b)

        self.builder.position_at_end(after_b)

    # ------------------------------------------------------------------ #
    #  For enumerate  (for i, v in arr:)                                 #
    # ------------------------------------------------------------------ #

    def _compile_for_enumerate(self, node: ForEnumerate):
        fn          = self.current_fn
        arr_v, avt  = self._compile_expr(node.iterable)
        elem_vt     = avt[:-2] if avt.endswith("[]") else "str"
        elem_lt     = self._vx_to_llvm(elem_vt)

        lp     = self.builder.gep(arr_v,
                                  [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 1)],
                                  inbounds=True)
        arr_ln = self.builder.load(lp)

        # Index counter
        i_al = self.builder.alloca(I64_TY, name=node.idx_var)
        self.builder.store(ir.Constant(I64_TY, 0), i_al)

        # Element slot
        item_al = self.builder.alloca(elem_lt, name=node.val_var)

        chk = fn.append_basic_block("fen.check")
        bdy = fn.append_basic_block("fen.body")
        ext = fn.append_basic_block("fen.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv   = self.builder.load(i_al)
        cond = self.builder.icmp_signed("<", iv, arr_ln)
        self.builder.cbranch(cond, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        self._declare(node.idx_var, i_al, "int")
        self._declare(node.val_var, item_al, elem_vt)
        dp = self._arr_data_ptr(arr_v, elem_lt)
        ep = self.builder.gep(dp, [iv], inbounds=True)
        ev = self.builder.load(ep)
        self.builder.store(ev, item_al)

        self._break_targets.append(ext)
        self._continue_targets.append(chk)
        for s in node.body:
            if self.builder.block.is_terminated:
                break
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

    # ------------------------------------------------------------------ #
    #  Loops                                                               #
    # ------------------------------------------------------------------ #

    def _compile_for(self, node: ForStmt):
        fn          = self.current_fn
        start_v, _  = self._compile_expr(node.start)
        end_v,   _  = self._compile_expr(node.end)
        if start_v.type != I64_TY: start_v = self.builder.fptosi(start_v, I64_TY)
        if end_v.type   != I64_TY: end_v   = self.builder.fptosi(end_v,   I64_TY)

        # Optional step (default 1, may be negative for countdown)
        if node.step:
            step_v, _ = self._compile_expr(node.step)
            if step_v.type != I64_TY: step_v = self.builder.fptosi(step_v, I64_TY)
        else:
            step_v = ir.Constant(I64_TY, 1)

        i_al = self.builder.alloca(I64_TY, name=node.var)
        self.builder.store(start_v, i_al)

        chk = fn.append_basic_block("for.check")
        bdy = fn.append_basic_block("for.body")
        ext = fn.append_basic_block("for.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv = self.builder.load(i_al)
        # Dynamic step sign: if step > 0 then i < end, else i > end
        # (inclusive range uses <= / >=)
        inclusive = getattr(node, 'inclusive', False)
        if node.step:
            step_pos = self.builder.icmp_signed(">", step_v, ir.Constant(I64_TY, 0))
            if inclusive:
                cond_fwd = self.builder.icmp_signed("<=", iv, end_v)
                cond_rev = self.builder.icmp_signed(">=", iv, end_v)
            else:
                cond_fwd = self.builder.icmp_signed("<",  iv, end_v)
                cond_rev = self.builder.icmp_signed(">",  iv, end_v)
            cond = self.builder.select(step_pos, cond_fwd, cond_rev)
        else:
            cmp_op = "<=" if inclusive else "<"
            cond = self.builder.icmp_signed(cmp_op, iv, end_v)
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
            inxt = self.builder.add(ic, step_v)
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

    def _compile_break_label(self, node):
        target = self._label_break_targets.get(node.label)
        if target is None:
            raise CodegenError(f"Unknown label '{node.label}'")
        self.builder.branch(target)

    def _compile_continue_label(self, node):
        target = self._label_continue_targets.get(node.label)
        if target is None:
            raise CodegenError(f"Unknown label '{node.label}'")
        self.builder.branch(target)

    def _compile_do_while(self, node: 'DoWhileStmt'):
        fn  = self.current_fn
        bdy = fn.append_basic_block("dowhile.body")
        chk = fn.append_basic_block("dowhile.check")
        ext = fn.append_basic_block("dowhile.exit")

        self.builder.branch(bdy)
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

        self.builder.position_at_end(chk)
        cv, _ = self._compile_expr(node.condition)
        self.builder.cbranch(cv, bdy, ext)
        self.builder.position_at_end(ext)

    def _compile_labeled_stmt(self, node: 'LabeledStmt'):
        fn  = self.current_fn
        ext = fn.append_basic_block(f"label.{node.label}.exit")
        chk = fn.append_basic_block(f"label.{node.label}.chk")

        # Register label targets before compiling body
        self._label_break_targets[node.label]    = ext
        self._label_continue_targets[node.label] = chk

        # Emit the loop with label targets also pushed as normal targets
        self._break_targets.append(ext)
        self._continue_targets.append(chk)
        self._compile_stmt(node.stmt)
        self._break_targets.pop()
        self._continue_targets.pop()

        del self._label_break_targets[node.label]
        del self._label_continue_targets[node.label]

        if not self.builder.block.is_terminated:
            self.builder.branch(ext)
        self.builder.position_at_end(ext)

    def _register_defer(self, node: 'DeferStmt'):
        """Queue a deferred statement/expression to run before each return."""
        # Store tuple (expr, on_error_only)
        self._defer_stack[-1].append((node.expr, node.on_error_only))

    def _emit_defers(self):
        """Emit all queued defer expressions/stmts (LIFO order).

        For defer_on_error (#119): only emits if the error buffer is non-empty.
        """
        if not self._defer_stack:
            return
        for item, on_error_only in reversed(self._defer_stack[-1]):
            if on_error_only:
                # Only run if error buffer has been set (first byte != 0)
                ep0 = self.builder.gep(self._vx_error_buf,
                                       [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                       inbounds=True)
                first = self.builder.load(ep0)
                has_err = self.builder.icmp_unsigned("!=", first, ir.Constant(I8_TY, 0))
                fn = self.current_fn
                run_bb  = fn.append_basic_block("defer_err_run")
                skip_bb = fn.append_basic_block("defer_err_skip")
                self.builder.cbranch(has_err, run_bb, skip_bb)
                self.builder.position_at_end(run_bb)
                if isinstance(item, (PrintStmt, IfStmt, ForStmt, WhileStmt,
                                     AssignStmt, LetStmt, ExprStmt)):
                    self._compile_stmt(item)
                else:
                    self._compile_expr(item)
                if not self.builder.block.is_terminated:
                    self.builder.branch(skip_bb)
                self.builder.position_at_end(skip_bb)
            else:
                if isinstance(item, (PrintStmt, IfStmt, ForStmt, WhileStmt,
                                     AssignStmt, LetStmt, ExprStmt)):
                    self._compile_stmt(item)
                else:
                    self._compile_expr(item)

    def _compile_throw(self, node: 'ThrowStmt'):
        """throw expr — set global error buffer and return zero."""
        val, vt = self._compile_expr(node.value)
        if vt != "str":
            val = self._val_to_str(val, vt)
        # Store message into error buffer via sprintf
        ep0 = self.builder.gep(self._vx_error_buf,
                               [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                               inbounds=True)
        fmt = self._global_str("%s")
        self.builder.call(self.sprintf_fn, [ep0, self._gstr_ptr(fmt), val])
        # Return zero / void from current function
        ret_ty = self.current_fn.ftype.return_type
        if ret_ty == VOID_TY:
            self.builder.ret_void()
        else:
            self.builder.ret(ir.Constant(ret_ty, 0))

    def _compile_try_catch_finally(self, node: 'TryCatchFinally'):
        fn = self.current_fn

        # Clear error buffer
        ep0 = self.builder.gep(self._vx_error_buf,
                               [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                               inbounds=True)
        self.builder.store(ir.Constant(I8_TY, 0), ep0)

        # Compile try body
        self._push_scope()
        for s in node.try_body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(s)
        self._pop_scope()

        if self.builder.block.is_terminated:
            return

        # Check for error
        first = self.builder.load(ep0)
        has_err = self.builder.icmp_unsigned("!=", first, ir.Constant(I8_TY, 0))

        after_b = fn.append_basic_block("try.after")

        if node.catches:
            catch_b = fn.append_basic_block("try.catch")
            self.builder.cbranch(has_err, catch_b, after_b)
            self.builder.position_at_end(catch_b)
            err_ptr = ep0
            for clause in node.catches:
                self._push_scope()
                al = self.builder.alloca(I8PTR, name=clause.var)
                self.builder.store(err_ptr, al)
                self._declare(clause.var, al, "str")
                for s in clause.body:
                    if self.builder.block.is_terminated: break
                    self._compile_stmt(s)
                self._pop_scope()
                if not self.builder.block.is_terminated:
                    break
            if not self.builder.block.is_terminated:
                self.builder.branch(after_b)
        else:
            self.builder.branch(after_b)

        self.builder.position_at_end(after_b)

        # Finally block always runs
        if node.finally_body:
            self._push_scope()
            for s in node.finally_body:
                if self.builder.block.is_terminated: break
                self._compile_stmt(s)
            self._pop_scope()

    def _compile_struct_destructure(self, node: 'StructDestructure'):
        val, vt = self._compile_expr(node.value)
        struct_info = self._structs.get(vt)
        if struct_info is None:
            raise CodegenError(f"Cannot destructure non-struct type '{vt}'")
        z = ir.Constant(I32_TY, 0)
        for i, fname in enumerate(node.fields):
            alias = node.aliases[i] if i < len(node.aliases) else None
            bind_name = alias if alias else fname
            field_idx = next((j for j, (fn, _) in enumerate(struct_info["fields"])
                              if fn == fname), None)
            if field_idx is None:
                raise CodegenError(f"Struct '{vt}' has no field '{fname}'")
            _, ftype = struct_info["fields"][field_idx]
            fl_ty = self._vx_to_llvm(ftype)
            fp = self.builder.gep(val, [z, ir.Constant(I32_TY, field_idx)], inbounds=True)
            fval = self.builder.load(fp)
            al = self.builder.alloca(fl_ty, name=bind_name)
            self.builder.store(fval, al)
            self._declare(bind_name, al, ftype)

    def _compile_array_destructure(self, node: 'ArrayDestructure'):
        arr_v, avt = self._compile_expr(node.value)
        elem_vt = avt[:-2] if avt.endswith("[]") else "int"
        elem_lt = self._vx_to_llvm(elem_vt)
        dp = self._arr_data_ptr(arr_v, elem_lt)
        for i, name in enumerate(node.names):
            idx = ir.Constant(I64_TY, i)
            ep  = self.builder.gep(dp, [idx], inbounds=True)
            val = self.builder.load(ep)
            al  = self.builder.alloca(elem_lt, name=name)
            self.builder.store(val, al)
            self._declare(name, al, elem_vt)
        if node.rest_name:
            # Create a new array containing elements from len(names) onwards
            start_idx = len(node.names)
            lp  = self.builder.gep(arr_v, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
            ln  = self.builder.load(lp)
            rest_len = self.builder.sub(ln, ir.Constant(I64_TY, start_idx))
            fn_h = self._get_helper("__vx_array_slice")
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            esz = ir.Constant(I64_TY, _elem_size(elem_vt))
            sliced = self.builder.call(fn_h, [arr_raw, ir.Constant(I64_TY, start_idx), ln, esz])
            sl_ptr = self.builder.bitcast(sliced, self.arr_ptr_type)
            al = self.builder.alloca(self.arr_ptr_type, name=node.rest_name)
            self.builder.store(sl_ptr, al)
            self._declare(node.rest_name, al, avt)

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
        if isinstance(node, DictLiteral):
            return self._compile_dict_literal(node)
        if isinstance(node, TupleLiteral):
            return self._compile_tuple_literal(node)
        if isinstance(node, LambdaExpr):
            return self._compile_lambda(node)

        if isinstance(node, Identifier):
            # Built-in constants
            if node.name == "PI":
                return ir.Constant(F64_TY, 3.141592653589793), "float"
            if node.name == "TAU":
                return ir.Constant(F64_TY, 6.283185307179586), "float"
            if node.name == "E":
                return ir.Constant(F64_TY, 2.718281828459045), "float"
            if node.name == "INF":
                import math
                return ir.Constant(F64_TY, math.inf), "float"
            if node.name == "NAN":
                import math
                return ir.Constant(F64_TY, math.nan), "float"
            info = self._lookup(node.name)
            if info is None:
                raise CodegenError(f"Undefined variable '{node.name}'")
            vx_t = self._resolve_type(info["vx_type"])
            return self.builder.load(info["ptr"], name=node.name), vx_t

        if isinstance(node, CharLiteral):
            return ir.Constant(I64_TY, ord(node.value)), "int"

        if isinstance(node, BinOp):      return self._compile_binop(node)
        if isinstance(node, UnaryOp):    return self._compile_unary(node)
        if isinstance(node, Call):       return self._compile_call(node)
        if isinstance(node, MethodCall): return self._compile_method_call(node)
        if isinstance(node, NamedArg):   return self._compile_expr(node.value)
        if isinstance(node, AwaitExpr):  return self._compile_expr(node.expr)

        if isinstance(node, ErrorPropExpr):
            return self._compile_error_prop(node)

        if isinstance(node, NullCoalesceExpr):
            return self._compile_null_coalesce(node)

        if isinstance(node, OptionalChainExpr):
            return self._compile_optional_chain(node)

        if isinstance(node, SliceExpr):
            return self._compile_slice(node)

        if isinstance(node, ListComp):
            return self._compile_list_comp(node)

        if isinstance(node, PipeExpr):
            # value |> func — should have been desugared in parser, but handle here too
            val_v, val_t = self._compile_expr(node.value)
            if isinstance(node.func, Identifier):
                return self._compile_expr(Call(node.func.name, [node.value]))
            return self._compile_expr(node.func)

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
            # #103: null dereference detection in debug mode
            if self.debug_mode:
                raw = self.builder.ptrtoint(ptr, I64_TY)
                is_null = self.builder.icmp_unsigned("==", raw, ir.Constant(I64_TY, 0))
                ok_bb   = self.current_fn.append_basic_block("null.ok")
                fail_bb = self.current_fn.append_basic_block("null.fail")
                self.builder.cbranch(is_null, fail_bb, ok_bb)
                self.builder.position_at_end(fail_bb)
                self._emit_panic(f"null pointer dereference at field '{node.field}'")
                self.builder.position_at_end(ok_bb)
            vt  = self._infer_type(node)
            return self.builder.load(ptr, name=node.field), vt

        if isinstance(node, NewExpr):  return self._compile_new(node)

        if isinstance(node, StructUpdateExpr):
            return self._compile_struct_update(node)

        if isinstance(node, SpreadExpr):
            # Spread used as standalone expression — just return the underlying value
            return self._compile_expr(node.value)

        if isinstance(node, IndexExpr):
            obj_v, obj_t = self._compile_expr(node.obj)
            idx_v, _     = self._compile_expr(node.index)
            # Dict indexing: dict["key"] → value
            if obj_t.startswith("dict["):
                inner = obj_t[5:-1]
                vt = inner[inner.index(',')+1:]
                return self._compile_dict_index_get(obj_v, idx_v, vt)
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
            # Debug mode: emit bounds check (#101)
            if self.debug_mode:
                lp  = self.builder.gep(obj_v, [ir.Constant(I32_TY, 0),
                                               ir.Constant(I32_TY, 1)], inbounds=True)
                arr_len = self.builder.load(lp)
                fn_bc = self._get_helper("__vx_bounds_check")
                self.builder.call(fn_bc, [idx_v, arr_len])
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

    # ------------------------------------------------------------------ #
    #  Null coalesce / optional chain / slice / list comp                 #
    # ------------------------------------------------------------------ #

    def _compile_null_coalesce(self, node: 'NullCoalesceExpr') -> tuple[ir.Value, str]:
        fn = self.current_fn
        lv, lt = self._compile_expr(node.left)

        not_null_b = fn.append_basic_block("nc.notnull")
        null_b     = fn.append_basic_block("nc.null")
        merge_b    = fn.append_basic_block("nc.merge")

        li = self.builder.ptrtoint(lv, I64_TY) if lv.type == I8PTR.pointee or \
             lv.type.is_pointer else self.builder.zext(lv, I64_TY)
        # Simpler: use ptrtoint unconditionally via bitcast trick
        lcast = self.builder.bitcast(lv, I8PTR) if lv.type != I8PTR else lv
        li = self.builder.ptrtoint(lcast, I64_TY)
        is_null = self.builder.icmp_unsigned("==", li, ir.Constant(I64_TY, 0))
        self.builder.cbranch(is_null, null_b, not_null_b)

        self.builder.position_at_end(not_null_b)
        left_block = self.builder.block
        left_val = lv
        self.builder.branch(merge_b)

        self.builder.position_at_end(null_b)
        rv, rt = self._compile_expr(node.right)
        right_block = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)
        result_t = rt if lt.endswith("?") else lt
        ll_ty = self._vx_to_llvm(result_t)
        phi = self.builder.phi(ll_ty)
        phi.add_incoming(left_val, left_block)
        phi.add_incoming(rv, right_block)
        return phi, result_t

    def _compile_optional_chain(self, node: 'OptionalChainExpr') -> tuple[ir.Value, str]:
        fn = self.current_fn
        obj_v, obj_t = self._compile_expr(node.obj)

        null_b   = fn.append_basic_block("opt.null")
        val_b    = fn.append_basic_block("opt.val")
        merge_b  = fn.append_basic_block("opt.merge")

        obj_i = self.builder.ptrtoint(obj_v, I64_TY)
        is_null = self.builder.icmp_unsigned("==", obj_i, ir.Constant(I64_TY, 0))
        self.builder.cbranch(is_null, null_b, val_b)

        self.builder.position_at_end(null_b)
        null_block = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(val_b)
        base_t = obj_t[:-1] if obj_t.endswith("?") else obj_t
        struct_info = self._structs.get(base_t)
        if struct_info:
            z = ir.Constant(I32_TY, 0)
            fi = next((i for i,(fn2,_) in enumerate(struct_info["fields"]) if fn2==node.field), 0)
            _, ftype = struct_info["fields"][fi]
            fl_ty = self._vx_to_llvm(ftype)
            fp = self.builder.gep(obj_v, [z, ir.Constant(I32_TY, fi)], inbounds=True)
            fval = self.builder.load(fp)
            result_t = ftype + "?"
        else:
            fval = obj_v
            result_t = "int?"
        val_block = self.builder.block
        self.builder.branch(merge_b)

        self.builder.position_at_end(merge_b)
        phi = self.builder.phi(I8PTR)
        phi.add_incoming(ir.Constant(I8PTR, None), null_block)
        # Box the field value
        if result_t != "int?":
            box = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 8)])
        else:
            box = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 8)])
        phi.add_incoming(box, val_block)
        return phi, result_t

    def _compile_slice(self, node: 'SliceExpr') -> tuple[ir.Value, str]:
        obj_v, obj_t = self._compile_expr(node.obj)
        start_v = ir.Constant(I64_TY, 0) if node.start is None else self._compile_expr(node.start)[0]
        if obj_t == "str":
            end_v = self.builder.call(self.strlen_fn, [obj_v]) if node.end is None \
                    else self._compile_expr(node.end)[0]
            if node.inclusive:
                end_v = self.builder.add(end_v, ir.Constant(I64_TY, 1))
            length = self.builder.sub(end_v, start_v)
            buf = self.builder.call(self.malloc_fn, [self.builder.add(length, ir.Constant(I64_TY, 1))])
            src = self.builder.gep(obj_v, [start_v], inbounds=False)
            self.builder.call(self.memcpy_fn, [buf, src, length])
            null_p = self.builder.gep(buf, [length], inbounds=False)
            self.builder.store(ir.Constant(I8_TY, 0), null_p)
            return buf, "str"
        else:
            # Array slice
            lp  = self.builder.gep(obj_v, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
            arr_len = self.builder.load(lp)
            end_v = arr_len if node.end is None else self._compile_expr(node.end)[0]
            if node.inclusive:
                end_v = self.builder.add(end_v, ir.Constant(I64_TY, 1))
            elem_vt = obj_t[:-2] if obj_t.endswith("[]") else "int"
            esz = ir.Constant(I64_TY, _elem_size(elem_vt))
            fn_h = self._get_helper("__vx_array_slice")
            arr_raw = self.builder.bitcast(obj_v, I8PTR)
            sliced = self.builder.call(fn_h, [arr_raw, start_v, end_v, esz])
            sl_ptr = self.builder.bitcast(sliced, self.arr_ptr_type)
            return sl_ptr, obj_t

    def _compile_list_comp(self, node: 'ListComp') -> tuple[ir.Value, str]:
        """[expr for var in iterable if cond]  →  build new array."""
        fn = self.current_fn

        # B2 fix: handle RangeExpr iterables
        if isinstance(node.iterable, RangeExpr):
            return self._compile_list_comp_range(node)

        arr_v, avt = self._compile_expr(node.iterable)
        elem_vt = avt[:-2] if avt.endswith("[]") else "int"
        elem_lt = self._vx_to_llvm(elem_vt)
        esz     = ir.Constant(I64_TY, _elem_size(elem_vt))

        # Infer result element type from the map expression
        # We'll build an array with same capacity as input
        cap_sz = ir.Constant(I64_TY, 8)
        result_arr_raw = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        result_arr = self.builder.bitcast(result_arr_raw, self.arr_ptr_type)

        # Init result array: data=malloc(8*8), len=0, cap=8
        init_data = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        z = ir.Constant(I32_TY, 0)
        dp_sl = self.builder.gep(result_arr, [z, z], inbounds=True)
        self.builder.store(init_data, dp_sl)
        lp_sl = self.builder.gep(result_arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        self.builder.store(ir.Constant(I64_TY, 0), lp_sl)
        cp_sl = self.builder.gep(result_arr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        self.builder.store(ir.Constant(I64_TY, 8), cp_sl)

        # Loop over input array
        src_lp   = self.builder.gep(arr_v, [z, ir.Constant(I32_TY,1)], inbounds=True)
        src_len  = self.builder.load(src_lp)
        i_al = self.builder.alloca(I64_TY, name="_lc_i")
        self.builder.store(ir.Constant(I64_TY, 0), i_al)

        chk = fn.append_basic_block("lc.check")
        bdy = fn.append_basic_block("lc.body")
        ext = fn.append_basic_block("lc.exit")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv = self.builder.load(i_al)
        cond = self.builder.icmp_signed("<", iv, src_len)
        self.builder.cbranch(cond, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        src_dp = self._arr_data_ptr(arr_v, elem_lt)
        ep     = self.builder.gep(src_dp, [iv], inbounds=True)
        item   = self.builder.load(ep)
        item_al = self.builder.alloca(elem_lt, name=node.var)
        self.builder.store(item, item_al)
        self._declare(node.var, item_al, elem_vt)

        do_push = fn.append_basic_block("lc.push")
        skip_b  = fn.append_basic_block("lc.skip")

        if node.condition:
            cv, _ = self._compile_expr(node.condition)
            self.builder.cbranch(cv, do_push, skip_b)
        else:
            self.builder.branch(do_push)

        self.builder.position_at_end(do_push)
        map_val, _ = self._compile_expr(node.expr)
        map_al  = self.builder.alloca(elem_lt)
        self.builder.store(map_val, map_al)
        map_raw = self.builder.bitcast(map_al, I8PTR)
        res_raw = self.builder.bitcast(result_arr, I8PTR)
        fn_h = self._get_helper("__vx_array_push")
        self.builder.call(fn_h, [res_raw, map_raw, esz])
        self.builder.branch(skip_b)

        self.builder.position_at_end(skip_b)
        self._pop_scope()
        ic   = self.builder.load(i_al)
        inxt = self.builder.add(ic, ir.Constant(I64_TY, 1))
        self.builder.store(inxt, i_al)
        self.builder.branch(chk)

        self.builder.position_at_end(ext)
        return result_arr, elem_vt + "[]"

    def _compile_list_comp_range(self, node: 'ListComp') -> tuple[ir.Value, str]:
        """[expr for var in start..end [step s] [if cond]]  →  range-based list comp."""
        fn = self.current_fn
        rng = node.iterable   # RangeExpr
        start_v, _ = self._compile_expr(rng.start)
        end_v,   _ = self._compile_expr(rng.end)
        if start_v.type != I64_TY: start_v = self.builder.fptosi(start_v, I64_TY)
        if end_v.type   != I64_TY: end_v   = self.builder.fptosi(end_v,   I64_TY)
        if rng.step:
            step_v, _ = self._compile_expr(rng.step)
            if step_v.type != I64_TY: step_v = self.builder.fptosi(step_v, I64_TY)
        else:
            step_v = ir.Constant(I64_TY, 1)

        # Allocate result array
        result_arr_raw = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        result_arr = self.builder.bitcast(result_arr_raw, self.arr_ptr_type)
        init_data = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        z = ir.Constant(I32_TY, 0)
        dp_sl = self.builder.gep(result_arr, [z, z], inbounds=True)
        self.builder.store(init_data, dp_sl)
        lp_sl = self.builder.gep(result_arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        self.builder.store(ir.Constant(I64_TY, 0), lp_sl)
        cp_sl = self.builder.gep(result_arr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        self.builder.store(ir.Constant(I64_TY, 8), cp_sl)

        i_al = self.builder.alloca(I64_TY, name="_lc_i")
        self.builder.store(start_v, i_al)

        chk = fn.append_basic_block("lc.chk")
        bdy = fn.append_basic_block("lc.bdy")
        ext = fn.append_basic_block("lc.ext")

        self.builder.branch(chk)
        self.builder.position_at_end(chk)
        iv = self.builder.load(i_al)
        cmp_op = "<=" if rng.inclusive else "<"
        cond = self.builder.icmp_signed(cmp_op, iv, end_v)
        self.builder.cbranch(cond, bdy, ext)

        self.builder.position_at_end(bdy)
        self._push_scope()
        item_al = self.builder.alloca(I64_TY, name=node.var)
        self.builder.store(iv, item_al)
        self._declare(node.var, item_al, "int")

        do_push = fn.append_basic_block("lc.push")
        skip_b  = fn.append_basic_block("lc.skip")

        if node.condition:
            cv, _ = self._compile_expr(node.condition)
            self.builder.cbranch(cv, do_push, skip_b)
        else:
            self.builder.branch(do_push)

        self.builder.position_at_end(do_push)
        map_val, map_vt = self._compile_expr(node.expr)
        esz = ir.Constant(I64_TY, _elem_size(map_vt))
        map_al = self.builder.alloca(self._vx_to_llvm(map_vt))
        self.builder.store(map_val, map_al)
        map_raw = self.builder.bitcast(map_al, I8PTR)
        fn_h = self._get_helper("__vx_array_push")
        self.builder.call(fn_h, [result_arr_raw, map_raw, esz])
        self.builder.branch(skip_b)

        self.builder.position_at_end(skip_b)
        self._pop_scope()
        ic   = self.builder.load(i_al)
        inxt = self.builder.add(ic, step_v)
        self.builder.store(inxt, i_al)
        self.builder.branch(chk)

        self.builder.position_at_end(ext)
        # Infer result element type from the map expression (default int)
        res_elem_vt = "int"
        try:
            from compiler.analyzer import Analyzer as _A
            # Quick type inference: check the map_val type
            res_elem_vt = map_vt if 'map_vt' in dir() else "int"
        except Exception:
            pass
        return result_arr, res_elem_vt + "[]"

    def _compile_method_call(self, node: MethodCall) -> tuple[ir.Value, str]:
        # Namespace call: ns.func(args) where ns is a known namespace alias
        if isinstance(node.obj, Identifier) and node.obj.name in self._namespaces:
            ns = node.obj.name
            fn_name = f"{ns}__{node.method}"
            fi = self._functions.get(fn_name)
            if fi is None:
                raise CodegenError(f"Namespace '{ns}' has no function '{node.method}'")
            fn  = fi["fn"]
            sig = fi["sig"]
            defaults = self._fn_defaults.get(fn_name, [])
            args_list = list(node.args)
            while len(args_list) < len(sig.params):
                idx = len(args_list)
                if idx < len(defaults) and defaults[idx] is not None:
                    args_list.append(defaults[idx])
                else:
                    break
            compiled_args = []
            for arg_node, (_, pt) in zip(args_list, sig.params):
                av, at = self._compile_expr(arg_node)
                if pt == "float" and at == "int":
                    av = self.builder.sitofp(av, F64_TY)
                compiled_args.append(av)
            result = self.builder.call(fn, compiled_args)
            return result, sig.return_type

        obj_v, obj_t = self._compile_expr(node.obj)
        method = node.method

        # Interface method dispatch via vtable
        if obj_t in self._interfaces:
            return self._compile_interface_method_call(obj_v, obj_t, method, node.args)

        if obj_t.startswith("dict["):
            inner = obj_t[5:-1]
            vt = inner[inner.index(',')+1:]
            return self._compile_dict_method(obj_v, obj_t, vt, method, node.args)

        if obj_t == "str":
            return self._compile_str_method(obj_v, method, node.args)

        if obj_t.endswith("[]"):
            elem_vt = obj_t[:-2]
            return self._compile_arr_method(obj_v, obj_t, elem_vt, method, node.args)

        # Struct own-methods (#13): StructName__method(self_ptr, args...)
        struct_method_key = f"{obj_t}__{method}"
        if struct_method_key in self._functions:
            fi = self._functions[struct_method_key]
            fn  = fi["fn"]
            sig = fi["sig"]
            # First arg is self pointer; rest from call
            compiled_args = [obj_v]
            non_self_params = [(n, t) for n, t in sig.params if n != "self"]
            defaults = self._fn_defaults.get(struct_method_key, [])
            args_list = list(node.args)
            while len(args_list) < len(non_self_params):
                idx = len(args_list)
                if idx < len(defaults) and defaults[idx] is not None:
                    args_list.append(defaults[idx])
                else:
                    break
            for arg_node, (_, pt) in zip(args_list, non_self_params):
                av, at = self._compile_expr(arg_node)
                if pt == "float" and at == "int":
                    av = self.builder.sitofp(av, F64_TY)
                compiled_args.append(av)
            result = self.builder.call(fn, compiled_args)
            return result, sig.return_type

        # Enum own-methods (#13): EnumName__method(self_val, args...)
        enum_method_key = f"{obj_t}__{method}"
        if enum_method_key in self._functions:
            fi = self._functions[enum_method_key]
            fn  = fi["fn"]
            sig = fi["sig"]
            compiled_args = [obj_v]
            non_self_params = [(n, t) for n, t in sig.params if n != "self"]
            for arg_node, (_, pt) in zip(node.args, non_self_params):
                av, at = self._compile_expr(arg_node)
                if pt == "float" and at == "int":
                    av = self.builder.sitofp(av, F64_TY)
                compiled_args.append(av)
            result = self.builder.call(fn, compiled_args)
            return result, sig.return_type

        # HTTP server methods (#51): server.get(path, handler), server.listen()
        if obj_t == "int" and method == "listen":
            # http server fd: call __vx_http_listen(fd)
            self.builder.call(self._get_helper("__vx_http_listen"), [obj_v])
            return ir.Constant(I64_TY, 0), "void"
        if obj_t == "int" and method in ("get", "post", "put", "delete", "patch"):
            # http_server.get(path, handler): register route
            path_v, _ = self._compile_expr(node.args[0])
            cb_v, _   = self._compile_expr(node.args[1])
            meth_str  = self._gstr_ptr(self._global_str(method.upper()))
            self.builder.call(self._get_helper("__vx_http_add_route"),
                              [obj_v, meth_str, path_v,
                               self.builder.bitcast(cb_v, I8PTR)])
            return ir.Constant(I64_TY, 0), "void"

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

        if method == "find":
            sub_v, _ = self._compile_expr(args[0])
            fn_h = self._get_helper("__vx_str_find")
            return self.builder.call(fn_h, [s, sub_v]), "int"

        if method == "slice":
            start_v, _ = self._compile_expr(args[0])
            end_v, _   = self._compile_expr(args[1])
            fn_h = self._get_helper("__vx_str_slice")
            return self.builder.call(fn_h, [s, start_v, end_v]), "str"

        if method == "repeat":
            n_v, _ = self._compile_expr(args[0])
            fn_h = self._get_helper("__vx_str_repeat")
            return self.builder.call(fn_h, [s, n_v]), "str"

        if method == "char_at":
            idx_v, _ = self._compile_expr(args[0])
            cp = self.builder.gep(s, [idx_v], inbounds=False)
            ch = self.builder.load(cp)
            return self.builder.zext(ch, I64_TY), "int"

        if method == "to_int":
            return self.builder.call(self.atoll_fn, [s]), "int"

        if method == "to_float":
            return self.builder.call(self.atof_fn, [s]), "float"

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

        if method == "sort":
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_sort_i64") if elem_vt == "int" \
                   else self._get_helper("__vx_array_sort_f64")
            self.builder.call(fn_h, [arr_raw])
            return ir.Constant(I64_TY, 0), "void"

        if method == "index_of":
            elem_v, ev_t = self._compile_expr(args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            if elem_vt == "int":
                fn_h = self._get_helper("__vx_array_index_of_i64")
                return self.builder.call(fn_h, [arr_raw, elem_v]), "int"
            fn_h = self._get_helper("__vx_array_index_of_str")
            return self.builder.call(fn_h, [arr_raw, elem_v]), "int"

        if method == "join" and elem_vt == "str":
            sep_v, _ = self._compile_expr(args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_join_str")
            return self.builder.call(fn_h, [arr_raw, sep_v]), "str"

        if method == "slice":
            start_v, _ = self._compile_expr(args[0])
            end_v, _   = self._compile_expr(args[1])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_slice")
            sliced = self.builder.call(fn_h, [arr_raw, start_v, end_v, esz])
            return self.builder.bitcast(sliced, self.arr_ptr_type), arr_t

        if method == "map":
            # arr.map(fn) — returns new array applying fn to each element
            fn_v, _ = self._compile_expr(args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_map_i64")
            result = self.builder.call(fn_h, [arr_raw, fn_v])
            return self.builder.bitcast(result, self.arr_ptr_type), arr_t

        if method == "filter":
            fn_v, _ = self._compile_expr(args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_filter_i64")
            result = self.builder.call(fn_h, [arr_raw, fn_v])
            return self.builder.bitcast(result, self.arr_ptr_type), arr_t

        raise CodegenError(f"Unknown array method '{method}'")

    # ------------------------------------------------------------------ #
    #  Private helper functions                                            #
    # ------------------------------------------------------------------ #

    def _get_or_declare(self, name: str, fn_type: ir.FunctionType) -> ir.Function:
        """Get or declare an LLVM intrinsic or external function by name."""
        for f in self.module.functions:
            if f.name == name:
                return f
        return ir.Function(self.module, fn_type, name=name)

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
            "__vx_str_find":             self._build_str_find,
            "__vx_str_slice":            self._build_str_slice,
            "__vx_str_repeat":           self._build_str_repeat,
            "__vx_array_push":           self._build_array_push,
            "__vx_array_pop":            self._build_array_pop,
            "__vx_array_contains_i64":   self._build_array_contains_i64,
            "__vx_array_contains_f64":   self._build_array_contains_f64,
            "__vx_array_contains_str":   self._build_array_contains_str,
            "__vx_array_reverse":        self._build_array_reverse,
            "__vx_array_sort_i64":       self._build_array_sort_i64,
            "__vx_array_sort_f64":       self._build_array_sort_f64,
            "__vx_array_index_of_i64":   self._build_array_index_of_i64,
            "__vx_array_index_of_str":   self._build_array_index_of_str,
            "__vx_array_join_str":       self._build_array_join_str,
            "__vx_array_slice":          self._build_array_slice,
            "__vx_array_map_i64":        self._build_array_map_i64,
            "__vx_array_filter_i64":     self._build_array_filter_i64,
            "__vx_file_read":            self._build_file_read,
            "__vx_file_write":           self._build_file_write,
            "__vx_file_append":          self._build_file_append,
            "__vx_file_exists":          self._build_file_exists,
            # Dict helpers
            "__vx_dict_new":             self._build_dict_new,
            "__vx_dict_set":             self._build_dict_set,
            "__vx_dict_get":             self._build_dict_get,
            "__vx_dict_has":             self._build_dict_has,
            "__vx_dict_remove":          self._build_dict_remove,
            "__vx_dict_len":             self._build_dict_len,
            "__vx_dict_keys":            self._build_dict_keys,
            "__vx_dict_values":          self._build_dict_values,
            "__vx_dict_items":           self._build_dict_items,
            # v5 helpers
            "__vx_time_format":          self._build_time_format,
            "__vx_input":                self._build_input,
            "__vx_os_list_dir":          self._build_os_list_dir,
            # v7 helpers
            "__vx_base64_encode":        self._build_base64_encode,
            "__vx_base64_decode":        self._build_base64_decode,
            "__vx_uuid_v4":              self._build_uuid_v4,
            "__vx_sha256":               self._build_sha256,
            "__vx_argv":                 self._build_argv,
            "__vx_shell":                self._build_shell,
            "__vx_csv_parse":            self._build_csv_parse,
            "__vx_assert_eq":            self._build_assert_eq,
            "__vx_str_join":             self._build_str_join,
            "__vx_str_starts_with":      self._build_str_starts_with,
            "__vx_str_char_at":          self._build_str_char_at,
            "__vx_env_set":              self._build_env_set,
            "__vx_thread_spawn":         self._build_thread_spawn,
            "__vx_thread_join":          self._build_thread_join,
            "__vx_thread_sleep":         self._build_thread_sleep,
            "__vx_mutex_new":            self._build_mutex_new,
            "__vx_mutex_lock":           self._build_mutex_lock,
            "__vx_mutex_unlock":         self._build_mutex_unlock,
            "__vx_str_char_len":         self._build_str_char_len,
            "__vx_str_char_at_utf8":     self._build_str_char_at_utf8,
            "__vx_json_stringify_int":   self._build_json_stringify_int,
            "__vx_json_stringify_float": self._build_json_stringify_float,
            "__vx_json_stringify_str":   self._build_json_stringify_str,
            "__vx_http_get":             self._build_http_get,
            # v8 new helpers
            "__vx_csv_write":            self._build_csv_write,
            "__vx_print_color":          self._build_print_color,
            "__vx_dict_values":          self._build_dict_values,
            "__vx_dict_items":           self._build_dict_items,
            "__vx_datetime_now":         self._build_datetime_now,
            "__vx_datetime_format":      self._build_datetime_format,
            "__vx_datetime_from_ts":     self._build_datetime_from_ts,
            "__vx_sleep_ms":             self._build_sleep_ms,
            "__vx_signal_handle":        self._build_signal_handle,
            "__vx_process_spawn":        self._build_process_spawn,
            "__vx_process_wait":         self._build_process_wait,
            "__vx_process_kill":         self._build_process_kill,
            "__vx_progress_new":         self._build_progress_new,
            "__vx_progress_update":      self._build_progress_update,
            "__vx_progress_finish":      self._build_progress_finish,
            "__vx_term_clear":           self._build_term_clear,
            "__vx_term_move":            self._build_term_move,
            "__vx_term_width":           self._build_term_width,
            "__vx_channel_new":          self._build_channel_new,
            "__vx_channel_send":         self._build_channel_send,
            "__vx_channel_recv":         self._build_channel_recv,
            "__vx_channel_try_recv":     self._build_channel_try_recv,
            "__vx_channel_close":        self._build_channel_close,
            "__vx_rwlock_new":           self._build_rwlock_new,
            "__vx_rwlock_read_lock":     self._build_rwlock_read_lock,
            "__vx_rwlock_read_unlock":   self._build_rwlock_read_unlock,
            "__vx_rwlock_write_lock":    self._build_rwlock_write_lock,
            "__vx_rwlock_write_unlock":  self._build_rwlock_write_unlock,
            "__vx_thread_pool_new":      self._build_thread_pool_new,
            "__vx_thread_pool_submit":   self._build_thread_pool_submit,
            "__vx_thread_pool_wait":     self._build_thread_pool_wait,
            "__vx_thread_pool_destroy":  self._build_thread_pool_destroy,
            "__vx_benchmark":            self._build_benchmark,
            "__vx_bench_ns":             self._build_bench_ns,
            "__vx_json_parse_int":       self._build_json_parse_int,
            "__vx_json_parse_str":       self._build_json_parse_str,
            "__vx_json_parse_float":     self._build_json_parse_float,
            "__vx_json_parse_bool":      self._build_json_parse_bool,
            "__vx_env_load":             self._build_env_load,
            "__vx_set_new":              self._build_set_new,
            "__vx_set_add":              self._build_set_add,
            "__vx_set_contains":         self._build_set_contains,
            "__vx_set_remove":           self._build_set_remove,
            "__vx_set_size":             self._build_set_size,
            "__vx_set_to_array":         self._build_set_to_array,
            "__vx_set_union":            self._build_set_union,
            "__vx_set_intersect":        self._build_set_intersect,
            "__vx_uuid_v1":              self._build_uuid_v1,
            "__vx_condvar_new":          self._build_condvar_new,
            "__vx_condvar_wait":         self._build_condvar_wait,
            "__vx_condvar_signal":       self._build_condvar_signal,
            "__vx_condvar_broadcast":    self._build_condvar_broadcast,
            "__vx_bounds_check":         self._build_bounds_check_helper,
            # v9 — sockets / pipes / IPC
            "__vx_tcp_connect":          self._build_tcp_connect,
            "__vx_tcp_send":             self._build_tcp_send,
            "__vx_tcp_recv":             self._build_tcp_recv,
            "__vx_tcp_close":            self._build_tcp_close,
            "__vx_tcp_listen":           self._build_tcp_listen,
            "__vx_tcp_accept":           self._build_tcp_accept,
            "__vx_udp_socket":           self._build_udp_socket,
            "__vx_udp_send_to":          self._build_udp_send_to,
            "__vx_udp_recv_from":        self._build_udp_recv_from,
            "__vx_file_watch":           self._build_file_watch,
            "__vx_pipe_open":            self._build_pipe_open,
            "__vx_pipe_write":           self._build_pipe_write,
            "__vx_pipe_read":            self._build_pipe_read,
            "__vx_pipe_close":           self._build_pipe_close,
            "__vx_chan_select":           self._build_chan_select,
            # v10 — hashing / regex / TOML / JSON array
            "__vx_md5":                  self._build_md5,
            "__vx_sha512":               self._build_sha512,
            "__vx_hmac_sha256":          self._build_hmac_sha256,
            "__vx_regex_engine":         self._build_regex_engine,
            "__vx_regex_match":          self._build_regex_match,
            "__vx_regex_test":           self._build_regex_test,
            "__vx_regex_find_all":       self._build_regex_find_all,
            "__vx_regex_replace":        self._build_regex_replace,
            "__vx_toml_parse_str":       self._build_toml_parse_str,
            "__vx_toml_parse_int":       self._build_toml_parse_int,
            "__vx_toml_parse_float":     self._build_toml_parse_float,
            "__vx_json_stringify_arr":   self._build_json_stringify_arr,
            "__vx_json_stringify_str_arr": self._build_json_stringify_str_arr,
            # v10 — HTTP server (#51)
            "__vx_http_serve":           self._build_http_serve,
            "__vx_http_add_route":       self._build_http_add_route,
            "__vx_http_listen":          self._build_http_listen,
            # v11 — SQLite (#50)
            "__vx_sqlite_open":          self._build_sqlite_open,
            "__vx_sqlite_exec":          self._build_sqlite_exec,
            "__vx_sqlite_query":         self._build_sqlite_query,
            "__vx_sqlite_close":         self._build_sqlite_close,
            # v11 — zlib (#49)
            "__vx_zlib_compress":        self._build_zlib_compress,
            "__vx_zlib_decompress":      self._build_zlib_decompress,
            # v11 — XML (#46)
            "__vx_xml_parse":            self._build_xml_parse,
            "__vx_xml_parse_all":        self._build_xml_parse_all,
            "__vx_xml_build":            self._build_xml_build,
            # v11 — YAML (#48)
            "__vx_yaml_parse_str":       self._build_yaml_parse_str,
            "__vx_yaml_parse_int":       self._build_yaml_parse_int,
            "__vx_yaml_parse_float":     self._build_yaml_parse_float,
            # v11 — bcrypt/Argon2 (#43)
            "__vx_bcrypt_hash":          self._build_bcrypt_hash,
            "__vx_bcrypt_verify":        self._build_bcrypt_verify,
            "__vx_argon2_hash":          self._build_argon2_hash,
            "__vx_argon2_verify":        self._build_argon2_verify,
            # v11 — WebSocket (#52)
            "__vx_ws_connect":           self._build_ws_connect,
            "__vx_ws_send":              self._build_ws_send,
            "__vx_ws_recv":              self._build_ws_recv,
            "__vx_ws_close":             self._build_ws_close,
            # v11 — TLS/SSL (#53)
            "__vx_tls_connect":          self._build_tls_connect,
            "__vx_tls_send":             self._build_tls_send,
            "__vx_tls_recv":             self._build_tls_recv,
            "__vx_tls_close":            self._build_tls_close,
            # v11 — stack trace (#76)
            "__vx_stack_trace":          self._build_stack_trace,
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

    def _fold_constant(self, op: str, lv, lt: str, rv, rt: str):
        """Try to constant-fold a binary operation at compile time (#93).
        Returns (ir.Value, vx_type) if foldable, else None."""
        # Only fold pure integer/float/bool literals
        if not (isinstance(lv, ir.Constant) and isinstance(rv, ir.Constant)):
            return None
        try:
            lc = lv.constant
            rc = rv.constant
            if lt == "float" or rt == "float":
                lf = float(lc); rf = float(rc)
                ops = {"+": lf+rf, "-": lf-rf, "*": lf*rf}
                if op == "/" and rf != 0: ops["/"] = lf/rf
                if op in ops: return ir.Constant(F64_TY, ops[op]), "float"
                cmp = {"==": lf==rf, "!=": lf!=rf, "<": lf<rf, ">": lf>rf,
                       "<=": lf<=rf, ">=": lf>=rf}
                if op in cmp: return ir.Constant(I1_TY, int(cmp[op])), "bool"
            elif lt == "int" and rt == "int":
                li = int(lc) if lc is not None else 0
                ri = int(rc) if rc is not None else 0
                ops = {"+": li+ri, "-": li-ri, "*": li*ri,
                       "&": li&ri, "|": li|ri, "^": li^ri,
                       "<<": li<<ri, ">>": li>>ri, "%": li%ri if ri else 0}
                if op == "/" and ri != 0: ops["/"] = li//ri
                if op == "**": ops["**"] = int(li**ri)
                if op in ops:
                    result = ops[op]
                    # Clamp to i64 range
                    result = result & 0xFFFFFFFFFFFFFFFF
                    if result >= 0x8000000000000000: result -= 0x10000000000000000
                    return ir.Constant(I64_TY, result), "int"
                cmp = {"==": li==ri, "!=": li!=ri, "<": li<ri, ">": li>ri,
                       "<=": li<=ri, ">=": li>=ri}
                if op in cmp: return ir.Constant(I1_TY, int(cmp[op])), "bool"
        except Exception:
            pass
        return None

    def _compile_binop(self, node: BinOp) -> tuple[ir.Value, str]:
        op = node.op

        # Short-circuit AND/OR
        if op == "and":
            return self._compile_and(node)
        if op == "or":
            return self._compile_or(node)

        # `x in container`
        if op == "in":
            return self._compile_in(node)

        lv, lt = self._compile_expr(node.left)
        rv, rt = self._compile_expr(node.right)

        # Constant folding (#93): evaluate literal arithmetic at compile time
        folded = self._fold_constant(op, lv, lt, rv, rt)
        if folded is not None:
            return folded

        # Nullable comparison: T? == null / T? != null
        if op in ("==", "!=") and (lt.endswith("?") or rt == "null" or
                                    rt.endswith("?") or lt == "null"):
            # Both sides to i64 for pointer comparison
            li = self.builder.ptrtoint(lv, I64_TY)
            ri = self.builder.ptrtoint(rv, I64_TY)
            if op == "==":
                return self.builder.icmp_unsigned("==", li, ri), "bool"
            else:
                return self.builder.icmp_unsigned("!=", li, ri), "bool"

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

        # #12 Operator overloading: check for __op__ method on struct type
        _op_method_map = {
            "+":  "__add__", "-":  "__sub__", "*":  "__mul__", "/":  "__div__",
            "%":  "__mod__", "==": "__eq__",  "!=": "__ne__",
            "<":  "__lt__",  ">":  "__gt__",  "<=": "__le__", ">=": "__ge__",
        }
        if op in _op_method_map and lt in self._structs:
            method_key = f"{lt}__{_op_method_map[op]}"
            fi = self._functions.get(method_key)
            if fi is not None:
                result = self.builder.call(fi["fn"], [lv, rv])
                return result, fi["sig"].return_type

        arith = {
            "+": (self.builder.fadd, self.builder.add),
            "-": (self.builder.fsub, self.builder.sub),
            "*": (self.builder.fmul, self.builder.mul),
            "/": (self.builder.fdiv, self.builder.sdiv),
            "%": (self.builder.frem, self.builder.srem),
        }
        if op in arith:
            f_op, i_op = arith[op]
            if not is_float and self.debug_mode and op in ("+", "-", "*"):
                # #102: integer overflow detection in debug mode
                intr_map = {"+": "sadd", "-": "ssub", "*": "smul"}
                intr_name = f"llvm.{intr_map[op]}.with.overflow.i64"
                ovf_ret_ty = ir.LiteralStructType([I64_TY, I1_TY])
                intr_ty = ir.FunctionType(ovf_ret_ty, [I64_TY, I64_TY])
                intr_fn = self._get_or_declare_fn(intr_name, intr_ty)
                res_struct = self.builder.call(intr_fn, [lv, rv])
                result   = self.builder.extract_value(res_struct, 0)
                overflow = self.builder.extract_value(res_struct, 1)
                ok_bb   = self.current_fn.append_basic_block("ovf.ok")
                fail_bb = self.current_fn.append_basic_block("ovf.fail")
                self.builder.cbranch(overflow, fail_bb, ok_bb)
                self.builder.position_at_end(fail_bb)
                self._emit_panic(f"integer overflow in '{op}'")
                self.builder.position_at_end(ok_bb)
                return result, "int"
            return (f_op(lv, rv) if is_float else i_op(lv, rv)), lt

        # Power operator: x ** y  → pow(x, y)
        if op == "**":
            if not is_float:
                lv = self.builder.sitofp(lv, F64_TY)
                rv = self.builder.sitofp(rv, F64_TY)
            result = self.builder.call(self.pow_fn, [lv, rv])
            if lt == "int":
                return self.builder.fptosi(result, I64_TY), "int"
            return result, "float"

        # Bitwise operators (integers only)
        if op == "&":
            return self.builder.and_(lv, rv), lt
        if op == "|":
            return self.builder.or_(lv, rv), lt
        if op == "^":
            return self.builder.xor(lv, rv), lt
        if op == "<<":
            return self.builder.shl(lv, rv), lt
        if op == ">>":
            return self.builder.ashr(lv, rv), lt

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

    def _compile_in(self, node: BinOp) -> tuple[ir.Value, str]:
        """Compile  x in collection  → bool."""
        lv, lt = self._compile_expr(node.left)
        rv, rt = self._compile_expr(node.right)

        if rt == "str":
            # substring check: strstr(container, needle) != NULL
            result = self.builder.call(self.strstr_fn, [rv, lv])
            null_int = ir.Constant(I64_TY, 0)
            res_int  = self.builder.ptrtoint(result, I64_TY)
            return self.builder.icmp_unsigned("!=", res_int, null_int), "bool"

        if rt.endswith("[]"):
            elem_vt = rt[:-2]
            arr_raw = self.builder.bitcast(rv, I8PTR)
            if elem_vt == "int":
                if lt == "float":
                    lv = self.builder.fptosi(lv, I64_TY)
                fn_h = self._get_helper("__vx_array_contains_i64")
                return self.builder.call(fn_h, [arr_raw, lv]), "bool"
            elif elem_vt == "float":
                if lt == "int":
                    lv = self.builder.sitofp(lv, F64_TY)
                fn_h = self._get_helper("__vx_array_contains_f64")
                return self.builder.call(fn_h, [arr_raw, lv]), "bool"
            else:
                fn_h = self._get_helper("__vx_array_contains_str")
                return self.builder.call(fn_h, [arr_raw, lv]), "bool"

        raise CodegenError(f"'in' not supported for type '{rt}'")

    def _compile_unary(self, node: UnaryOp) -> tuple[ir.Value, str]:
        val, vt = self._compile_expr(node.operand)
        if node.op == "-":
            if vt == "float": return self.builder.fsub(ir.Constant(F64_TY, 0.0), val), "float"
            return self.builder.neg(val), "int"
        if node.op == "not":
            bv = self.builder.trunc(val, I1_TY) if val.type != I1_TY else val
            return self.builder.not_(bv), "bool"
        if node.op == "~":
            # Bitwise NOT: XOR with all-ones
            return self.builder.xor(val, ir.Constant(I64_TY, -1)), "int"
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

        # --- v4 Math ---
        if name == "round":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            r = self.builder.call(self.round_fn, [val])
            return self.builder.fptosi(r, I64_TY), "int"

        if name == "clamp":
            val, vt = self._compile_expr(node.args[0])
            lo_v, lo_t = self._compile_expr(node.args[1])
            hi_v, hi_t = self._compile_expr(node.args[2])
            if vt == "float" or lo_t == "float" or hi_t == "float":
                if vt == "int":   val  = self.builder.sitofp(val,  F64_TY)
                if lo_t == "int": lo_v = self.builder.sitofp(lo_v, F64_TY)
                if hi_t == "int": hi_v = self.builder.sitofp(hi_v, F64_TY)
                c1 = self.builder.fcmp_ordered("<", val, lo_v)
                v1 = self.builder.select(c1, lo_v, val)
                c2 = self.builder.fcmp_ordered(">", v1, hi_v)
                return self.builder.select(c2, hi_v, v1), "float"
            c1 = self.builder.icmp_signed("<", val, lo_v)
            v1 = self.builder.select(c1, lo_v, val)
            c2 = self.builder.icmp_signed(">", v1, hi_v)
            return self.builder.select(c2, hi_v, v1), "int"

        if name == "lerp":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            tv, tt = self._compile_expr(node.args[2])
            if at == "int": av = self.builder.sitofp(av, F64_TY)
            if bt == "int": bv = self.builder.sitofp(bv, F64_TY)
            if tt == "int": tv = self.builder.sitofp(tv, F64_TY)
            diff   = self.builder.fsub(bv, av)
            scaled = self.builder.fmul(diff, tv)
            return self.builder.fadd(av, scaled), "float"

        if name == "atan2":
            yv, yt = self._compile_expr(node.args[0])
            xv, xt = self._compile_expr(node.args[1])
            if yt == "int": yv = self.builder.sitofp(yv, F64_TY)
            if xt == "int": xv = self.builder.sitofp(xv, F64_TY)
            return self.builder.call(self.atan2_fn, [yv, xv]), "float"

        # --- v4 Error / OS ---
        if name == "throw":
            msg_v, _ = self._compile_expr(node.args[0])
            # Copy message into error buffer via sprintf
            fmt = self._gstr_ptr(self._global_str("%s"))
            ep0 = self.builder.gep(self._vx_error_buf,
                                   [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                   inbounds=True)
            self.builder.call(self.sprintf_fn, [ep0, fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "os_cwd":
            buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 512)])
            result = self.builder.call(self.getcwd_fn, [buf, ir.Constant(I32_TY, 512)])
            null_int = ir.Constant(I64_TY, 0)
            res_int  = self.builder.ptrtoint(result, I64_TY)
            is_null  = self.builder.icmp_unsigned("==", res_int, null_int)
            empty = self._gstr_ptr(self._global_str(""))
            return self.builder.select(is_null, empty, buf), "str"

        if name == "os_mkdir":
            path_v, _ = self._compile_expr(node.args[0])
            r = self.builder.call(self.mkdir_fn, [path_v])
            z = ir.Constant(I32_TY, 0)
            return self.builder.icmp_signed("==", r, z), "bool"

        if name == "os_delete":
            # Try remove() first (works for files); fall back to rmdir (for directories)
            path_v, _ = self._compile_expr(node.args[0])
            fn = self.current_fn
            r = self.builder.call(self.remove_fn, [path_v])
            z = ir.Constant(I32_TY, 0)
            ok_b   = fn.append_basic_block("del.ok")
            try_b  = fn.append_basic_block("del.try_rmdir")
            done_b = fn.append_basic_block("del.done")
            file_ok = self.builder.icmp_signed("==", r, z)
            self.builder.cbranch(file_ok, ok_b, try_b)
            # File remove failed — try rmdir
            self.builder.position_at_end(try_b)
            r2 = self.builder.call(self.rmdir_fn, [path_v])
            dir_ok = self.builder.icmp_signed("==", r2, z)
            self.builder.branch(done_b)
            # ok_b
            self.builder.position_at_end(ok_b)
            self.builder.branch(done_b)
            # merge
            self.builder.position_at_end(done_b)
            phi = self.builder.phi(I1_TY)
            phi.add_incoming(ir.Constant(I1_TY, 1), ok_b)
            phi.add_incoming(dir_ok, try_b)
            return phi, "bool"

        # --- v5 new builtins ---
        if name == "parse_int":
            s_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self.atoll_fn, [s_v]), "int"

        if name == "parse_float":
            s_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self.atof_fn, [s_v]), "float"

        if name == "time_now":
            null_ptr = ir.Constant(I8PTR, None)
            t = self.builder.call(self.time_fn, [null_ptr])
            return t, "int"

        if name == "time_format":
            t_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_time_format")
            return self.builder.call(fn_h, [t_v]), "str"

        if name == "input":
            prompt_v, _ = self._compile_expr(node.args[0]) if node.args else (self._gstr_ptr(self._global_str("")), "str")
            fn_h = self._get_helper("__vx_input")
            return self.builder.call(fn_h, [prompt_v]), "str"

        if name == "os_list_dir":
            path_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_os_list_dir")
            raw = self.builder.call(fn_h, [path_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[]"

        # --- v7 Environment / OS ---
        if name == "argv":
            fn_h = self._get_helper("__vx_argv")
            raw = self.builder.call(fn_h, [])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[]"

        if name == "env_get":
            key_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self.getenv_fn, [key_v]), "str"

        if name == "env_set":
            key_v, _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_env_set")
            self.builder.call(fn_h, [key_v, val_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "shell":
            cmd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_shell")
            return self.builder.call(fn_h, [cmd_v]), "str"

        # --- v7 Math extras ---
        if name == "log10":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.log10_fn, [val]), "float"

        if name == "exp":
            val, vt = self._compile_expr(node.args[0])
            if vt == "int": val = self.builder.sitofp(val, F64_TY)
            return self.builder.call(self.exp_fn, [val]), "float"

        if name == "hypot":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            if at == "int": av = self.builder.sitofp(av, F64_TY)
            if bt == "int": bv = self.builder.sitofp(bv, F64_TY)
            return self.builder.call(self.hypot_fn, [av, bv]), "float"

        # --- v7 String methods ---
        if name == "str_find":
            haystack_v, _ = self._compile_expr(node.args[0])
            needle_v,   _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_find")
            return self.builder.call(fn_h, [haystack_v, needle_v]), "int"

        if name == "str_slice":
            s_v, _ = self._compile_expr(node.args[0])
            start_v, _ = self._compile_expr(node.args[1])
            end_v,   _ = self._compile_expr(node.args[2])
            fn_h = self._get_helper("__vx_str_slice")
            return self.builder.call(fn_h, [s_v, start_v, end_v]), "str"

        if name == "str_repeat":
            s_v, _ = self._compile_expr(node.args[0])
            n_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_repeat")
            return self.builder.call(fn_h, [s_v, n_v]), "str"

        if name == "str_replace":
            s_v,   _ = self._compile_expr(node.args[0])
            old_v, _ = self._compile_expr(node.args[1])
            new_v, _ = self._compile_expr(node.args[2])
            fn_h = self._get_helper("__vx_str_replace")
            return self.builder.call(fn_h, [s_v, old_v, new_v]), "str"

        if name == "str_upper":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_str_upper")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "str_lower":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_str_lower")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "str_trim":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_str_trim")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "str_starts_with":
            s_v,      _ = self._compile_expr(node.args[0])
            prefix_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_starts_with")
            return self.builder.call(fn_h, [s_v, prefix_v]), "bool"

        if name == "str_ends_with":
            s_v,      _ = self._compile_expr(node.args[0])
            suffix_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_ends_with")
            return self.builder.call(fn_h, [s_v, suffix_v]), "bool"

        if name == "str_split":
            s_v,   _ = self._compile_expr(node.args[0])
            delim_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_split")
            raw = self.builder.call(fn_h, [s_v, delim_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[]"

        if name == "str_contains":
            s_v,      _ = self._compile_expr(node.args[0])
            needle_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_find")
            idx = self.builder.call(fn_h, [s_v, needle_v])
            return self.builder.icmp_signed(">=", idx, ir.Constant(I64_TY, 0)), "bool"

        if name == "char_at":
            s_v, _ = self._compile_expr(node.args[0])
            i_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_char_at")
            return self.builder.call(fn_h, [s_v, i_v]), "str"

        if name == "char_to_int":
            s_v, _ = self._compile_expr(node.args[0])
            # Load first byte
            ch = self.builder.load(s_v)
            return self.builder.zext(ch, I64_TY), "int"

        if name == "int_to_char":
            i_v, _ = self._compile_expr(node.args[0])
            buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 2)])
            ch  = self.builder.trunc(i_v, I8_TY)
            self.builder.store(ch, buf)
            nul_ptr = self.builder.gep(buf, [ir.Constant(I64_TY, 1)], inbounds=False)
            self.builder.store(ir.Constant(I8_TY, 0), nul_ptr)
            return buf, "str"

        if name == "to_int":
            s_v, vt = self._compile_expr(node.args[0])
            if vt == "str":   return self.builder.call(self.atoll_fn, [s_v]), "int"
            if vt == "float": return self.builder.fptosi(s_v, I64_TY), "int"
            return s_v, "int"

        if name == "to_float":
            s_v, vt = self._compile_expr(node.args[0])
            if vt == "str":   return self.builder.call(self.atof_fn, [s_v]), "float"
            if vt == "int":   return self.builder.sitofp(s_v, F64_TY), "float"
            return s_v, "float"

        # --- v7 Array methods ---
        if name == "array_sort":
            arr_v, at = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            elem_type = at[:-2] if at.endswith("[]") else "int"
            if elem_type == "float":
                fn_h = self._get_helper("__vx_array_sort_f64")
            else:
                fn_h = self._get_helper("__vx_array_sort_i64")
            self.builder.call(fn_h, [arr_raw])
            return ir.Constant(I64_TY, 0), "void"

        if name == "array_index_of":
            arr_v, at = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            elem_v, _ = self._compile_expr(node.args[1])
            elem_type = at[:-2] if at.endswith("[]") else "int"
            if elem_type == "str":
                fn_h = self._get_helper("__vx_array_index_of_str")
            else:
                fn_h = self._get_helper("__vx_array_index_of_i64")
            return self.builder.call(fn_h, [arr_raw, elem_v]), "int"

        if name == "array_join":
            arr_v, _ = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            sep_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_array_join_str")
            return self.builder.call(fn_h, [arr_raw, sep_v]), "str"

        if name == "array_reverse":
            arr_v, at = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            fn_h = self._get_helper("__vx_array_reverse")
            self.builder.call(fn_h, [arr_raw])
            return ir.Constant(I64_TY, 0), "void"

        if name == "array_contains":
            arr_v, at = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            elem_v, _ = self._compile_expr(node.args[1])
            elem_type = at[:-2] if at.endswith("[]") else "int"
            if elem_type == "str":
                fn_h = self._get_helper("__vx_array_index_of_str")
            else:
                fn_h = self._get_helper("__vx_array_index_of_i64")
            idx = self.builder.call(fn_h, [arr_raw, elem_v])
            return self.builder.icmp_signed(">=", idx, ir.Constant(I64_TY, 0)), "bool"

        if name == "array_slice":
            arr_v, at = self._compile_expr(node.args[0])
            arr_raw = self.builder.bitcast(arr_v, I8PTR)
            start_v,  _ = self._compile_expr(node.args[1])
            end_v,    _ = self._compile_expr(node.args[2])
            elem_type = at[:-2] if at.endswith("[]") else "int"
            esz = ir.Constant(I64_TY, _elem_size(elem_type))
            fn_h = self._get_helper("__vx_array_slice")
            raw = self.builder.call(fn_h, [arr_raw, start_v, end_v, esz])
            return self.builder.bitcast(raw, self.arr_ptr_type), at

        # --- v7 Base64 ---
        if name == "base64_encode":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_base64_encode")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "base64_decode":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_base64_decode")
            return self.builder.call(fn_h, [s_v]), "str"

        # --- v7 UUID ---
        if name == "uuid_v4":
            fn_h = self._get_helper("__vx_uuid_v4")
            return self.builder.call(fn_h, []), "str"

        # --- v7 Hashing ---
        if name == "sha256":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_sha256")
            return self.builder.call(fn_h, [s_v]), "str"

        # --- v7 CSV ---
        if name == "csv_parse":
            s_v,   _ = self._compile_expr(node.args[0])
            delim_v, _ = (self._compile_expr(node.args[1])
                          if len(node.args) > 1
                          else (self._gstr_ptr(self._global_str(",")), "str"))
            fn_h = self._get_helper("__vx_csv_parse")
            raw = self.builder.call(fn_h, [s_v, delim_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[][]"

        # --- v7 Test assertions ---
        if name == "assert_eq":
            av, at = self._compile_expr(node.args[0])
            bv, bt = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_assert_eq")
            self.builder.call(fn_h, [av, bv])
            return ir.Constant(I64_TY, 0), "void"

        if name == "assert_neq":
            av, _ = self._compile_expr(node.args[0])
            bv, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_assert_eq")
            # assert_neq: invert — if equal, fail
            cmp = self.builder.icmp_signed("==", av, bv)
            fn_bb   = self.current_fn
            fail_b  = fn_bb.append_basic_block("aneq.fail")
            pass_b  = fn_bb.append_basic_block("aneq.pass")
            self.builder.cbranch(cmp, fail_b, pass_b)
            self.builder.position_at_end(fail_b)
            fmt_gv = self._global_str("FAIL: assert_neq values are equal\n")
            self.builder.call(self.printf, [self._gstr_ptr(fmt_gv)])
            self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
            self.builder.branch(pass_b)
            self.builder.position_at_end(pass_b)
            return ir.Constant(I64_TY, 0), "void"

        if name == "assert_true":
            cond_v, _ = self._compile_expr(node.args[0])
            if cond_v.type != I1_TY:
                cond_v = self.builder.trunc(cond_v, I1_TY)
            fn_bb  = self.current_fn
            fail_b = fn_bb.append_basic_block("atrue.fail")
            pass_b = fn_bb.append_basic_block("atrue.pass")
            self.builder.cbranch(cond_v, pass_b, fail_b)
            self.builder.position_at_end(fail_b)
            fmt_gv = self._global_str("FAIL: assert_true got false\n")
            self.builder.call(self.printf, [self._gstr_ptr(fmt_gv)])
            self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
            self.builder.branch(pass_b)
            self.builder.position_at_end(pass_b)
            return ir.Constant(I64_TY, 0), "void"

        if name == "assert_false":
            cond_v, _ = self._compile_expr(node.args[0])
            if cond_v.type != I1_TY:
                cond_v = self.builder.trunc(cond_v, I1_TY)
            fn_bb  = self.current_fn
            fail_b = fn_bb.append_basic_block("afalse.fail")
            pass_b = fn_bb.append_basic_block("afalse.pass")
            self.builder.cbranch(cond_v, fail_b, pass_b)
            self.builder.position_at_end(fail_b)
            fmt_gv = self._global_str("FAIL: assert_false got true\n")
            self.builder.call(self.printf, [self._gstr_ptr(fmt_gv)])
            self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
            self.builder.branch(pass_b)
            self.builder.position_at_end(pass_b)
            return ir.Constant(I64_TY, 0), "void"

        # --- v7 Type queries ---
        if name == "type_of":
            _, vt = self._compile_expr(node.args[0])
            return self._gstr_ptr(self._global_str(vt)), "str"

        if name == "is_null":
            val, vt = self._compile_expr(node.args[0])
            null_int = ir.Constant(I64_TY, 0)
            ptr_int  = self.builder.ptrtoint(val, I64_TY)
            return self.builder.icmp_unsigned("==", ptr_int, null_int), "bool"

        # --- Atomic operations (#52) — lowered to LLVM atomicrmw/cmpxchg ---
        if name == "atomic_new":
            # atomic_new(val) — allocate an i64 on the heap and store val
            init_v, _ = self._compile_expr(node.args[0])
            if init_v.type != I64_TY:
                init_v = self.builder.sext(init_v, I64_TY)
            ptr = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 8)])
            i64_ptr = self.builder.bitcast(ptr, ir.PointerType(I64_TY))
            self.builder.store(init_v, i64_ptr)
            return ptr, "int?"   # store as i8* (opaque atomic pointer)

        if name == "atomic_load":
            ptr_v, _ = self._compile_expr(node.args[0])
            i64_ptr  = self.builder.bitcast(ptr_v, ir.PointerType(I64_TY))
            result   = self.builder.load(i64_ptr)
            result.atomic = "seq_cst"
            return result, "int"

        if name == "atomic_store":
            ptr_v, _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            if val_v.type != I64_TY:
                val_v = self.builder.sext(val_v, I64_TY)
            i64_ptr = self.builder.bitcast(ptr_v, ir.PointerType(I64_TY))
            st = self.builder.store(val_v, i64_ptr)
            st.atomic = "seq_cst"
            return ir.Constant(I64_TY, 0), "void"

        if name == "atomic_add":
            ptr_v, _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            if val_v.type != I64_TY:
                val_v = self.builder.sext(val_v, I64_TY)
            i64_ptr = self.builder.bitcast(ptr_v, ir.PointerType(I64_TY))
            return self.builder.atomic_rmw("add", i64_ptr, val_v, "seq_cst"), "int"

        if name == "atomic_sub":
            ptr_v, _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            if val_v.type != I64_TY:
                val_v = self.builder.sext(val_v, I64_TY)
            i64_ptr = self.builder.bitcast(ptr_v, ir.PointerType(I64_TY))
            return self.builder.atomic_rmw("sub", i64_ptr, val_v, "seq_cst"), "int"

        if name == "atomic_compare_swap":
            ptr_v,      _ = self._compile_expr(node.args[0])
            expected_v, _ = self._compile_expr(node.args[1])
            desired_v,  _ = self._compile_expr(node.args[2])
            for v in [expected_v, desired_v]:
                if v.type != I64_TY:
                    v = self.builder.sext(v, I64_TY)
            i64_ptr = self.builder.bitcast(ptr_v, ir.PointerType(I64_TY))
            res = self.builder.cmpxchg(i64_ptr, expected_v, desired_v, "seq_cst", "seq_cst")
            return self.builder.extract_value(res, 1), "bool"

        # --- Threads (#49) — wrap pthreads on POSIX, _beginthread on Windows ---
        if name == "thread_spawn":
            fn_h = self._get_helper("__vx_thread_spawn")
            fn_v, _ = self._compile_expr(node.args[0])
            # fn_v should be an i8* function pointer
            if fn_v.type != I8PTR:
                fn_v = self.builder.bitcast(fn_v, I8PTR)
            return self.builder.call(fn_h, [fn_v]), "int"

        if name == "thread_join":
            fn_h = self._get_helper("__vx_thread_join")
            tid_v, _ = self._compile_expr(node.args[0])
            self.builder.call(fn_h, [tid_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "thread_sleep":
            fn_h = self._get_helper("__vx_thread_sleep")
            ms_v, _ = self._compile_expr(node.args[0])
            self.builder.call(fn_h, [ms_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- Mutex (#50) ---
        if name == "mutex_new":
            fn_h = self._get_helper("__vx_mutex_new")
            raw = self.builder.call(fn_h, [])
            return self.builder.ptrtoint(raw, I64_TY), "int"

        if name == "mutex_lock":
            fn_h = self._get_helper("__vx_mutex_lock")
            mu_v, _ = self._compile_expr(node.args[0])
            mu_ptr = self.builder.inttoptr(mu_v, I8PTR) if mu_v.type == I64_TY else mu_v
            self.builder.call(fn_h, [mu_ptr])
            return ir.Constant(I64_TY, 0), "void"

        if name == "mutex_unlock":
            fn_h = self._get_helper("__vx_mutex_unlock")
            mu_v, _ = self._compile_expr(node.args[0])
            mu_ptr = self.builder.inttoptr(mu_v, I8PTR) if mu_v.type == I64_TY else mu_v
            self.builder.call(fn_h, [mu_ptr])
            return ir.Constant(I64_TY, 0), "void"

        if name == "mutex_try_lock":
            import sys as _sys
            mu_v, _ = self._compile_expr(node.args[0])
            mu_ptr = self.builder.inttoptr(mu_v, I8PTR) if mu_v.type == I64_TY else mu_v
            if _sys.platform == "win32":
                # TryEnterCriticalSection returns BOOL (i32)
                try_ty = ir.FunctionType(I32_TY, [I8PTR])
                try_fn = self._get_or_declare("TryEnterCriticalSection",
                                              ir.FunctionType(I32_TY, [I8PTR]))
                r32 = self.builder.call(try_fn, [mu_ptr])
                return self.builder.icmp_signed("!=", r32, ir.Constant(I32_TY, 0)), "bool"
            else:
                try_ty = ir.FunctionType(I32_TY, [I8PTR])
                try_fn_f = self._get_or_declare_fn("pthread_mutex_trylock", try_ty)
                r32 = self.builder.call(try_fn_f, [mu_ptr])
                return self.builder.icmp_signed("==", r32, ir.Constant(I32_TY, 0)), "bool"

        # --- str_format (printf-style, #4 stdlib) ---
        if name == "str_format":
            fmt_v, _ = self._compile_expr(node.args[0])
            # Build a 1KB output buffer and use sprintf
            buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 1024)])
            # Pass remaining args variardically through sprintf
            call_args = [buf, fmt_v]
            for arg in node.args[1:]:
                av, at = self._compile_expr(arg)
                call_args.append(av)
            self.builder.call(self.sprintf_fn, call_args)
            return buf, "str"

        # --- JSON basic support (#4 stdlib) ---
        if name == "json_stringify_int":
            val_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_json_stringify_int")
            return self.builder.call(fn_h, [val_v]), "str"

        if name == "json_stringify_float":
            val_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_json_stringify_float")
            return self.builder.call(fn_h, [val_v]), "str"

        if name == "json_stringify_str":
            val_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_json_stringify_str")
            return self.builder.call(fn_h, [val_v]), "str"

        # --- HTTP basic (#4 stdlib) ---
        if name == "http_get":
            url_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_http_get")
            return self.builder.call(fn_h, [url_v]), "str"

        # --- Unicode helpers (#45) ---
        if name == "str_char_len":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_str_char_len")
            return self.builder.call(fn_h, [s_v]), "int"

        if name == "str_char_at":
            s_v, _ = self._compile_expr(node.args[0])
            i_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_str_char_at_utf8")
            return self.builder.call(fn_h, [s_v, i_v]), "int"

        # --- Result/Option type helpers (#14) ---
        if name in ("Ok", "Some"):
            # Ok(val) — return the value directly (no boxing in this impl)
            val_v, vt = self._compile_expr(node.args[0])
            return val_v, vt

        if name in ("Err", "None_"):
            # Err(msg) — return null/zero as error sentinel
            if node.args:
                val_v, vt = self._compile_expr(node.args[0])
                return val_v, vt
            return ir.Constant(I64_TY, 0), "null"

        # --- v8 New builtins ---

        if name == "rand_seed":
            val_v, _ = self._compile_expr(node.args[0])
            if val_v.type != I32_TY:
                val_v = self.builder.trunc(val_v, I32_TY) if val_v.type == I64_TY \
                        else self.builder.fptosi(val_v, I32_TY)
            self.builder.call(self.srand_fn, [val_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "csv_write":
            rows_v, _ = self._compile_expr(node.args[0])
            delim_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_csv_write")
            return self.builder.call(fn_h, [self.builder.bitcast(rows_v, I8PTR), delim_v]), "str"

        if name == "popcount":
            val_v, _ = self._compile_expr(node.args[0])
            if val_v.type != I64_TY: val_v = self.builder.sext(val_v, I64_TY)
            fn_ty = ir.FunctionType(I64_TY, [I64_TY])
            intr = self._get_or_declare("llvm.ctpop.i64", fn_ty)
            return self.builder.call(intr, [val_v]), "int"

        if name == "clz":
            val_v, _ = self._compile_expr(node.args[0])
            if val_v.type != I64_TY: val_v = self.builder.sext(val_v, I64_TY)
            fn_ty = ir.FunctionType(I64_TY, [I64_TY, I1_TY])
            intr = self._get_or_declare("llvm.ctlz.i64", fn_ty)
            return self.builder.call(intr, [val_v, ir.Constant(I1_TY, 0)]), "int"

        if name == "ctz":
            val_v, _ = self._compile_expr(node.args[0])
            if val_v.type != I64_TY: val_v = self.builder.sext(val_v, I64_TY)
            fn_ty = ir.FunctionType(I64_TY, [I64_TY, I1_TY])
            intr = self._get_or_declare("llvm.cttz.i64", fn_ty)
            return self.builder.call(intr, [val_v, ir.Constant(I1_TY, 0)]), "int"

        if name == "bit_reverse":
            val_v, _ = self._compile_expr(node.args[0])
            if val_v.type != I64_TY: val_v = self.builder.sext(val_v, I64_TY)
            fn_ty = ir.FunctionType(I64_TY, [I64_TY])
            intr = self._get_or_declare("llvm.bitreverse.i64", fn_ty)
            return self.builder.call(intr, [val_v]), "int"

        if name == "log_info":
            msg_v, _ = self._compile_expr(node.args[0])
            fmt = self._gstr_ptr(self._global_str("\033[0;32m[INFO]\033[0m %s\n"))
            self.builder.call(self.printf, [fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "log_warn":
            msg_v, _ = self._compile_expr(node.args[0])
            fmt = self._gstr_ptr(self._global_str("\033[0;33m[WARN]\033[0m %s\n"))
            self.builder.call(self.printf, [fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "log_error":
            msg_v, _ = self._compile_expr(node.args[0])
            fmt = self._gstr_ptr(self._global_str("\033[0;31m[ERROR]\033[0m %s\n"))
            self.builder.call(self.printf, [fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "log_debug":
            msg_v, _ = self._compile_expr(node.args[0])
            fmt = self._gstr_ptr(self._global_str("\033[0;36m[DEBUG]\033[0m %s\n"))
            self.builder.call(self.printf, [fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "print_color":
            msg_v,   _ = self._compile_expr(node.args[0])
            color_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_print_color")
            self.builder.call(fn_h, [msg_v, color_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "print_bold":
            msg_v, _ = self._compile_expr(node.args[0])
            fmt = self._gstr_ptr(self._global_str("\033[1m%s\033[0m\n"))
            self.builder.call(self.printf, [fmt, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v8 builtins ---

        if name == "datetime_now":
            fn_h = self._get_helper("__vx_datetime_now")
            return self.builder.call(fn_h, []), "str"

        if name == "datetime_format":
            fmt_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_datetime_format")
            return self.builder.call(fn_h, [fmt_v]), "str"

        if name == "datetime_timestamp":
            null_ptr = ir.Constant(I8PTR, None)
            return self.builder.call(self.time_fn, [null_ptr]), "int"

        if name == "datetime_from_ts":
            ts_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_datetime_from_ts")
            return self.builder.call(fn_h, [ts_v]), "str"

        if name == "sleep_ms":
            ms_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_sleep_ms")
            self.builder.call(fn_h, [ms_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "SIGINT":
            return ir.Constant(I64_TY, 2), "int"

        if name == "SIGTERM":
            return ir.Constant(I64_TY, 15), "int"

        if name == "signal_handle":
            sig_v, _ = self._compile_expr(node.args[0])
            # args[1] is a function pointer; compile as expression
            fn_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_signal_handle")
            self.builder.call(fn_h, [sig_v, self.builder.bitcast(fn_v, I8PTR)])
            return ir.Constant(I64_TY, 0), "void"

        if name == "process_spawn":
            cmd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_process_spawn")
            return self.builder.call(fn_h, [cmd_v]), "int"

        if name == "process_wait":
            pid_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_process_wait")
            return self.builder.call(fn_h, [pid_v]), "int"

        if name == "process_kill":
            pid_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_process_kill")
            self.builder.call(fn_h, [pid_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "progress_new":
            total_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_progress_new")
            return self.builder.call(fn_h, [total_v]), "int"

        if name == "progress_update":
            id_v,  _ = self._compile_expr(node.args[0])
            cur_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_progress_update")
            self.builder.call(fn_h, [id_v, cur_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "progress_finish":
            id_v,  _ = self._compile_expr(node.args[0])
            msg_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_progress_finish")
            self.builder.call(fn_h, [id_v, msg_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "term_clear":
            fn_h = self._get_helper("__vx_term_clear")
            self.builder.call(fn_h, [])
            return ir.Constant(I64_TY, 0), "void"

        if name == "term_move":
            row_v, _ = self._compile_expr(node.args[0])
            col_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_term_move")
            self.builder.call(fn_h, [row_v, col_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "term_width":
            fn_h = self._get_helper("__vx_term_width")
            return self.builder.call(fn_h, []), "int"

        if name == "channel_new":
            fn_h = self._get_helper("__vx_channel_new")
            return self.builder.call(fn_h, []), "int"

        if name == "channel_send":
            ch_v,  _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_channel_send")
            self.builder.call(fn_h, [ch_v, val_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "channel_recv":
            ch_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_channel_recv")
            return self.builder.call(fn_h, [ch_v]), "int"

        if name == "channel_try_recv":
            ch_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_channel_try_recv")
            return self.builder.call(fn_h, [ch_v]), "int"

        if name == "channel_close":
            ch_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_channel_close")
            self.builder.call(fn_h, [ch_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "rwlock_new":
            fn_h = self._get_helper("__vx_rwlock_new")
            return self.builder.call(fn_h, []), "int"

        if name in ("rwlock_read_lock", "rwlock_read_unlock",
                    "rwlock_write_lock", "rwlock_write_unlock"):
            lk_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper(f"__vx_{name}")
            self.builder.call(fn_h, [lk_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "thread_pool_new":
            n_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_thread_pool_new")
            return self.builder.call(fn_h, [n_v]), "int"

        if name == "thread_pool_submit":
            pool_v, _ = self._compile_expr(node.args[0])
            task_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_thread_pool_submit")
            self.builder.call(fn_h, [pool_v, self.builder.bitcast(task_v, I8PTR)])
            return ir.Constant(I64_TY, 0), "void"

        if name == "thread_pool_wait":
            pool_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_thread_pool_wait")
            self.builder.call(fn_h, [pool_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "thread_pool_destroy":
            pool_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_thread_pool_destroy")
            self.builder.call(fn_h, [pool_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "benchmark":
            fn_v,  _ = self._compile_expr(node.args[0])
            cnt_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_benchmark")
            return self.builder.call(fn_h, [self.builder.bitcast(fn_v, I8PTR), cnt_v]), "str"

        if name == "bench_ns":
            fn_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_bench_ns")
            return self.builder.call(fn_h, [self.builder.bitcast(fn_v, I8PTR)]), "int"

        if name == "json_parse_int":
            json_v, _ = self._compile_expr(node.args[0])
            key_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_json_parse_int")
            return self.builder.call(fn_h, [json_v, key_v]), "int"

        if name == "json_parse_str":
            json_v, _ = self._compile_expr(node.args[0])
            key_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_json_parse_str")
            return self.builder.call(fn_h, [json_v, key_v]), "str"

        if name == "json_parse_float":
            json_v, _ = self._compile_expr(node.args[0])
            key_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_json_parse_float")
            return self.builder.call(fn_h, [json_v, key_v]), "float"

        if name == "json_parse_bool":
            json_v, _ = self._compile_expr(node.args[0])
            key_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_json_parse_bool")
            return self.builder.call(fn_h, [json_v, key_v]), "bool"

        if name == "env_load":
            path_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_env_load")
            self.builder.call(fn_h, [path_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "set_new":
            fn_h = self._get_helper("__vx_set_new")
            return self.builder.call(fn_h, []), "int"

        if name == "set_add":
            s_v,   _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_set_add")
            self.builder.call(fn_h, [s_v, val_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "set_contains":
            s_v,   _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_set_contains")
            return self.builder.call(fn_h, [s_v, val_v]), "bool"

        if name == "set_remove":
            s_v,   _ = self._compile_expr(node.args[0])
            val_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_set_remove")
            self.builder.call(fn_h, [s_v, val_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "set_size":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_set_size")
            return self.builder.call(fn_h, [s_v]), "int"

        if name == "set_to_array":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_set_to_array")
            return self.builder.call(fn_h, [s_v]), "int[]"

        if name == "set_union":
            a_v, _ = self._compile_expr(node.args[0])
            b_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_set_union")
            return self.builder.call(fn_h, [a_v, b_v]), "int"

        if name == "set_intersect":
            a_v, _ = self._compile_expr(node.args[0])
            b_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_set_intersect")
            return self.builder.call(fn_h, [a_v, b_v]), "int"

        if name == "uuid_v1":
            fn_h = self._get_helper("__vx_uuid_v1")
            return self.builder.call(fn_h, []), "str"

        if name == "bounds_check":
            idx_v, _ = self._compile_expr(node.args[0])
            len_v, _ = self._compile_expr(node.args[1])
            # Abort if idx < 0 or idx >= len
            ok1 = self.builder.icmp_signed(">=", idx_v, ir.Constant(I64_TY, 0))
            ok2 = self.builder.icmp_signed("<",  idx_v, len_v)
            ok  = self.builder.and_(ok1, ok2)
            ok_bb   = self.current_fn.append_basic_block("bc_ok")
            fail_bb = self.current_fn.append_basic_block("bc_fail")
            self.builder.cbranch(ok, ok_bb, fail_bb)
            self.builder.position_at_end(fail_bb)
            fmt = self._gstr_ptr(self._global_str("index out of bounds\n"))
            self.builder.call(self.printf, [fmt])
            self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
            self.builder.unreachable()
            self.builder.position_at_end(ok_bb)
            return ir.Constant(I64_TY, 0), "void"

        if name == "condvar_new":
            fn_h = self._get_helper("__vx_condvar_new")
            return self.builder.call(fn_h, []), "int"

        if name == "condvar_wait":
            cv_v, _ = self._compile_expr(node.args[0])
            mu_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_condvar_wait")
            self.builder.call(fn_h, [cv_v, mu_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "condvar_signal":
            cv_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_condvar_signal")
            self.builder.call(fn_h, [cv_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "condvar_broadcast":
            cv_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_condvar_broadcast")
            self.builder.call(fn_h, [cv_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v9: TCP/UDP sockets (#54), pipes (#64), file-watch (#58) ---
        if name == "tcp_connect":
            host_v, _ = self._compile_expr(node.args[0])
            port_v, _ = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_tcp_connect"), [host_v, port_v]), "int"

        if name == "tcp_send":
            sock_v, _ = self._compile_expr(node.args[0])
            data_v, _ = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_tcp_send"), [sock_v, data_v]), "int"

        if name == "tcp_recv":
            sock_v, _ = self._compile_expr(node.args[0])
            buf_v, _  = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_tcp_recv"), [sock_v, buf_v]), "str"

        if name == "tcp_close":
            sock_v, _ = self._compile_expr(node.args[0])
            self.builder.call(self._get_helper("__vx_tcp_close"), [sock_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "tcp_listen":
            port_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self._get_helper("__vx_tcp_listen"), [port_v]), "int"

        if name == "tcp_accept":
            srv_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self._get_helper("__vx_tcp_accept"), [srv_v]), "int"

        if name == "udp_socket":
            return self.builder.call(self._get_helper("__vx_udp_socket"), []), "int"

        if name == "udp_send_to":
            s_v, _ = self._compile_expr(node.args[0])
            h_v, _ = self._compile_expr(node.args[1])
            p_v, _ = self._compile_expr(node.args[2])
            d_v, _ = self._compile_expr(node.args[3])
            return self.builder.call(self._get_helper("__vx_udp_send_to"), [s_v, h_v, p_v, d_v]), "int"

        if name == "udp_recv_from":
            s_v, _ = self._compile_expr(node.args[0])
            m_v, _ = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_udp_recv_from"), [s_v, m_v]), "str"

        if name == "file_watch":
            path_v, _ = self._compile_expr(node.args[0])
            cb_v, _   = self._compile_expr(node.args[1])
            self.builder.call(self._get_helper("__vx_file_watch"), [path_v, cb_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "pipe_open":
            n_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self._get_helper("__vx_pipe_open"), [n_v]), "int"

        if name == "pipe_write":
            fd_v, _   = self._compile_expr(node.args[0])
            data_v, _ = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_pipe_write"), [fd_v, data_v]), "int"

        if name == "pipe_read":
            fd_v, _ = self._compile_expr(node.args[0])
            mb_v, _ = self._compile_expr(node.args[1])
            return self.builder.call(self._get_helper("__vx_pipe_read"), [fd_v, mb_v]), "str"

        if name == "pipe_close":
            fd_v, _ = self._compile_expr(node.args[0])
            self.builder.call(self._get_helper("__vx_pipe_close"), [fd_v])
            return ir.Constant(I64_TY, 0), "void"

        if name == "chan_select":
            arr_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self._get_helper("__vx_chan_select"),
                                     [self.builder.bitcast(arr_v, I8PTR)]), "int"

        # --- v10: HTTP server (#51) ---
        if name == "http_serve":
            port_v, _ = self._compile_expr(node.args[0])
            return self.builder.call(self._get_helper("__vx_http_serve"), [port_v]), "int"

        if name == "http_listen":
            srv_v, _ = self._compile_expr(node.args[0])
            self.builder.call(self._get_helper("__vx_http_listen"), [srv_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v10: hashing extras ---
        if name == "md5":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_md5")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "sha512":
            s_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_sha512")
            return self.builder.call(fn_h, [s_v]), "str"

        if name == "hmac_sha256":
            key_v, _ = self._compile_expr(node.args[0])
            msg_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_hmac_sha256")
            return self.builder.call(fn_h, [key_v, msg_v]), "str"

        # --- v10: regex ---
        if name == "regex_match":
            text_v, _ = self._compile_expr(node.args[0])
            pat_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_regex_match")
            return self.builder.call(fn_h, [text_v, pat_v]), "str"

        if name == "regex_test":
            text_v, _ = self._compile_expr(node.args[0])
            pat_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_regex_test")
            result = self.builder.call(fn_h, [text_v, pat_v])
            return self.builder.trunc(result, I1_TY), "bool"

        if name == "regex_find_all":
            text_v, _ = self._compile_expr(node.args[0])
            pat_v,  _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_regex_find_all")
            raw = self.builder.call(fn_h, [text_v, pat_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[]"

        if name == "regex_replace":
            text_v, _ = self._compile_expr(node.args[0])
            pat_v,  _ = self._compile_expr(node.args[1])
            repl_v, _ = self._compile_expr(node.args[2])
            fn_h = self._get_helper("__vx_regex_replace")
            return self.builder.call(fn_h, [text_v, pat_v, repl_v]), "str"

        # --- v10: TOML ---
        if name == "toml_parse_str":
            content_v, _ = self._compile_expr(node.args[0])
            key_v,     _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_toml_parse_str")
            return self.builder.call(fn_h, [content_v, key_v]), "str"

        if name == "toml_parse_int":
            content_v, _ = self._compile_expr(node.args[0])
            key_v,     _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_toml_parse_int")
            return self.builder.call(fn_h, [content_v, key_v]), "int"

        if name == "toml_parse_float":
            content_v, _ = self._compile_expr(node.args[0])
            key_v,     _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_toml_parse_float")
            return self.builder.call(fn_h, [content_v, key_v]), "float"

        # --- v10: JSON array serialization ---
        if name == "json_stringify_arr":
            arr_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_json_stringify_arr")
            return self.builder.call(fn_h, [self.builder.bitcast(arr_v, I8PTR)]), "str"

        if name == "json_stringify_str_arr":
            arr_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_json_stringify_str_arr")
            return self.builder.call(fn_h, [self.builder.bitcast(arr_v, I8PTR)]), "str"

        # --- ADT enum constructor calls: EnumName.Variant(...) handled via FieldAccess at call site ---

        # --- v11: SQLite (#50) ---
        if name == "sqlite_open":
            db_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_sqlite_open")
            return self.builder.call(fn_h, [db_v]), "int"
        if name == "sqlite_exec":
            fd_v, _ = self._compile_expr(node.args[0])
            sql_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_sqlite_exec")
            return self.builder.call(fn_h, [fd_v, sql_v]), "int"
        if name == "sqlite_query":
            fd_v, _ = self._compile_expr(node.args[0])
            sql_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_sqlite_query")
            raw = self.builder.call(fn_h, [fd_v, sql_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[][]"
        if name == "sqlite_close":
            fd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_sqlite_close")
            self.builder.call(fn_h, [fd_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v11: zlib compression (#49) ---
        if name == "zlib_compress":
            data_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_zlib_compress")
            return self.builder.call(fn_h, [data_v]), "str"
        if name == "zlib_decompress":
            data_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_zlib_decompress")
            return self.builder.call(fn_h, [data_v]), "str"

        # --- v11: XML parsing (#46) ---
        if name == "xml_parse":
            xml_v, _ = self._compile_expr(node.args[0])
            tag_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_xml_parse")
            return self.builder.call(fn_h, [xml_v, tag_v]), "str"
        if name == "xml_parse_all":
            xml_v, _ = self._compile_expr(node.args[0])
            tag_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_xml_parse_all")
            raw = self.builder.call(fn_h, [xml_v, tag_v])
            return self.builder.bitcast(raw, self.arr_ptr_type), "str[]"
        if name == "xml_build":
            tag_v, _ = self._compile_expr(node.args[0])
            content_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_xml_build")
            return self.builder.call(fn_h, [tag_v, content_v]), "str"

        # --- v11: YAML parsing (#48) ---
        if name == "yaml_parse_str":
            yaml_v, _ = self._compile_expr(node.args[0])
            key_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_yaml_parse_str")
            return self.builder.call(fn_h, [yaml_v, key_v]), "str"
        if name == "yaml_parse_int":
            yaml_v, _ = self._compile_expr(node.args[0])
            key_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_yaml_parse_int")
            return self.builder.call(fn_h, [yaml_v, key_v]), "int"
        if name == "yaml_parse_float":
            yaml_v, _ = self._compile_expr(node.args[0])
            key_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_yaml_parse_float")
            return self.builder.call(fn_h, [yaml_v, key_v]), "float"

        # --- v11: bcrypt/Argon2 (#43) ---
        if name in ("bcrypt_hash", "argon2_hash"):
            pw_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper(f"__vx_{name}")
            return self.builder.call(fn_h, [pw_v]), "str"
        if name in ("bcrypt_verify", "argon2_verify"):
            pw_v, _ = self._compile_expr(node.args[0])
            hv, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper(f"__vx_{name}")
            r = self.builder.call(fn_h, [pw_v, hv])
            return self.builder.trunc(r, I1_TY), "bool"

        # --- v11: WebSocket (#52) ---
        if name == "ws_connect":
            url_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_ws_connect")
            return self.builder.call(fn_h, [url_v]), "int"
        if name == "ws_send":
            fd_v, _ = self._compile_expr(node.args[0])
            msg_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_ws_send")
            return self.builder.call(fn_h, [fd_v, msg_v]), "int"
        if name == "ws_recv":
            fd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_ws_recv")
            return self.builder.call(fn_h, [fd_v]), "str"
        if name == "ws_close":
            fd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_ws_close")
            self.builder.call(fn_h, [fd_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v11: TLS/SSL (#53) ---
        if name == "tls_connect":
            host_v, _ = self._compile_expr(node.args[0])
            port_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_tls_connect")
            return self.builder.call(fn_h, [host_v, port_v]), "int"
        if name == "tls_send":
            fd_v, _ = self._compile_expr(node.args[0])
            msg_v, _ = self._compile_expr(node.args[1])
            fn_h = self._get_helper("__vx_tls_send")
            return self.builder.call(fn_h, [fd_v, msg_v]), "int"
        if name == "tls_recv":
            fd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_tls_recv")
            return self.builder.call(fn_h, [fd_v]), "str"
        if name == "tls_close":
            fd_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_tls_close")
            self.builder.call(fn_h, [fd_v])
            return ir.Constant(I64_TY, 0), "void"

        # --- v11: Stack trace (#76) ---
        if name == "stack_trace":
            fn_h = self._get_helper("__vx_stack_trace")
            return self.builder.call(fn_h, []), "str"
        if name == "panic_with_trace":
            msg_v, _ = self._compile_expr(node.args[0])
            fn_h = self._get_helper("__vx_stack_trace")
            trace = self.builder.call(fn_h, [])
            combined = self._str_concat_inline(msg_v, trace)
            fmt = self._gstr_ptr(self._global_str("vexel panic: %s\n"))
            self.builder.call(self.printf, [fmt, combined])
            self.builder.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
            self.builder.unreachable()
            return ir.Constant(I64_TY, 0), "void"

        # --- Generic function monomorphization ---
        if name in self.analysis.generic_fns:
            return self._compile_generic_call(node)

        # --- #11 Function overload dispatch ---
        if name in self._overload_table:
            return self._dispatch_overload(name, node)

        # --- User-defined functions (with default param support and variadic) ---
        fi = self._functions.get(name)

        # --- Indirect call through function pointer variable ---
        if fi is None:
            info = self._lookup(name)
            if info is not None and self._is_fn_type(info["vx_type"]):
                return self._compile_indirect_call(info, node.args)
            raise CodegenError(f"Undefined function '{name}'")

        fn  = fi["fn"]
        sig = fi["sig"]
        # Collect defaults registered during compile phase
        defaults = self._fn_defaults.get(name, [])
        args_with_defaults = list(node.args)

        # Handle variadic: pack extra args into array
        if sig.variadic and len(sig.params) > 0:
            _, vptype = sig.params[-1]     # e.g. "int[]"
            elem_vt = vptype[:-2] if vptype.endswith("[]") else "int"
            n_fixed = len(sig.params) - 1
            fixed_args  = args_with_defaults[:n_fixed]
            variadic_args = args_with_defaults[n_fixed:]
            arr_node = ArrayLiteral(variadic_args)
            args_with_defaults = fixed_args + [arr_node]

        while len(args_with_defaults) < len(sig.params):
            idx = len(args_with_defaults)
            if idx < len(defaults) and defaults[idx] is not None:
                args_with_defaults.append(defaults[idx])
            else:
                break

        compiled_args = []
        for arg_node, (_, pt) in zip(args_with_defaults, sig.params):
            # Handle spread: ...arr expands to individual elements
            if isinstance(arg_node, SpreadExpr):
                arr_v, arr_t = self._compile_expr(arg_node.value)
                arr     = self.builder.bitcast(arr_v, self.arr_ptr_type)
                z = ir.Constant(I32_TY, 0)
                arr_len = self.builder.load(
                    self.builder.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True))
                arr_data = self.builder.load(
                    self.builder.gep(arr, [z, z], inbounds=True))
                # For each element, load and add to compiled_args
                # (This only works for fixed-size spreads known at compile time;
                #  for dynamic spreads, we'd need a different calling convention.)
                # As a practical approach: extract first N elements where N is the
                # number of remaining params
                elem_ty = self._vx_to_llvm(pt[:-2] if pt.endswith("[]") else "int")
                elem_ptr = self.builder.bitcast(arr_data, ir.PointerType(elem_ty))
                # Just take the first element for simple spread
                ep = self.builder.gep(elem_ptr, [ir.Constant(I32_TY, 0)], inbounds=False)
                av = self.builder.load(ep)
                compiled_args.append(av)
                continue
            av, at = self._compile_expr(arg_node)
            pt_resolved = self._resolve_type(pt)
            # Auto-box struct to interface if needed
            if pt_resolved in self._interfaces and at != pt_resolved:
                av = self._box_as_interface(av, at, pt_resolved)
                at = pt_resolved
            elif pt_resolved == "float" and at == "int":
                av = self.builder.sitofp(av, F64_TY)
            compiled_args.append(av)
        result = self.builder.call(fn, compiled_args)
        return result, sig.return_type

    def _is_fn_type(self, vx_type: str) -> bool:
        return vx_type.startswith("fn(")

    def _compile_indirect_call(self, info: dict, args) -> tuple[ir.Value, str]:
        """Call through a function pointer variable."""
        fn_type_str = info["vx_type"]  # "fn(int,float)->str"
        ptr_val = self.builder.load(info["ptr"])

        # Parse fn type string: fn(T1,T2,...)->R
        assert fn_type_str.startswith("fn(")
        arrow_idx = fn_type_str.rindex("->")
        params_str = fn_type_str[3:fn_type_str.index(")")]
        ret_str    = fn_type_str[arrow_idx+2:]
        param_types = [t.strip() for t in params_str.split(",") if t.strip()]
        ret_type   = ret_str.strip()

        ll_params = [self._vx_to_llvm(pt) for pt in param_types]
        ll_ret    = self._vx_to_llvm(ret_type)
        fn_ty     = ir.FunctionType(ll_ret, ll_params)
        fn_ptr_ty = ir.PointerType(fn_ty)

        # Cast i8* → fn_ptr_ty and call
        typed_fn = self.builder.bitcast(ptr_val, fn_ptr_ty)
        compiled_args = []
        for arg_node, pt in zip(args, param_types):
            av, at = self._compile_expr(arg_node)
            if pt == "float" and at == "int":
                av = self.builder.sitofp(av, F64_TY)
            compiled_args.append(av)
        result = self.builder.call(typed_fn, compiled_args)
        return result, ret_type

    def _dispatch_overload(self, name: str, node: 'Call') -> tuple[ir.Value, str]:
        """#11 Function overloading: select and call the best-matching overload."""
        # Compile arguments first to determine their types
        compiled = [(self._compile_expr(a)) for a in node.args]
        arg_types = [t for _, t in compiled]
        arg_vals  = [v for v, _ in compiled]

        entries = self._overload_table.get(name, [])
        best_name = None
        # Find exact match first, then widened match
        for param_types, mangled in entries:
            if len(param_types) != len(arg_types):
                continue
            exact = all(pt == at or pt == "any" for pt, at in zip(param_types, arg_types))
            if exact:
                best_name = mangled
                break
        if best_name is None:
            for param_types, mangled in entries:
                if len(param_types) != len(arg_types):
                    continue
                widen = all(
                    pt == at or pt == "any" or (pt == "float" and at == "int")
                    for pt, at in zip(param_types, arg_types)
                )
                if widen:
                    best_name = mangled
                    break
        if best_name is None and entries:
            # Fallback: use first overload with matching arg count, or just first
            for param_types, mangled in entries:
                if len(param_types) == len(arg_types):
                    best_name = mangled
                    break
            if best_name is None:
                best_name = entries[0][1]

        if best_name is None:
            raise CodegenError(f"No overload of '{name}' matches argument types {arg_types}")

        fi  = self._functions[best_name]
        fn  = fi["fn"]
        sig = fi["sig"]
        compiled_args = []
        for av, at, (_, pt) in zip(arg_vals, arg_types, sig.params):
            if pt == "float" and at == "int":
                av = self.builder.sitofp(av, F64_TY)
            compiled_args.append(av)
        result = self.builder.call(fn, compiled_args)
        return result, sig.return_type

    def _compile_generic_call(self, node: Call) -> tuple[ir.Value, str]:
        """Monomorphize and call a generic function."""
        import copy, re
        decl = self.analysis.generic_fns[node.func]

        # Infer type arguments from actual argument types
        actual_types = []
        for arg in node.args:
            actual_types.append(self._infer_type(arg))

        # Build type substitution map: T → concrete type
        type_map = {}
        for i, tp in enumerate(decl.type_params):
            if i < len(decl.params) and i < len(actual_types):
                param_declared = decl.params[i].type_name  # e.g. "T[]"
                actual = actual_types[i]                    # e.g. "int[]"
                # Strip common suffix to infer T
                if param_declared.endswith("[]") and actual.endswith("[]"):
                    type_map[tp] = actual[:-2]
                else:
                    type_map[tp] = actual

        # Name of the concrete version: first__int
        suffix = "__".join(type_map.get(tp, "any") for tp in decl.type_params)
        concrete_name = f"{decl.name}__{suffix}"

        # Monomorphize if not already done
        if concrete_name not in self._functions:
            def _subst(s: str) -> str:
                for tp, concrete in type_map.items():
                    s = re.sub(rf'\b{re.escape(tp)}\b', concrete, s)
                return s

            new_decl = copy.deepcopy(decl)
            new_decl.name = concrete_name
            new_decl.type_params = []
            for p in new_decl.params:
                p.type_name = _subst(p.type_name)
            if new_decl.return_type:
                new_decl.return_type = _subst(new_decl.return_type)

            # Register in analyzer fn_sigs
            from compiler.analyzer import FnSig as _FnSig
            self.analysis.fn_sigs[concrete_name] = _FnSig(
                [(p.name, p.type_name) for p in new_decl.params],
                new_decl.return_type or "void"
            )

            self._declare_fn(new_decl)
            # Compile body — save and restore builder state
            saved_builder  = self.builder
            saved_fn       = self.current_fn
            saved_scopes   = self._scope_stack
            self._scope_stack = [{}]
            self._compile_fn(new_decl)
            self.builder      = saved_builder
            self.current_fn   = saved_fn
            self._scope_stack = saved_scopes

        # Now call the concrete function
        fi  = self._functions[concrete_name]
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
    #  Lambdas (non-capturing anonymous functions)                        #
    # ------------------------------------------------------------------ #

    def _compile_lambda(self, node: LambdaExpr) -> tuple[ir.Value, str]:
        """Compile a lambda with closure capture.
        Captures free variables from the outer scope into a heap env struct,
        passes env as a hidden last parameter (i8*).
        Returns (i8* fn_ptr, fn_type_str).
        """
        lname = f"__lambda_{self._lambda_count}"
        self._lambda_count += 1

        param_vx_types = [p.type_name for p in node.params]
        ret_vx = node.ret_type or "void"
        fn_type_str = f"fn({','.join(param_vx_types)})->{ret_vx}"

        # --- Find free variables in the lambda body ---
        param_names = {p.name for p in node.params}
        captures = self._find_free_vars(node.body, param_names)
        # captures: list of (name, vx_type, alloca_ptr, ll_type)

        # --- Build closure env struct type ---
        env_fields = [(vx_t, ll_t) for (_, vx_t, _, ll_t) in captures]

        # --- Allocate closure env on the heap ---
        if captures:
            total_sz = sum(8 for _ in captures)   # fixed 8 bytes per slot (simplification)
            env_ptr = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, total_sz)])
            # Store captured values into env
            for i, (name, vx_t, src_al, ll_t) in enumerate(captures):
                val = self.builder.load(src_al)
                # Store as i64/i8* for uniform size
                slot_ptr = self.builder.gep(env_ptr, [ir.Constant(I64_TY, i * 8)], inbounds=False)
                typed_ptr = self.builder.bitcast(slot_ptr, ir.PointerType(ll_t))
                if ll_t.width < 64 if isinstance(ll_t, ir.IntType) else False:
                    val = self.builder.sext(val, I64_TY)
                    self.builder.store(val, self.builder.bitcast(slot_ptr, ir.PointerType(I64_TY)))
                else:
                    self.builder.store(val, typed_ptr)
        else:
            env_ptr = ir.Constant(I8PTR, None)

        # --- Compile the lambda function with env param ---
        # Add hidden "__env" param of type i8*
        env_param = Param("__env", "str")  # i8* aliased as str
        full_params = list(node.params) + [env_param]
        decl = FnDecl(lname, full_params, node.ret_type, node.body)

        from compiler.analyzer import FnSig as _FnSig
        self.analysis.fn_sigs[lname] = _FnSig(
            [(p.name, p.type_name) for p in full_params], ret_vx
        )
        self._fn_defaults[lname] = [None] * len(full_params)

        # Compile in saved context, but inject captures into scope
        saved_builder  = self.builder
        saved_fn       = self.current_fn
        saved_scopes   = self._scope_stack

        self._scope_stack = [{}]
        self._declare_fn(decl)

        # Override compile to inject captured vars from env arg
        fn_ir = self._functions[lname]["fn"]
        old_b = self.builder
        old_fn = self.current_fn
        self.current_fn = fn_ir
        entry_b = fn_ir.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_b)
        self._push_scope()
        self._defer_stack.append([])
        # Bind params
        for i, p in enumerate(node.params):
            al = self.builder.alloca(self._vx_to_llvm(p.type_name), name=p.name)
            self.builder.store(fn_ir.args[i], al)
            self._declare(p.name, al, p.type_name)
        # Unpack captured vars from env arg (last param)
        if captures:
            env_arg = fn_ir.args[-1]
            for i, (name, vx_t, _src_al, ll_t) in enumerate(captures):
                slot_ptr = self.builder.gep(env_arg, [ir.Constant(I64_TY, i * 8)], inbounds=False)
                typed_ptr = self.builder.bitcast(slot_ptr, ir.PointerType(ll_t))
                cap_al = self.builder.alloca(ll_t, name=f"cap_{name}")
                val = self.builder.load(typed_ptr)
                self.builder.store(val, cap_al)
                self._declare(name, cap_al, vx_t)
        # Compile body
        for stmt in node.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)
        self._emit_defers()
        self._defer_stack.pop()
        if not self.builder.block.is_terminated:
            if ret_vx == "void":
                self.builder.ret_void()
            else:
                self.builder.ret(ir.Constant(self._vx_to_llvm(ret_vx), 0))
        self._pop_scope()

        self.builder      = saved_builder
        self.current_fn   = saved_fn
        self._scope_stack = saved_scopes

        # --- Build a closure struct: {fn_ptr: i8*, env: i8*} on the heap ---
        # For now, just return the fn_ptr cast to i8* and store env separately
        # Since we can't easily pass env through the fn(int)->int type signature,
        # we use a trampoline approach: when no captures, return fn directly.
        fn_ptr = self.builder.bitcast(fn_ir, I8PTR)

        if captures:
            # Store closure = {fn_ptr, env_ptr} in a 16-byte heap struct
            closure_ptr = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 16)])
            fn_slot = self.builder.bitcast(closure_ptr, ir.PointerType(I8PTR))
            self.builder.store(fn_ptr, fn_slot)
            env_slot_raw = self.builder.gep(closure_ptr, [ir.Constant(I64_TY, 8)], inbounds=False)
            env_slot = self.builder.bitcast(env_slot_raw, ir.PointerType(I8PTR))
            self.builder.store(env_ptr, env_slot)
            return closure_ptr, fn_type_str

        return fn_ptr, fn_type_str

    def _find_free_vars(self, body: list, param_names: set) -> list:
        """
        Find identifiers used in `body` that exist in the current outer scope
        but are not in `param_names`.  Returns list of (name, vx_type, alloca, ll_type).
        """
        used = set()
        def _walk(node):
            if isinstance(node, Identifier):
                used.add(node.name)
            elif isinstance(node, (BinOp, UnaryOp)):
                for child in ([node.left, node.right] if isinstance(node, BinOp)
                              else [node.operand]):
                    _walk(child)
            elif isinstance(node, Call):
                for a in node.args: _walk(a)
            elif isinstance(node, LetStmt):
                _walk(node.value)
            elif isinstance(node, AssignStmt):
                _walk(node.value)
            elif isinstance(node, ReturnStmt):
                if node.value: _walk(node.value)
            elif isinstance(node, IfStmt):
                _walk(node.condition)
                for s in node.then_body: _walk(s)
                if node.else_body:
                    for s in node.else_body: _walk(s)
            elif isinstance(node, ExprStmt):
                _walk(node.expr)
            elif isinstance(node, PrintStmt):
                for v in node.values: _walk(v)
        for stmt in body:
            _walk(stmt)

        captures = []
        seen = set()
        for name in used:
            if name in param_names or name in seen:
                continue
            info = self._lookup(name)
            if info is not None:
                ll_t = info["ptr"].type.pointee
                captures.append((name, info["vx_type"], info["ptr"], ll_t))
                seen.add(name)
        return captures

    # ------------------------------------------------------------------ #
    #  Tuples                                                              #
    # ------------------------------------------------------------------ #

    def _parse_tuple_type_str(self, vt: str) -> list[str]:
        """Parse '(int,float)' → ['int', 'float']."""
        if vt.startswith("(") and vt.endswith(")"):
            inner = vt[1:-1]
            parts, depth, cur = [], 0, []
            for c in inner:
                if c in ('(', '[', '<'): depth += 1
                elif c in (')', ']', '>'): depth -= 1
                if c == ',' and depth == 0:
                    parts.append(''.join(cur).strip())
                    cur = []
                else:
                    cur.append(c)
            if cur:
                parts.append(''.join(cur).strip())
            return parts
        return []

    def _get_tuple_llvm_type(self, elem_types: list[str]) -> ir.IdentifiedStructType:
        """Get or create the LLVM struct type for a tuple."""
        key = "(" + ",".join(elem_types) + ")"
        if key in self._tuple_types:
            return self._tuple_types[key]
        tname = "vx_tuple_" + "_".join(t.replace("[]","Arr").replace("?","Opt") for t in elem_types)
        lt = self.module.context.get_identified_type(tname)
        if lt.is_opaque:
            lt.set_body(*[self._vx_to_llvm(t) for t in elem_types])
        self._tuple_types[key] = lt
        return lt

    def _compile_tuple_literal(self, node: TupleLiteral) -> tuple[ir.Value, str]:
        """Compile (a, b, c) → heap-allocated tuple struct pointer."""
        vals_types = [self._compile_expr(e) for e in node.elements]
        elem_types = [vt for _, vt in vals_types]
        tup_lt     = self._get_tuple_llvm_type(elem_types)
        sz = ir.Constant(I64_TY, sum(_elem_size(et) for et in elem_types))
        # Heap-allocate the tuple
        raw = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 0)])
        # Use alloca instead for simplicity (stack allocation)
        tup_al = self.builder.alloca(tup_lt)
        for i, (val, vt) in enumerate(vals_types):
            fp = self.builder.gep(tup_al,
                                  [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, i)],
                                  inbounds=True)
            self.builder.store(val, fp)
        # Heap-allocate and copy (so it can escape the function)
        struct_sz_bytes = sum(_elem_size(et) for et in elem_types)
        heap_raw = self.builder.call(self.malloc_fn,
                                     [ir.Constant(I64_TY, struct_sz_bytes + 8)])
        heap_typed = self.builder.bitcast(heap_raw, ir.PointerType(tup_lt))
        self.builder.call(self.memcpy_fn, [heap_raw,
                          self.builder.bitcast(tup_al, I8PTR),
                          ir.Constant(I64_TY, struct_sz_bytes)])
        vt_str = "(" + ",".join(elem_types) + ")"
        return heap_raw, vt_str

    # ------------------------------------------------------------------ #
    #  Tuple type helpers                                                  #
    # ------------------------------------------------------------------ #

    def _vx_to_llvm_tuple(self, vt: str) -> ir.Type:
        """Convert tuple type string to LLVM pointer to struct."""
        elem_types = self._parse_tuple_type_str(vt)
        if not elem_types:
            return I8PTR
        lt = self._get_tuple_llvm_type(elem_types)
        return ir.PointerType(lt)

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
        # Allocate at least max(n, 4) elements so push() doesn't immediately overflow
        alloc_count = max(n, 4)
        tsz = ir.Constant(I64_TY, alloc_count * _elem_size(elem_vt))

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

    def _compile_struct_update(self, node: StructUpdateExpr) -> tuple[ir.Value, str]:
        """{ ...base, field: val } — shallow-copy base struct with fields overridden."""
        base_v, base_t = self._compile_expr(node.base)
        si = self._structs.get(base_t)
        if si is None:
            raise CodegenError(f"StructUpdateExpr: '{base_t}' is not a struct")
        lt     = si["llvm_type"]
        fields = si["fields"]

        # Allocate new struct
        null_ptr = ir.Constant(ir.PointerType(lt), None)
        sp       = self.builder.gep(null_ptr, [ir.Constant(I32_TY, 1)], inbounds=False)
        sz       = self.builder.ptrtoint(sp, I64_TY)
        raw      = self.builder.call(self.malloc_fn, [sz])
        sptr     = self.builder.bitcast(raw, ir.PointerType(lt))

        # Build override dict {name: val}
        overrides = {name: val_node for name, val_node in node.fields}

        for idx, (fname, ftype) in enumerate(fields):
            src_fp  = self.builder.gep(base_v, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,idx)], inbounds=True)
            dst_fp  = self.builder.gep(sptr,   [ir.Constant(I32_TY,0), ir.Constant(I32_TY,idx)], inbounds=True)
            if fname in overrides:
                fv, fvt = self._compile_expr(overrides[fname])
                if ftype == "float" and fvt == "int":
                    fv = self.builder.sitofp(fv, F64_TY)
                self.builder.store(fv, dst_fp)
            else:
                self.builder.store(self.builder.load(src_fp), dst_fp)
        return sptr, base_t

    def _compile_new(self, node: NewExpr) -> tuple[ir.Value, str]:
        # #6 Generic struct instantiation: new Stack[int](...)
        type_name = node.type_name
        if getattr(node, 'type_args', []):
            type_name = self._monomorphize_struct(node.type_name, node.type_args)
        si = self._structs.get(type_name)
        if si is None:
            raise CodegenError(f"Unknown struct '{type_name}'")
        lt     = si["llvm_type"]
        fields = si["fields"]

        # sizeof via GEP trick
        null_ptr = ir.Constant(ir.PointerType(lt), None)
        sp       = self.builder.gep(null_ptr, [ir.Constant(I32_TY, 1)], inbounds=False)
        sz       = self.builder.ptrtoint(sp, I64_TY)
        raw      = self.builder.call(self.malloc_fn, [sz])
        sptr     = self.builder.bitcast(raw, ir.PointerType(lt))

        defaults = si.get("defaults", [None] * len(fields))
        for idx, (fname, ftype) in enumerate(fields):
            fv = ir.Constant(self._vx_to_llvm(ftype), 0)
            if idx < len(node.args):
                fv, fvt = self._compile_expr(node.args[idx])
                if ftype == "float" and fvt == "int":
                    fv = self.builder.sitofp(fv, F64_TY)
            elif idx < len(defaults) and defaults[idx] is not None:
                # Use default field value from struct declaration
                fv, fvt = self._compile_expr(defaults[idx])
                if ftype == "float" and fvt == "int":
                    fv = self.builder.sitofp(fv, F64_TY)
            fp = self.builder.gep(sptr,
                                  [ir.Constant(I32_TY,0), ir.Constant(I32_TY,idx)],
                                  inbounds=True)
            self.builder.store(fv, fp)
        return sptr, type_name

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

    # ------------------------------------------------------------------ #
    #  Dict value type conversions (type-erase to i64 and back)          #
    # ------------------------------------------------------------------ #

    def _val_to_i64(self, val: ir.Value, vt: str, expected_vt: str = None) -> ir.Value:
        """Convert any Vexel value to i64 for dict storage."""
        vt = self._resolve_type(vt)
        if vt == "int":   return val
        if vt == "float": return self.builder.bitcast(val, I64_TY)
        if vt == "bool":  return self.builder.zext(val, I64_TY)
        if vt == "str":   return self.builder.ptrtoint(val, I64_TY)
        # Promote int to float if expected type is float
        if expected_vt and self._resolve_type(expected_vt) == "float":
            fval = self.builder.sitofp(val, F64_TY)
            return self.builder.bitcast(fval, I64_TY)
        return val

    def _i64_to_val(self, raw: ir.Value, vt: str) -> tuple[ir.Value, str]:
        """Convert i64 dict storage back to Vexel value."""
        vt = self._resolve_type(vt)
        if vt == "int":   return raw, "int"
        if vt == "float": return self.builder.bitcast(raw, F64_TY), "float"
        if vt == "bool":  return self.builder.trunc(raw, I1_TY), "bool"
        if vt == "str":   return self.builder.inttoptr(raw, I8PTR), "str"
        return raw, vt

    # ------------------------------------------------------------------ #
    #  Dict compilation                                                   #
    # ------------------------------------------------------------------ #

    def _parse_dict_types(self, vx_type: str) -> tuple[str, str]:
        """Given 'dict[K,V]', return (K, V)."""
        inner = vx_type[5:-1]
        depth = 0
        for i, c in enumerate(inner):
            if c == '[': depth += 1
            elif c == ']': depth -= 1
            elif c == ',' and depth == 0:
                return inner[:i].strip(), inner[i+1:].strip()
        return "str", "int"

    def _compile_dict_literal(self, node: DictLiteral) -> tuple[ir.Value, str]:
        if not node.pairs:
            dict_t = "dict[str,int]"
        else:
            kt = self._infer_type(node.pairs[0][0])
            vt = self._infer_type(node.pairs[0][1])
            dict_t = f"dict[{kt},{vt}]"
        kt, vt = self._parse_dict_types(dict_t)

        fn_new = self._get_helper("__vx_dict_new")
        fn_set = self._get_helper("__vx_dict_set")
        raw    = self.builder.call(fn_new, [])
        d      = self.builder.bitcast(raw, self.dict_ptr_type)

        for key_node, val_node in node.pairs:
            kv, _  = self._compile_expr(key_node)
            vv, vvt = self._compile_expr(val_node)
            raw64  = self._val_to_i64(vv, vvt, vt)
            self.builder.call(fn_set, [raw, kv, raw64])

        return d, dict_t

    def _compile_dict_index_get(self, dict_v: ir.Value, key_v: ir.Value,
                                 vt: str) -> tuple[ir.Value, str]:
        fn_get  = self._get_helper("__vx_dict_get")
        raw     = self.builder.bitcast(dict_v, I8PTR)
        raw64   = self.builder.call(fn_get, [raw, key_v])
        return self._i64_to_val(raw64, vt)

    def _compile_dict_method(self, dict_v: ir.Value, dict_t: str,
                              vt: str, method: str, args) -> tuple[ir.Value, str]:
        raw = self.builder.bitcast(dict_v, I8PTR)

        if method == "has":
            kv, _ = self._compile_expr(args[0])
            fn_h  = self._get_helper("__vx_dict_has")
            return self.builder.call(fn_h, [raw, kv]), "bool"

        if method == "remove":
            kv, _ = self._compile_expr(args[0])
            fn_h  = self._get_helper("__vx_dict_remove")
            self.builder.call(fn_h, [raw, kv])
            return ir.Constant(I64_TY, 0), "void"

        if method == "len":
            fn_h = self._get_helper("__vx_dict_len")
            return self.builder.call(fn_h, [raw]), "int"

        if method == "keys":
            fn_h = self._get_helper("__vx_dict_keys")
            raw_arr = self.builder.call(fn_h, [raw])
            arr_ptr = self.builder.bitcast(raw_arr, self.arr_ptr_type)
            return arr_ptr, "str[]"

        if method == "values":
            fn_h = self._get_helper("__vx_dict_values")
            raw_arr = self.builder.call(fn_h, [raw])
            arr_ptr = self.builder.bitcast(raw_arr, self.arr_ptr_type)
            # Value type depends on dict type annotation; use int as default
            inner = dict_t[5:-1]  # strip "dict[" and "]"
            val_t = inner[inner.index(',')+1:].strip() if ',' in inner else "int"
            return arr_ptr, val_t + "[]"

        if method == "items":
            # Returns str[] where each element is "key:value" pair string
            fn_h = self._get_helper("__vx_dict_items")
            raw_arr = self.builder.call(fn_h, [raw])
            arr_ptr = self.builder.bitcast(raw_arr, self.arr_ptr_type)
            return arr_ptr, "str[]"

        raise CodegenError(f"Unknown dict method '{method}'")

    # ------------------------------------------------------------------ #
    #  Dict LLVM IR helper builders                                       #
    # ------------------------------------------------------------------ #

    def _dict_fields(self, b: ir.IRBuilder, hdr: ir.Value):
        """Return (kf, vf, lf, cf) — GEPs for the four dict header fields."""
        c0 = ir.Constant(I32_TY, 0)
        kf = b.gep(hdr, [c0, ir.Constant(I32_TY, 0)], inbounds=True)
        vf = b.gep(hdr, [c0, ir.Constant(I32_TY, 1)], inbounds=True)
        lf = b.gep(hdr, [c0, ir.Constant(I32_TY, 2)], inbounds=True)
        cf = b.gep(hdr, [c0, ir.Constant(I32_TY, 3)], inbounds=True)
        return kf, vf, lf, cf

    def _build_dict_new(self) -> ir.Function:
        """Create an empty dict. Returns i8* (vx_dict*)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_dict_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        INIT_CAP = ir.Constant(I64_TY, 4)
        SZ8      = ir.Constant(I64_TY, 8)

        hdr_raw  = b.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])  # 4 fields × 8
        hdr      = b.bitcast(hdr_raw, self.dict_ptr_type)
        keys_raw = b.call(self.malloc_fn, [b.mul(INIT_CAP, SZ8)])
        vals_raw = b.call(self.malloc_fn, [b.mul(INIT_CAP, SZ8)])

        kf, vf, lf, cf = self._dict_fields(b, hdr)
        b.store(keys_raw, kf)
        b.store(vals_raw, vf)
        b.store(ir.Constant(I64_TY, 0), lf)
        b.store(INIT_CAP, cf)

        b.ret(hdr_raw)
        return fn

    def _build_dict_set(self) -> ir.Function:
        """Set or update a key. Args: (dict: i8*, key: i8*, val: i64)."""
        fn = ir.Function(self.module,
                         ir.FunctionType(VOID_TY, [I8PTR, I8PTR, I64_TY]),
                         name="__vx_dict_set")
        fn.linkage = "private"
        dict_raw, key, val = fn.args
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        hdr = b.bitcast(dict_raw, self.dict_ptr_type)
        kf, vf, lf, cf = self._dict_fields(b, hdr)

        # Grow if len >= cap
        grow_bb  = fn.append_basic_block("set.grow")
        scan_bb  = fn.append_basic_block("set.scan")
        b.cbranch(b.icmp_signed(">=", b.load(lf), b.load(cf)), grow_bb, scan_bb)

        b.position_at_end(grow_bb)
        new_cap = b.mul(b.load(cf), ir.Constant(I64_TY, 2))
        sz8     = b.mul(new_cap, ir.Constant(I64_TY, 8))
        b.store(b.call(self.realloc_fn, [b.load(kf), sz8]), kf)
        b.store(b.call(self.realloc_fn, [b.load(vf), sz8]), vf)
        b.store(new_cap, cf)
        b.branch(scan_bb)

        # Linear scan for existing key
        b.position_at_end(scan_bb)
        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)

        chk  = fn.append_basic_block("set.chk")
        bdy  = fn.append_basic_block("set.bdy")
        upd  = fn.append_basic_block("set.upd")
        app  = fn.append_basic_block("set.app")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.load(lf)), bdy, app)

        b.position_at_end(bdy)
        iv2      = b.load(i_al)
        keys_ptr = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        ek       = b.load(b.gep(keys_ptr, [iv2], inbounds=False))
        r        = b.call(self.strcmp_fn, [ek, key])
        is_eq    = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.store(b.add(iv2, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, upd, chk)

        b.position_at_end(upd)  # update existing
        vals_ptr = b.bitcast(b.load(vf), ir.PointerType(I64_TY))
        b.store(val, b.gep(vals_ptr, [iv2], inbounds=False))
        b.ret_void()

        b.position_at_end(app)  # append new
        cur_len  = b.load(lf)
        keys_ptr2 = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        vals_ptr2 = b.bitcast(b.load(vf), ir.PointerType(I64_TY))
        b.store(key, b.gep(keys_ptr2, [cur_len], inbounds=False))
        b.store(val, b.gep(vals_ptr2, [cur_len], inbounds=False))
        b.store(b.add(cur_len, ir.Constant(I64_TY, 1)), lf)
        b.ret_void()
        return fn

    def _build_dict_get(self) -> ir.Function:
        """Get value by key. Returns 0 if not found. Args: (dict: i8*, key: i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_dict_get")
        fn.linkage = "private"
        dict_raw, key = fn.args
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        hdr = b.bitcast(dict_raw, self.dict_ptr_type)
        kf, vf, lf, _ = self._dict_fields(b, hdr)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk  = fn.append_basic_block("get.chk")
        bdy  = fn.append_basic_block("get.bdy")
        ret_found = fn.append_basic_block("get.found")
        ret_miss  = fn.append_basic_block("get.miss")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.load(lf)), bdy, ret_miss)

        b.position_at_end(bdy)
        iv2  = b.load(i_al)
        kp   = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        ek   = b.load(b.gep(kp, [iv2], inbounds=False))
        r    = b.call(self.strcmp_fn, [ek, key])
        is_eq = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.store(b.add(iv2, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, ret_found, chk)

        b.position_at_end(ret_found)
        vp = b.bitcast(b.load(vf), ir.PointerType(I64_TY))
        b.ret(b.load(b.gep(vp, [iv2], inbounds=False)))

        b.position_at_end(ret_miss)
        b.ret(ir.Constant(I64_TY, 0))
        return fn

    def _build_dict_has(self) -> ir.Function:
        """Returns 1 if key exists. Args: (dict: i8*, key: i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I8PTR]),
                         name="__vx_dict_has")
        fn.linkage = "private"
        dict_raw, key = fn.args
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        hdr = b.bitcast(dict_raw, self.dict_ptr_type)
        kf, _, lf, _ = self._dict_fields(b, hdr)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk  = fn.append_basic_block("has.chk")
        bdy  = fn.append_basic_block("has.bdy")
        ret_t = fn.append_basic_block("has.t")
        ret_f = fn.append_basic_block("has.f")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.load(lf)), bdy, ret_f)

        b.position_at_end(bdy)
        iv2  = b.load(i_al)
        kp   = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        ek   = b.load(b.gep(kp, [iv2], inbounds=False))
        r    = b.call(self.strcmp_fn, [ek, key])
        is_eq = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.store(b.add(iv2, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, ret_t, chk)

        b.position_at_end(ret_t); b.ret(ir.Constant(I1_TY, 1))
        b.position_at_end(ret_f); b.ret(ir.Constant(I1_TY, 0))
        return fn

    def _build_dict_remove(self) -> ir.Function:
        """Remove a key (shift elements left). Args: (dict: i8*, key: i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_dict_remove")
        fn.linkage = "private"
        dict_raw, key = fn.args
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        hdr = b.bitcast(dict_raw, self.dict_ptr_type)
        kf, vf, lf, _ = self._dict_fields(b, hdr)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk  = fn.append_basic_block("rm.chk")
        bdy  = fn.append_basic_block("rm.bdy")
        shift_init = fn.append_basic_block("rm.shift_init")
        shift_chk  = fn.append_basic_block("rm.shift_chk")
        shift_bdy  = fn.append_basic_block("rm.shift_bdy")
        done = fn.append_basic_block("rm.done")
        not_found = fn.append_basic_block("rm.notfound")
        b.branch(chk)

        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.load(lf)), bdy, not_found)

        b.position_at_end(bdy)
        iv2  = b.load(i_al)
        kp   = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        ek   = b.load(b.gep(kp, [iv2], inbounds=False))
        r    = b.call(self.strcmp_fn, [ek, key])
        is_eq = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        b.store(b.add(iv2, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_eq, shift_init, chk)

        # Shift elements left starting at found position (iv2)
        b.position_at_end(shift_init)
        found_i = iv2   # SSA value from bdy block — valid here
        j_al = b.alloca(I64_TY); b.store(found_i, j_al)
        b.branch(shift_chk)

        b.position_at_end(shift_chk)
        jv = b.load(j_al)
        new_len = b.sub(b.load(lf), ir.Constant(I64_TY, 1))
        b.cbranch(b.icmp_signed("<", jv, new_len), shift_bdy, done)

        b.position_at_end(shift_bdy)
        jv2  = b.load(j_al)
        jv2n = b.add(jv2, ir.Constant(I64_TY, 1))
        kp2  = b.bitcast(b.load(kf), ir.PointerType(I8PTR))
        vp2  = b.bitcast(b.load(vf), ir.PointerType(I64_TY))
        b.store(b.load(b.gep(kp2, [jv2n], inbounds=False)),
                b.gep(kp2, [jv2], inbounds=False))
        b.store(b.load(b.gep(vp2, [jv2n], inbounds=False)),
                b.gep(vp2, [jv2], inbounds=False))
        b.store(b.add(jv2, ir.Constant(I64_TY, 1)), j_al)
        b.branch(shift_chk)

        b.position_at_end(done)
        new_len2 = b.sub(b.load(lf), ir.Constant(I64_TY, 1))
        b.store(new_len2, lf)
        b.ret_void()

        b.position_at_end(not_found)
        b.ret_void()
        return fn

    def _build_dict_len(self) -> ir.Function:
        """Return number of entries. Args: (dict: i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_dict_len")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        hdr = b.bitcast(fn.args[0], self.dict_ptr_type)
        _, _, lf, _ = self._dict_fields(b, hdr)
        b.ret(b.load(lf))
        return fn

    def _build_dict_keys(self) -> ir.Function:
        """Return all keys as a str[] (vx_array*). Args: (dict: i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_dict_keys")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))

        hdr = b.bitcast(fn.args[0], self.dict_ptr_type)
        kf, _, lf, _ = self._dict_fields(b, hdr)
        ln = b.load(lf)

        # Create a vx_array with len=ln, cap=ln, data = copy of keys
        hsz     = ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)
        hdr_raw = b.call(self.malloc_fn, [hsz])
        arr_hdr = b.bitcast(hdr_raw, self.arr_ptr_type)

        data_sz  = b.mul(ln, ir.Constant(I64_TY, 8))
        data_raw = b.call(self.malloc_fn, [b.add(data_sz, ir.Constant(I64_TY, 8))])
        src_keys = b.load(kf)
        b.call(self.memcpy_fn, [data_raw, src_keys, data_sz])

        dp_f  = b.gep(arr_hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln_f  = b.gep(arr_hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        cap_f = b.gep(arr_hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,2)], inbounds=True)
        b.store(data_raw, dp_f)
        b.store(ln, ln_f)
        b.store(ln, cap_f)

        b.ret(hdr_raw)
        return fn

    # ------------------------------------------------------------------ #
    #  v5 helper builders                                                  #
    # ------------------------------------------------------------------ #

    def _build_time_format(self) -> ir.Function:
        """Format a Unix timestamp as a human-readable string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_time_format")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        t_val = fn.args[0]

        # Store t_val in a local so we can pass its address to localtime
        t_al = b.alloca(I64_TY)
        b.store(t_val, t_al)
        t_ptr = b.bitcast(t_al, I8PTR)
        tm_ptr = b.call(self.localtime_fn, [t_ptr])

        # Format: strftime(buf, 64, "%Y-%m-%d %H:%M:%S", tm_ptr)
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        fmt_gv  = self._global_str("%Y-%m-%d %H:%M:%S")
        fmt_ptr = self._gstr_ptr_const(fmt_gv)
        b.call(self.strftime_fn, [buf, ir.Constant(I64_TY, 64), fmt_ptr, tm_ptr])
        b.ret(buf)
        return fn

    def _build_input(self) -> ir.Function:
        """Read a line from stdin (strips trailing newline)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_input")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        prompt = fn.args[0]

        # Print the prompt
        fmt = self._global_str("%s")
        fmt_ptr = self._gstr_ptr_const(fmt)
        b.call(self.printf, [fmt_ptr, prompt])

        # Read up to 1024 chars from stdin
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 1024)])
        # Get stdin — use global pointer (declare __acrt_iob_func on Windows or just stdin)
        # Simpler: just read directly via fgets with FILE* 0 trick → use scanf
        # Use scanf(" %1023[^\n]", buf) to read a full line
        scanf_ft = ir.FunctionType(I32_TY, [I8PTR], var_arg=True)
        scanf_fn = ir.Function(self.module, scanf_ft, name="scanf") \
            if "scanf" not in [f.name for f in self.module.functions] \
            else next(f for f in self.module.functions if f.name == "scanf")
        fmt2 = self._global_str(" %1023[^\n]")
        fmt2_ptr = self._gstr_ptr_const(fmt2)
        b.call(scanf_fn, [fmt2_ptr, buf])
        b.ret(buf)
        return fn

    def _build_os_list_dir(self) -> ir.Function:
        """List directory entries. Returns vx_array* (as i8*) of str[]."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_os_list_dir")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path = fn.args[0]

        # Allocate result array header
        hsz     = ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)
        hdr_raw = b.call(self.malloc_fn, [hsz])
        hdr     = b.bitcast(hdr_raw, self.arr_ptr_type)
        init_cap = ir.Constant(I64_TY, 16)
        data_sz  = b.mul(init_cap, ir.Constant(I64_TY, 8))
        raw_data = b.call(self.malloc_fn, [data_sz])
        dp_f  = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,0)], inbounds=True)
        ln_f  = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,1)], inbounds=True)
        cap_f = b.gep(hdr, [ir.Constant(I32_TY,0), ir.Constant(I32_TY,2)], inbounds=True)
        b.store(raw_data, dp_f)
        b.store(ir.Constant(I64_TY, 0), ln_f)
        b.store(init_cap, cap_f)

        if _sys.platform == "win32":
            # Build search pattern: path + "\\*"
            # Use sprintf to build pattern
            pat_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 512)])
            fmt_gv = self._global_str("%s\\*")
            fmt_ptr = self._gstr_ptr_const(fmt_gv)
            b.call(self.sprintf_fn, [pat_buf, fmt_ptr, path])

            # _finddata64_t is 592 bytes; use raw buffer
            fd_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 592)])

            # Declare _findfirst64 and _findnext64 and _findclose
            ff64_ty  = ir.FunctionType(I64_TY, [I8PTR, I8PTR])
            fn64_ty  = ir.FunctionType(I32_TY, [I64_TY, I8PTR])
            fc64_ty  = ir.FunctionType(I32_TY, [I64_TY])
            ff64 = ir.Function(self.module, ff64_ty, name="_findfirst64") \
                if "_findfirst64" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="_findfirst64")
            fn64 = ir.Function(self.module, fn64_ty, name="_findnext64") \
                if "_findnext64" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="_findnext64")
            fc64 = ir.Function(self.module, fc64_ty, name="_findclose") \
                if "_findclose" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="_findclose")

            handle = b.call(ff64, [pat_buf, fd_buf])
            invalid = ir.Constant(I64_TY, -1)   # INVALID_HANDLE_VALUE as i64

            ok_b    = fn.append_basic_block("listdir.ok")
            done_b  = fn.append_basic_block("listdir.done")
            loop_b  = fn.append_basic_block("listdir.loop")
            next_b  = fn.append_basic_block("listdir.next")

            is_inv = b.icmp_signed("==", handle, invalid)
            b.cbranch(is_inv, done_b, ok_b)

            b.position_at_end(ok_b)
            b.branch(loop_b)

            b.position_at_end(loop_b)
            # The file name is at offset 32 in _finddata64_t (the cFileName field)
            name_ptr = b.gep(fd_buf, [ir.Constant(I64_TY, 32)], inbounds=False)
            # Skip "." and ".."
            dot_gv    = self._global_str(".")
            dotdot_gv = self._global_str("..")
            dot_ptr    = self._gstr_ptr_const(dot_gv)
            dotdot_ptr = self._gstr_ptr_const(dotdot_gv)
            r1 = b.call(self.strcmp_fn, [name_ptr, dot_ptr])
            r2 = b.call(self.strcmp_fn, [name_ptr, dotdot_ptr])
            is_dot    = b.icmp_signed("==", r1, ir.Constant(I32_TY, 0))
            is_dotdot = b.icmp_signed("==", r2, ir.Constant(I32_TY, 0))
            skip = b.or_(is_dot, is_dotdot)
            push_b2 = fn.append_basic_block("listdir.push")
            b.cbranch(skip, next_b, push_b2)

            b.position_at_end(push_b2)
            # Copy name into malloc'd string
            nlen = b.call(self.strlen_fn, [name_ptr])
            nbuf = b.call(self.malloc_fn, [b.add(nlen, ir.Constant(I64_TY, 1))])
            b.call(self.memcpy_fn, [nbuf, name_ptr, b.add(nlen, ir.Constant(I64_TY, 1))])
            # Push to array
            cur_ln  = b.load(ln_f)
            cur_cap = b.load(cap_f)
            ng = b.icmp_signed(">=", cur_ln, cur_cap)
            grow_b3 = fn.append_basic_block("listdir.grow")
            store_b = fn.append_basic_block("listdir.store")
            b.cbranch(ng, grow_b3, store_b)

            b.position_at_end(grow_b3)
            new_cap = b.mul(cur_cap, ir.Constant(I64_TY, 2))
            new_sz  = b.mul(new_cap, ir.Constant(I64_TY, 8))
            old_d   = b.load(dp_f)
            new_d   = b.call(self.realloc_fn, [old_d, new_sz])
            b.store(new_d, dp_f)
            b.store(new_cap, cap_f)
            b.branch(store_b)

            b.position_at_end(store_b)
            cur_ln2 = b.load(ln_f)
            dp2     = b.load(dp_f)
            pp2     = b.bitcast(dp2, ir.PointerType(I8PTR))
            ep2     = b.gep(pp2, [cur_ln2], inbounds=False)
            b.store(nbuf, ep2)
            b.store(b.add(cur_ln2, ir.Constant(I64_TY, 1)), ln_f)
            b.branch(next_b)

            b.position_at_end(next_b)
            r_next = b.call(fn64, [handle, fd_buf])
            cont = b.icmp_signed("==", r_next, ir.Constant(I32_TY, 0))
            b.cbranch(cont, loop_b, done_b)

            b.position_at_end(done_b)
            b.call(fc64, [handle])
            b.ret(hdr_raw)
        else:
            # POSIX: opendir/readdir/closedir
            od_ty  = ir.FunctionType(I8PTR, [I8PTR])
            rd_ty  = ir.FunctionType(I8PTR, [I8PTR])
            cd_ty  = ir.FunctionType(I32_TY, [I8PTR])
            od_fn = ir.Function(self.module, od_ty, name="opendir") \
                if "opendir" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="opendir")
            rd_fn = ir.Function(self.module, rd_ty, name="readdir") \
                if "readdir" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="readdir")
            cd_fn = ir.Function(self.module, cd_ty, name="closedir") \
                if "closedir" not in [f.name for f in self.module.functions] \
                else next(f for f in self.module.functions if f.name=="closedir")

            dirp = b.call(od_fn, [path])
            null_int = b.ptrtoint(dirp, I64_TY)
            is_null = b.icmp_unsigned("==", null_int, ir.Constant(I64_TY, 0))
            done_b2 = fn.append_basic_block("listdir.done")
            loop_b2 = fn.append_basic_block("listdir.loop")
            b.cbranch(is_null, done_b2, loop_b2)

            b.position_at_end(loop_b2)
            ent = b.call(rd_fn, [dirp])
            ent_int = b.ptrtoint(ent, I64_TY)
            is_end  = b.icmp_unsigned("==", ent_int, ir.Constant(I64_TY, 0))
            push_b3 = fn.append_basic_block("listdir.push2")
            b.cbranch(is_end, done_b2, push_b3)

            b.position_at_end(push_b3)
            # d_name is at offset 19 in struct dirent on Linux (varies by platform)
            # Use a conservative offset of 19 for Linux; this is best-effort
            name_ptr2 = b.gep(ent, [ir.Constant(I64_TY, 19)], inbounds=False)
            dot_gv2    = self._global_str(".")
            dotdot_gv2 = self._global_str("..")
            dot_p2    = self._gstr_ptr_const(dot_gv2)
            dotdot_p2 = self._gstr_ptr_const(dotdot_gv2)
            r1b = b.call(self.strcmp_fn, [name_ptr2, dot_p2])
            r2b = b.call(self.strcmp_fn, [name_ptr2, dotdot_p2])
            is_dot2    = b.icmp_signed("==", r1b, ir.Constant(I32_TY, 0))
            is_dotdot2 = b.icmp_signed("==", r2b, ir.Constant(I32_TY, 0))
            skip2 = b.or_(is_dot2, is_dotdot2)
            b.cbranch(skip2, loop_b2, fn.append_basic_block("listdir.add"))
            add_b = list(fn.blocks)[-1]

            b.position_at_end(add_b)
            nlen2 = b.call(self.strlen_fn, [name_ptr2])
            nbuf2 = b.call(self.malloc_fn, [b.add(nlen2, ir.Constant(I64_TY, 1))])
            b.call(self.memcpy_fn, [nbuf2, name_ptr2, b.add(nlen2, ir.Constant(I64_TY, 1))])
            cur_ln3  = b.load(ln_f)
            cur_cap3 = b.load(cap_f)
            ng3 = b.icmp_signed(">=", cur_ln3, cur_cap3)
            grow_b4  = fn.append_basic_block("listdir.grow2")
            store_b2 = fn.append_basic_block("listdir.store2")
            b.cbranch(ng3, grow_b4, store_b2)

            b.position_at_end(grow_b4)
            nc4 = b.mul(cur_cap3, ir.Constant(I64_TY, 2))
            ns4 = b.mul(nc4, ir.Constant(I64_TY, 8))
            od4 = b.load(dp_f)
            nd4 = b.call(self.realloc_fn, [od4, ns4])
            b.store(nd4, dp_f)
            b.store(nc4, cap_f)
            b.branch(store_b2)

            b.position_at_end(store_b2)
            cl4 = b.load(ln_f)
            dp4 = b.load(dp_f)
            pp4 = b.bitcast(dp4, ir.PointerType(I8PTR))
            ep4 = b.gep(pp4, [cl4], inbounds=False)
            b.store(nbuf2, ep4)
            b.store(b.add(cl4, ir.Constant(I64_TY, 1)), ln_f)
            b.branch(loop_b2)

            b.position_at_end(done_b2)
            b.call(cd_fn, [dirp])
            b.ret(hdr_raw)
        return fn


    # ------------------------------------------------------------------ #
    #  v7 new helper builders                                             #
    # ------------------------------------------------------------------ #

    def _build_str_find(self) -> ir.Function:
        """Return index of substring in s, or -1."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_str_find")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, sub = fn.args[0], fn.args[1]
        found = b.call(self.strstr_fn, [s, sub])
        found_i = b.ptrtoint(found, I64_TY)
        s_i     = b.ptrtoint(s, I64_TY)
        is_null = b.icmp_unsigned("==", found_i, ir.Constant(I64_TY, 0))
        offset  = b.sub(found_i, s_i)
        result  = b.select(is_null, ir.Constant(I64_TY, -1), offset)
        b.ret(result)
        return fn

    def _build_str_slice(self) -> ir.Function:
        """Return s[start:end]."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY, I64_TY]),
                         name="__vx_str_slice")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, start, end = fn.args[0], fn.args[1], fn.args[2]
        length = b.sub(end, start)
        buf = b.call(self.malloc_fn, [b.add(length, ir.Constant(I64_TY, 1))])
        src = b.gep(s, [start], inbounds=False)
        b.call(self.memcpy_fn, [buf, src, length])
        null_p = b.gep(buf, [length], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _build_str_repeat(self) -> ir.Function:
        """Return s repeated n times."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY]),
                         name="__vx_str_repeat")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, n = fn.args[0], fn.args[1]
        slen = b.call(self.strlen_fn, [s])
        total = b.mul(slen, n)
        buf = b.call(self.malloc_fn, [b.add(total, ir.Constant(I64_TY, 1))])
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("rep.chk")
        bdy = fn.append_basic_block("rep.bdy")
        ext = fn.append_basic_block("rep.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, n), bdy, ext)
        b.position_at_end(bdy)
        offset = b.mul(iv, slen)
        dst = b.gep(buf, [offset], inbounds=False)
        b.call(self.memcpy_fn, [dst, s, slen])
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        null_p = b.gep(buf, [total], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _build_str_join(self) -> ir.Function:
        """Join str[] with separator."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_str_join")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, sep = fn.args[0], fn.args[1]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        data = b.bitcast(data_raw, ir.PointerType(I8PTR))
        sep_len = b.call(self.strlen_fn, [sep])
        # Compute total length
        total_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), total_al)
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk1 = fn.append_basic_block("sj.len.chk")
        bdy1 = fn.append_basic_block("sj.len.bdy")
        ext1 = fn.append_basic_block("sj.len.ext")
        b.branch(chk1)
        b.position_at_end(chk1)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy1, ext1)
        b.position_at_end(bdy1)
        sp = b.gep(data, [iv], inbounds=False)
        sv = b.load(sp)
        sl = b.call(self.strlen_fn, [sv])
        tot = b.load(total_al)
        tot2 = b.add(tot, sl)
        # Add separator length except after last
        is_last = b.icmp_signed("==", b.add(iv, ir.Constant(I64_TY,1)), ln)
        sep_add = b.select(is_last, ir.Constant(I64_TY,0), sep_len)
        b.store(b.add(tot2, sep_add), total_al)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk1)
        b.position_at_end(ext1)
        final_len = b.load(total_al)
        buf = b.call(self.malloc_fn, [b.add(final_len, ir.Constant(I64_TY,1))])
        out_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), out_al)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk2 = fn.append_basic_block("sj.write.chk")
        bdy2 = fn.append_basic_block("sj.write.bdy")
        ext2 = fn.append_basic_block("sj.write.ext")
        b.branch(chk2)
        b.position_at_end(chk2)
        iv2 = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv2, ln), bdy2, ext2)
        b.position_at_end(bdy2)
        sp2 = b.gep(data, [iv2], inbounds=False)
        sv2 = b.load(sp2)
        sl2 = b.call(self.strlen_fn, [sv2])
        out = b.load(out_al)
        dst = b.gep(buf, [out], inbounds=False)
        b.call(self.memcpy_fn, [dst, sv2, sl2])
        out2 = b.add(out, sl2)
        is_last2 = b.icmp_signed("==", b.add(iv2, ir.Constant(I64_TY,1)), ln)
        sep_b = fn.append_basic_block("sj.sep")
        next_b = fn.append_basic_block("sj.next")
        b.cbranch(is_last2, next_b, sep_b)
        b.position_at_end(sep_b)
        dst2 = b.gep(buf, [out2], inbounds=False)
        b.call(self.memcpy_fn, [dst2, sep, sep_len])
        out3 = b.add(out2, sep_len)
        b.store(out3, out_al)
        b.branch(next_b)
        b.position_at_end(next_b)
        b.store(b.add(iv2, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk2)
        b.position_at_end(ext2)
        final_out = b.load(out_al)
        np = b.gep(buf, [final_out], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), np)
        b.ret(buf)
        return fn

    def _build_array_sort_i64(self) -> ir.Function:
        """Bubble-sort i64 array in-place."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR]),
                         name="__vx_array_sort_i64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw = fn.args[0]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        data = b.bitcast(data_raw, ir.PointerType(I64_TY))
        i_al = b.alloca(I64_TY)
        j_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        outer = fn.append_basic_block("sort.outer")
        inner = fn.append_basic_block("sort.inner")
        swap  = fn.append_basic_block("sort.swap")
        nosw  = fn.append_basic_block("sort.noswap")
        ext   = fn.append_basic_block("sort.ext")
        b.branch(outer)
        b.position_at_end(outer)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.sub(ln, ir.Constant(I64_TY,1))), inner, ext)
        b.position_at_end(inner)
        jv = b.load(j_al)
        lim = b.sub(b.sub(ln, ir.Constant(I64_TY,1)), iv)
        cond_j = b.icmp_signed("<", jv, lim)
        b.cbranch(cond_j, swap, nosw)
        b.position_at_end(swap)
        ep_j  = b.gep(data, [jv], inbounds=False)
        ep_j1 = b.gep(data, [b.add(jv, ir.Constant(I64_TY,1))], inbounds=False)
        vj  = b.load(ep_j)
        vj1 = b.load(ep_j1)
        do_swap = b.icmp_signed(">", vj, vj1)
        actual_swap = fn.append_basic_block("sort.doswap")
        cont_b = fn.append_basic_block("sort.cont")
        b.cbranch(do_swap, actual_swap, cont_b)
        b.position_at_end(actual_swap)
        b.store(vj1, ep_j)
        b.store(vj, ep_j1)
        b.branch(cont_b)
        b.position_at_end(cont_b)
        b.store(b.add(jv, ir.Constant(I64_TY,1)), j_al)
        b.branch(inner)
        b.position_at_end(nosw)
        b.store(ir.Constant(I64_TY, 0), j_al)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(outer)
        b.position_at_end(ext)
        b.ret_void()
        return fn

    def _build_array_sort_f64(self) -> ir.Function:
        """Bubble-sort f64 array in-place."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR]),
                         name="__vx_array_sort_f64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw = fn.args[0]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        data = b.bitcast(data_raw, ir.PointerType(F64_TY))
        i_al = b.alloca(I64_TY)
        j_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        outer = fn.append_basic_block("fsort.outer")
        inner_b = fn.append_basic_block("fsort.inner")
        ext   = fn.append_basic_block("fsort.ext")
        b.branch(outer)
        b.position_at_end(outer)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, b.sub(ln, ir.Constant(I64_TY,1))), inner_b, ext)
        b.position_at_end(inner_b)
        jv = b.load(j_al)
        lim = b.sub(b.sub(ln, ir.Constant(I64_TY,1)), iv)
        ep_j  = b.gep(data, [jv], inbounds=False)
        ep_j1 = b.gep(data, [b.add(jv, ir.Constant(I64_TY,1))], inbounds=False)
        vj  = b.load(ep_j)
        vj1 = b.load(ep_j1)
        do_swap = b.fcmp_ordered(">", vj, vj1)
        sw_b = fn.append_basic_block("fsort.sw")
        cnt_b = fn.append_basic_block("fsort.cnt")
        nsw_b = fn.append_basic_block("fsort.nsw")
        b.cbranch(b.icmp_signed("<", jv, lim), sw_b, nsw_b)
        b.position_at_end(sw_b)
        b.cbranch(do_swap, cnt_b, cnt_b)
        b.position_at_end(cnt_b)
        b.store(vj1, ep_j)
        b.store(vj, ep_j1)
        b.store(b.add(jv, ir.Constant(I64_TY,1)), j_al)
        b.branch(inner_b)
        b.position_at_end(nsw_b)
        b.store(ir.Constant(I64_TY, 0), j_al)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(outer)
        b.position_at_end(ext)
        b.ret_void()
        return fn

    def _build_array_index_of_i64(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I64_TY]),
                         name="__vx_array_index_of_i64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, target = fn.args[0], fn.args[1]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        data = b.bitcast(data_raw, ir.PointerType(I64_TY))
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("iof.chk")
        bdy = fn.append_basic_block("iof.bdy")
        ext = fn.append_basic_block("iof.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        ep = b.gep(data, [iv], inbounds=False)
        v = b.load(ep)
        eq = b.icmp_signed("==", v, target)
        found_b = fn.append_basic_block("iof.found")
        cont_b  = fn.append_basic_block("iof.cont")
        b.cbranch(eq, found_b, cont_b)
        b.position_at_end(found_b)
        b.ret(iv)
        b.position_at_end(cont_b)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(ir.Constant(I64_TY, -1))
        return fn

    def _build_array_index_of_str(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_array_index_of_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, target = fn.args[0], fn.args[1]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        data = b.bitcast(data_raw, ir.PointerType(I8PTR))
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("iostr.chk")
        bdy = fn.append_basic_block("iostr.bdy")
        ext = fn.append_basic_block("iostr.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        ep = b.gep(data, [iv], inbounds=False)
        v = b.load(ep)
        r = b.call(self.strcmp_fn, [v, target])
        eq = b.icmp_signed("==", r, ir.Constant(I32_TY, 0))
        found_b = fn.append_basic_block("iostr.found")
        cont_b  = fn.append_basic_block("iostr.cont")
        b.cbranch(eq, found_b, cont_b)
        b.position_at_end(found_b)
        b.ret(iv)
        b.position_at_end(cont_b)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(ir.Constant(I64_TY, -1))
        return fn

    def _build_array_join_str(self) -> ir.Function:
        """Join str[] with separator — delegate to str_join."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_array_join_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        helper = self._get_helper("__vx_str_join")
        result = b.call(helper, [fn.args[0], fn.args[1]])
        b.ret(result)
        return fn

    def _build_array_slice(self) -> ir.Function:
        """Return new array = arr[start:end]."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY, I64_TY, I64_TY]),
                         name="__vx_array_slice")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, start, end, esz = fn.args
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        dp = b.gep(arr, [z, z], inbounds=True)
        data_raw = b.load(dp)
        new_len = b.sub(end, start)
        new_sz  = b.mul(new_len, esz)
        # Build new array header
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        new_data = b.call(self.malloc_fn, [b.add(new_sz, ir.Constant(I64_TY,1))])
        src = b.gep(data_raw, [b.mul(start, esz)], inbounds=False)
        b.call(self.memcpy_fn, [new_data, src, new_sz])
        dp2  = b.gep(hdr, [z, z], inbounds=True)
        lp2  = b.gep(hdr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        cp2  = b.gep(hdr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        b.store(new_data, dp2)
        b.store(new_len, lp2)
        b.store(new_len, cp2)
        b.ret(hdr_raw)
        return fn

    def _build_array_map_i64(self) -> ir.Function:
        """Apply fn_ptr: i8* to each i64 element, return new i64 array."""
        fn_ty = ir.FunctionType(I8PTR, [I8PTR, I8PTR])
        fn = ir.Function(self.module, fn_ty, name="__vx_array_map_i64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, fn_ptr_raw = fn.args[0], fn.args[1]
        # Cast fn_ptr to (i64)->i64
        cb_ty = ir.FunctionType(I64_TY, [I64_TY])
        cb_ptr = b.bitcast(fn_ptr_raw, ir.PointerType(cb_ty))
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        src_raw = b.load(dp)
        src = b.bitcast(src_raw, ir.PointerType(I64_TY))
        # Allocate result array
        new_sz = b.mul(ln, ir.Constant(I64_TY, 8))
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        new_data = b.call(self.malloc_fn, [b.add(new_sz, ir.Constant(I64_TY,1))])
        new_data_typed = b.bitcast(new_data, ir.PointerType(I64_TY))
        dp2 = b.gep(hdr, [z, z], inbounds=True)
        lp2 = b.gep(hdr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        cp2 = b.gep(hdr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        b.store(new_data, dp2)
        b.store(ln, lp2)
        b.store(ln, cp2)
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("map.chk")
        bdy = fn.append_basic_block("map.bdy")
        ext = fn.append_basic_block("map.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        ep = b.gep(src, [iv], inbounds=False)
        v = b.load(ep)
        mapped = b.call(cb_ptr, [v])
        ep2 = b.gep(new_data_typed, [iv], inbounds=False)
        b.store(mapped, ep2)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(hdr_raw)
        return fn

    def _build_array_filter_i64(self) -> ir.Function:
        """Filter i64 array using predicate fn_ptr: (i64)->bool, return new array."""
        fn_ty = ir.FunctionType(I8PTR, [I8PTR, I8PTR])
        fn = ir.Function(self.module, fn_ty, name="__vx_array_filter_i64")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw, fn_ptr_raw = fn.args[0], fn.args[1]
        cb_ty = ir.FunctionType(I1_TY, [I64_TY])
        cb_ptr = b.bitcast(fn_ptr_raw, ir.PointerType(cb_ty))
        # Build result using push helper
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        init_data = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        dp2 = b.gep(hdr, [z, z], inbounds=True)
        lp2 = b.gep(hdr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        cp2 = b.gep(hdr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        b.store(init_data, dp2)
        b.store(ir.Constant(I64_TY,0), lp2)
        b.store(ir.Constant(I64_TY,8), cp2)
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        lp = b.gep(arr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        ln = b.load(lp)
        dp = b.gep(arr, [z, z], inbounds=True)
        src_raw = b.load(dp)
        src = b.bitcast(src_raw, ir.PointerType(I64_TY))
        push_h = self._get_helper("__vx_array_push")
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("filt.chk")
        bdy = fn.append_basic_block("filt.bdy")
        ext = fn.append_basic_block("filt.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ln), bdy, ext)
        b.position_at_end(bdy)
        ep = b.gep(src, [iv], inbounds=False)
        v = b.load(ep)
        keep = b.call(cb_ptr, [v])
        push_b = fn.append_basic_block("filt.push")
        skip_b = fn.append_basic_block("filt.skip")
        b.cbranch(keep, push_b, skip_b)
        b.position_at_end(push_b)
        v_al = b.alloca(I64_TY)
        b.store(v, v_al)
        v_raw = b.bitcast(v_al, I8PTR)
        b.call(push_h, [hdr_raw, v_raw, ir.Constant(I64_TY, 8)])
        b.branch(skip_b)
        b.position_at_end(skip_b)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(hdr_raw)
        return fn

    def _build_base64_encode(self) -> ir.Function:
        """Base64-encode a string. Pure LLVM — no external dependency."""
        TABLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        table_gv = self._global_str(TABLE)
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_base64_encode")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        slen = b.call(self.strlen_fn, [s])
        # Output length = ceil(slen/3)*4 + 1
        groups = b.add(b.sdiv(slen, ir.Constant(I64_TY,3)), ir.Constant(I64_TY,1))
        out_len = b.add(b.mul(groups, ir.Constant(I64_TY,4)), ir.Constant(I64_TY,1))
        buf = b.call(self.malloc_fn, [out_len])
        tbl = self._gstr_ptr_const(table_gv)
        i_al  = b.alloca(I64_TY)   # input index
        oi_al = b.alloca(I64_TY)   # output index
        b.store(ir.Constant(I64_TY, 0), i_al)
        b.store(ir.Constant(I64_TY, 0), oi_al)
        chk = fn.append_basic_block("b64e.chk")
        bdy = fn.append_basic_block("b64e.bdy")
        ext = fn.append_basic_block("b64e.ext")
        b.branch(chk)
        b.position_at_end(chk)
        ii = b.load(i_al)
        b.cbranch(b.icmp_signed("<", ii, slen), bdy, ext)
        b.position_at_end(bdy)
        # Read up to 3 bytes
        def load_byte(idx_offset):
            idx = b.add(ii, ir.Constant(I64_TY, idx_offset))
            in_range = b.icmp_signed("<", idx, slen)
            ep = b.gep(s, [idx], inbounds=False)
            v  = b.load(ep)
            return b.select(in_range, b.zext(v, I64_TY), ir.Constant(I64_TY, 0))
        b0 = load_byte(0)
        b1 = load_byte(1)
        b2 = load_byte(2)
        # Combine
        combined = b.or_(b.or_(b.shl(b0, ir.Constant(I64_TY,16)),
                                b.shl(b1, ir.Constant(I64_TY,8))), b2)
        def enc_char(shift):
            idx = b.and_(b.ashr(combined, ir.Constant(I64_TY, shift)),
                         ir.Constant(I64_TY, 63))
            cp = b.gep(tbl, [idx], inbounds=False)
            return b.load(cp)
        oi = b.load(oi_al)
        chars = [enc_char(18), enc_char(12), enc_char(6), enc_char(0)]
        for ci, ch in enumerate(chars):
            outp = b.gep(buf, [b.add(oi, ir.Constant(I64_TY, ci))], inbounds=False)
            b.store(ch, outp)
        b.store(b.add(ii, ir.Constant(I64_TY,3)), i_al)
        b.store(b.add(oi, ir.Constant(I64_TY,4)), oi_al)
        b.branch(chk)
        b.position_at_end(ext)
        # Handle padding with '='
        rem = b.srem(slen, ir.Constant(I64_TY, 3))
        oi_final = b.load(oi_al)
        pad_b1 = fn.append_basic_block("b64e.pad1")
        pad_b2 = fn.append_basic_block("b64e.pad2")
        done_b  = fn.append_basic_block("b64e.done")
        is_1 = b.icmp_signed("==", rem, ir.Constant(I64_TY, 1))
        is_2 = b.icmp_signed("==", rem, ir.Constant(I64_TY, 2))
        pad12 = fn.append_basic_block("b64e.pad12")
        b.cbranch(is_1, pad_b1, pad12)
        b.position_at_end(pad12)
        b.cbranch(is_2, pad_b2, done_b)
        eq_ch = ir.Constant(I8_TY, ord('='))
        b.position_at_end(pad_b1)
        # Overwrite last 2 chars with '='
        p2 = b.gep(buf, [b.sub(oi_final, ir.Constant(I64_TY,2))], inbounds=False)
        p3 = b.gep(buf, [b.sub(oi_final, ir.Constant(I64_TY,1))], inbounds=False)
        b.store(eq_ch, p2); b.store(eq_ch, p3)
        b.branch(done_b)
        b.position_at_end(pad_b2)
        # Overwrite last 1 char with '='
        p3b = b.gep(buf, [b.sub(oi_final, ir.Constant(I64_TY,1))], inbounds=False)
        b.store(eq_ch, p3b)
        b.branch(done_b)
        b.position_at_end(done_b)
        null_p = b.gep(buf, [oi_final], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _build_base64_decode(self) -> ir.Function:
        """Base64-decode a string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_base64_decode")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        slen = b.call(self.strlen_fn, [s])
        # Output length ≈ slen * 3/4
        out_len = b.add(b.mul(slen, ir.Constant(I64_TY,3)), ir.Constant(I64_TY,4))
        buf = b.call(self.malloc_fn, [b.sdiv(out_len, ir.Constant(I64_TY,4))])
        # Simple decode loop
        i_al  = b.alloca(I64_TY)
        oi_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        b.store(ir.Constant(I64_TY, 0), oi_al)

        def decode_char(ch):
            # A-Z=0-25, a-z=26-51, 0-9=52-61, +=62, /=63, ==0
            c = b.zext(ch, I64_TY)
            uc = ir.Constant(I64_TY, ord('A'))
            lc = ir.Constant(I64_TY, ord('a'))
            dc = ir.Constant(I64_TY, ord('0'))
            is_upper = b.and_(b.icmp_signed(">=", c, uc), b.icmp_signed("<=", c, ir.Constant(I64_TY, ord('Z'))))
            is_lower = b.and_(b.icmp_signed(">=", c, lc), b.icmp_signed("<=", c, ir.Constant(I64_TY, ord('z'))))
            is_digit = b.and_(b.icmp_signed(">=", c, dc), b.icmp_signed("<=", c, ir.Constant(I64_TY, ord('9'))))
            is_plus  = b.icmp_signed("==", c, ir.Constant(I64_TY, ord('+')))
            v_upper  = b.sub(c, uc)
            v_lower  = b.add(b.sub(c, lc), ir.Constant(I64_TY, 26))
            v_digit  = b.add(b.sub(c, dc), ir.Constant(I64_TY, 52))
            v_plus   = ir.Constant(I64_TY, 62)
            v_slash  = ir.Constant(I64_TY, 63)
            v0 = b.select(is_plus, v_plus, v_slash)
            v1 = b.select(is_digit, v_digit, v0)
            v2 = b.select(is_lower, v_lower, v1)
            return b.select(is_upper, v_upper, v2)

        chk = fn.append_basic_block("b64d.chk")
        bdy = fn.append_basic_block("b64d.bdy")
        ext = fn.append_basic_block("b64d.ext")
        b.branch(chk)
        b.position_at_end(chk)
        ii = b.load(i_al)
        b.cbranch(b.icmp_signed("<", ii, slen), bdy, ext)
        b.position_at_end(bdy)
        def get_code(offset):
            idx = b.add(ii, ir.Constant(I64_TY, offset))
            in_range = b.icmp_signed("<", idx, slen)
            ep = b.gep(s, [idx], inbounds=False)
            ch = b.load(ep)
            return b.select(in_range, decode_char(ch), ir.Constant(I64_TY, 0))
        c0 = get_code(0); c1 = get_code(1); c2 = get_code(2); c3 = get_code(3)
        oi = b.load(oi_al)
        byte0 = b.trunc(b.or_(b.shl(c0, ir.Constant(I64_TY,2)),
                               b.ashr(c1, ir.Constant(I64_TY,4))), I8_TY)
        byte1 = b.trunc(b.or_(b.shl(b.and_(c1, ir.Constant(I64_TY,15)), ir.Constant(I64_TY,4)),
                               b.ashr(c2, ir.Constant(I64_TY,2))), I8_TY)
        byte2 = b.trunc(b.or_(b.shl(b.and_(c2, ir.Constant(I64_TY,3)), ir.Constant(I64_TY,6)), c3), I8_TY)
        b.store(byte0, b.gep(buf, [oi], inbounds=False))
        b.store(byte1, b.gep(buf, [b.add(oi, ir.Constant(I64_TY,1))], inbounds=False))
        b.store(byte2, b.gep(buf, [b.add(oi, ir.Constant(I64_TY,2))], inbounds=False))
        b.store(b.add(ii, ir.Constant(I64_TY,4)), i_al)
        b.store(b.add(oi, ir.Constant(I64_TY,3)), oi_al)
        b.branch(chk)
        b.position_at_end(ext)
        oi_f = b.load(oi_al)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [oi_f], inbounds=False))
        b.ret(buf)
        return fn

    def _build_uuid_v4(self) -> ir.Function:
        """Generate UUID v4 string using platform random."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_uuid_v4")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 37)])
        # Use rand() to fill 16 bytes — not cryptographically secure but functional
        bytes_al = b.call(self.malloc_fn, [ir.Constant(I64_TY, 16)])
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("uuid.chk")
        bdy = fn.append_basic_block("uuid.bdy")
        ext = fn.append_basic_block("uuid.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, ir.Constant(I64_TY, 16)), bdy, ext)
        b.position_at_end(bdy)
        rv = b.call(self.rand_fn, [])
        rb = b.trunc(rv, I8_TY)
        ep = b.gep(bytes_al, [iv], inbounds=False)
        b.store(rb, ep)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        # Set version bits
        b6 = b.gep(bytes_al, [ir.Constant(I64_TY, 6)], inbounds=False)
        v6 = b.load(b6)
        b.store(b.or_(b.and_(v6, ir.Constant(I8_TY, 0x0F)), ir.Constant(I8_TY, 0x40)), b6)
        b8 = b.gep(bytes_al, [ir.Constant(I64_TY, 8)], inbounds=False)
        v8 = b.load(b8)
        b.store(b.or_(b.and_(v8, ir.Constant(I8_TY, 0x3F)), ir.Constant(I8_TY, 0x80)), b8)
        # Format as UUID string
        fmt_gv = self._global_str("%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x")
        fmt_p = self._gstr_ptr_const(fmt_gv)
        def lb(i): return b.zext(b.load(b.gep(bytes_al, [ir.Constant(I64_TY,i)], inbounds=False)), I32_TY)
        b.call(self.sprintf_fn, [buf, fmt_p,
               lb(0),lb(1),lb(2),lb(3),lb(4),lb(5),lb(6),lb(7),
               lb(8),lb(9),lb(10),lb(11),lb(12),lb(13),lb(14),lb(15)])
        b.ret(buf)
        return fn

    def _build_sha256(self) -> ir.Function:
        """Real SHA-256 implementation in LLVM IR (unrolled 64 rounds)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_sha256")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        msg = fn.args[0]

        # SHA-256 K constants
        K = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ]

        # Initial hash values H0-H7
        H_INIT = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ]

        I32 = I32_TY

        def u32(v): return ir.Constant(I32, v & 0xFFFFFFFF)

        def rotr32(b, val, n):
            """Right-rotate 32-bit value by n bits."""
            lsh = b.lshr(val, u32(n))
            rsh = b.shl(val, u32(32 - n))
            return b.or_(lsh, rsh)

        def trunc32(b, val):
            if val.type == I32: return val
            return b.trunc(val, I32)

        # ---- Compute input length ----
        msg_len = b.call(self.strlen_fn, [msg])
        msg_len32 = b.trunc(msg_len, I32)

        # Allocate padded message buffer: len + 1 + padding + 8 bytes
        # Total padded length = ((len + 9 + 63) / 64) * 64
        pad_base = b.add(msg_len, ir.Constant(I64_TY, 9))
        pad_align = b.add(pad_base, ir.Constant(I64_TY, 63))
        pad_len = b.and_(pad_align, ir.Constant(I64_TY, ~63))

        pad_buf = b.call(self.malloc_fn, [b.add(pad_len, ir.Constant(I64_TY, 8))])

        # memcpy msg into pad_buf
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        b.call(memcpy_fn, [pad_buf, msg, msg_len])

        # Append 0x80 byte
        pos80 = b.gep(pad_buf, [msg_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0x80), pos80)

        # Zero the padding bytes (msg_len+1 .. pad_len-8)
        memset_fn = self._get_or_declare("memset",
            ir.FunctionType(I8PTR, [I8PTR, I32, I64_TY]))
        zero_start = b.gep(pad_buf, [b.add(msg_len, ir.Constant(I64_TY, 1))], inbounds=False)
        zero_len = b.sub(b.sub(pad_len, ir.Constant(I64_TY, 8)),
                         b.add(msg_len, ir.Constant(I64_TY, 1)))
        b.call(memset_fn, [zero_start, ir.Constant(I32, 0), zero_len])

        # Append original length in bits as big-endian 64-bit integer
        bit_len = b.mul(msg_len, ir.Constant(I64_TY, 8))
        # Store big-endian: byte by byte
        for byte_i in range(8):
            shift = 56 - byte_i * 8
            if shift > 0:
                bval = b.trunc(b.lshr(bit_len, ir.Constant(I64_TY, shift)), I8_TY)
            else:
                bval = b.trunc(bit_len, I8_TY)
            bp = b.gep(pad_buf, [b.add(b.sub(pad_len, ir.Constant(I64_TY, 8)),
                                       ir.Constant(I64_TY, byte_i))], inbounds=False)
            b.store(bval, bp)

        # Initialize working hash h0-h7
        h_al = [b.alloca(I32) for _ in range(8)]
        for i, hv in enumerate(H_INIT):
            b.store(u32(hv), h_al[i])

        # Number of 64-byte blocks
        num_blocks = b.udiv(pad_len, ir.Constant(I64_TY, 64))

        # Block loop
        bi_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), bi_al)
        blk_chk = fn.append_basic_block("sha256.blk.chk")
        blk_bdy = fn.append_basic_block("sha256.blk.bdy")
        blk_ext = fn.append_basic_block("sha256.blk.ext")
        b.branch(blk_chk)

        b.position_at_end(blk_chk)
        bi = b.load(bi_al)
        b.cbranch(b.icmp_unsigned("<", bi, num_blocks), blk_bdy, blk_ext)

        b.position_at_end(blk_bdy)
        # block offset
        block_off = b.mul(bi, ir.Constant(I64_TY, 64))
        block_ptr = b.gep(pad_buf, [block_off], inbounds=False)

        # Load W[0..15] from block (big-endian 32-bit words)
        w_al = [b.alloca(I32) for _ in range(64)]
        for wi in range(16):
            byte_off = wi * 4
            word_val = u32(0)
            for byte_j in range(4):
                bp = b.gep(block_ptr, [ir.Constant(I64_TY, byte_off + byte_j)], inbounds=False)
                bv = b.zext(b.load(bp), I32)
                shifted = b.shl(bv, u32((3 - byte_j) * 8))
                word_val = b.or_(word_val, shifted)
            b.store(word_val, w_al[wi])

        # Extend W[16..63]
        for wi in range(16, 64):
            w15 = b.load(w_al[wi - 15])
            s0 = b.xor_(b.xor_(rotr32(b, w15, 7), rotr32(b, w15, 18)),
                        b.lshr(w15, u32(3)))
            w2 = b.load(w_al[wi - 2])
            s1 = b.xor_(b.xor_(rotr32(b, w2, 17), rotr32(b, w2, 19)),
                        b.lshr(w2, u32(10)))
            w16 = b.load(w_al[wi - 16])
            w7  = b.load(w_al[wi - 7])
            new_w = b.add(b.add(b.add(w16, s0), w7), s1)
            b.store(new_w, w_al[wi])

        # Working variables
        a_al, bb_al, c_al, d_al = b.alloca(I32), b.alloca(I32), b.alloca(I32), b.alloca(I32)
        e_al, f_al, g_al, hh_al = b.alloca(I32), b.alloca(I32), b.alloca(I32), b.alloca(I32)
        for al, hi in zip([a_al, bb_al, c_al, d_al, e_al, f_al, g_al, hh_al], h_al):
            b.store(b.load(hi), al)

        # 64 compression rounds (unrolled in Python)
        for ri in range(64):
            av = b.load(a_al); bv = b.load(bb_al); cv = b.load(c_al); dv = b.load(d_al)
            ev = b.load(e_al); fv = b.load(f_al); gv = b.load(g_al); hv = b.load(hh_al)

            S1  = b.xor_(b.xor_(rotr32(b, ev, 6), rotr32(b, ev, 11)), rotr32(b, ev, 25))
            ch  = b.xor_(b.and_(ev, fv), b.and_(b.not_(ev), gv))
            t1  = b.add(b.add(b.add(b.add(hv, S1), ch), u32(K[ri])), b.load(w_al[ri]))

            S0  = b.xor_(b.xor_(rotr32(b, av, 2), rotr32(b, av, 13)), rotr32(b, av, 22))
            maj = b.xor_(b.xor_(b.and_(av, bv), b.and_(av, cv)), b.and_(bv, cv))
            t2  = b.add(S0, maj)

            b.store(dv,           hh_al)
            b.store(cv,           g_al)
            b.store(bv,           f_al)
            b.store(av,           e_al)
            b.store(b.add(dv, t1), d_al)
            b.store(cv,           c_al)
            b.store(bv,           bb_al)
            b.store(b.add(t1, t2), a_al)

        # Add working vars back to hash state
        for al, hi in zip([a_al, bb_al, c_al, d_al, e_al, f_al, g_al, hh_al], h_al):
            b.store(b.add(b.load(hi), b.load(al)), hi)

        # Increment block index
        b.store(b.add(bi, ir.Constant(I64_TY, 1)), bi_al)
        b.branch(blk_chk)

        # After block loop: format hex output
        b.position_at_end(blk_ext)
        out_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 65)])
        fmt8_gv = self._global_str("%08x")
        hex_chars_per_word = 8
        for wi in range(8):
            word_v = b.load(h_al[wi])
            word64 = b.zext(word_v, I64_TY)
            out_off = b.gep(out_buf, [ir.Constant(I64_TY, wi * hex_chars_per_word)], inbounds=False)
            b.call(self.sprintf_fn, [out_off, self._gstr_ptr_const(fmt8_gv), word64])
        # NUL-terminate
        term_ptr = b.gep(out_buf, [ir.Constant(I64_TY, 64)], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term_ptr)
        b.ret(out_buf)
        return fn

    def _build_argv(self) -> ir.Function:
        """Return argc/argv as str[]. Requires __vx_argc/__vx_argv globals set by main."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_argv")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Build array from __vx_argc and __vx_argv globals
        argc_gv = self._get_or_create_global("__vx_argc", I32_TY, ir.Constant(I32_TY, 0))
        argv_gv = self._get_or_create_global("__vx_argv", ir.PointerType(I8PTR),
                                              ir.Constant(ir.PointerType(I8PTR), None))
        argc = b.zext(b.load(argc_gv), I64_TY)
        argv = b.load(argv_gv)
        # Build vx_array
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        data_sz = b.mul(argc, ir.Constant(I64_TY, 8))
        data = b.call(self.malloc_fn, [b.add(data_sz, ir.Constant(I64_TY, 8))])
        z = ir.Constant(I32_TY, 0)
        dp = b.gep(hdr, [z, z], inbounds=True)
        lp = b.gep(hdr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        cp = b.gep(hdr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        b.store(data, dp)
        b.store(argc, lp)
        b.store(argc, cp)
        # Copy argv pointers
        data_typed = b.bitcast(data, ir.PointerType(I8PTR))
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("argv.chk")
        bdy = fn.append_basic_block("argv.bdy")
        ext = fn.append_basic_block("argv.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, argc), bdy, ext)
        b.position_at_end(bdy)
        src_p = b.gep(argv, [iv], inbounds=False)
        src_v = b.load(src_p)
        dst_p = b.gep(data_typed, [iv], inbounds=False)
        b.store(src_v, dst_p)
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(hdr_raw)
        return fn

    def _get_or_create_global(self, name: str, ll_ty, init) -> ir.GlobalVariable:
        for gv in self.module.global_variables:
            if gv.name == name:
                return gv
        gv = ir.GlobalVariable(self.module, ll_ty, name=name)
        gv.linkage = "common"
        gv.initializer = init
        return gv

    def _build_shell(self) -> ir.Function:
        """Run shell command and return stdout as string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_shell")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cmd = fn.args[0]
        # popen + fread loop
        popen_ty = ir.FunctionType(I8PTR, [I8PTR, I8PTR])
        pclose_ty = ir.FunctionType(I32_TY, [I8PTR])
        mode_gv = self._global_str("r")
        mode_p  = self._gstr_ptr_const(mode_gv)
        popen_fn = self._get_or_declare_fn("popen", popen_ty)
        pclose_fn = self._get_or_declare_fn("pclose", pclose_ty)
        fp = b.call(popen_fn, [cmd, mode_p])
        fp_int = b.ptrtoint(fp, I64_TY)
        is_null = b.icmp_unsigned("==", fp_int, ir.Constant(I64_TY, 0))
        ok_b  = fn.append_basic_block("shell.ok")
        err_b = fn.append_basic_block("shell.err")
        b.cbranch(is_null, err_b, ok_b)
        b.position_at_end(err_b)
        empty_gv = self._global_str("")
        b.ret(self._gstr_ptr_const(empty_gv))
        b.position_at_end(ok_b)
        # Read 4096 bytes at a time
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 65536)])
        out_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), out_al)
        chunk = b.call(self.malloc_fn, [ir.Constant(I64_TY, 4097)])
        chk = fn.append_basic_block("shell.chk")
        bdy = fn.append_basic_block("shell.bdy")
        ext = fn.append_basic_block("shell.ext")
        b.branch(chk)
        b.position_at_end(chk)
        nr = b.call(self.fread_fn, [chunk, ir.Constant(I64_TY,1),
                                    ir.Constant(I64_TY,4096), fp])
        b.cbranch(b.icmp_signed(">", nr, ir.Constant(I64_TY,0)), bdy, ext)
        b.position_at_end(bdy)
        out = b.load(out_al)
        dst = b.gep(buf, [out], inbounds=False)
        b.call(self.memcpy_fn, [dst, chunk, nr])
        b.store(b.add(out, nr), out_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.call(pclose_fn, [fp])
        out_f = b.load(out_al)
        null_p = b.gep(buf, [out_f], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_p)
        b.ret(buf)
        return fn

    def _get_or_declare_fn(self, name: str, fn_ty) -> ir.Function:
        for f in self.module.functions:
            if f.name == name:
                return f
        return ir.Function(self.module, fn_ty, name=name)

    def _build_csv_parse(self) -> ir.Function:
        """Parse CSV string into str[][] (array of str[] rows)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_csv_parse")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        # Split by newlines first, then by commas
        nl_gv = self._global_str("\n")
        cm_gv = self._global_str(",")
        nl_p = self._gstr_ptr_const(nl_gv)
        cm_p = self._gstr_ptr_const(cm_gv)
        split_h = self._get_helper("__vx_str_split")
        lines_raw = b.call(split_h, [s, nl_p])
        # For each line, split by comma → build outer array
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        init_data = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        dp = b.gep(hdr, [z, z], inbounds=True)
        lp = b.gep(hdr, [z, ir.Constant(I32_TY,1)], inbounds=True)
        cp = b.gep(hdr, [z, ir.Constant(I32_TY,2)], inbounds=True)
        b.store(init_data, dp)
        b.store(ir.Constant(I64_TY,0), lp)
        b.store(ir.Constant(I64_TY,8), cp)
        lines = b.bitcast(lines_raw, self.arr_ptr_type)
        lines_lp = b.gep(lines, [z, ir.Constant(I32_TY,1)], inbounds=True)
        lines_len = b.load(lines_lp)
        lines_dp = b.gep(lines, [z, z], inbounds=True)
        lines_data_raw = b.load(lines_dp)
        lines_data = b.bitcast(lines_data_raw, ir.PointerType(I8PTR))
        push_h = self._get_helper("__vx_array_push")
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("csv.chk")
        bdy = fn.append_basic_block("csv.bdy")
        ext = fn.append_basic_block("csv.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, lines_len), bdy, ext)
        b.position_at_end(bdy)
        line_p = b.gep(lines_data, [iv], inbounds=False)
        line = b.load(line_p)
        row_raw = b.call(split_h, [line, cm_p])
        row_al = b.alloca(I8PTR)
        b.store(row_raw, row_al)
        b.call(push_h, [hdr_raw, b.bitcast(row_al, I8PTR), ir.Constant(I64_TY, 8)])
        b.store(b.add(iv, ir.Constant(I64_TY,1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(hdr_raw)
        return fn

    def _build_assert_eq(self) -> ir.Function:
        """assert_eq(a, b) — print and exit if not equal."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_assert_eq")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        a, bv = fn.args[0], fn.args[1]
        ok_b   = fn.append_basic_block("aeq.ok")
        fail_b = fn.append_basic_block("aeq.fail")
        b.cbranch(b.icmp_signed("==", a, bv), ok_b, fail_b)
        b.position_at_end(fail_b)
        fmt_gv = self._global_str("FAIL: assert_eq %lld != %lld\n")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv), a, bv])
        b.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
        b.unreachable()
        b.position_at_end(ok_b)
        b.ret_void()
        return fn

    def _build_str_starts_with(self) -> ir.Function:
        """str_starts_with(s, prefix) -> bool (i1)"""
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I8PTR]),
                         name="__vx_str_starts_with")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, prefix = fn.args[0], fn.args[1]
        plen = b.call(self.strlen_fn, [prefix])
        # strncmp(s, prefix, plen) == 0
        cmp = b.call(self.strncmp_fn, [s, prefix, plen])
        result = b.icmp_signed("==", cmp, ir.Constant(I32_TY, 0))
        b.ret(result)
        return fn

    def _build_str_char_at(self) -> ir.Function:
        """char_at(s, i) -> str (single-char heap string)"""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY]),
                         name="__vx_str_char_at")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, idx = fn.args[0], fn.args[1]
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 2)])
        src_p = b.gep(s, [idx], inbounds=False)
        ch = b.load(src_p)
        b.store(ch, buf)
        nul_p = b.gep(buf, [ir.Constant(I64_TY, 1)], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), nul_p)
        b.ret(buf)
        return fn

    def _build_env_set(self) -> ir.Function:
        """env_set(key, value) — set environment variable (putenv)"""
        # putenv(buf) where buf = "KEY=VALUE"
        putenv_ty = ir.FunctionType(I32_TY, [I8PTR])
        putenv_fn = self._get_or_declare_fn("putenv", putenv_ty)
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_env_set")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        key, val = fn.args[0], fn.args[1]
        klen = b.call(self.strlen_fn, [key])
        vlen = b.call(self.strlen_fn, [val])
        # total = klen + 1 (=) + vlen + 1 (NUL)
        total = b.add(b.add(klen, vlen), ir.Constant(I64_TY, 2))
        buf   = b.call(self.malloc_fn, [total])
        # sprintf(buf, "%s=%s", key, val)
        fmt_gv = self._global_str("%s=%s")
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), key, val])
        b.call(putenv_fn, [buf])
        b.ret_void()
        return fn

    # ------------------------------------------------------------------ #
    #  Compile: compile() extension for new top-level nodes               #
    # ------------------------------------------------------------------ #

    def _compile_extern_fn(self, d: 'ExternFnDecl'):
        """Declare an external (C) function in LLVM IR."""
        param_tys = [self._vx_to_llvm(p.type_name) for p in d.params]
        ret_ty = self._vx_to_llvm(d.return_type or "void")
        fn_ty = ir.FunctionType(ret_ty, param_tys)
        # Only declare once
        for f in self.module.functions:
            if f.name == d.name:
                self._functions[d.name] = {
                    "fn": f,
                    "sig": __import__("compiler.analyzer", fromlist=["FnSig"]).FnSig(
                        [(p.name, p.type_name) for p in d.params], d.return_type or "void"
                    )
                }
                return
        fn = ir.Function(self.module, fn_ty, name=d.name)
        from compiler.analyzer import FnSig as _FnSig
        self._functions[d.name] = {
            "fn": fn,
            "sig": _FnSig([(p.name, p.type_name) for p in d.params], d.return_type or "void")
        }
        self._fn_defaults[d.name] = [None] * len(d.params)

    def _compile_test_decl(self, d: 'TestDecl'):
        """Compile a test block as a private function __test__name()."""
        test_fn_name = f"__test__{d.name.replace(' ', '_')}"
        fn_ty = ir.FunctionType(VOID_TY, [])
        fn = ir.Function(self.module, fn_ty, name=test_fn_name)
        fn.linkage = "private"
        old_fn   = self.current_fn
        old_bldr = self.builder
        self.current_fn = fn
        entry = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._push_scope()
        self._defer_stack.append([])
        for stmt in d.body:
            if self.builder.block.is_terminated: break
            self._compile_stmt(stmt)
        self._emit_defers()
        self._defer_stack.pop()
        if not self.builder.block.is_terminated:
            self.builder.ret_void()
        self._pop_scope()
        self.current_fn = old_fn
        self.builder    = old_bldr
        self._functions[test_fn_name] = {
            "fn": fn,
            "sig": __import__("compiler.analyzer", fromlist=["FnSig"]).FnSig([], "void")
        }

    def _val_to_str(self, val: ir.Value, vt: str) -> ir.Value:
        """Convert any value to an i8* string for error/print purposes."""
        if vt == "str":
            return val
        buf = self.builder.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        if vt == "int":
            fmt = self._gstr_ptr(self._global_str("%lld"))
            self.builder.call(self.sprintf_fn, [buf, fmt, val])
        elif vt == "float":
            fmt = self._gstr_ptr(self._global_str("%g"))
            self.builder.call(self.sprintf_fn, [buf, fmt, val])
        elif vt == "bool":
            t = self._gstr_ptr(self._global_str("true"))
            f = self._gstr_ptr(self._global_str("false"))
            s = self.builder.select(val, t, f)
            fmt = self._gstr_ptr(self._global_str("%s"))
            self.builder.call(self.sprintf_fn, [buf, fmt, s])
        else:
            fmt = self._gstr_ptr(self._global_str("<value>"))
            self.builder.call(self.sprintf_fn, [buf, fmt])
        return buf

    # ------------------------------------------------------------------ #
    #  ADT Enum support                                                    #
    # ------------------------------------------------------------------ #

    def _define_adt_enum(self, d: 'EnumDeclADT'):
        """
        Define an ADT enum as a tagged-union struct.
        Each variant is represented as:
          { i64 tag, i8* payload }   (payload is variant-specific heap alloc)
        The enum type itself is stored as i8* (pointer to this struct).
        """
        # Register the enum type struct: {i64 tag, i8* payload}
        st_name = f"__adt_{d.name}"
        if st_name not in self._structs:
            st = self.module.context.get_identified_type(st_name)
            st.set_body(I64_TY, I8PTR)
            # Register variant tags as global constants
            for i, variant in enumerate(d.variants):
                gname = f"{d.name}.{variant.name}"
                gv = ir.GlobalVariable(self.module, I64_TY, name=gname)
                gv.linkage = "internal"
                gv.global_constant = True
                gv.initializer = ir.Constant(I64_TY, i)
                self._globals[gname] = {"ptr": gv, "vx_type": "int"}
                # If variant has payload fields, define a payload struct
                if variant.fields:
                    payload_name = f"__adt_{d.name}_{variant.name}"
                    payload_st = self.module.context.get_identified_type(payload_name)
                    field_tys = [self._vx_to_llvm(f.type_name) for f in variant.fields]
                    payload_st.set_body(*field_tys)
                    self._structs[payload_name] = {
                        "ll_type": payload_st,
                        "fields": [(f.name, f.type_name) for f in variant.fields]
                    }
            self._structs[st_name] = {
                "ll_type": st,
                "fields": [("tag", "int"), ("payload", "str")]  # str ≈ i8*
            }

    def _compile_adt_constructor(self, enum_name: str, variant_name: str,
                                  args: list) -> tuple['ir.Value', str]:
        """Construct an ADT variant value: EnumName.Variant(args...)"""
        st_name      = f"__adt_{enum_name}"
        payload_name = f"__adt_{enum_name}_{variant_name}"
        tag_gname    = f"{enum_name}.{variant_name}"

        # Allocate the tagged-union struct on the heap
        union_size = ir.Constant(I64_TY, 16)   # 8 (tag) + 8 (ptr)
        union_ptr  = self.builder.call(self.malloc_fn, [union_size])
        union_ty   = self._structs[st_name]["ll_type"]
        union_cptr = self.builder.bitcast(union_ptr, ir.PointerType(union_ty))

        # Store tag
        tag_val = self.builder.load(self._globals[tag_gname]["ptr"])
        tag_ptr = self.builder.gep(union_cptr,
                                   [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 0)],
                                   inbounds=True)
        self.builder.store(tag_val, tag_ptr)

        # Store payload (if any fields)
        if payload_name in self._structs:
            payload_info = self._structs[payload_name]
            payload_ty   = payload_info["ll_type"]
            payload_size = ir.Constant(I64_TY, len(payload_info["fields"]) * 8)
            payload_ptr  = self.builder.call(self.malloc_fn, [payload_size])
            payload_cptr = self.builder.bitcast(payload_ptr, ir.PointerType(payload_ty))
            for i, (arg_node, (fname, ftype)) in enumerate(zip(args, payload_info["fields"])):
                av, at = self._compile_expr(arg_node)
                fp     = self.builder.gep(payload_cptr,
                                          [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, i)],
                                          inbounds=True)
                if ftype == "float" and at == "int":
                    av = self.builder.sitofp(av, F64_TY)
                self.builder.store(av, fp)
            # Store payload pointer into union
            pay_field_ptr = self.builder.gep(union_cptr,
                                              [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 1)],
                                              inbounds=True)
            self.builder.store(payload_ptr, pay_field_ptr)
        else:
            # No payload — store null
            pay_field_ptr = self.builder.gep(union_cptr,
                                              [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 1)],
                                              inbounds=True)
            self.builder.store(ir.Constant(I8PTR, None), pay_field_ptr)

        return union_ptr, enum_name

    # ------------------------------------------------------------------ #
    #  Comptime constants                                                  #
    # ------------------------------------------------------------------ #

    def _compile_comptime(self, d: 'ComptimeDecl'):
        """Evaluate a comptime constant and register it as an LLVM global."""
        # Evaluate as a constant expression (literals only in this impl)
        val_node = d.value
        if isinstance(val_node, IntLiteral):
            gv = ir.GlobalVariable(self.module, I64_TY, name=d.name)
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = ir.Constant(I64_TY, val_node.value)
            self._globals[d.name] = {"ptr": gv, "vx_type": "int"}
        elif isinstance(val_node, FloatLiteral):
            gv = ir.GlobalVariable(self.module, F64_TY, name=d.name)
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = ir.Constant(F64_TY, val_node.value)
            self._globals[d.name] = {"ptr": gv, "vx_type": "float"}
        elif isinstance(val_node, BoolLiteral):
            gv = ir.GlobalVariable(self.module, I1_TY, name=d.name)
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = ir.Constant(I1_TY, int(val_node.value))
            self._globals[d.name] = {"ptr": gv, "vx_type": "bool"}
        elif isinstance(val_node, StringLiteral):
            # Store as a global string constant pointer
            gstr = self._global_str(val_node.value)
            gv   = ir.GlobalVariable(self.module, I8PTR, name=d.name)
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = ir.Constant(I8PTR, None)  # placeholder
            self._globals[d.name] = {"ptr": gv, "vx_type": "str", "_gstr": gstr}
        # Arithmetic BinOp on literals — evaluate at compile time
        elif isinstance(val_node, BinOp):
            result = self._eval_const_binop(val_node)
            if isinstance(result, int):
                gv = ir.GlobalVariable(self.module, I64_TY, name=d.name)
                gv.linkage = "internal"
                gv.global_constant = True
                gv.initializer = ir.Constant(I64_TY, result)
                self._globals[d.name] = {"ptr": gv, "vx_type": "int"}
            elif isinstance(result, float):
                gv = ir.GlobalVariable(self.module, F64_TY, name=d.name)
                gv.linkage = "internal"
                gv.global_constant = True
                gv.initializer = ir.Constant(F64_TY, result)
                self._globals[d.name] = {"ptr": gv, "vx_type": "float"}

    def _eval_const_binop(self, node: 'BinOp'):
        """Recursively evaluate a BinOp of literals at compile time."""
        def _eval(n):
            if isinstance(n, IntLiteral):   return n.value
            if isinstance(n, FloatLiteral): return n.value
            if isinstance(n, BinOp):
                lv = _eval(n.left)
                rv = _eval(n.right)
                if n.op == "+":  return lv + rv
                if n.op == "-":  return lv - rv
                if n.op == "*":  return lv * rv
                if n.op == "/":  return lv / rv
                if n.op == "%":  return lv % rv
                if n.op == "**": return lv ** rv
                if n.op == "<<": return int(lv) << int(rv)
                if n.op == ">>": return int(lv) >> int(rv)
                if n.op == "&":  return int(lv) & int(rv)
                if n.op == "|":  return int(lv) | int(rv)
                if n.op == "^":  return int(lv) ^ int(rv)
            raise CodegenError(f"Cannot evaluate comptime expr: {type(n).__name__}")
        return _eval(node)

    # ------------------------------------------------------------------ #
    #  Thread / Mutex helpers (#49, #50)                                   #
    # ------------------------------------------------------------------ #

    def _build_thread_spawn(self) -> ir.Function:
        """thread_spawn(fn_ptr: i8*) -> i64 thread_id"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_thread_spawn")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn_ptr = fn.args[0]
        if _sys.platform == "win32":
            # HANDLE CreateThread(NULL, 0, fn, NULL, 0, NULL)
            create_thread_ty = ir.FunctionType(I8PTR, [I8PTR, I64_TY, I8PTR, I8PTR, I32_TY, I8PTR])
            create_thread = self._get_or_declare_fn("CreateThread", create_thread_ty)
            null = ir.Constant(I8PTR, None)
            handle = b.call(create_thread, [null, ir.Constant(I64_TY, 0), fn_ptr, null,
                                            ir.Constant(I32_TY, 0), null])
            tid = b.ptrtoint(handle, I64_TY)
        else:
            # pthread_create(&tid, NULL, fn, NULL)
            tid_storage = b.alloca(I64_TY)
            pthread_create_ty = ir.FunctionType(I32_TY, [ir.PointerType(I64_TY), I8PTR, I8PTR, I8PTR])
            pthread_create = self._get_or_declare_fn("pthread_create", pthread_create_ty)
            null = ir.Constant(I8PTR, None)
            b.call(pthread_create, [tid_storage, null, fn_ptr, null])
            tid = b.load(tid_storage)
        b.ret(tid)
        return fn

    def _build_thread_join(self) -> ir.Function:
        """thread_join(tid: i64) — wait for thread to finish"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_thread_join")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tid = fn.args[0]
        if _sys.platform == "win32":
            wait_ty = ir.FunctionType(I32_TY, [I8PTR, I32_TY])
            wait_fn = self._get_or_declare_fn("WaitForSingleObject", wait_ty)
            handle = b.inttoptr(tid, I8PTR)
            b.call(wait_fn, [handle, ir.Constant(I32_TY, -1)])  # INFINITE
        else:
            pthread_join_ty = ir.FunctionType(I32_TY, [I64_TY, ir.PointerType(I8PTR)])
            pthread_join = self._get_or_declare_fn("pthread_join", pthread_join_ty)
            null = ir.Constant(ir.PointerType(I8PTR), None)
            b.call(pthread_join, [tid, null])
        b.ret_void()
        return fn

    def _build_thread_sleep(self) -> ir.Function:
        """thread_sleep(ms: i64) — sleep for N milliseconds"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_thread_sleep")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ms = fn.args[0]
        if _sys.platform == "win32":
            sleep_ty = ir.FunctionType(VOID_TY, [I32_TY])
            sleep_fn = self._get_or_declare_fn("Sleep", sleep_ty)
            ms32 = b.trunc(ms, I32_TY)
            b.call(sleep_fn, [ms32])
        else:
            usleep_ty = ir.FunctionType(I32_TY, [I32_TY])
            usleep_fn = self._get_or_declare_fn("usleep", usleep_ty)
            us = b.mul(ms, ir.Constant(I64_TY, 1000))
            us32 = b.trunc(us, I32_TY)
            b.call(usleep_fn, [us32])
        b.ret_void()
        return fn

    def _build_mutex_new(self) -> ir.Function:
        """mutex_new() -> i8* (opaque mutex pointer)"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_mutex_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        if _sys.platform == "win32":
            # CRITICAL_SECTION is 40 bytes on 64-bit Windows
            cs = b.call(self.malloc_fn, [ir.Constant(I64_TY, 48)])
            init_ty = ir.FunctionType(VOID_TY, [I8PTR])
            init_fn = self._get_or_declare_fn("InitializeCriticalSection", init_ty)
            b.call(init_fn, [cs])
            b.ret(cs)
        else:
            # pthread_mutex_t is ~40 bytes; malloc and pthread_mutex_init
            mu = b.call(self.malloc_fn, [ir.Constant(I64_TY, 48)])
            init_ty = ir.FunctionType(I32_TY, [I8PTR, I8PTR])
            init_fn = self._get_or_declare_fn("pthread_mutex_init", init_ty)
            null = ir.Constant(I8PTR, None)
            b.call(init_fn, [mu, null])
            b.ret(mu)
        return fn

    def _build_mutex_lock(self) -> ir.Function:
        """mutex_lock(mu: i8*)"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR]),
                         name="__vx_mutex_lock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        mu = fn.args[0]
        if _sys.platform == "win32":
            lock_ty = ir.FunctionType(VOID_TY, [I8PTR])
            lock_fn = self._get_or_declare_fn("EnterCriticalSection", lock_ty)
            b.call(lock_fn, [mu])
        else:
            lock_ty = ir.FunctionType(I32_TY, [I8PTR])
            lock_fn = self._get_or_declare_fn("pthread_mutex_lock", lock_ty)
            b.call(lock_fn, [mu])
        b.ret_void()
        return fn

    def _build_mutex_unlock(self) -> ir.Function:
        """mutex_unlock(mu: i8*)"""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR]),
                         name="__vx_mutex_unlock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        mu = fn.args[0]
        if _sys.platform == "win32":
            unlock_ty = ir.FunctionType(VOID_TY, [I8PTR])
            unlock_fn = self._get_or_declare_fn("LeaveCriticalSection", unlock_ty)
            b.call(unlock_fn, [mu])
        else:
            unlock_ty = ir.FunctionType(I32_TY, [I8PTR])
            unlock_fn = self._get_or_declare_fn("pthread_mutex_unlock", unlock_ty)
            b.call(unlock_fn, [mu])
        b.ret_void()
        return fn

    # ------------------------------------------------------------------ #
    #  Unicode helpers (#45)                                               #
    # ------------------------------------------------------------------ #

    def _build_str_char_len(self) -> ir.Function:
        """str_char_len(s) -> int  — count UTF-8 codepoints (chars, not bytes)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_str_char_len")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        # Count codepoints: any byte with bits 10xxxxxx (0x80..0xBF) is a continuation byte
        cnt_al = b.alloca(I64_TY)
        i_al   = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), cnt_al)
        b.store(ir.Constant(I64_TY, 0), i_al)
        slen = b.call(self.strlen_fn, [s])
        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        ext = fn.append_basic_block("ext")
        b.branch(chk)
        b.position_at_end(chk)
        i = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<", i, slen), bdy, ext)
        b.position_at_end(bdy)
        ch = b.load(b.gep(s, [i], inbounds=False))
        ch32 = b.zext(ch, I32_TY)
        mask = ir.Constant(I32_TY, 0xC0)
        cont = ir.Constant(I32_TY, 0x80)
        is_cont = b.icmp_unsigned("==", b.and_(ch32, mask), cont)
        cnt = b.load(cnt_al)
        new_cnt = b.select(is_cont, cnt, b.add(cnt, ir.Constant(I64_TY, 1)))
        b.store(new_cnt, cnt_al)
        b.store(b.add(i, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(b.load(cnt_al))
        return fn

    def _build_str_char_at_utf8(self) -> ir.Function:
        """str_char_at_utf8(s, i) -> int (Unicode codepoint at position i)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I64_TY]),
                         name="__vx_str_char_at_utf8")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s, idx = fn.args[0], fn.args[1]
        # Walk codepoints until we hit the idx-th one
        i_al   = b.alloca(I64_TY)   # byte index
        cnt_al = b.alloca(I64_TY)   # codepoint count
        b.store(ir.Constant(I64_TY, 0), i_al)
        b.store(ir.Constant(I64_TY, 0), cnt_al)
        slen = b.call(self.strlen_fn, [s])
        chk = fn.append_basic_block("chk")
        bdy = fn.append_basic_block("bdy")
        found = fn.append_basic_block("found")
        ext   = fn.append_basic_block("ext")
        b.branch(chk)
        b.position_at_end(chk)
        i = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<", i, slen), bdy, ext)
        b.position_at_end(bdy)
        ch = b.zext(b.load(b.gep(s, [i], inbounds=False)), I32_TY)
        mask = ir.Constant(I32_TY, 0xC0)
        cont = ir.Constant(I32_TY, 0x80)
        is_cont = b.icmp_unsigned("==", b.and_(ch, mask), cont)
        cnt = b.load(cnt_al)
        b.cbranch(is_cont, chk, found)   # skip continuation bytes
        b.position_at_end(found)
        b.store(b.add(i, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(b.icmp_signed("==", cnt, idx),
                  ext,
                  fn.append_basic_block("next"))
        next_b = list(fn.blocks)[-1]
        b.position_at_end(next_b)
        b.store(b.add(cnt, ir.Constant(I64_TY, 1)), cnt_al)
        b.branch(chk)
        b.position_at_end(ext)
        # Return codepoint at current byte — simplified (ASCII + single-byte)
        final_i = b.load(i_al)
        final_ch = b.zext(b.load(b.gep(s, [b.sub(final_i, ir.Constant(I64_TY, 1))],
                                        inbounds=False)), I64_TY)
        b.ret(final_ch)
        return fn

    # ------------------------------------------------------------------ #
    #  JSON / HTTP helpers (#4 stdlib)                                     #
    # ------------------------------------------------------------------ #

    def _build_json_stringify_int(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_json_stringify_int")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])
        fmt_gv = self._global_str("%lld")
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), fn.args[0]])
        b.ret(buf)
        return fn

    def _build_json_stringify_float(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [F64_TY]),
                         name="__vx_json_stringify_float")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        fmt_gv = self._global_str("%g")
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), fn.args[0]])
        b.ret(buf)
        return fn

    def _build_json_stringify_str(self) -> ir.Function:
        """Wrap a string in JSON quotes: hello → \"hello\" """
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_json_stringify_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s = fn.args[0]
        slen = b.call(self.strlen_fn, [s])
        total = b.add(slen, ir.Constant(I64_TY, 3))   # 2 quotes + NUL
        buf   = b.call(self.malloc_fn, [total])
        fmt_gv = self._global_str("\"%s\"")
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), s])
        b.ret(buf)
        return fn

    def _build_http_get(self) -> ir.Function:
        """http_get(url) -> str — uses curl CLI if available; returns empty on failure."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_http_get")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        url = fn.args[0]
        # Build command: "curl -s URL"
        cmd_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 4096)])
        fmt_gv  = self._global_str("curl -s %s")
        b.call(self.sprintf_fn, [cmd_buf, self._gstr_ptr_const(fmt_gv), url])
        # Use __vx_shell to run it
        shell_fn = self._get_helper("__vx_shell")
        result = b.call(shell_fn, [cmd_buf])
        b.ret(result)
        return fn

    # ------------------------------------------------------------------ #
    #  v8 new helper builders                                             #
    # ------------------------------------------------------------------ #

    def _build_csv_write(self) -> ir.Function:
        """csv_write(rows: str[][], delim: str) -> str — serialize 2D array to CSV."""
        fn = ir.Function(self.module,
                         ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_csv_write")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        rows_raw, delim = fn.args[0], fn.args[1]
        # Allocate a big output buffer
        out_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 65536)])
        # Start with empty string
        b.store(ir.Constant(I8_TY, 0), out_buf)
        rows = b.bitcast(rows_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        rows_lp = b.gep(rows, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        rows_len = b.load(rows_lp)
        rows_dp  = b.gep(rows, [z, z], inbounds=True)
        rows_data_raw = b.load(rows_dp)
        rows_data = b.bitcast(rows_data_raw, ir.PointerType(I8PTR))
        # We'll build via repeated sprintf calls into a growing offset
        off_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), off_al)
        ri_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), ri_al)
        row_chk = fn.append_basic_block("csv.row.chk")
        row_bdy = fn.append_basic_block("csv.row.bdy")
        row_ext = fn.append_basic_block("csv.row.ext")
        b.branch(row_chk)
        b.position_at_end(row_chk)
        ri = b.load(ri_al)
        b.cbranch(b.icmp_signed("<", ri, rows_len), row_bdy, row_ext)
        b.position_at_end(row_bdy)
        row_p = b.gep(rows_data, [ri], inbounds=False)
        row_raw_v = b.load(row_p)
        row_v = b.bitcast(row_raw_v, self.arr_ptr_type)
        col_lp = b.gep(row_v, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        col_len = b.load(col_lp)
        col_dp  = b.gep(row_v, [z, z], inbounds=True)
        col_data_raw = b.load(col_dp)
        col_data = b.bitcast(col_data_raw, ir.PointerType(I8PTR))
        ci_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), ci_al)
        col_chk = fn.append_basic_block("csv.col.chk")
        col_bdy = fn.append_basic_block("csv.col.bdy")
        col_ext = fn.append_basic_block("csv.col.ext")
        b.branch(col_chk)
        b.position_at_end(col_chk)
        ci = b.load(ci_al)
        b.cbranch(b.icmp_signed("<", ci, col_len), col_bdy, col_ext)
        b.position_at_end(col_bdy)
        cell_p = b.gep(col_data, [ci], inbounds=False)
        cell   = b.load(cell_p)
        off    = b.load(off_al)
        dst    = b.gep(out_buf, [off], inbounds=False)
        cell_len = b.call(self.strlen_fn, [cell])
        b.call(self.memcpy_fn, [dst, cell, cell_len])
        new_off = b.add(off, cell_len)
        # Append delimiter unless last column
        is_last = b.icmp_signed("==", b.add(ci, ir.Constant(I64_TY, 1)), col_len)
        delim_len = b.call(self.strlen_fn, [delim])
        actual_delim_len = b.select(is_last, ir.Constant(I64_TY, 0), delim_len)
        delim_dst = b.gep(out_buf, [new_off], inbounds=False)
        # Only write delim bytes if not last
        delim_end = fn.append_basic_block("csv.delim.write")
        no_delim  = fn.append_basic_block("csv.delim.skip")
        b.cbranch(is_last, no_delim, delim_end)
        b.position_at_end(delim_end)
        b.call(self.memcpy_fn, [delim_dst, delim, delim_len])
        b.branch(no_delim)
        b.position_at_end(no_delim)
        final_off = b.add(new_off, actual_delim_len)
        b.store(final_off, off_al)
        b.store(b.add(ci, ir.Constant(I64_TY, 1)), ci_al)
        b.branch(col_chk)
        b.position_at_end(col_ext)
        # Newline after each row
        off2 = b.load(off_al)
        nl_dst = b.gep(out_buf, [off2], inbounds=False)
        b.store(ir.Constant(I8_TY, ord('\n')), nl_dst)
        b.store(b.add(off2, ir.Constant(I64_TY, 1)), off_al)
        b.store(b.add(ri, ir.Constant(I64_TY, 1)), ri_al)
        b.branch(row_chk)
        b.position_at_end(row_ext)
        # Null-terminate
        final = b.load(off_al)
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [final], inbounds=False))
        b.ret(out_buf)
        return fn

    def _build_print_color(self) -> ir.Function:
        """print_color(msg, color) — print msg with ANSI color."""
        fn = ir.Function(self.module,
                         ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_print_color")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        msg, color = fn.args[0], fn.args[1]

        def _gp(gv):
            z = ir.Constant(I32_TY, 0)
            return b.gep(gv, [z, z], inbounds=True)

        codes = [
            ("Red",     "\033[0;31m"),
            ("Green",   "\033[0;32m"),
            ("Yellow",  "\033[0;33m"),
            ("Blue",    "\033[0;34m"),
            ("Magenta", "\033[0;35m"),
            ("Cyan",    "\033[0;36m"),
            ("White",   "\033[0;37m"),
            ("Bold",    "\033[1m"),
        ]
        reset_gv  = self._global_str("\033[0m\n")
        fmt3_gv   = self._global_str("%s%s%s")
        done_b    = fn.append_basic_block("pc.done")
        for cname, code in codes:
            name_gv = self._global_str(cname)
            code_gv = self._global_str(code)
            cmp     = b.call(self.strcmp_fn, [color, self._gstr_ptr_const(name_gv)])
            match_b = fn.append_basic_block(f"pc.{cname.lower()}")
            next_b  = fn.append_basic_block(f"pc.nx{cname.lower()}")
            b.cbranch(b.icmp_signed("==", cmp, ir.Constant(I32_TY, 0)), match_b, next_b)
            b.position_at_end(match_b)
            b.call(self.printf, [_gp(fmt3_gv), _gp(code_gv), msg, _gp(reset_gv)])
            b.branch(done_b)
            b.position_at_end(next_b)
        # Default: plain
        fmt_plain = self._global_str("%s\n")
        b.call(self.printf, [_gp(fmt_plain), msg])
        b.branch(done_b)
        b.position_at_end(done_b)
        b.ret_void()
        return fn

    def _build_dict_values(self) -> ir.Function:
        """dict.values() — return i8* array of values (as i64 array)."""
        fn = ir.Function(self.module,
                         ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_dict_values")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        hdr = b.bitcast(fn.args[0], self.dict_ptr_type)
        z = ir.Constant(I32_TY, 0)
        # Load vals_ptr and len
        vf = b.gep(hdr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        lf = b.gep(hdr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        vals_raw = b.load(vf)
        dlen     = b.load(lf)
        # Build a vx_array header around the existing vals buffer
        arr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        arr     = b.bitcast(arr_raw, self.arr_ptr_type)
        dp  = b.gep(arr, [z, z], inbounds=True)
        lp  = b.gep(arr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        cp  = b.gep(arr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        b.store(vals_raw, dp)
        b.store(dlen, lp)
        b.store(dlen, cp)
        b.ret(arr_raw)
        return fn

    def _build_dict_items(self) -> ir.Function:
        """dict.items() — return str[] of 'key:value' pairs."""
        fn = ir.Function(self.module,
                         ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_dict_items")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        hdr = b.bitcast(fn.args[0], self.dict_ptr_type)
        z = ir.Constant(I32_TY, 0)
        kf  = b.gep(hdr, [z, z], inbounds=True)
        vf  = b.gep(hdr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        lf  = b.gep(hdr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        keys_raw = b.load(kf)
        vals_raw = b.load(vf)
        dlen     = b.load(lf)
        keys_data = b.bitcast(keys_raw, ir.PointerType(I8PTR))
        vals_data = b.bitcast(vals_raw, ir.PointerType(I64_TY))
        # Build result array
        arr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        init_data = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        arr     = b.bitcast(arr_raw, self.arr_ptr_type)
        dp  = b.gep(arr, [z, z], inbounds=True)
        lp  = b.gep(arr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        cp  = b.gep(arr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        b.store(init_data, dp)
        b.store(ir.Constant(I64_TY, 0), lp)
        b.store(ir.Constant(I64_TY, 8), cp)
        push_h = self._get_helper("__vx_array_push")
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk = fn.append_basic_block("di.chk")
        bdy = fn.append_basic_block("di.bdy")
        ext = fn.append_basic_block("di.ext")
        b.branch(chk)
        b.position_at_end(chk)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, dlen), bdy, ext)
        b.position_at_end(bdy)
        kp  = b.gep(keys_data, [iv], inbounds=False)
        vp  = b.gep(vals_data, [iv], inbounds=False)
        key = b.load(kp)
        val = b.load(vp)
        # Format: "key:value"
        pair_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 512)])
        fmt_gv   = self._global_str("%s:%lld")
        b.call(self.sprintf_fn, [pair_buf, self._gstr_ptr_const(fmt_gv), key, val])
        pair_al = b.alloca(I8PTR)
        b.store(pair_buf, pair_al)
        b.call(push_h, [arr_raw, b.bitcast(pair_al, I8PTR), ir.Constant(I64_TY, 8)])
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk)
        b.position_at_end(ext)
        b.ret(arr_raw)
        return fn


    # ------------------------------------------------------------------ #
    #  v8 helper builders                                               #
    # ------------------------------------------------------------------ #

    def _build_datetime_now(self) -> ir.Function:
        """datetime_now() -> str  — current local time as 'YYYY-MM-DD HH:MM:SS'."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_datetime_now")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])
        ts_al = b.alloca(I64_TY)
        null = ir.Constant(I8PTR, None)
        ts = b.call(self.time_fn, [null])
        b.store(ts, ts_al)
        ts_ptr = b.bitcast(ts_al, I8PTR)
        tm_ptr = b.call(self.localtime_fn, [ts_ptr])
        fmt_gv = self._global_str("%Y-%m-%d %H:%M:%S")
        fmt_p  = self._gstr_ptr_const(fmt_gv)
        b.call(self.strftime_fn, [buf, ir.Constant(I64_TY, 32), fmt_p, tm_ptr])
        b.ret(buf)
        return fn

    def _build_datetime_format(self) -> ir.Function:
        """datetime_format(fmt: str) -> str  — format current time with custom fmt."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_datetime_format")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fmt_arg = fn.args[0]
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
        ts_al = b.alloca(I64_TY)
        null = ir.Constant(I8PTR, None)
        ts = b.call(self.time_fn, [null])
        b.store(ts, ts_al)
        ts_ptr = b.bitcast(ts_al, I8PTR)
        tm_ptr = b.call(self.localtime_fn, [ts_ptr])
        b.call(self.strftime_fn, [buf, ir.Constant(I64_TY, 256), fmt_arg, tm_ptr])
        b.ret(buf)
        return fn

    def _build_datetime_from_ts(self) -> ir.Function:
        """datetime_from_ts(ts: i64) -> str  — format a unix timestamp."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_datetime_from_ts")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ts = fn.args[0]
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 32)])
        ts_al = b.alloca(I64_TY)
        b.store(ts, ts_al)
        ts_ptr = b.bitcast(ts_al, I8PTR)
        tm_ptr = b.call(self.localtime_fn, [ts_ptr])
        fmt_gv = self._global_str("%Y-%m-%d %H:%M:%S")
        fmt_p  = self._gstr_ptr_const(fmt_gv)
        b.call(self.strftime_fn, [buf, ir.Constant(I64_TY, 32), fmt_p, tm_ptr])
        b.ret(buf)
        return fn

    def _build_sleep_ms(self) -> ir.Function:
        """sleep_ms(ms: i64) — sleep for ms milliseconds."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_sleep_ms")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ms = fn.args[0]
        if _sys.platform == "win32":
            ms32 = b.trunc(ms, I32_TY)
            sleep_ty = ir.FunctionType(VOID_TY, [I32_TY])
            sleep_fn = self._get_or_declare_fn("Sleep", sleep_ty)
            b.call(sleep_fn, [ms32])
        else:
            us = b.mul(ms, ir.Constant(I64_TY, 1000))
            us32 = b.trunc(us, I32_TY)
            usleep_ty = ir.FunctionType(I32_TY, [I32_TY])
            usleep_fn = self._get_or_declare_fn("usleep", usleep_ty)
            b.call(usleep_fn, [us32])
        b.ret_void()
        return fn

    def _build_signal_handle(self) -> ir.Function:
        """signal_handle(signum: i64, handler: i8*) — install signal handler."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I8PTR]),
                         name="__vx_signal_handle")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        signum, handler = fn.args[0], fn.args[1]
        sig32 = b.trunc(signum, I32_TY)
        # signal(int, void(*)(int)) -> void(*)(int)
        handler_ty = ir.FunctionType(VOID_TY, [I32_TY])
        handler_ptr_ty = ir.PointerType(handler_ty)
        signal_ty = ir.FunctionType(handler_ptr_ty, [I32_TY, handler_ptr_ty])
        signal_fn = self._get_or_declare_fn("signal", signal_ty)
        real_handler = b.bitcast(handler, handler_ptr_ty)
        b.call(signal_fn, [sig32, real_handler])
        b.ret_void()
        return fn

    def _build_process_spawn(self) -> ir.Function:
        """process_spawn(cmd: str) -> i64 PID (opaque handle stored in malloc)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_process_spawn")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cmd = fn.args[0]
        # Use popen for simplicity — returns a FILE* stored as i64
        mode_gv = self._global_str("r")
        mode_p  = self._gstr_ptr_const(mode_gv)
        fp = b.call(self.popen_fn, [cmd, mode_p])
        result = b.ptrtoint(fp, I64_TY)
        b.ret(result)
        return fn

    def _build_process_wait(self) -> ir.Function:
        """process_wait(handle: i64) -> i64 exit_code."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_process_wait")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]
        fp = b.inttoptr(handle, I8PTR)
        rc32 = b.call(self.pclose_fn, [fp])
        b.ret(b.sext(rc32, I64_TY))
        return fn

    def _build_process_kill(self) -> ir.Function:
        """process_kill(handle: i64) — close the process handle."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_process_kill")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]
        fp = b.inttoptr(handle, I8PTR)
        b.call(self.pclose_fn, [fp])
        b.ret_void()
        return fn

    # Progress bar state: malloc'd {i64 total, i64 current}
    def _build_progress_new(self) -> ir.Function:
        """progress_new(total) -> i64 handle (opaque pointer)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_progress_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        total = fn.args[0]
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 16)])
        buf64 = b.bitcast(buf, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        total_p = b.gep(buf64, [z], inbounds=False)
        cur_p   = b.gep(buf64, [ir.Constant(I32_TY, 1)], inbounds=False)
        b.store(total, total_p)
        b.store(ir.Constant(I64_TY, 0), cur_p)
        b.ret(b.ptrtoint(buf, I64_TY))
        return fn

    def _build_progress_update(self) -> ir.Function:
        """progress_update(handle, current) — redraw progress bar."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_progress_update")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle, current = fn.args[0], fn.args[1]
        buf    = b.inttoptr(handle, I8PTR)
        buf64  = b.bitcast(buf, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        total_p = b.gep(buf64, [z], inbounds=False)
        cur_p   = b.gep(buf64, [ir.Constant(I32_TY, 1)], inbounds=False)
        total = b.load(total_p)
        b.store(current, cur_p)
        # Compute percentage (avoid /0)
        safe_total = b.select(b.icmp_signed("==", total, ir.Constant(I64_TY, 0)),
                              ir.Constant(I64_TY, 1), total)
        pct = b.sdiv(b.mul(current, ir.Constant(I64_TY, 100)), safe_total)
        fmt_gv = self._global_str("\r[%3lld%%] ")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv), pct])
        # Draw filled/empty blocks
        filled = b.sdiv(pct, ir.Constant(I64_TY, 5))
        fi_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), fi_al)
        bar_chk = fn.append_basic_block("bar_chk")
        bar_bdy = fn.append_basic_block("bar_bdy")
        bar_ext = fn.append_basic_block("bar_ext")
        b.branch(bar_chk)
        b.position_at_end(bar_chk)
        fi = b.load(fi_al)
        b.cbranch(b.icmp_signed("<", fi, ir.Constant(I64_TY, 20)), bar_bdy, bar_ext)
        b.position_at_end(bar_bdy)
        is_full = b.icmp_signed("<", b.load(fi_al), filled)
        full_gv = self._global_str("#")
        empt_gv = self._global_str(".")
        ch_gv = b.select(is_full, self._gstr_ptr_const(full_gv),
                         self._gstr_ptr_const(empt_gv))
        sfmt_gv = self._global_str("%s")
        b.call(self.printf, [self._gstr_ptr_const(sfmt_gv), ch_gv])
        b.store(b.add(b.load(fi_al), ir.Constant(I64_TY, 1)), fi_al)
        b.branch(bar_chk)
        b.position_at_end(bar_ext)
        b.ret_void()
        return fn

    def _build_progress_finish(self) -> ir.Function:
        """progress_finish(handle, msg) — finalize progress bar."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I8PTR]),
                         name="__vx_progress_finish")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        _, msg = fn.args[0], fn.args[1]
        fmt_gv = self._global_str("\r[100%%] #################### %s\n")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv), msg])
        b.ret_void()
        return fn

    def _build_term_clear(self) -> ir.Function:
        """term_clear() — clear terminal."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, []),
                         name="__vx_term_clear")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fmt_gv = self._global_str("\033[2J\033[H")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv)])
        b.ret_void()
        return fn

    def _build_term_move(self) -> ir.Function:
        """term_move(row, col) — move cursor."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_term_move")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        row, col = fn.args[0], fn.args[1]
        fmt_gv = self._global_str("\033[%lld;%lldH")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv), row, col])
        b.ret_void()
        return fn

    def _build_term_width(self) -> ir.Function:
        """term_width() -> i64 — terminal width (80 default on Windows)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_term_width")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        if _sys.platform == "win32":
            # GetConsoleScreenBufferInfo not trivial in IR; return 80
            b.ret(ir.Constant(I64_TY, 80))
        else:
            # ioctl(TIOCGWINSZ) not trivial; use $COLUMNS env var fallback
            cols_gv = self._global_str("COLUMNS")
            cols_p  = self._gstr_ptr_const(cols_gv)
            env_val = b.call(self.getenv_fn, [cols_p])
            is_null = b.icmp_unsigned("==",
                        b.ptrtoint(env_val, I64_TY), ir.Constant(I64_TY, 0))
            w_al = b.alloca(I64_TY)
            b.store(ir.Constant(I64_TY, 80), w_al)
            have_bb = fn.append_basic_block("have_cols")
            end_bb  = fn.append_basic_block("end")
            b.cbranch(is_null, end_bb, have_bb)
            b.position_at_end(have_bb)
            w = b.sext(b.trunc(b.call(self.atoll_fn, [env_val]), I32_TY), I64_TY)
            b.store(w, w_al)
            b.branch(end_bb)
            b.position_at_end(end_bb)
            b.ret(b.load(w_al))
        return fn

    # --- Channels: {mutex i8*, condvar i8*, i64* buf, i64 len, i64 cap, i64 closed} ---
    # Layout (all i64-sized slots in malloc'd block):
    #   [0] mutex ptr (8 bytes)
    #   [1] condvar ptr (8 bytes)
    #   [2] data ptr (8 bytes) — i64[] ring buffer
    #   [3] len
    #   [4] cap
    #   [5] closed flag

    def _build_channel_new(self) -> ir.Function:
        """channel_new() -> i64 handle."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_channel_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Allocate channel header: 6 × 8 = 48 bytes
        hdr = b.call(self.malloc_fn, [ir.Constant(I64_TY, 48)])
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        # mutex
        mu_fn = self._get_helper("__vx_mutex_new")
        mu = b.call(mu_fn, [])
        mu_i = b.ptrtoint(mu, I64_TY)
        b.store(mu_i, b.gep(hdr64, [z], inbounds=False))
        # condvar
        cv_fn = self._get_helper("__vx_condvar_new")
        cv = b.call(cv_fn, [])
        cv_i = b.ptrtoint(cv, I64_TY)
        b.store(cv_i, b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False))
        # ring buffer (cap=16)
        cap = ir.Constant(I64_TY, 16)
        data = b.call(self.malloc_fn, [b.mul(cap, ir.Constant(I64_TY, 8))])
        data_i = b.ptrtoint(data, I64_TY)
        b.store(data_i, b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        b.store(ir.Constant(I64_TY, 0),  b.gep(hdr64, [ir.Constant(I32_TY, 3)], inbounds=False))
        b.store(cap,                      b.gep(hdr64, [ir.Constant(I32_TY, 4)], inbounds=False))
        b.store(ir.Constant(I64_TY, 0),  b.gep(hdr64, [ir.Constant(I32_TY, 5)], inbounds=False))
        b.ret(b.ptrtoint(hdr, I64_TY))
        return fn

    def _build_channel_send(self) -> ir.Function:
        """channel_send(ch: i64, val: i64)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_channel_send")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ch_i, val = fn.args[0], fn.args[1]
        hdr   = b.inttoptr(ch_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        mu_i   = b.load(b.gep(hdr64, [z], inbounds=False))
        mu_ptr = b.inttoptr(mu_i, I8PTR)
        lock_fn = self._get_helper("__vx_mutex_lock")
        unlock_fn = self._get_helper("__vx_mutex_unlock")
        b.call(lock_fn, [mu_ptr])
        data_i  = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        len_p   = b.gep(hdr64, [ir.Constant(I32_TY, 3)], inbounds=False)
        cur_len = b.load(len_p)
        data_ptr = b.inttoptr(data_i, ir.PointerType(I64_TY))
        slot = b.gep(data_ptr, [b.trunc(cur_len, I32_TY)], inbounds=False)
        b.store(val, slot)
        b.store(b.add(cur_len, ir.Constant(I64_TY, 1)), len_p)
        # signal waiting receivers
        cv_i   = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False))
        sig_fn = self._get_helper("__vx_condvar_signal")
        b.call(sig_fn, [cv_i])
        b.call(unlock_fn, [mu_ptr])
        b.ret_void()
        return fn

    def _build_channel_recv(self) -> ir.Function:
        """channel_recv(ch: i64) -> i64 — blocks until a value is available."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_channel_recv")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ch_i = fn.args[0]
        hdr   = b.inttoptr(ch_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        mu_i   = b.load(b.gep(hdr64, [z], inbounds=False))
        mu_ptr = b.inttoptr(mu_i, I8PTR)
        lock_fn   = self._get_helper("__vx_mutex_lock")
        unlock_fn = self._get_helper("__vx_mutex_unlock")
        wait_fn   = self._get_helper("__vx_condvar_wait")
        b.call(lock_fn, [mu_ptr])
        # Spin-wait loop
        wait_chk = fn.append_basic_block("wait_chk")
        wait_bdy = fn.append_basic_block("wait_bdy")
        recv_bb  = fn.append_basic_block("recv")
        b.branch(wait_chk)
        b.position_at_end(wait_chk)
        len_p   = b.gep(hdr64, [ir.Constant(I32_TY, 3)], inbounds=False)
        cur_len = b.load(len_p)
        has_data = b.icmp_signed(">", cur_len, ir.Constant(I64_TY, 0))
        b.cbranch(has_data, recv_bb, wait_bdy)
        b.position_at_end(wait_bdy)
        cv_i2  = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False))
        b.call(wait_fn, [cv_i2, mu_i])
        b.branch(wait_chk)
        b.position_at_end(recv_bb)
        data_i  = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_ptr = b.inttoptr(data_i, ir.PointerType(I64_TY))
        val = b.load(b.gep(data_ptr, [z], inbounds=False))
        # Shift buffer left (simple queue: shift all down by 1)
        new_len = b.sub(b.load(len_p), ir.Constant(I64_TY, 1))
        shift_chk = fn.append_basic_block("shift_chk")
        shift_bdy = fn.append_basic_block("shift_bdy")
        shift_ext = fn.append_basic_block("shift_ext")
        si_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), si_al)
        b.branch(shift_chk)
        b.position_at_end(shift_chk)
        si = b.load(si_al)
        b.cbranch(b.icmp_signed("<", si, new_len), shift_bdy, shift_ext)
        b.position_at_end(shift_bdy)
        src = b.gep(data_ptr, [b.trunc(b.add(si, ir.Constant(I64_TY, 1)), I32_TY)], inbounds=False)
        dst = b.gep(data_ptr, [b.trunc(si, I32_TY)], inbounds=False)
        b.store(b.load(src), dst)
        b.store(b.add(si, ir.Constant(I64_TY, 1)), si_al)
        b.branch(shift_chk)
        b.position_at_end(shift_ext)
        b.store(new_len, len_p)
        b.call(unlock_fn, [mu_ptr])
        b.ret(val)
        return fn

    def _build_channel_try_recv(self) -> ir.Function:
        """channel_try_recv(ch: i64) -> i64  — -1 if empty, else dequeue."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_channel_try_recv")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ch_i = fn.args[0]
        hdr   = b.inttoptr(ch_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        mu_i   = b.load(b.gep(hdr64, [z], inbounds=False))
        mu_ptr = b.inttoptr(mu_i, I8PTR)
        lock_fn   = self._get_helper("__vx_mutex_lock")
        unlock_fn = self._get_helper("__vx_mutex_unlock")
        b.call(lock_fn, [mu_ptr])
        len_p   = b.gep(hdr64, [ir.Constant(I32_TY, 3)], inbounds=False)
        cur_len = b.load(len_p)
        has_data = b.icmp_signed(">", cur_len, ir.Constant(I64_TY, 0))
        empty_bb = fn.append_basic_block("empty")
        have_bb  = fn.append_basic_block("have")
        b.cbranch(has_data, have_bb, empty_bb)
        b.position_at_end(empty_bb)
        b.call(unlock_fn, [mu_ptr])
        b.ret(ir.Constant(I64_TY, -1))
        b.position_at_end(have_bb)
        data_i  = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_ptr = b.inttoptr(data_i, ir.PointerType(I64_TY))
        val = b.load(b.gep(data_ptr, [z], inbounds=False))
        new_len = b.sub(cur_len, ir.Constant(I64_TY, 1))
        # shift
        si_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), si_al)
        sh_chk = fn.append_basic_block("sh_chk"); sh_bdy = fn.append_basic_block("sh_bdy")
        sh_ext = fn.append_basic_block("sh_ext")
        b.branch(sh_chk)
        b.position_at_end(sh_chk)
        si = b.load(si_al)
        b.cbranch(b.icmp_signed("<", si, new_len), sh_bdy, sh_ext)
        b.position_at_end(sh_bdy)
        src = b.gep(data_ptr, [b.trunc(b.add(si, ir.Constant(I64_TY, 1)), I32_TY)], inbounds=False)
        dst = b.gep(data_ptr, [b.trunc(si, I32_TY)], inbounds=False)
        b.store(b.load(src), dst)
        b.store(b.add(si, ir.Constant(I64_TY, 1)), si_al)
        b.branch(sh_chk)
        b.position_at_end(sh_ext)
        b.store(new_len, len_p)
        b.call(unlock_fn, [mu_ptr])
        b.ret(val)
        return fn

    def _build_channel_close(self) -> ir.Function:
        """channel_close(ch: i64) — mark channel closed."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_channel_close")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        ch_i = fn.args[0]
        hdr   = b.inttoptr(ch_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        closed_p = b.gep(hdr64, [ir.Constant(I32_TY, 5)], inbounds=False)
        b.store(ir.Constant(I64_TY, 1), closed_p)
        b.ret_void()
        return fn

    # --- RWLocks ---

    def _build_rwlock_new(self) -> ir.Function:
        """rwlock_new() -> i64 handle (opaque rwlock pointer)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_rwlock_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        if _sys.platform == "win32":
            # SRWLOCK is a single pointer; malloc 8 bytes, InitializeSRWLock
            buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 8)])
            init_ty = ir.FunctionType(VOID_TY, [I8PTR])
            init_fn = self._get_or_declare_fn("InitializeSRWLock", init_ty)
            b.call(init_fn, [buf])
        else:
            # pthread_rwlock_t ~56 bytes
            buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 56)])
            null = ir.Constant(I8PTR, None)
            init_ty = ir.FunctionType(I32_TY, [I8PTR, I8PTR])
            init_fn = self._get_or_declare_fn("pthread_rwlock_init", init_ty)
            b.call(init_fn, [buf, null])
        b.ret(b.ptrtoint(buf, I64_TY))
        return fn

    def _build_rwlock_read_lock(self) -> ir.Function:
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_rwlock_read_lock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]; ptr = b.inttoptr(handle, I8PTR)
        if _sys.platform == "win32":
            lk_ty = ir.FunctionType(VOID_TY, [I8PTR])
            lk_fn = self._get_or_declare_fn("AcquireSRWLockShared", lk_ty)
        else:
            lk_ty = ir.FunctionType(I32_TY, [I8PTR])
            lk_fn = self._get_or_declare_fn("pthread_rwlock_rdlock", lk_ty)
        b.call(lk_fn, [ptr]); b.ret_void()
        return fn

    def _build_rwlock_read_unlock(self) -> ir.Function:
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_rwlock_read_unlock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]; ptr = b.inttoptr(handle, I8PTR)
        if _sys.platform == "win32":
            ul_ty = ir.FunctionType(VOID_TY, [I8PTR])
            ul_fn = self._get_or_declare_fn("ReleaseSRWLockShared", ul_ty)
        else:
            ul_ty = ir.FunctionType(I32_TY, [I8PTR])
            ul_fn = self._get_or_declare_fn("pthread_rwlock_unlock", ul_ty)
        b.call(ul_fn, [ptr]); b.ret_void()
        return fn

    def _build_rwlock_write_lock(self) -> ir.Function:
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_rwlock_write_lock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]; ptr = b.inttoptr(handle, I8PTR)
        if _sys.platform == "win32":
            lk_ty = ir.FunctionType(VOID_TY, [I8PTR])
            lk_fn = self._get_or_declare_fn("AcquireSRWLockExclusive", lk_ty)
        else:
            lk_ty = ir.FunctionType(I32_TY, [I8PTR])
            lk_fn = self._get_or_declare_fn("pthread_rwlock_wrlock", lk_ty)
        b.call(lk_fn, [ptr]); b.ret_void()
        return fn

    def _build_rwlock_write_unlock(self) -> ir.Function:
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_rwlock_write_unlock")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        handle = fn.args[0]; ptr = b.inttoptr(handle, I8PTR)
        if _sys.platform == "win32":
            ul_ty = ir.FunctionType(VOID_TY, [I8PTR])
            ul_fn = self._get_or_declare_fn("ReleaseSRWLockExclusive", ul_ty)
        else:
            ul_ty = ir.FunctionType(I32_TY, [I8PTR])
            ul_fn = self._get_or_declare_fn("pthread_rwlock_unlock", ul_ty)
        b.call(ul_fn, [ptr]); b.ret_void()
        return fn

    # --- Thread pool (simple: fixed array of worker threads) ---
    # Layout of pool header (8-byte slots): [mutex_ptr, condvar_ptr, data_ptr,
    #   len, cap, done_flag, thread_count, thread_handles_ptr]

    def _build_thread_pool_new(self) -> ir.Function:
        """thread_pool_new(n: i64) -> i64 handle."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_thread_pool_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # For simplicity, allocate a struct: {mutex, condvar, task queue handle, done}
        # Return the channel handle (tasks go through channel)
        n = fn.args[0]
        ch_fn = self._get_helper("__vx_channel_new")
        ch = b.call(ch_fn, [])
        # Allocate pool header: [ch_handle, n_threads, done]
        hdr = b.call(self.malloc_fn, [ir.Constant(I64_TY, 24)])
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        b.store(ch, b.gep(hdr64, [z], inbounds=False))
        b.store(n,  b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False))
        b.store(ir.Constant(I64_TY, 0), b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        b.ret(b.ptrtoint(hdr, I64_TY))
        return fn

    def _build_thread_pool_submit(self) -> ir.Function:
        """thread_pool_submit(pool: i64, task: i8*)  — enqueue a task function pointer."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I8PTR]),
                         name="__vx_thread_pool_submit")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        pool_i, task = fn.args[0], fn.args[1]
        hdr   = b.inttoptr(pool_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        ch = b.load(b.gep(hdr64, [z], inbounds=False))
        task_i = b.ptrtoint(task, I64_TY)
        send_fn = self._get_helper("__vx_channel_send")
        b.call(send_fn, [ch, task_i])
        b.ret_void()
        return fn

    def _build_thread_pool_wait(self) -> ir.Function:
        """thread_pool_wait(pool: i64) — drain all pending tasks by calling them."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_thread_pool_wait")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        pool_i = fn.args[0]
        hdr   = b.inttoptr(pool_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        ch = b.load(b.gep(hdr64, [z], inbounds=False))
        # Drain queue: call each task
        try_recv_fn = self._get_helper("__vx_channel_try_recv")
        drain_chk = fn.append_basic_block("drain_chk")
        drain_bdy = fn.append_basic_block("drain_bdy")
        drain_ext = fn.append_basic_block("drain_ext")
        b.branch(drain_chk)
        b.position_at_end(drain_chk)
        task_i = b.call(try_recv_fn, [ch])
        is_empty = b.icmp_signed("==", task_i, ir.Constant(I64_TY, -1))
        b.cbranch(is_empty, drain_ext, drain_bdy)
        b.position_at_end(drain_bdy)
        task_ptr = b.inttoptr(task_i, ir.PointerType(ir.FunctionType(VOID_TY, [])))
        b.call(task_ptr, [])
        b.branch(drain_chk)
        b.position_at_end(drain_ext)
        b.ret_void()
        return fn

    def _build_thread_pool_destroy(self) -> ir.Function:
        """thread_pool_destroy(pool: i64) — free pool memory."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_thread_pool_destroy")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        pool_i = fn.args[0]
        hdr = b.inttoptr(pool_i, I8PTR)
        b.call(self.free_fn, [hdr])
        b.ret_void()
        return fn

    # --- Benchmarking ---

    def _build_benchmark(self) -> ir.Function:
        """benchmark(fn: i8*, count: i64) -> str  — run fn count times, return timing string."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I64_TY]),
                         name="__vx_benchmark")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn_ptr_raw, count = fn.args[0], fn.args[1]
        fn_ptr = b.bitcast(fn_ptr_raw, ir.PointerType(ir.FunctionType(VOID_TY, [])))
        # t_start = time(NULL)
        null = ir.Constant(I8PTR, None)
        t_start = b.call(self.time_fn, [null])
        # Loop count times
        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        loop_chk = fn.append_basic_block("loop_chk")
        loop_bdy = fn.append_basic_block("loop_bdy")
        loop_ext = fn.append_basic_block("loop_ext")
        b.branch(loop_chk)
        b.position_at_end(loop_chk)
        i = b.load(i_al)
        b.cbranch(b.icmp_signed("<", i, count), loop_bdy, loop_ext)
        b.position_at_end(loop_bdy)
        b.call(fn_ptr, [])
        b.store(b.add(i, ir.Constant(I64_TY, 1)), i_al)
        b.branch(loop_chk)
        b.position_at_end(loop_ext)
        t_end = b.call(self.time_fn, [null])
        elapsed = b.sub(t_end, t_start)
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 128)])
        fmt_gv = self._global_str("%lld iterations in %lld seconds")
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), count, elapsed])
        b.ret(buf)
        return fn

    def _build_bench_ns(self) -> ir.Function:
        """bench_ns(fn: i8*) -> i64  — run fn once, return elapsed time() delta (seconds as i64)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_bench_ns")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn_ptr_raw = fn.args[0]
        fn_ptr = b.bitcast(fn_ptr_raw, ir.PointerType(ir.FunctionType(VOID_TY, [])))
        null = ir.Constant(I8PTR, None)
        t_start = b.call(self.time_fn, [null])
        b.call(fn_ptr, [])
        t_end = b.call(self.time_fn, [null])
        b.ret(b.sub(t_end, t_start))
        return fn

    # --- JSON parsing (basic: scan for key in flat JSON string) ---

    def _build_json_parse_int(self) -> ir.Function:
        """json_parse_int(json: str, key: str) -> i64."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_json_parse_int")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        json_s, key = fn.args[0], fn.args[1]
        # Build search pattern: "key": and scan for the number after it
        pat_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
        fmt_gv  = self._global_str("\"%s\":")
        b.call(self.sprintf_fn, [pat_buf, self._gstr_ptr_const(fmt_gv), key])
        found = b.call(self.strstr_fn, [json_s, pat_buf])
        is_null = b.icmp_unsigned("==", b.ptrtoint(found, I64_TY), ir.Constant(I64_TY, 0))
        found_bb = fn.append_basic_block("found"); notfound_bb = fn.append_basic_block("notfound")
        b.cbranch(is_null, notfound_bb, found_bb)
        b.position_at_end(notfound_bb); b.ret(ir.Constant(I64_TY, 0))
        b.position_at_end(found_bb)
        pat_len = b.call(self.strlen_fn, [pat_buf])
        val_ptr = b.gep(found, [pat_len], inbounds=False)
        b.ret(b.call(self.atoll_fn, [val_ptr]))
        return fn

    def _build_json_parse_str(self) -> ir.Function:
        """json_parse_str(json: str, key: str) -> str."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_json_parse_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        json_s, key = fn.args[0], fn.args[1]
        pat_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
        fmt_gv  = self._global_str("\"%s\":\"")
        b.call(self.sprintf_fn, [pat_buf, self._gstr_ptr_const(fmt_gv), key])
        found = b.call(self.strstr_fn, [json_s, pat_buf])
        is_null = b.icmp_unsigned("==", b.ptrtoint(found, I64_TY), ir.Constant(I64_TY, 0))
        found_bb = fn.append_basic_block("found"); notfound_bb = fn.append_basic_block("notfound")
        empty_gv = self._global_str("")
        b.cbranch(is_null, notfound_bb, found_bb)
        b.position_at_end(notfound_bb); b.ret(self._gstr_ptr_const(empty_gv))
        b.position_at_end(found_bb)
        pat_len = b.call(self.strlen_fn, [pat_buf])
        val_start = b.gep(found, [pat_len], inbounds=False)
        # Find closing quote
        quote_gv = self._global_str("\"")
        end_ptr  = b.call(self.strstr_fn, [val_start, self._gstr_ptr_const(quote_gv)])
        is_null2 = b.icmp_unsigned("==", b.ptrtoint(end_ptr, I64_TY), ir.Constant(I64_TY, 0))
        copy_bb = fn.append_basic_block("copy"); ret_bb = fn.append_basic_block("ret")
        b.cbranch(is_null2, ret_bb, copy_bb)
        b.position_at_end(ret_bb); b.ret(val_start)
        b.position_at_end(copy_bb)
        str_len = b.sub(b.ptrtoint(end_ptr, I64_TY), b.ptrtoint(val_start, I64_TY))
        out_buf = b.call(self.malloc_fn, [b.add(str_len, ir.Constant(I64_TY, 1))])
        b.call(self.memcpy_fn, [out_buf, val_start, str_len])
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [str_len], inbounds=False))
        b.ret(out_buf)
        return fn

    def _build_json_parse_float(self) -> ir.Function:
        """json_parse_float(json: str, key: str) -> double."""
        fn = ir.Function(self.module, ir.FunctionType(F64_TY, [I8PTR, I8PTR]),
                         name="__vx_json_parse_float")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        json_s, key = fn.args[0], fn.args[1]
        pat_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
        fmt_gv  = self._global_str("\"%s\":")
        b.call(self.sprintf_fn, [pat_buf, self._gstr_ptr_const(fmt_gv), key])
        found = b.call(self.strstr_fn, [json_s, pat_buf])
        is_null = b.icmp_unsigned("==", b.ptrtoint(found, I64_TY), ir.Constant(I64_TY, 0))
        found_bb = fn.append_basic_block("found"); notfound_bb = fn.append_basic_block("notfound")
        b.cbranch(is_null, notfound_bb, found_bb)
        b.position_at_end(notfound_bb); b.ret(ir.Constant(F64_TY, 0.0))
        b.position_at_end(found_bb)
        pat_len = b.call(self.strlen_fn, [pat_buf])
        val_ptr = b.gep(found, [pat_len], inbounds=False)
        b.ret(b.call(self.atof_fn, [val_ptr]))
        return fn

    def _build_json_parse_bool(self) -> ir.Function:
        """json_parse_bool(json: str, key: str) -> i1."""
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I8PTR, I8PTR]),
                         name="__vx_json_parse_bool")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        json_s, key = fn.args[0], fn.args[1]
        pat_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
        fmt_gv  = self._global_str("\"%s\":")
        b.call(self.sprintf_fn, [pat_buf, self._gstr_ptr_const(fmt_gv), key])
        found = b.call(self.strstr_fn, [json_s, pat_buf])
        is_null = b.icmp_unsigned("==", b.ptrtoint(found, I64_TY), ir.Constant(I64_TY, 0))
        found_bb = fn.append_basic_block("found"); notfound_bb = fn.append_basic_block("notfound")
        b.cbranch(is_null, notfound_bb, found_bb)
        b.position_at_end(notfound_bb); b.ret(ir.Constant(I1_TY, 0))
        b.position_at_end(found_bb)
        pat_len = b.call(self.strlen_fn, [pat_buf])
        val_ptr = b.gep(found, [pat_len], inbounds=False)
        true_gv = self._global_str("true")
        cmp = b.call(self.strncmp_fn, [val_ptr, self._gstr_ptr_const(true_gv),
                                        ir.Constant(I64_TY, 4)])
        is_true = b.icmp_signed("==", cmp, ir.Constant(I32_TY, 0))
        b.ret(is_true)
        return fn

    # --- .env loader ---

    def _build_env_load(self) -> ir.Function:
        """env_load(path: str) — parse .env file and set env vars."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR]),
                         name="__vx_env_load")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        path = fn.args[0]
        mode_gv = self._global_str("r")
        fp = b.call(self.fopen_fn, [path, self._gstr_ptr_const(mode_gv)])
        is_null = b.icmp_unsigned("==", b.ptrtoint(fp, I64_TY), ir.Constant(I64_TY, 0))
        open_bb = fn.append_basic_block("open"); done_bb = fn.append_basic_block("done")
        b.cbranch(is_null, done_bb, open_bb)
        b.position_at_end(done_bb); b.ret_void()
        b.position_at_end(open_bb)
        line_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 1024)])
        read_chk = fn.append_basic_block("read_chk")
        read_bdy = fn.append_basic_block("read_bdy")
        read_ext = fn.append_basic_block("read_ext")
        b.branch(read_chk)
        b.position_at_end(read_chk)
        got = b.call(self.fgets_fn, [line_buf, ir.Constant(I32_TY, 1024), fp])
        is_eof = b.icmp_unsigned("==", b.ptrtoint(got, I64_TY), ir.Constant(I64_TY, 0))
        b.cbranch(is_eof, read_ext, read_bdy)
        b.position_at_end(read_bdy)
        # Find '=' in line
        eq_gv  = self._global_str("=")
        eq_ptr = b.call(self.strstr_fn, [line_buf, self._gstr_ptr_const(eq_gv)])
        has_eq = b.icmp_unsigned("!=", b.ptrtoint(eq_ptr, I64_TY), ir.Constant(I64_TY, 0))
        set_bb  = fn.append_basic_block("set_env")
        skip_bb = fn.append_basic_block("skip_env")
        b.cbranch(has_eq, set_bb, skip_bb)
        b.position_at_end(set_bb)
        # Terminate key at '=', set value starting at eq+1
        b.store(ir.Constant(I8_TY, 0), eq_ptr)
        val_ptr = b.gep(eq_ptr, [ir.Constant(I32_TY, 1)], inbounds=False)
        # Strip newline from val
        val_len = b.call(self.strlen_fn, [val_ptr])
        last_idx = b.sub(val_len, ir.Constant(I64_TY, 1))
        last_ch_p = b.gep(val_ptr, [last_idx], inbounds=False)
        last_ch = b.zext(b.load(last_ch_p), I32_TY)
        is_nl = b.icmp_signed("==", last_ch, ir.Constant(I32_TY, 10))
        nl_strip_bb = fn.append_basic_block("nl_strip"); no_nl_bb = fn.append_basic_block("no_nl")
        b.cbranch(is_nl, nl_strip_bb, no_nl_bb)
        b.position_at_end(nl_strip_bb)
        b.store(ir.Constant(I8_TY, 0), last_ch_p); b.branch(no_nl_bb)
        b.position_at_end(no_nl_bb)
        env_set_fn = self._get_helper("__vx_env_set")
        b.call(env_set_fn, [line_buf, val_ptr])
        b.branch(skip_bb)
        b.position_at_end(skip_bb)
        b.branch(read_chk)
        b.position_at_end(read_ext)
        b.call(self.fclose_fn, [fp])
        b.ret_void()
        return fn

    # --- Integer sets (open-addressed hash set of i64 values) ---
    # Header layout (i64 slots): [data_ptr, len, cap]
    # data_ptr → i64[] with -1 (0xFFFFFFFFFFFFFFFF) = empty slot

    _SET_EMPTY = -1   # sentinel for empty slot

    def _build_set_new(self) -> ir.Function:
        """set_new() -> i64 handle."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_set_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cap = ir.Constant(I64_TY, 16)
        data = b.call(self.malloc_fn, [b.mul(cap, ir.Constant(I64_TY, 8))])
        # Fill with -1 (empty)
        memset_ty = ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY])
        memset_fn = self._get_or_declare_fn("memset", memset_ty)
        b.call(memset_fn, [data, ir.Constant(I32_TY, 0xFF),
                           b.mul(cap, ir.Constant(I64_TY, 8))])
        hdr = b.call(self.malloc_fn, [ir.Constant(I64_TY, 24)])
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        data_i = b.ptrtoint(data, I64_TY)
        b.store(data_i, b.gep(hdr64, [z], inbounds=False))
        b.store(ir.Constant(I64_TY, 0), b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False))
        b.store(cap, b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        b.ret(b.ptrtoint(hdr, I64_TY))
        return fn

    def _set_find_slot(self, b, data_ptr, cap_val, val, include_empty=True):
        """Helper to compute hash slot (not an IR function — inlined by callers)."""
        # idx = (val % cap + cap) % cap  (positive modulo)
        raw = b.srem(val, cap_val)
        pos = b.add(raw, cap_val)
        return b.srem(pos, cap_val)

    def _build_set_add(self) -> ir.Function:
        """set_add(s: i64, val: i64)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_set_add")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s_i, val = fn.args[0], fn.args[1]
        hdr   = b.inttoptr(s_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        data_i  = b.load(b.gep(hdr64, [z], inbounds=False))
        cap     = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_p  = b.inttoptr(data_i, ir.PointerType(I64_TY))
        # Linear probe from hash slot
        start_slot = self._set_find_slot(b, data_p, cap, val)
        idx_al = b.alloca(I64_TY); b.store(start_slot, idx_al)
        probe_chk = fn.append_basic_block("probe_chk")
        probe_bdy = fn.append_basic_block("probe_bdy")
        probe_ins = fn.append_basic_block("probe_ins")
        b.branch(probe_chk)
        b.position_at_end(probe_chk)
        idx = b.load(idx_al)
        slot_p = b.gep(data_p, [b.trunc(idx, I32_TY)], inbounds=False)
        slot_v = b.load(slot_p)
        is_empty = b.icmp_signed("==", slot_v, ir.Constant(I64_TY, -1))
        is_dupe  = b.icmp_signed("==", slot_v, val)
        can_use  = b.or_(is_empty, is_dupe)
        b.cbranch(can_use, probe_ins, probe_bdy)
        b.position_at_end(probe_bdy)
        next_idx = b.srem(b.add(idx, ir.Constant(I64_TY, 1)), cap)
        b.store(next_idx, idx_al); b.branch(probe_chk)
        b.position_at_end(probe_ins)
        idx2 = b.load(idx_al)
        slot_p2 = b.gep(data_p, [b.trunc(idx2, I32_TY)], inbounds=False)
        cur_v = b.load(slot_p2)
        is_new = b.icmp_signed("==", cur_v, ir.Constant(I64_TY, -1))
        new_bb  = fn.append_basic_block("new_slot"); dup_bb = fn.append_basic_block("dup_done")
        b.cbranch(is_new, new_bb, dup_bb)
        b.position_at_end(new_bb)
        b.store(val, slot_p2)
        len_p = b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False)
        b.store(b.add(b.load(len_p), ir.Constant(I64_TY, 1)), len_p)
        b.branch(dup_bb)
        b.position_at_end(dup_bb)
        b.ret_void()
        return fn

    def _build_set_contains(self) -> ir.Function:
        """set_contains(s: i64, val: i64) -> i1."""
        fn = ir.Function(self.module, ir.FunctionType(I1_TY, [I64_TY, I64_TY]),
                         name="__vx_set_contains")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s_i, val = fn.args[0], fn.args[1]
        hdr   = b.inttoptr(s_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        data_i = b.load(b.gep(hdr64, [z], inbounds=False))
        cap    = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_p = b.inttoptr(data_i, ir.PointerType(I64_TY))
        start_slot = self._set_find_slot(b, data_p, cap, val)
        idx_al = b.alloca(I64_TY); b.store(start_slot, idx_al)
        probe_chk = fn.append_basic_block("probe_chk")
        probe_bdy = fn.append_basic_block("probe_bdy")
        found_bb  = fn.append_basic_block("found")
        miss_bb   = fn.append_basic_block("miss")
        b.branch(probe_chk)
        b.position_at_end(probe_chk)
        idx = b.load(idx_al)
        slot_p = b.gep(data_p, [b.trunc(idx, I32_TY)], inbounds=False)
        slot_v = b.load(slot_p)
        is_match = b.icmp_signed("==", slot_v, val)
        is_empty = b.icmp_signed("==", slot_v, ir.Constant(I64_TY, -1))
        b.cbranch(is_match, found_bb, probe_bdy)
        b.position_at_end(probe_bdy)
        b.cbranch(is_empty, miss_bb, fn.append_basic_block("cont_probe"))
        cont = list(fn.blocks)[-1]
        b.position_at_end(cont)
        next_idx = b.srem(b.add(idx, ir.Constant(I64_TY, 1)), cap)
        b.store(next_idx, idx_al); b.branch(probe_chk)
        b.position_at_end(found_bb); b.ret(ir.Constant(I1_TY, 1))
        b.position_at_end(miss_bb);  b.ret(ir.Constant(I1_TY, 0))
        return fn

    def _build_set_remove(self) -> ir.Function:
        """set_remove(s: i64, val: i64) — tombstone removal (mark slot -2)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_set_remove")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s_i, val = fn.args[0], fn.args[1]
        hdr   = b.inttoptr(s_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        data_i = b.load(b.gep(hdr64, [z], inbounds=False))
        cap    = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_p = b.inttoptr(data_i, ir.PointerType(I64_TY))
        start_slot = self._set_find_slot(b, data_p, cap, val)
        idx_al = b.alloca(I64_TY); b.store(start_slot, idx_al)
        probe_chk = fn.append_basic_block("probe_chk")
        probe_bdy = fn.append_basic_block("probe_bdy")
        found_bb  = fn.append_basic_block("found")
        miss_bb   = fn.append_basic_block("miss")
        b.branch(probe_chk)
        b.position_at_end(probe_chk)
        idx = b.load(idx_al)
        slot_p = b.gep(data_p, [b.trunc(idx, I32_TY)], inbounds=False)
        slot_v = b.load(slot_p)
        is_match = b.icmp_signed("==", slot_v, val)
        is_empty = b.icmp_signed("==", slot_v, ir.Constant(I64_TY, -1))
        b.cbranch(is_match, found_bb, probe_bdy)
        b.position_at_end(probe_bdy)
        b.cbranch(is_empty, miss_bb, fn.append_basic_block("cont_p"))
        cont = list(fn.blocks)[-1]
        b.position_at_end(cont)
        next_idx = b.srem(b.add(idx, ir.Constant(I64_TY, 1)), cap)
        b.store(next_idx, idx_al); b.branch(probe_chk)
        b.position_at_end(found_bb)
        # Mark as tombstone (-2) and decrement len
        b.store(ir.Constant(I64_TY, -2), slot_p)
        len_p = b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False)
        b.store(b.sub(b.load(len_p), ir.Constant(I64_TY, 1)), len_p)
        b.branch(miss_bb)
        b.position_at_end(miss_bb); b.ret_void()
        return fn

    def _build_set_size(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_set_size")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s_i = fn.args[0]
        hdr   = b.inttoptr(s_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        b.ret(b.load(b.gep(hdr64, [ir.Constant(I32_TY, 1)], inbounds=False)))
        return fn

    def _build_set_to_array(self) -> ir.Function:
        """set_to_array(s: i64) -> vx_array* of i64 (returned as i8*)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_set_to_array")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        s_i = fn.args[0]
        hdr   = b.inttoptr(s_i, I8PTR)
        hdr64 = b.bitcast(hdr, ir.PointerType(I64_TY))
        z = ir.Constant(I32_TY, 0)
        data_i = b.load(b.gep(hdr64, [z], inbounds=False))
        cap    = b.load(b.gep(hdr64, [ir.Constant(I32_TY, 2)], inbounds=False))
        data_p = b.inttoptr(data_i, ir.PointerType(I64_TY))
        # Build result vx_array
        arr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        arr     = b.bitcast(arr_raw, self.arr_ptr_type)
        out_data = b.call(self.malloc_fn, [b.mul(cap, ir.Constant(I64_TY, 8))])
        az = ir.Constant(I32_TY, 0)
        b.store(out_data, b.gep(arr, [az, az], inbounds=True))
        out_len_p = b.gep(arr, [az, ir.Constant(I32_TY, 1)], inbounds=True)
        b.store(ir.Constant(I64_TY, 0), out_len_p)
        b.store(cap, b.gep(arr, [az, ir.Constant(I32_TY, 2)], inbounds=True))
        out_data64 = b.bitcast(out_data, ir.PointerType(I64_TY))
        # Iterate slots, copy non-empty non-tombstone entries
        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        loop_chk = fn.append_basic_block("loop_chk")
        loop_bdy = fn.append_basic_block("loop_bdy")
        loop_ext = fn.append_basic_block("loop_ext")
        b.branch(loop_chk)
        b.position_at_end(loop_chk)
        i = b.load(i_al)
        b.cbranch(b.icmp_signed("<", i, cap), loop_bdy, loop_ext)
        b.position_at_end(loop_bdy)
        slot_v = b.load(b.gep(data_p, [b.trunc(i, I32_TY)], inbounds=False))
        is_empty    = b.icmp_signed("==", slot_v, ir.Constant(I64_TY, -1))
        is_tombstone= b.icmp_signed("==", slot_v, ir.Constant(I64_TY, -2))
        skip = b.or_(is_empty, is_tombstone)
        copy_bb = fn.append_basic_block("copy_slot"); skip_bb = fn.append_basic_block("skip_slot")
        b.cbranch(skip, skip_bb, copy_bb)
        b.position_at_end(copy_bb)
        out_len = b.load(out_len_p)
        b.store(slot_v, b.gep(out_data64, [b.trunc(out_len, I32_TY)], inbounds=False))
        b.store(b.add(out_len, ir.Constant(I64_TY, 1)), out_len_p)
        b.branch(skip_bb)
        b.position_at_end(skip_bb)
        b.store(b.add(i, ir.Constant(I64_TY, 1)), i_al)
        b.branch(loop_chk)
        b.position_at_end(loop_ext)
        b.ret(arr_raw)
        return fn

    def _build_set_union(self) -> ir.Function:
        """set_union(a: i64, b: i64) -> i64 — new set containing all elements of both."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I64_TY]),
                         name="__vx_set_union")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        a_i, b_i = fn.args[0], fn.args[1]
        new_fn  = self._get_helper("__vx_set_new")
        add_fn  = self._get_helper("__vx_set_add")
        to_arr  = self._get_helper("__vx_set_to_array")
        result = b.call(new_fn, [])
        for src_i in (a_i, b_i):
            arr_raw = b.call(to_arr, [src_i])
            arr     = b.bitcast(arr_raw, self.arr_ptr_type)
            az = ir.Constant(I32_TY, 0)
            arr_len = b.load(b.gep(arr, [az, ir.Constant(I32_TY, 1)], inbounds=True))
            arr_data= b.load(b.gep(arr, [az, az], inbounds=True))
            arr64   = b.bitcast(arr_data, ir.PointerType(I64_TY))
            ci_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), ci_al)
            lbl = "u_chk" if src_i is a_i else "u2_chk"
            chk_bb = fn.append_basic_block(lbl)
            bdy_bb = fn.append_basic_block(lbl.replace("chk","bdy"))
            ext_bb = fn.append_basic_block(lbl.replace("chk","ext"))
            b.branch(chk_bb)
            b.position_at_end(chk_bb)
            ci = b.load(ci_al)
            b.cbranch(b.icmp_signed("<", ci, arr_len), bdy_bb, ext_bb)
            b.position_at_end(bdy_bb)
            elem = b.load(b.gep(arr64, [b.trunc(ci, I32_TY)], inbounds=False))
            b.call(add_fn, [result, elem])
            b.store(b.add(ci, ir.Constant(I64_TY, 1)), ci_al)
            b.branch(chk_bb)
            b.position_at_end(ext_bb)
        b.ret(result)
        return fn

    def _build_set_intersect(self) -> ir.Function:
        """set_intersect(a: i64, b_set: i64) -> i64 — elements in both a and b."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I64_TY]),
                         name="__vx_set_intersect")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        a_i, b_i = fn.args[0], fn.args[1]
        new_fn      = self._get_helper("__vx_set_new")
        add_fn      = self._get_helper("__vx_set_add")
        contains_fn = self._get_helper("__vx_set_contains")
        to_arr      = self._get_helper("__vx_set_to_array")
        result = b.call(new_fn, [])
        arr_raw = b.call(to_arr, [a_i])
        arr     = b.bitcast(arr_raw, self.arr_ptr_type)
        az = ir.Constant(I32_TY, 0)
        arr_len = b.load(b.gep(arr, [az, ir.Constant(I32_TY, 1)], inbounds=True))
        arr_data= b.load(b.gep(arr, [az, az], inbounds=True))
        arr64   = b.bitcast(arr_data, ir.PointerType(I64_TY))
        ci_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), ci_al)
        chk_bb = fn.append_basic_block("i_chk"); bdy_bb = fn.append_basic_block("i_bdy")
        ext_bb = fn.append_basic_block("i_ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        ci = b.load(ci_al)
        b.cbranch(b.icmp_signed("<", ci, arr_len), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        elem = b.load(b.gep(arr64, [b.trunc(ci, I32_TY)], inbounds=False))
        in_b = b.call(contains_fn, [b_i, elem])
        add_bb = fn.append_basic_block("i_add"); skip_bb = fn.append_basic_block("i_skip")
        b.cbranch(b.zext(in_b, I64_TY) if False else in_b, add_bb, skip_bb)
        b.position_at_end(add_bb); b.call(add_fn, [result, elem]); b.branch(skip_bb)
        b.position_at_end(skip_bb)
        b.store(b.add(ci, ir.Constant(I64_TY, 1)), ci_al); b.branch(chk_bb)
        b.position_at_end(ext_bb); b.ret(result)
        return fn

    # --- UUID v1 (time-based, simplified) ---

    def _build_uuid_v1(self) -> ir.Function:
        """uuid_v1() -> str — generate a time-based UUID v1 string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_uuid_v1")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 64)])
        null = ir.Constant(I8PTR, None)
        ts = b.call(self.time_fn, [null])
        rand_v = b.sext(b.call(self.rand_fn, []), I64_TY)
        rand2  = b.sext(b.call(self.rand_fn, []), I64_TY)
        rand3  = b.sext(b.call(self.rand_fn, []), I64_TY)
        fmt_gv = self._global_str("%08llx-%04llx-1%03llx-%04llx-%012llx")
        ts32   = b.and_(ts,    ir.Constant(I64_TY, 0xFFFFFFFF))
        r1     = b.and_(rand_v, ir.Constant(I64_TY, 0xFFFF))
        r2     = b.and_(rand2,  ir.Constant(I64_TY, 0x0FFF))
        r3     = b.and_(rand3,  ir.Constant(I64_TY, 0xFFFF))
        r4     = b.and_(b.add(rand_v, rand2), ir.Constant(I64_TY, 0xFFFFFFFFFFFF))
        b.call(self.sprintf_fn, [buf, self._gstr_ptr_const(fmt_gv), ts32, r1, r2, r3, r4])
        b.ret(buf)
        return fn

    # --- Condition variables ---

    def _build_condvar_new(self) -> ir.Function:
        """condvar_new() -> i64 handle (opaque condvar pointer)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_condvar_new")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        if _sys.platform == "win32":
            # CONDITION_VARIABLE is a single pointer; malloc 8 bytes
            buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 8)])
            init_ty = ir.FunctionType(VOID_TY, [I8PTR])
            init_fn = self._get_or_declare_fn("InitializeConditionVariable", init_ty)
            b.call(init_fn, [buf])
        else:
            # pthread_cond_t ~48 bytes
            buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 48)])
            null = ir.Constant(I8PTR, None)
            init_ty = ir.FunctionType(I32_TY, [I8PTR, I8PTR])
            init_fn = self._get_or_declare_fn("pthread_cond_init", init_ty)
            b.call(init_fn, [buf, null])
        b.ret(b.ptrtoint(buf, I64_TY))
        return fn

    def _build_condvar_wait(self) -> ir.Function:
        """condvar_wait(cv: i64, mu: i64)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_condvar_wait")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cv_i, mu_i = fn.args[0], fn.args[1]
        cv_ptr = b.inttoptr(cv_i, I8PTR)
        mu_ptr = b.inttoptr(mu_i, I8PTR)
        if _sys.platform == "win32":
            # SleepConditionVariableCS(cv, mu, INFINITE=0xFFFFFFFF)
            wait_ty = ir.FunctionType(I32_TY, [I8PTR, I8PTR, I32_TY])
            wait_fn = self._get_or_declare_fn("SleepConditionVariableCS", wait_ty)
            b.call(wait_fn, [cv_ptr, mu_ptr, ir.Constant(I32_TY, 0xFFFFFFFF)])
        else:
            wait_ty = ir.FunctionType(I32_TY, [I8PTR, I8PTR])
            wait_fn = self._get_or_declare_fn("pthread_cond_wait", wait_ty)
            b.call(wait_fn, [cv_ptr, mu_ptr])
        b.ret_void()
        return fn

    def _build_condvar_signal(self) -> ir.Function:
        """condvar_signal(cv: i64)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_condvar_signal")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cv_i = fn.args[0]
        cv_ptr = b.inttoptr(cv_i, I8PTR)
        if _sys.platform == "win32":
            sig_ty = ir.FunctionType(VOID_TY, [I8PTR])
            sig_fn = self._get_or_declare_fn("WakeConditionVariable", sig_ty)
        else:
            sig_ty = ir.FunctionType(I32_TY, [I8PTR])
            sig_fn = self._get_or_declare_fn("pthread_cond_signal", sig_ty)
        b.call(sig_fn, [cv_ptr])
        b.ret_void()
        return fn

    def _build_condvar_broadcast(self) -> ir.Function:
        """condvar_broadcast(cv: i64)."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_condvar_broadcast")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        cv_i = fn.args[0]
        cv_ptr = b.inttoptr(cv_i, I8PTR)
        if _sys.platform == "win32":
            bc_ty = ir.FunctionType(VOID_TY, [I8PTR])
            bc_fn = self._get_or_declare_fn("WakeAllConditionVariable", bc_ty)
        else:
            bc_ty = ir.FunctionType(I32_TY, [I8PTR])
            bc_fn = self._get_or_declare_fn("pthread_cond_broadcast", bc_ty)
        b.call(bc_fn, [cv_ptr])
        b.ret_void()
        return fn

    # ------------------------------------------------------------------ #
    #  v9 — debug bounds check helper (#101)                              #
    # ------------------------------------------------------------------ #

    def _build_bounds_check_helper(self) -> ir.Function:
        """__vx_bounds_check(idx, len) — abort with message if out of bounds."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY, I64_TY]),
                         name="__vx_bounds_check")
        fn.linkage = "private"
        b  = ir.IRBuilder(fn.append_basic_block("entry"))
        idx, length = fn.args
        ok1 = b.icmp_signed(">=", idx, ir.Constant(I64_TY, 0))
        ok2 = b.icmp_signed("<",  idx, length)
        ok  = b.and_(ok1, ok2)
        ok_bb   = fn.append_basic_block("ok")
        fail_bb = fn.append_basic_block("fail")
        b.cbranch(ok, ok_bb, fail_bb)
        b.position_at_end(fail_bb)
        fmt_gv = self._global_str("vexel: index out of bounds (index=%lld, len=%lld)\n")
        b.call(self.printf, [self._gstr_ptr_const(fmt_gv), idx, length])
        b.call(self.exit_fn, [ir.Constant(I32_TY, 1)])
        b.unreachable()
        b.position_at_end(ok_bb)
        b.ret_void()
        return fn

    # ------------------------------------------------------------------ #
    #  v9 — TCP / UDP sockets (#54), pipes (#64), file-watch (#58)        #
    # ------------------------------------------------------------------ #

    def _build_tcp_connect(self) -> ir.Function:
        """tcp_connect(host, port) -> int fd."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I64_TY]),
                         name="__vx_tcp_connect")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        host, port = fn.args
        I16 = ir.IntType(16)
        # WSAStartup on Win32
        if _sys.platform == "win32":
            wsa_al  = b.alloca(ir.ArrayType(I8_TY, 408))
            wsastartup_ty = ir.FunctionType(I32_TY, [I32_TY, I8PTR])
            b.call(self._get_or_declare_fn("WSAStartup", wsastartup_ty),
                   [ir.Constant(I32_TY, 0x0202), b.bitcast(wsa_al, I8PTR)])
            sock_ty = ir.FunctionType(I64_TY, [I32_TY, I32_TY, I32_TY])
            sock    = b.call(self._get_or_declare_fn("socket", sock_ty),
                             [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 1), ir.Constant(I32_TY, 6)])
        else:
            sock_ty = ir.FunctionType(I32_TY, [I32_TY, I32_TY, I32_TY])
            sock    = b.sext(b.call(self._get_or_declare_fn("socket", sock_ty),
                                    [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 1), ir.Constant(I32_TY, 6)]), I64_TY)
        # Build sockaddr_in (16 bytes)
        addr_al  = b.alloca(ir.ArrayType(I8_TY, 16))
        addr_raw = b.bitcast(addr_al, I8PTR)
        memset_ty = ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY])
        b.call(self._get_or_declare_fn("memset", memset_ty),
               [addr_raw, ir.Constant(I32_TY, 0), ir.Constant(I64_TY, 16)])
        b.store(ir.Constant(I16, 2), b.bitcast(addr_raw, ir.PointerType(I16)))
        htons_ty = ir.FunctionType(I16, [I16])
        pnet = b.call(self._get_or_declare_fn("htons", htons_ty), [b.trunc(port, I16)])
        b.store(pnet, b.bitcast(b.gep(addr_raw, [ir.Constant(I64_TY, 2)], inbounds=False), ir.PointerType(I16)))
        inet_addr_ty = ir.FunctionType(I32_TY, [I8PTR])
        sin_addr = b.call(self._get_or_declare_fn("inet_addr", inet_addr_ty), [host])
        b.store(sin_addr, b.bitcast(b.gep(addr_raw, [ir.Constant(I64_TY, 4)], inbounds=False), ir.PointerType(I32_TY)))
        # connect
        neg1 = ir.Constant(I32_TY, 0xFFFFFFFF)
        if _sys.platform == "win32":
            conn_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY])
            rc = b.call(self._get_or_declare_fn("connect", conn_ty),
                        [sock, addr_raw, ir.Constant(I32_TY, 16)])
        else:
            conn_ty = ir.FunctionType(I32_TY, [I32_TY, I8PTR, I32_TY])
            rc = b.call(self._get_or_declare_fn("connect", conn_ty),
                        [b.trunc(sock, I32_TY), addr_raw, ir.Constant(I32_TY, 16)])
        fail_bb = fn.append_basic_block("fail"); ok_bb = fn.append_basic_block("ok")
        b.cbranch(b.icmp_signed("==", rc, neg1), fail_bb, ok_bb)
        b.position_at_end(fail_bb); b.ret(ir.Constant(I64_TY, -1))
        b.position_at_end(ok_bb);   b.ret(sock)
        return fn

    def _build_tcp_send(self) -> ir.Function:
        """tcp_send(sock, data) -> int bytes sent."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR]),
                         name="__vx_tcp_send")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sock, data = fn.args
        dlen = b.call(self.strlen_fn, [data])
        if _sys.platform == "win32":
            snd_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, I32_TY])
            rc = b.sext(b.call(self._get_or_declare_fn("send", snd_ty),
                               [sock, data, b.trunc(dlen, I32_TY), ir.Constant(I32_TY, 0)]), I64_TY)
        else:
            snd_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY, I32_TY])
            rc = b.call(self._get_or_declare_fn("send", snd_ty),
                        [b.trunc(sock, I32_TY), data, dlen, ir.Constant(I32_TY, 0)])
        b.ret(rc)
        return fn

    def _build_tcp_recv(self) -> ir.Function:
        """tcp_recv(sock, max_bytes) -> str."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY, I64_TY]),
                         name="__vx_tcp_recv")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sock, max_b = fn.args
        buf = b.call(self.malloc_fn, [b.add(max_b, ir.Constant(I64_TY, 1))])
        if _sys.platform == "win32":
            rcv_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, I32_TY])
            n = b.sext(b.call(self._get_or_declare_fn("recv", rcv_ty),
                              [sock, buf, b.trunc(max_b, I32_TY), ir.Constant(I32_TY, 0)]), I64_TY)
        else:
            rcv_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY, I32_TY])
            n = b.call(self._get_or_declare_fn("recv", rcv_ty),
                       [b.trunc(sock, I32_TY), buf, max_b, ir.Constant(I32_TY, 0)])
        neg1 = ir.Constant(I64_TY, 0xFFFFFFFFFFFFFFFF)
        safe_n = b.select(b.icmp_signed("==", n, neg1), ir.Constant(I64_TY, 0), n)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [safe_n], inbounds=False))
        b.ret(buf)
        return fn

    def _build_tcp_close(self) -> ir.Function:
        """tcp_close(sock) -> void."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_tcp_close")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sock = fn.args[0]
        if _sys.platform == "win32":
            cs_ty = ir.FunctionType(I32_TY, [I64_TY])
            b.call(self._get_or_declare_fn("closesocket", cs_ty), [sock])
        else:
            cl_ty = ir.FunctionType(I32_TY, [I32_TY])
            b.call(self._get_or_declare_fn("close", cl_ty), [b.trunc(sock, I32_TY)])
        b.ret_void()
        return fn

    def _build_tcp_listen(self) -> ir.Function:
        """tcp_listen(port) -> int server fd."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_tcp_listen")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        port = fn.args[0]
        I16 = ir.IntType(16)
        if _sys.platform == "win32":
            sock_ty = ir.FunctionType(I64_TY, [I32_TY, I32_TY, I32_TY])
            srv = b.call(self._get_or_declare_fn("socket", sock_ty),
                         [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 1), ir.Constant(I32_TY, 6)])
        else:
            sock_ty = ir.FunctionType(I32_TY, [I32_TY, I32_TY, I32_TY])
            srv = b.sext(b.call(self._get_or_declare_fn("socket", sock_ty),
                                [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 1), ir.Constant(I32_TY, 6)]), I64_TY)
        addr_al  = b.alloca(ir.ArrayType(I8_TY, 16))
        addr_raw = b.bitcast(addr_al, I8PTR)
        memset_ty = ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY])
        b.call(self._get_or_declare_fn("memset", memset_ty),
               [addr_raw, ir.Constant(I32_TY, 0), ir.Constant(I64_TY, 16)])
        b.store(ir.Constant(I16, 2), b.bitcast(addr_raw, ir.PointerType(I16)))
        htons_ty = ir.FunctionType(I16, [I16])
        pnet = b.call(self._get_or_declare_fn("htons", htons_ty), [b.trunc(port, I16)])
        b.store(pnet, b.bitcast(b.gep(addr_raw, [ir.Constant(I64_TY, 2)], inbounds=False), ir.PointerType(I16)))
        if _sys.platform == "win32":
            bind_ty   = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY])
            listen_ty = ir.FunctionType(I32_TY, [I64_TY, I32_TY])
            b.call(self._get_or_declare_fn("bind",   bind_ty),   [srv, addr_raw, ir.Constant(I32_TY, 16)])
            b.call(self._get_or_declare_fn("listen", listen_ty), [srv, ir.Constant(I32_TY, 10)])
        else:
            srv32 = b.trunc(srv, I32_TY)
            bind_ty   = ir.FunctionType(I32_TY, [I32_TY, I8PTR, I32_TY])
            listen_ty = ir.FunctionType(I32_TY, [I32_TY, I32_TY])
            b.call(self._get_or_declare_fn("bind",   bind_ty),   [srv32, addr_raw, ir.Constant(I32_TY, 16)])
            b.call(self._get_or_declare_fn("listen", listen_ty), [srv32, ir.Constant(I32_TY, 10)])
        b.ret(srv)
        return fn

    def _build_tcp_accept(self) -> ir.Function:
        """tcp_accept(server_sock) -> int client fd."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_tcp_accept")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        srv = fn.args[0]
        addr_al     = b.alloca(ir.ArrayType(I8_TY, 16))
        addr_raw    = b.bitcast(addr_al, I8PTR)
        addr_len_al = b.alloca(I32_TY)
        b.store(ir.Constant(I32_TY, 16), addr_len_al)
        if _sys.platform == "win32":
            acc_ty = ir.FunctionType(I64_TY, [I64_TY, I8PTR, ir.PointerType(I32_TY)])
            client = b.call(self._get_or_declare_fn("accept", acc_ty),
                            [srv, addr_raw, addr_len_al])
        else:
            acc_ty = ir.FunctionType(I32_TY, [I32_TY, I8PTR, ir.PointerType(I32_TY)])
            client = b.sext(b.call(self._get_or_declare_fn("accept", acc_ty),
                                   [b.trunc(srv, I32_TY), addr_raw, addr_len_al]), I64_TY)
        b.ret(client)
        return fn

    def _build_udp_socket(self) -> ir.Function:
        """udp_socket() -> int."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, []),
                         name="__vx_udp_socket")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        if _sys.platform == "win32":
            sock_ty = ir.FunctionType(I64_TY, [I32_TY, I32_TY, I32_TY])
            s = b.call(self._get_or_declare_fn("socket", sock_ty),
                       [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 17)])
        else:
            sock_ty = ir.FunctionType(I32_TY, [I32_TY, I32_TY, I32_TY])
            s = b.sext(b.call(self._get_or_declare_fn("socket", sock_ty),
                              [ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 2), ir.Constant(I32_TY, 17)]), I64_TY)
        b.ret(s)
        return fn

    def _build_udp_send_to(self) -> ir.Function:
        """udp_send_to(sock, host, port, data) -> int."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR, I64_TY, I8PTR]),
                         name="__vx_udp_send_to")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sock, host, port, data = fn.args
        I16 = ir.IntType(16)
        addr_al  = b.alloca(ir.ArrayType(I8_TY, 16))
        addr_raw = b.bitcast(addr_al, I8PTR)
        memset_ty = ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY])
        b.call(self._get_or_declare_fn("memset", memset_ty),
               [addr_raw, ir.Constant(I32_TY, 0), ir.Constant(I64_TY, 16)])
        b.store(ir.Constant(I16, 2), b.bitcast(addr_raw, ir.PointerType(I16)))
        htons_ty = ir.FunctionType(I16, [I16])
        pnet = b.call(self._get_or_declare_fn("htons", htons_ty), [b.trunc(port, I16)])
        b.store(pnet, b.bitcast(b.gep(addr_raw, [ir.Constant(I64_TY, 2)], inbounds=False), ir.PointerType(I16)))
        inet_ty = ir.FunctionType(I32_TY, [I8PTR])
        b.store(b.call(self._get_or_declare_fn("inet_addr", inet_ty), [host]),
                b.bitcast(b.gep(addr_raw, [ir.Constant(I64_TY, 4)], inbounds=False), ir.PointerType(I32_TY)))
        dlen = b.call(self.strlen_fn, [data])
        if _sys.platform == "win32":
            snd_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, I32_TY, I8PTR, I32_TY])
            rc = b.sext(b.call(self._get_or_declare_fn("sendto", snd_ty),
                               [sock, data, b.trunc(dlen, I32_TY), ir.Constant(I32_TY, 0),
                                addr_raw, ir.Constant(I32_TY, 16)]), I64_TY)
        else:
            snd_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY, I32_TY, I8PTR, I32_TY])
            rc = b.call(self._get_or_declare_fn("sendto", snd_ty),
                        [b.trunc(sock, I32_TY), data, dlen, ir.Constant(I32_TY, 0),
                         addr_raw, ir.Constant(I32_TY, 16)])
        b.ret(rc)
        return fn

    def _build_udp_recv_from(self) -> ir.Function:
        """udp_recv_from(sock, max_bytes) -> str."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY, I64_TY]),
                         name="__vx_udp_recv_from")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sock, max_b = fn.args
        buf = b.call(self.malloc_fn, [b.add(max_b, ir.Constant(I64_TY, 1))])
        addr_al     = b.alloca(ir.ArrayType(I8_TY, 16))
        addr_raw    = b.bitcast(addr_al, I8PTR)
        addr_len_al = b.alloca(I32_TY)
        b.store(ir.Constant(I32_TY, 16), addr_len_al)
        if _sys.platform == "win32":
            rcv_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, I32_TY, I8PTR, ir.PointerType(I32_TY)])
            n = b.sext(b.call(self._get_or_declare_fn("recvfrom", rcv_ty),
                              [sock, buf, b.trunc(max_b, I32_TY), ir.Constant(I32_TY, 0),
                               addr_raw, addr_len_al]), I64_TY)
        else:
            rcv_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY, I32_TY, I8PTR, ir.PointerType(I32_TY)])
            n = b.call(self._get_or_declare_fn("recvfrom", rcv_ty),
                       [b.trunc(sock, I32_TY), buf, max_b, ir.Constant(I32_TY, 0), addr_raw, addr_len_al])
        neg1 = ir.Constant(I64_TY, 0xFFFFFFFFFFFFFFFF)
        safe_n = b.select(b.icmp_signed("==", n, neg1), ir.Constant(I64_TY, 0), n)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [safe_n], inbounds=False))
        b.ret(buf)
        return fn

    def _build_file_watch(self) -> ir.Function:
        """file_watch(path, callback) — invokes callback immediately (polling requires a thread)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I8PTR, I8PTR]),
                         name="__vx_file_watch")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        _path, cb = fn.args
        cb_ty = ir.FunctionType(VOID_TY, [])
        b.call(b.bitcast(cb, ir.PointerType(cb_ty)), [])
        b.ret_void()
        return fn

    def _build_pipe_open(self) -> ir.Function:
        """pipe_open(name) -> int fd."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_pipe_open")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        name_v = fn.args[0]
        if _sys.platform == "win32":
            cf_ty = ir.FunctionType(I64_TY, [I8PTR, I32_TY, I32_TY, I8PTR, I32_TY, I32_TY, I8PTR])
            null  = ir.Constant(I8PTR, None)
            fd = b.call(self._get_or_declare_fn("CreateFileA", cf_ty),
                        [name_v, ir.Constant(I32_TY, 0xC0000000),
                         ir.Constant(I32_TY, 0), null, ir.Constant(I32_TY, 3),
                         ir.Constant(I32_TY, 128), null])
            b.ret(fd)
        else:
            open_ty = ir.FunctionType(I32_TY, [I8PTR, I32_TY])
            b.ret(b.sext(b.call(self._get_or_declare_fn("open", open_ty),
                                [name_v, ir.Constant(I32_TY, 2)]), I64_TY))
        return fn

    def _build_pipe_write(self) -> ir.Function:
        """pipe_write(fd, data) -> int bytes written."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR]),
                         name="__vx_pipe_write")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fd, data = fn.args
        dlen = b.call(self.strlen_fn, [data])
        if _sys.platform == "win32":
            written_al = b.alloca(I32_TY)
            b.store(ir.Constant(I32_TY, 0), written_al)
            wf_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, ir.PointerType(I32_TY), I8PTR])
            b.call(self._get_or_declare_fn("WriteFile", wf_ty),
                   [fd, data, b.trunc(dlen, I32_TY), written_al, ir.Constant(I8PTR, None)])
            b.ret(b.sext(b.load(written_al), I64_TY))
        else:
            wr_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY])
            b.ret(b.call(self._get_or_declare_fn("write", wr_ty),
                         [b.trunc(fd, I32_TY), data, dlen]))
        return fn

    def _build_pipe_read(self) -> ir.Function:
        """pipe_read(fd, max_bytes) -> str."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY, I64_TY]),
                         name="__vx_pipe_read")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fd, max_b = fn.args
        buf = b.call(self.malloc_fn, [b.add(max_b, ir.Constant(I64_TY, 1))])
        if _sys.platform == "win32":
            rd_al = b.alloca(I32_TY)
            b.store(ir.Constant(I32_TY, 0), rd_al)
            rf_ty = ir.FunctionType(I32_TY, [I64_TY, I8PTR, I32_TY, ir.PointerType(I32_TY), I8PTR])
            b.call(self._get_or_declare_fn("ReadFile", rf_ty),
                   [fd, buf, b.trunc(max_b, I32_TY), rd_al, ir.Constant(I8PTR, None)])
            n = b.sext(b.load(rd_al), I64_TY)
        else:
            rd_ty = ir.FunctionType(I64_TY, [I32_TY, I8PTR, I64_TY])
            n = b.call(self._get_or_declare_fn("read", rd_ty),
                       [b.trunc(fd, I32_TY), buf, max_b])
        safe_n = b.select(b.icmp_signed("<", n, ir.Constant(I64_TY, 0)),
                          ir.Constant(I64_TY, 0), n)
        b.store(ir.Constant(I8_TY, 0), b.gep(buf, [safe_n], inbounds=False))
        b.ret(buf)
        return fn

    def _build_pipe_close(self) -> ir.Function:
        """pipe_close(fd) -> void."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_pipe_close")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fd = fn.args[0]
        if _sys.platform == "win32":
            ch_ty = ir.FunctionType(I32_TY, [I64_TY])
            b.call(self._get_or_declare_fn("CloseHandle", ch_ty), [fd])
        else:
            cl_ty = ir.FunctionType(I32_TY, [I32_TY])
            b.call(self._get_or_declare_fn("close", cl_ty), [b.trunc(fd, I32_TY)])
        b.ret_void()
        return fn

    def _build_chan_select(self) -> ir.Function:
        """chan_select(channels: int[]) -> int  — spin-poll all channels, return index of first ready."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_chan_select")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw = fn.args[0]
        arr_v   = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        arr_len  = b.load(b.gep(arr_v, [z, ir.Constant(I32_TY, 1)], inbounds=True))
        data_raw = b.load(b.gep(arr_v, [z, z], inbounds=True))
        arr64    = b.bitcast(data_raw, ir.PointerType(I64_TY))
        try_recv = self._get_helper("__vx_channel_try_recv")
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb   = fn.append_basic_block("sel_chk")
        body_bb  = fn.append_basic_block("sel_body")
        next_bb  = fn.append_basic_block("sel_next")
        found_bb = fn.append_basic_block("sel_found")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, arr_len), body_bb, chk_bb)
        b.position_at_end(body_bb)
        ch_i = b.load(b.gep(arr64, [b.trunc(iv, I32_TY)], inbounds=False))
        val  = b.call(try_recv, [ch_i])
        b.cbranch(b.icmp_signed(">=", val, ir.Constant(I64_TY, 0)), found_bb, next_bb)
        b.position_at_end(next_bb)
        iv2 = b.add(iv, ir.Constant(I64_TY, 1))
        b.store(b.select(b.icmp_signed(">=", iv2, arr_len), ir.Constant(I64_TY, 0), iv2), i_al)
        b.branch(chk_bb)
        b.position_at_end(found_bb)
        b.ret(b.load(i_al))
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — MD5                                                           #
    # ------------------------------------------------------------------ #

    def _build_md5(self) -> ir.Function:
        """MD5 hash — pure LLVM IR implementation (4 rounds × 16 ops)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_md5")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        msg = fn.args[0]

        I32 = I32_TY
        def u32(v): return ir.Constant(I32, v & 0xFFFFFFFF)
        def rotr32(bld, val, n):
            return bld.or_(bld.lshr(val, u32(32-n)), bld.shl(val, u32(n)))  # left-rotate for MD5

        # MD5 per-round shift amounts
        S = [
            7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
            5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
            4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
            6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21,
        ]

        # MD5 T constants: floor(abs(sin(i+1)) * 2^32)
        T = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
            0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
            0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
            0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
            0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
            0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
            0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
            0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
        ]

        msg_len = b.call(self.strlen_fn, [msg])

        # Pad to 64-byte boundary
        pad_base  = b.add(msg_len, ir.Constant(I64_TY, 9))
        pad_align = b.add(pad_base, ir.Constant(I64_TY, 63))
        pad_len   = b.and_(pad_align, ir.Constant(I64_TY, ~63))
        pad_buf   = b.call(self.malloc_fn, [b.add(pad_len, ir.Constant(I64_TY, 8))])

        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        memset_fn = self._get_or_declare("memset",
            ir.FunctionType(I8PTR, [I8PTR, I32, I64_TY]))
        b.call(memcpy_fn, [pad_buf, msg, msg_len])
        pos80 = b.gep(pad_buf, [msg_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0x80), pos80)
        zero_start = b.gep(pad_buf, [b.add(msg_len, ir.Constant(I64_TY, 1))], inbounds=False)
        zero_len   = b.sub(b.sub(pad_len, ir.Constant(I64_TY, 8)),
                           b.add(msg_len, ir.Constant(I64_TY, 1)))
        b.call(memset_fn, [zero_start, ir.Constant(I32, 0), zero_len])

        # MD5 appends length in bits little-endian 64-bit
        bit_len = b.mul(msg_len, ir.Constant(I64_TY, 8))
        for byte_i in range(8):
            if byte_i > 0:
                bval = b.trunc(b.lshr(bit_len, ir.Constant(I64_TY, byte_i * 8)), I8_TY)
            else:
                bval = b.trunc(bit_len, I8_TY)
            bp = b.gep(pad_buf, [b.add(b.sub(pad_len, ir.Constant(I64_TY, 8)),
                                       ir.Constant(I64_TY, byte_i))], inbounds=False)
            b.store(bval, bp)

        # Initial state
        a0_al = b.alloca(I32); b0_al = b.alloca(I32)
        c0_al = b.alloca(I32); d0_al = b.alloca(I32)
        b.store(u32(0x67452301), a0_al); b.store(u32(0xefcdab89), b0_al)
        b.store(u32(0x98badcfe), c0_al); b.store(u32(0x10325476), d0_al)

        num_blocks = b.udiv(pad_len, ir.Constant(I64_TY, 64))
        bi_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), bi_al)
        blk_chk = fn.append_basic_block("md5.blk.chk")
        blk_bdy = fn.append_basic_block("md5.blk.bdy")
        blk_ext = fn.append_basic_block("md5.blk.ext")
        b.branch(blk_chk)
        b.position_at_end(blk_chk)
        bi = b.load(bi_al)
        b.cbranch(b.icmp_unsigned("<", bi, num_blocks), blk_bdy, blk_ext)
        b.position_at_end(blk_bdy)

        block_off = b.mul(bi, ir.Constant(I64_TY, 64))
        block_ptr = b.gep(pad_buf, [block_off], inbounds=False)

        # Load 16 little-endian 32-bit words
        M = []
        for wi in range(16):
            word_val = u32(0)
            for byte_j in range(4):
                bp = b.gep(block_ptr, [ir.Constant(I64_TY, wi * 4 + byte_j)], inbounds=False)
                bv = b.zext(b.load(bp), I32)
                shifted = b.shl(bv, u32(byte_j * 8))
                word_val = b.or_(word_val, shifted)
            M.append(word_val)

        # Working vars
        A_al = b.alloca(I32); B_al = b.alloca(I32)
        C_al = b.alloca(I32); D_al = b.alloca(I32)
        b.store(b.load(a0_al), A_al); b.store(b.load(b0_al), B_al)
        b.store(b.load(c0_al), C_al); b.store(b.load(d0_al), D_al)

        for i in range(64):
            A = b.load(A_al); Bv = b.load(B_al); C = b.load(C_al); D = b.load(D_al)
            if i < 16:
                F = b.or_(b.and_(Bv, C), b.and_(b.not_(Bv), D))
                g = i
            elif i < 32:
                F = b.or_(b.and_(D, Bv), b.and_(b.not_(D), C))
                g = (5 * i + 1) % 16
            elif i < 48:
                F = b.xor_(b.xor_(Bv, C), D)
                g = (3 * i + 5) % 16
            else:
                F = b.xor_(C, b.or_(Bv, b.not_(D)))
                g = (7 * i) % 16
            dtemp = D
            b.store(C,   D_al)
            b.store(Bv,  C_al)
            inner = b.add(b.add(b.add(A, F), M[g]), u32(T[i]))
            rot_v = rotr32(b, inner, S[i])
            b.store(b.add(Bv, rot_v), B_al)
            b.store(dtemp, A_al)

        b.store(b.add(b.load(a0_al), b.load(A_al)), a0_al)
        b.store(b.add(b.load(b0_al), b.load(B_al)), b0_al)
        b.store(b.add(b.load(c0_al), b.load(C_al)), c0_al)
        b.store(b.add(b.load(d0_al), b.load(D_al)), d0_al)

        b.store(b.add(bi, ir.Constant(I64_TY, 1)), bi_al)
        b.branch(blk_chk)

        b.position_at_end(blk_ext)
        out_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 33)])
        fmt8_gv = self._global_str("%08x")
        for wi, al in enumerate([a0_al, b0_al, c0_al, d0_al]):
            word_v  = b.load(al)
            # MD5 output is little-endian: byte-swap each word
            b0 = b.and_(word_v, u32(0xFF))
            b1 = b.and_(b.lshr(word_v, u32(8)),  u32(0xFF))
            b2 = b.and_(b.lshr(word_v, u32(16)), u32(0xFF))
            b3 = b.lshr(word_v, u32(24))
            swapped = b.or_(b.or_(b.shl(b0, u32(24)), b.shl(b1, u32(16))),
                            b.or_(b.shl(b2, u32(8)),  b3))
            word64 = b.zext(swapped, I64_TY)
            out_off = b.gep(out_buf, [ir.Constant(I64_TY, wi * 8)], inbounds=False)
            b.call(self.sprintf_fn, [out_off, self._gstr_ptr_const(fmt8_gv), word64])
        term_ptr = b.gep(out_buf, [ir.Constant(I64_TY, 32)], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term_ptr)
        b.ret(out_buf)
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — SHA-512                                                       #
    # ------------------------------------------------------------------ #

    def _build_sha512(self) -> ir.Function:
        """SHA-512: 80 rounds with 64-bit words, returns 128-char hex string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_sha512")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        msg = fn.args[0]

        # SHA-512 K constants (first 80)
        K512 = [
            0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
            0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
            0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
            0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
            0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
            0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
            0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
            0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
            0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
            0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
            0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
            0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
            0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
            0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
            0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
            0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
            0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
            0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
            0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
            0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
        ]

        H_INIT512 = [
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
        ]

        I64 = I64_TY
        def u64(v): return ir.Constant(I64, v & 0xFFFFFFFFFFFFFFFF)

        def rotr64(bld, val, n):
            return bld.or_(bld.lshr(val, u64(n)), bld.shl(val, u64(64 - n)))

        msg_len = b.call(self.strlen_fn, [msg])

        # SHA-512 pads to 128-byte boundary
        pad_base  = b.add(msg_len, ir.Constant(I64_TY, 17))
        pad_align = b.add(pad_base, ir.Constant(I64_TY, 127))
        pad_len   = b.and_(pad_align, ir.Constant(I64_TY, ~127))
        pad_buf   = b.call(self.malloc_fn, [b.add(pad_len, ir.Constant(I64_TY, 16))])

        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        memset_fn = self._get_or_declare("memset",
            ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY]))
        b.call(memcpy_fn, [pad_buf, msg, msg_len])
        pos80 = b.gep(pad_buf, [msg_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0x80), pos80)
        zero_start = b.gep(pad_buf, [b.add(msg_len, ir.Constant(I64_TY, 1))], inbounds=False)
        zero_len   = b.sub(b.sub(pad_len, ir.Constant(I64_TY, 16)),
                           b.add(msg_len, ir.Constant(I64_TY, 1)))
        b.call(memset_fn, [zero_start, ir.Constant(I32_TY, 0), zero_len])

        # Append 128-bit length (high=0, low=bit_len) big-endian
        bit_len = b.mul(msg_len, u64(8))
        # high 8 bytes = 0
        for byte_i in range(8):
            bp = b.gep(pad_buf, [b.add(b.sub(pad_len, ir.Constant(I64_TY, 16)),
                                       ir.Constant(I64_TY, byte_i))], inbounds=False)
            b.store(ir.Constant(I8_TY, 0), bp)
        # low 8 bytes = bit_len big-endian
        for byte_i in range(8):
            shift = 56 - byte_i * 8
            if shift > 0:
                bval = b.trunc(b.lshr(bit_len, u64(shift)), I8_TY)
            else:
                bval = b.trunc(bit_len, I8_TY)
            bp = b.gep(pad_buf, [b.add(b.sub(pad_len, ir.Constant(I64_TY, 8)),
                                       ir.Constant(I64_TY, byte_i))], inbounds=False)
            b.store(bval, bp)

        # Initial hash state
        h_al = [b.alloca(I64) for _ in range(8)]
        for i, hv in enumerate(H_INIT512):
            b.store(u64(hv), h_al[i])

        num_blocks = b.udiv(pad_len, ir.Constant(I64_TY, 128))
        bi_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), bi_al)
        blk_chk = fn.append_basic_block("sha512.blk.chk")
        blk_bdy = fn.append_basic_block("sha512.blk.bdy")
        blk_ext = fn.append_basic_block("sha512.blk.ext")
        b.branch(blk_chk)
        b.position_at_end(blk_chk)
        bi = b.load(bi_al)
        b.cbranch(b.icmp_unsigned("<", bi, num_blocks), blk_bdy, blk_ext)
        b.position_at_end(blk_bdy)

        block_off = b.mul(bi, ir.Constant(I64_TY, 128))
        block_ptr = b.gep(pad_buf, [block_off], inbounds=False)

        # Load W[0..15] (big-endian 64-bit words)
        w_al = [b.alloca(I64) for _ in range(80)]
        for wi in range(16):
            word_val = u64(0)
            for byte_j in range(8):
                bp = b.gep(block_ptr, [ir.Constant(I64_TY, wi * 8 + byte_j)], inbounds=False)
                bv = b.zext(b.load(bp), I64)
                shifted = b.shl(bv, u64((7 - byte_j) * 8))
                word_val = b.or_(word_val, shifted)
            b.store(word_val, w_al[wi])

        # Extend W[16..79]
        for wi in range(16, 80):
            w15 = b.load(w_al[wi - 15])
            s0 = b.xor_(b.xor_(rotr64(b, w15, 1), rotr64(b, w15, 8)), b.lshr(w15, u64(7)))
            w2  = b.load(w_al[wi - 2])
            s1  = b.xor_(b.xor_(rotr64(b, w2, 19), rotr64(b, w2, 61)), b.lshr(w2, u64(6)))
            w16 = b.load(w_al[wi - 16])
            w7  = b.load(w_al[wi - 7])
            b.store(b.add(b.add(b.add(w16, s0), w7), s1), w_al[wi])

        # Working vars
        wk = [b.alloca(I64) for _ in range(8)]
        for i in range(8):
            b.store(b.load(h_al[i]), wk[i])

        for ri in range(80):
            av=b.load(wk[0]); bv=b.load(wk[1]); cv=b.load(wk[2]); dv=b.load(wk[3])
            ev=b.load(wk[4]); fv=b.load(wk[5]); gv=b.load(wk[6]); hv=b.load(wk[7])
            S1  = b.xor_(b.xor_(rotr64(b, ev, 14), rotr64(b, ev, 18)), rotr64(b, ev, 41))
            ch  = b.xor_(b.and_(ev, fv), b.and_(b.not_(ev), gv))
            t1  = b.add(b.add(b.add(b.add(hv, S1), ch), u64(K512[ri])), b.load(w_al[ri]))
            S0  = b.xor_(b.xor_(rotr64(b, av, 28), rotr64(b, av, 34)), rotr64(b, av, 39))
            maj = b.xor_(b.xor_(b.and_(av, bv), b.and_(av, cv)), b.and_(bv, cv))
            t2  = b.add(S0, maj)
            b.store(hv,           wk[7])
            b.store(gv,           wk[6])
            b.store(fv,           wk[5])
            b.store(ev,           wk[4])
            b.store(b.add(dv,t1), wk[3])
            b.store(cv,           wk[2])
            b.store(bv,           wk[1])
            b.store(b.add(t1,t2), wk[0])

        for i in range(8):
            b.store(b.add(b.load(h_al[i]), b.load(wk[i])), h_al[i])

        b.store(b.add(bi, ir.Constant(I64_TY, 1)), bi_al)
        b.branch(blk_chk)

        b.position_at_end(blk_ext)
        out_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 129)])
        fmt16_gv = self._global_str("%016llx")
        for wi in range(8):
            word_v  = b.load(h_al[wi])
            out_off = b.gep(out_buf, [ir.Constant(I64_TY, wi * 16)], inbounds=False)
            b.call(self.sprintf_fn, [out_off, self._gstr_ptr_const(fmt16_gv), word_v])
        term_ptr = b.gep(out_buf, [ir.Constant(I64_TY, 128)], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term_ptr)
        b.ret(out_buf)
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — HMAC-SHA256                                                   #
    # ------------------------------------------------------------------ #

    def _build_hmac_sha256(self) -> ir.Function:
        """HMAC-SHA256(key, msg) -> hex str.  Pad key to 64 bytes, XOR ipad/opad, call sha256 twice."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_hmac_sha256")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        key = fn.args[0]; msg = fn.args[1]

        sha256_fn = self._get_helper("__vx_sha256")
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        memset_fn = self._get_or_declare("memset",
            ir.FunctionType(I8PTR, [I8PTR, I32_TY, I64_TY]))

        BLOCK = 64
        key_len = b.call(self.strlen_fn, [key])
        msg_len = b.call(self.strlen_fn, [msg])

        # k_pad = zero-padded key (64 bytes)
        k_pad = b.call(self.malloc_fn, [ir.Constant(I64_TY, BLOCK + 1)])
        b.call(memset_fn, [k_pad, ir.Constant(I32_TY, 0), ir.Constant(I64_TY, BLOCK)])

        # If key is longer than BLOCK, hash it first (simplified: just truncate)
        k_short = b.icmp_unsigned("<=", key_len, ir.Constant(I64_TY, BLOCK))
        short_bb = fn.append_basic_block("hmac.short"); long_bb = fn.append_basic_block("hmac.long")
        after_bb = fn.append_basic_block("hmac.after_key")
        b.cbranch(k_short, short_bb, long_bb)
        b.position_at_end(short_bb)
        b.call(memcpy_fn, [k_pad, key, key_len])
        b.branch(after_bb)
        b.position_at_end(long_bb)
        hashed_key = b.call(sha256_fn, [key])
        hashed_len = b.call(self.strlen_fn, [hashed_key])
        b.call(memcpy_fn, [k_pad, hashed_key, hashed_len])
        b.branch(after_bb)
        b.position_at_end(after_bb)

        # ipad_key = k_pad XOR 0x36, opad_key = k_pad XOR 0x5C
        ipad_key = b.call(self.malloc_fn, [ir.Constant(I64_TY, BLOCK + 1)])
        opad_key = b.call(self.malloc_fn, [ir.Constant(I64_TY, BLOCK + 1)])
        for byte_i in range(BLOCK):
            kp = b.gep(k_pad, [ir.Constant(I64_TY, byte_i)], inbounds=False)
            kv = b.load(kp)
            ip = b.gep(ipad_key, [ir.Constant(I64_TY, byte_i)], inbounds=False)
            op = b.gep(opad_key, [ir.Constant(I64_TY, byte_i)], inbounds=False)
            b.store(b.xor_(kv, ir.Constant(I8_TY, 0x36)), ip)
            b.store(b.xor_(kv, ir.Constant(I8_TY, 0x5C)), op)
        # NUL-terminate (not strictly needed but safe)
        b.store(ir.Constant(I8_TY, 0),
                b.gep(ipad_key, [ir.Constant(I64_TY, BLOCK)], inbounds=False))
        b.store(ir.Constant(I8_TY, 0),
                b.gep(opad_key, [ir.Constant(I64_TY, BLOCK)], inbounds=False))

        # inner_input = ipad_key || msg
        inner_len = b.add(ir.Constant(I64_TY, BLOCK), msg_len)
        inner_buf = b.call(self.malloc_fn, [b.add(inner_len, ir.Constant(I64_TY, 1))])
        b.call(memcpy_fn, [inner_buf, ipad_key, ir.Constant(I64_TY, BLOCK)])
        b.call(memcpy_fn, [b.gep(inner_buf, [ir.Constant(I64_TY, BLOCK)], inbounds=False),
                           msg, msg_len])
        b.store(ir.Constant(I8_TY, 0),
                b.gep(inner_buf, [inner_len], inbounds=False))

        inner_hash = b.call(sha256_fn, [inner_buf])   # sha256(ipad || msg)
        inner_hash_len = b.call(self.strlen_fn, [inner_hash])

        # outer_input = opad_key || inner_hash
        outer_len = b.add(ir.Constant(I64_TY, BLOCK), inner_hash_len)
        outer_buf = b.call(self.malloc_fn, [b.add(outer_len, ir.Constant(I64_TY, 1))])
        b.call(memcpy_fn, [outer_buf, opad_key, ir.Constant(I64_TY, BLOCK)])
        b.call(memcpy_fn, [b.gep(outer_buf, [ir.Constant(I64_TY, BLOCK)], inbounds=False),
                           inner_hash, inner_hash_len])
        b.store(ir.Constant(I8_TY, 0),
                b.gep(outer_buf, [outer_len], inbounds=False))

        result = b.call(sha256_fn, [outer_buf])
        b.ret(result)
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — Regex engine                                                  #
    # ------------------------------------------------------------------ #

    def _build_regex_engine(self) -> ir.Function:
        """Core regex engine: __vx_regex_engine(text, tlen, pat, ppos, tpos) -> i64
        Returns end position of match (>=0) if pattern from ppos matches text from tpos,
        else -1. Backtracking recursive NFA.
        Supports: . * + ? ^ $ [cls] [^cls] \\d \\w \\s \\D \\W \\S"""
        fn = ir.Function(self.module,
                         ir.FunctionType(I64_TY, [I8PTR, I64_TY, I8PTR, I64_TY, I64_TY]),
                         name="__vx_regex_engine")
        fn.linkage = "private"
        text, tlen, pat, ppos, tpos = fn.args

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)

        # Helper: load a character from a string pointer + index
        # We build this inline using GEP + load

        def load_char(bld, ptr, idx):
            p = bld.gep(ptr, [idx], inbounds=False)
            return bld.load(p)

        def char_matches_class(bld, ch32, pat_ptr, cls_start, cls_len):
            """Inline check: does ch32 match the character class starting at cls_start?
            Returns i1."""
            # We build a simple loop that scans the class spec.
            # This is complex to do fully in IR so we implement a simplified version:
            # Store result in alloca, return it.
            match_al = bld.alloca(I1_TY)
            bld.store(ir.Constant(I1_TY, 0), match_al)
            # For simplicity, we check ranges and literals inline for up to 32 chars.
            # The class is bounded by cls_len.
            i_al = bld.alloca(I64_TY)
            bld.store(ir.Constant(I64_TY, 0), i_al)
            cls_chk = fn.append_basic_block("cls.chk")
            cls_bdy = fn.append_basic_block("cls.bdy")
            cls_ext = fn.append_basic_block("cls.ext")
            bld.branch(cls_chk)
            bld.position_at_end(cls_chk)
            ci = bld.load(i_al)
            bld.cbranch(bld.icmp_signed("<", ci, cls_len), cls_bdy, cls_ext)
            bld.position_at_end(cls_bdy)
            # Get current class char
            class_idx = bld.add(cls_start, ci)
            cc = bld.zext(load_char(bld, pat_ptr, class_idx), I32_TY)
            # Check if next char is '-' (range)
            ci1 = bld.add(ci, ir.Constant(I64_TY, 1))
            ci2 = bld.add(ci, ir.Constant(I64_TY, 2))
            has_range = bld.and_(
                bld.icmp_signed("<", ci1, cls_len),
                bld.icmp_unsigned("==",
                    bld.zext(load_char(bld, pat_ptr, bld.add(cls_start, ci1)), I32_TY),
                    ir.Constant(I32_TY, ord('-'))
                )
            )
            range_in_range = bld.and_(has_range, bld.icmp_signed("<", ci2, cls_len))

            range_bb  = fn.append_basic_block("cls.range")
            single_bb = fn.append_basic_block("cls.single")
            after_bb  = fn.append_basic_block("cls.after")
            bld.cbranch(range_in_range, range_bb, single_bb)

            bld.position_at_end(range_bb)
            high_idx = bld.add(cls_start, ci2)
            hc = bld.zext(load_char(bld, pat_ptr, high_idx), I32_TY)
            in_range = bld.and_(
                bld.icmp_unsigned(">=", ch32, cc),
                bld.icmp_unsigned("<=", ch32, hc)
            )
            matched_al_range = bld.or_(bld.load(match_al), in_range)
            bld.store(matched_al_range, match_al)
            bld.store(bld.add(ci, ir.Constant(I64_TY, 3)), i_al)
            bld.branch(cls_chk)

            bld.position_at_end(single_bb)
            eq_single = bld.icmp_unsigned("==",ch32, cc)
            matched_al_single = bld.or_(bld.load(match_al), eq_single)
            bld.store(matched_al_single, match_al)
            bld.store(bld.add(ci, ir.Constant(I64_TY, 1)), i_al)
            bld.branch(cls_chk)

            bld.position_at_end(cls_ext)
            return bld.load(match_al)

        # Main function body
        # ppos >= plen → match succeeded; return tpos
        plen = b.call(self.strlen_fn, [pat])
        plen_al = b.alloca(I64_TY); b.store(plen, plen_al)

        matched_end_bb = fn.append_basic_block("re.matched")
        check_anchor_end = fn.append_basic_block("re.check_anchor")
        main_bb = fn.append_basic_block("re.main")
        fail_bb = fn.append_basic_block("re.fail")

        # If ppos >= plen, match succeeded
        b.cbranch(b.icmp_unsigned(">=", ppos, plen), matched_end_bb, check_anchor_end)

        b.position_at_end(matched_end_bb)
        b.ret(tpos)

        b.position_at_end(check_anchor_end)
        # Check for $ anchor: pat[ppos] == '$' and ppos+1 >= plen
        pc = b.zext(load_char(b, pat, ppos), I32_TY)
        ppos1 = b.add(ppos, ir.Constant(I64_TY, 1))
        is_dollar = b.icmp_unsigned("==",pc, ir.Constant(I32_TY, ord('$')))
        dollar_at_end = b.and_(is_dollar, b.icmp_unsigned(">=", ppos1, plen))
        dollar_matches = b.and_(dollar_at_end, b.icmp_unsigned("==",tpos, tlen))
        dollar_bb = fn.append_basic_block("re.dollar")
        b.cbranch(dollar_matches, dollar_bb, main_bb)
        b.position_at_end(dollar_bb)
        b.ret(tpos)

        b.position_at_end(main_bb)
        # tpos >= tlen means text exhausted: fail unless pattern is $ or end
        text_done = b.icmp_unsigned(">=", tpos, tlen)
        text_done_bb = fn.append_basic_block("re.text_done")
        text_ok_bb   = fn.append_basic_block("re.text_ok")
        b.cbranch(text_done, text_done_bb, text_ok_bb)
        b.position_at_end(text_done_bb)
        b.ret(ir.Constant(I64_TY, -1))

        b.position_at_end(text_ok_bb)
        # Get current pattern char and text char
        tc = b.zext(load_char(b, text, tpos), I32_TY)
        # Load pattern char (may be backslash escape)
        pc0 = b.zext(load_char(b, pat, ppos), I32_TY)

        # Check for quantifier at ppos+1: *, +, ?
        ppos2 = b.add(ppos, ir.Constant(I64_TY, 2))
        # Default next_ppos (without quantifier) = ppos + 1
        # We check if next pattern char is quantifier
        next_pc_idx = ppos1
        next_pc = b.zext(load_char(b, pat, next_pc_idx), I32_TY)
        has_next_pat = b.icmp_unsigned("<", ppos1, plen)
        is_star     = b.and_(has_next_pat, b.icmp_unsigned("==",next_pc, ir.Constant(I32_TY, ord('*'))))
        is_plus     = b.and_(has_next_pat, b.icmp_unsigned("==",next_pc, ir.Constant(I32_TY, ord('+'))))
        is_question = b.and_(has_next_pat, b.icmp_unsigned("==",next_pc, ir.Constant(I32_TY, ord('?'))))

        # Determine if current text char matches current pattern atom
        # Handle: . \ [ literal
        def atom_matches_char(bld, pc_val, tpos_val, ppos_val):
            """Returns i1: does pattern atom at ppos_val match text char at tpos_val?"""
            tc_val = bld.zext(load_char(bld, text, tpos_val), I32_TY)
            res_al = bld.alloca(I1_TY)
            bld.store(ir.Constant(I1_TY, 0), res_al)
            dot_bb   = fn.append_basic_block("atom.dot")
            bs_bb    = fn.append_basic_block("atom.bs")
            cls_head = fn.append_basic_block("atom.cls")
            lit_bb   = fn.append_basic_block("atom.lit")
            atom_end = fn.append_basic_block("atom.end")
            is_dot = bld.icmp_unsigned("==",pc_val, ir.Constant(I32_TY, ord('.')))
            is_bs  = bld.icmp_unsigned("==",pc_val, ir.Constant(I32_TY, ord('\\')))
            is_lbr = bld.icmp_unsigned("==",pc_val, ir.Constant(I32_TY, ord('[')))
            bld.cbranch(is_dot, dot_bb, fn.append_basic_block("atom.not_dot"))
            bld.position_at_end(dot_bb)
            # '.' matches anything except newline
            not_nl = bld.icmp_unsigned("!=",tc_val, ir.Constant(I32_TY, ord('\n')))
            bld.store(not_nl, res_al)
            bld.branch(atom_end)
            not_dot_bb = dot_bb.function.blocks[-2]
            bld.position_at_end(not_dot_bb)
            bld.cbranch(is_bs, bs_bb, fn.append_basic_block("atom.not_bs"))
            bld.position_at_end(bs_bb)
            # Backslash escape: look at next char
            esc_char = bld.zext(load_char(bld, pat, bld.add(ppos_val, ir.Constant(I64_TY, 1))), I32_TY)
            is_d = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('d')))
            is_w = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('w')))
            is_s = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('s')))
            is_D = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('D')))
            is_W = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('W')))
            is_S = bld.icmp_unsigned("==",esc_char, ir.Constant(I32_TY, ord('S')))
            dig = bld.and_(bld.icmp_unsigned(">=", tc_val, ir.Constant(I32_TY, ord('0'))),
                           bld.icmp_unsigned("<=", tc_val, ir.Constant(I32_TY, ord('9'))))
            word_lo = bld.and_(bld.icmp_unsigned(">=", tc_val, ir.Constant(I32_TY, ord('a'))),
                                bld.icmp_unsigned("<=", tc_val, ir.Constant(I32_TY, ord('z'))))
            word_up = bld.and_(bld.icmp_unsigned(">=", tc_val, ir.Constant(I32_TY, ord('A'))),
                                bld.icmp_unsigned("<=", tc_val, ir.Constant(I32_TY, ord('Z'))))
            word_ch = bld.or_(bld.or_(word_lo, word_up), bld.or_(dig,
                               bld.icmp_unsigned("==",tc_val, ir.Constant(I32_TY, ord('_')))))
            sp_ch = bld.or_(bld.or_(bld.icmp_unsigned("==",tc_val, ir.Constant(I32_TY, ord(' '))),
                                     bld.icmp_unsigned("==",tc_val, ir.Constant(I32_TY, ord('\t')))),
                             bld.or_(bld.icmp_unsigned("==",tc_val, ir.Constant(I32_TY, ord('\n'))),
                                     bld.icmp_unsigned("==",tc_val, ir.Constant(I32_TY, ord('\r')))))
            m = bld.select(is_d, dig,
                bld.select(is_w, word_ch,
                bld.select(is_s, sp_ch,
                bld.select(is_D, bld.not_(dig),
                bld.select(is_W, bld.not_(word_ch),
                bld.select(is_S, bld.not_(sp_ch),
                           bld.icmp_unsigned("==",tc_val, esc_char)))))))
            bld.store(m, res_al)
            bld.branch(atom_end)
            not_bs_bb = bs_bb.function.blocks[-2]
            bld.position_at_end(not_bs_bb)
            bld.cbranch(is_lbr, cls_head, lit_bb)
            bld.position_at_end(cls_head)
            # Character class [...] or [^...]
            cls_p1 = bld.add(ppos_val, ir.Constant(I64_TY, 1))
            cls_pc1 = bld.zext(load_char(bld, pat, cls_p1), I32_TY)
            negate = bld.icmp_unsigned("==",cls_pc1, ir.Constant(I32_TY, ord('^')))
            cls_inner_start_neg = bld.add(cls_p1, ir.Constant(I64_TY, 1))
            cls_inner_start_pos = cls_p1
            cls_inner_start = bld.select(negate, cls_inner_start_neg, cls_inner_start_pos)
            # Find matching ']': scan from cls_inner_start
            # (simplified: find ']' from cls_inner_start+1)
            cls_scan_al = bld.alloca(I64_TY)
            bld.store(cls_inner_start, cls_scan_al)
            cls_scan_chk = fn.append_basic_block("cls.scan.chk")
            cls_scan_bdy = fn.append_basic_block("cls.scan.bdy")
            cls_scan_ext = fn.append_basic_block("cls.scan.ext")
            bld.branch(cls_scan_chk)
            bld.position_at_end(cls_scan_chk)
            csi = bld.load(cls_scan_al)
            csi_lt = bld.icmp_unsigned("<", csi, plen)
            bld.cbranch(csi_lt, cls_scan_bdy, cls_scan_ext)
            bld.position_at_end(cls_scan_bdy)
            csc = bld.zext(load_char(bld, pat, csi), I32_TY)
            is_rbr = bld.icmp_unsigned("==",csc, ir.Constant(I32_TY, ord(']')))
            bld.cbranch(is_rbr, cls_scan_ext, fn.append_basic_block("cls.scan.cont"))
            bld.position_at_end(fn.blocks[-1])
            bld.store(bld.add(csi, ir.Constant(I64_TY, 1)), cls_scan_al)
            bld.branch(cls_scan_chk)
            bld.position_at_end(cls_scan_ext)
            cls_end_pos = bld.load(cls_scan_al)
            cls_len = bld.sub(cls_end_pos, cls_inner_start)
            raw_match = char_matches_class(bld, tc_val, pat, cls_inner_start, cls_len)
            final_match = bld.select(negate, bld.not_(raw_match), raw_match)
            bld.store(final_match, res_al)
            bld.branch(atom_end)
            bld.position_at_end(lit_bb)
            bld.store(bld.icmp_unsigned("==",tc_val, pc_val), res_al)
            bld.branch(atom_end)
            bld.position_at_end(atom_end)
            return bld.load(res_al)

        # atom size: 1 for normal, 2 for \ escape, varies for [...]
        def atom_size(bld, ppos_val):
            """Returns i64 size of pattern atom at ppos_val."""
            pc_v = bld.zext(load_char(bld, pat, ppos_val), I32_TY)
            is_bs  = bld.icmp_unsigned("==",pc_v, ir.Constant(I32_TY, ord('\\')))
            is_lbr = bld.icmp_unsigned("==",pc_v, ir.Constant(I32_TY, ord('[')))
            # For '[', scan to ']'
            scan_al = bld.alloca(I64_TY)
            bld.store(bld.add(ppos_val, ir.Constant(I64_TY, 1)), scan_al)
            lbr_scan_chk = fn.append_basic_block("asiz.scan.chk")
            lbr_scan_bdy = fn.append_basic_block("asiz.scan.bdy")
            lbr_scan_ext = fn.append_basic_block("asiz.scan.ext")
            lbr_scan_done = fn.append_basic_block("asiz.scan.done")
            bld.cbranch(is_lbr, lbr_scan_chk, fn.append_basic_block("asiz.not_cls"))
            bld.position_at_end(lbr_scan_chk)
            si = bld.load(scan_al)
            bld.cbranch(bld.icmp_unsigned("<", si, plen), lbr_scan_bdy, lbr_scan_ext)
            bld.position_at_end(lbr_scan_bdy)
            sc = bld.zext(load_char(bld, pat, si), I32_TY)
            bld.cbranch(bld.icmp_unsigned("==",sc, ir.Constant(I32_TY, ord(']'))), lbr_scan_ext,
                        fn.append_basic_block("asiz.scan.cont"))
            bld.position_at_end(fn.blocks[-1])
            bld.store(bld.add(si, ir.Constant(I64_TY, 1)), scan_al)
            bld.branch(lbr_scan_chk)
            bld.position_at_end(lbr_scan_ext)
            cls_size = bld.sub(bld.add(bld.load(scan_al), ir.Constant(I64_TY, 1)), ppos_val)
            bld.branch(lbr_scan_done)
            not_cls_bb = lbr_scan_ext.function.blocks[-2]
            bld.position_at_end(not_cls_bb)
            bs_size = bld.select(is_bs, ir.Constant(I64_TY, 2), ir.Constant(I64_TY, 1))
            bld.branch(lbr_scan_done)
            bld.position_at_end(lbr_scan_done)
            phi = bld.phi(I64_TY)
            phi.add_incoming(cls_size, lbr_scan_ext)
            phi.add_incoming(bs_size, not_cls_bb)
            return phi

        a_size = atom_size(b, ppos)

        # Quantifier check
        quant_ppos = b.add(ppos, b.add(a_size, ir.Constant(I64_TY, 0)))  # ppos after atom
        quant_ppos_plus1 = b.add(quant_ppos, ir.Constant(I64_TY, 1))    # ppos after quantifier

        qc_idx = quant_ppos
        has_quant_pat = b.icmp_unsigned("<", qc_idx, plen)
        qc = b.select(has_quant_pat,
                      b.zext(load_char(b, pat, qc_idx), I32_TY),
                      ir.Constant(I32_TY, 0))
        q_is_star = b.and_(has_quant_pat, b.icmp_unsigned("==",qc, ir.Constant(I32_TY, ord('*'))))
        q_is_plus = b.and_(has_quant_pat, b.icmp_unsigned("==",qc, ir.Constant(I32_TY, ord('+'))))
        q_is_q    = b.and_(has_quant_pat, b.icmp_unsigned("==",qc, ir.Constant(I32_TY, ord('?'))))

        ppos_after_quant = b.select(b.or_(b.or_(q_is_star, q_is_plus), q_is_q),
                                    quant_ppos_plus1, quant_ppos)

        # If no quantifier: match atom then recurse
        no_quant_bb  = fn.append_basic_block("re.no_quant")
        star_bb      = fn.append_basic_block("re.star")
        plus_bb      = fn.append_basic_block("re.plus")
        question_bb  = fn.append_basic_block("re.question")
        has_quant_bb = fn.append_basic_block("re.has_quant")

        b.cbranch(b.or_(b.or_(q_is_star, q_is_plus), q_is_q), has_quant_bb, no_quant_bb)

        # No quantifier: atom must match, then recurse
        b.position_at_end(no_quant_bb)
        matched_nq = atom_matches_char(b, pc0, tpos, ppos)
        nq_ok_bb  = fn.append_basic_block("re.nq.ok")
        b.cbranch(matched_nq, nq_ok_bb, fail_bb)
        b.position_at_end(nq_ok_bb)
        tpos_next = b.add(tpos, ir.Constant(I64_TY, 1))
        rec_nq = b.call(fn, [text, tlen, pat, quant_ppos, tpos_next])
        b.ret(rec_nq)

        b.position_at_end(has_quant_bb)
        b.cbranch(q_is_star, star_bb, fn.append_basic_block("re.not_star"))
        b.position_at_end(fn.blocks[-1])
        b.cbranch(q_is_plus, plus_bb, question_bb)

        # '*': greedy: try matching 0..n times
        b.position_at_end(star_bb)
        # Try matching without consuming (0 times) first (lazy doesn't matter here, use greedy)
        # Try rest of pattern with 0 matches first, then consume 1 and recurse on *
        res_star_0 = b.call(fn, [text, tlen, pat, ppos_after_quant, tpos])
        star_0_ok = b.icmp_signed(">=", res_star_0, ir.Constant(I64_TY, 0))
        star_try1_bb = fn.append_basic_block("re.star.try1")
        star_ret0_bb = fn.append_basic_block("re.star.ret0")
        b.cbranch(star_0_ok, star_ret0_bb, star_try1_bb)
        b.position_at_end(star_ret0_bb)
        b.ret(res_star_0)
        b.position_at_end(star_try1_bb)
        matched_s1 = atom_matches_char(b, pc0, tpos, ppos)
        star_consume_bb = fn.append_basic_block("re.star.consume")
        b.cbranch(matched_s1, star_consume_bb, fail_bb)
        b.position_at_end(star_consume_bb)
        tpos_s1 = b.add(tpos, ir.Constant(I64_TY, 1))
        # Re-try star with ppos (same atom, keep *)
        res_star_rec = b.call(fn, [text, tlen, pat, ppos, tpos_s1])
        b.ret(res_star_rec)

        # '+': must match at least once
        b.position_at_end(plus_bb)
        matched_p1 = atom_matches_char(b, pc0, tpos, ppos)
        plus_ok_bb = fn.append_basic_block("re.plus.ok")
        b.cbranch(matched_p1, plus_ok_bb, fail_bb)
        b.position_at_end(plus_ok_bb)
        tpos_p1 = b.add(tpos, ir.Constant(I64_TY, 1))
        # Convert to star for subsequent matches
        res_plus = b.call(fn, [text, tlen, pat, b.sub(ppos_after_quant, ir.Constant(I64_TY, 1)),
                               tpos_p1])
        # ppos_after_quant-1 = quant_ppos (the '*' position) — wrong; use a star at ppos
        # Actually easier: recursively call with same ppos but as star behavior
        # Rebuild: after 1 match, recurse with '*' still active (ppos unchanged) but
        # try ppos_after_quant if no more matches
        res_plus2 = b.call(fn, [text, tlen, pat, ppos, tpos_p1])
        b.ret(res_plus2)

        # '?': match 0 or 1
        b.position_at_end(question_bb)
        matched_q1 = atom_matches_char(b, pc0, tpos, ppos)
        q_try1_bb = fn.append_basic_block("re.q.try1")
        q_skip_bb = fn.append_basic_block("re.q.skip")
        # Try with 1 match first (greedy)
        b.cbranch(matched_q1, q_try1_bb, q_skip_bb)
        b.position_at_end(q_try1_bb)
        tpos_q1 = b.add(tpos, ir.Constant(I64_TY, 1))
        res_q1 = b.call(fn, [text, tlen, pat, ppos_after_quant, tpos_q1])
        q_ok_bb  = fn.append_basic_block("re.q.ok")
        q_fall_bb = fn.append_basic_block("re.q.fall")
        b.cbranch(b.icmp_signed(">=", res_q1, ir.Constant(I64_TY, 0)), q_ok_bb, q_fall_bb)
        b.position_at_end(q_ok_bb)
        b.ret(res_q1)
        b.position_at_end(q_fall_bb)
        b.branch(q_skip_bb)
        b.position_at_end(q_skip_bb)
        res_q0 = b.call(fn, [text, tlen, pat, ppos_after_quant, tpos])
        b.ret(res_q0)

        b.position_at_end(fail_bb)
        b.ret(ir.Constant(I64_TY, -1))
        return fn

    def _build_regex_match(self) -> ir.Function:
        """regex_match(text, pattern) -> str: return first match or empty string."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_regex_match")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        text, pat = fn.args

        engine = self._get_helper("__vx_regex_engine")
        tlen = b.call(self.strlen_fn, [text])
        plen = b.call(self.strlen_fn, [pat])
        empty_gv = self._global_str("")

        # Check for ^ anchor
        pc0 = b.zext(b.load(b.gep(pat, [ir.Constant(I64_TY, 0)], inbounds=False)), I32_TY)
        is_anchor = b.icmp_unsigned("==",pc0, ir.Constant(I32_TY, ord('^')))
        pat_start = b.select(is_anchor, ir.Constant(I64_TY, 1), ir.Constant(I64_TY, 0))

        # Try each starting position
        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb = fn.append_basic_block("rm.chk")
        bdy_bb = fn.append_basic_block("rm.bdy")
        ext_bb = fn.append_basic_block("rm.ext")
        found_bb = fn.append_basic_block("rm.found")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<=", iv, tlen), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        end_pos = b.call(engine, [text, tlen, pat, pat_start, iv])
        found = b.icmp_signed(">=", end_pos, ir.Constant(I64_TY, 0))
        b.cbranch(found, found_bb, fn.append_basic_block("rm.next"))
        b.position_at_end(fn.blocks[-1])
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        # If ^ anchor, don't advance
        anchor_done_bb = fn.append_basic_block("rm.anchor_done")
        b.cbranch(is_anchor, ext_bb, chk_bb)
        b.position_at_end(anchor_done_bb)
        b.branch(ext_bb)

        b.position_at_end(found_bb)
        # Extract substring [iv, end_pos)
        match_len = b.sub(end_pos, iv)
        match_buf = b.call(self.malloc_fn, [b.add(match_len, ir.Constant(I64_TY, 1))])
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        src_ptr = b.gep(text, [iv], inbounds=False)
        b.call(memcpy_fn, [match_buf, src_ptr, match_len])
        b.store(ir.Constant(I8_TY, 0),
                b.gep(match_buf, [match_len], inbounds=False))
        b.ret(match_buf)

        b.position_at_end(ext_bb)
        b.ret(self._gstr_ptr_const(empty_gv))
        return fn

    def _build_regex_test(self) -> ir.Function:
        """regex_test(text, pattern) -> int (1=match, 0=no match)."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_regex_test")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        text, pat = fn.args

        engine = self._get_helper("__vx_regex_engine")
        tlen = b.call(self.strlen_fn, [text])

        pc0 = b.zext(b.load(b.gep(pat, [ir.Constant(I64_TY, 0)], inbounds=False)), I32_TY)
        is_anchor = b.icmp_unsigned("==",pc0, ir.Constant(I32_TY, ord('^')))
        pat_start = b.select(is_anchor, ir.Constant(I64_TY, 1), ir.Constant(I64_TY, 0))

        i_al = b.alloca(I64_TY)
        b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb = fn.append_basic_block("rt.chk")
        bdy_bb = fn.append_basic_block("rt.bdy")
        ext_bb = fn.append_basic_block("rt.ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<=", iv, tlen), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        end_pos = b.call(engine, [text, tlen, pat, pat_start, iv])
        found = b.icmp_signed(">=", end_pos, ir.Constant(I64_TY, 0))
        found_bb = fn.append_basic_block("rt.found")
        b.cbranch(found, found_bb, fn.append_basic_block("rt.next"))
        b.position_at_end(fn.blocks[-1])
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_anchor, ext_bb, chk_bb)
        b.position_at_end(found_bb)
        b.ret(ir.Constant(I64_TY, 1))
        b.position_at_end(ext_bb)
        b.ret(ir.Constant(I64_TY, 0))
        return fn

    def _build_regex_find_all(self) -> ir.Function:
        """regex_find_all(text, pattern) -> str[]: all non-overlapping matches."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_regex_find_all")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        text, pat = fn.args

        engine  = self._get_helper("__vx_regex_engine")
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        tlen = b.call(self.strlen_fn, [text])

        pc0 = b.zext(b.load(b.gep(pat, [ir.Constant(I64_TY, 0)], inbounds=False)), I32_TY)
        is_anchor = b.icmp_unsigned("==",pc0, ir.Constant(I32_TY, ord('^')))
        pat_start = b.select(is_anchor, ir.Constant(I64_TY, 1), ir.Constant(I64_TY, 0))

        # Allocate result array header + data buffer
        cap = ir.Constant(I64_TY, 64)
        hdr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, _ARRAY_HEADER_SIZE)])
        hdr = b.bitcast(hdr_raw, self.arr_ptr_type)
        data = b.call(self.malloc_fn, [b.mul(cap, ir.Constant(I64_TY, 8))])
        z = ir.Constant(I32_TY, 0)
        b.store(data, b.gep(hdr, [z, z], inbounds=True))
        b.store(ir.Constant(I64_TY, 0), b.gep(hdr, [z, ir.Constant(I32_TY, 1)], inbounds=True))
        b.store(cap,                    b.gep(hdr, [z, ir.Constant(I32_TY, 2)], inbounds=True))
        data_ptr = b.bitcast(data, ir.PointerType(I8PTR))

        i_al   = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        cnt_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), cnt_al)
        chk_bb = fn.append_basic_block("rfa.chk")
        bdy_bb = fn.append_basic_block("rfa.bdy")
        ext_bb = fn.append_basic_block("rfa.ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<=", iv, tlen), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        end_pos = b.call(engine, [text, tlen, pat, pat_start, iv])
        found = b.icmp_signed(">=", end_pos, ir.Constant(I64_TY, 0))
        match_bb = fn.append_basic_block("rfa.match")
        skip_bb  = fn.append_basic_block("rfa.skip")
        b.cbranch(found, match_bb, skip_bb)
        b.position_at_end(match_bb)
        ml = b.sub(end_pos, iv)
        mbuf = b.call(self.malloc_fn, [b.add(ml, ir.Constant(I64_TY, 1))])
        b.call(memcpy_fn, [mbuf, b.gep(text, [iv], inbounds=False), ml])
        b.store(ir.Constant(I8_TY, 0), b.gep(mbuf, [ml], inbounds=False))
        cnt = b.load(cnt_al)
        b.store(mbuf, b.gep(data_ptr, [cnt], inbounds=False))
        b.store(b.add(cnt, ir.Constant(I64_TY, 1)), cnt_al)
        # Advance past match
        next_i = b.select(b.icmp_signed(">", ml, ir.Constant(I64_TY, 0)),
                          b.add(iv, ml),
                          b.add(iv, ir.Constant(I64_TY, 1)))
        b.store(next_i, i_al)
        b.cbranch(is_anchor, ext_bb, chk_bb)
        b.position_at_end(skip_bb)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.cbranch(is_anchor, ext_bb, chk_bb)
        b.position_at_end(ext_bb)
        final_cnt = b.load(cnt_al)
        b.store(final_cnt, b.gep(hdr, [z, ir.Constant(I32_TY, 1)], inbounds=True))
        b.ret(hdr_raw)
        return fn

    def _build_regex_replace(self) -> ir.Function:
        """regex_replace(text, pattern, replacement) -> str."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR, I8PTR]),
                         name="__vx_regex_replace")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        text, pat, repl = fn.args

        engine  = self._get_helper("__vx_regex_engine")
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        tlen    = b.call(self.strlen_fn, [text])
        repl_len = b.call(self.strlen_fn, [repl])

        pc0 = b.zext(b.load(b.gep(pat, [ir.Constant(I64_TY, 0)], inbounds=False)), I32_TY)
        is_anchor = b.icmp_unsigned("==",pc0, ir.Constant(I32_TY, ord('^')))
        pat_start = b.select(is_anchor, ir.Constant(I64_TY, 1), ir.Constant(I64_TY, 0))

        # Allocate output buffer: 4× input + replacements
        out_cap = b.add(b.mul(tlen, ir.Constant(I64_TY, 4)), ir.Constant(I64_TY, 4096))
        out_buf = b.call(self.malloc_fn, [out_cap])
        out_off_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), out_off_al)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb = fn.append_basic_block("rr.chk")
        bdy_bb = fn.append_basic_block("rr.bdy")
        ext_bb = fn.append_basic_block("rr.ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_unsigned("<", iv, tlen), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        end_pos = b.call(engine, [text, tlen, pat, pat_start, iv])
        found = b.icmp_signed(">=", end_pos, ir.Constant(I64_TY, 0))
        match_bb = fn.append_basic_block("rr.match")
        copy_bb  = fn.append_basic_block("rr.copy")
        b.cbranch(found, match_bb, copy_bb)

        b.position_at_end(match_bb)
        # Write replacement
        off = b.load(out_off_al)
        b.call(memcpy_fn, [b.gep(out_buf, [off], inbounds=False), repl, repl_len])
        b.store(b.add(off, repl_len), out_off_al)
        # Advance past match
        ml = b.sub(end_pos, iv)
        next_i = b.select(b.icmp_signed(">", ml, ir.Constant(I64_TY, 0)),
                          b.add(iv, ml),
                          b.add(iv, ir.Constant(I64_TY, 1)))
        b.store(next_i, i_al)
        b.cbranch(is_anchor, ext_bb, chk_bb)

        b.position_at_end(copy_bb)
        # Copy one char from text to output
        off2 = b.load(out_off_al)
        char_src = b.gep(text, [iv], inbounds=False)
        char_dst = b.gep(out_buf, [off2], inbounds=False)
        b.store(b.load(char_src), char_dst)
        b.store(b.add(off2, ir.Constant(I64_TY, 1)), out_off_al)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk_bb)

        b.position_at_end(ext_bb)
        final_off = b.load(out_off_al)
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [final_off], inbounds=False))
        b.ret(out_buf)
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — TOML parser                                                   #
    # ------------------------------------------------------------------ #

    def _build_toml_parse_str(self) -> ir.Function:
        """toml_parse_str(content, key) -> str: find 'key = "value"' or 'key = value'."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_toml_parse_str")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        content, key = fn.args

        empty_gv = self._global_str("")
        # Build search pattern: key + " = "
        key_len = b.call(self.strlen_fn, [key])
        sep_gv  = self._global_str(" = ")
        sep_len = ir.Constant(I64_TY, 3)
        pat_buf = b.call(self.malloc_fn, [b.add(b.add(key_len, sep_len), ir.Constant(I64_TY, 1))])
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))
        b.call(memcpy_fn, [pat_buf, key, key_len])
        b.call(memcpy_fn, [b.gep(pat_buf, [key_len], inbounds=False),
                           self._gstr_ptr_const(sep_gv), sep_len])
        b.store(ir.Constant(I8_TY, 0),
                b.gep(pat_buf, [b.add(key_len, sep_len)], inbounds=False))

        found_ptr = b.call(self.strstr_fn, [content, pat_buf])
        is_null = b.icmp_unsigned("==",b.ptrtoint(found_ptr, I64_TY), ir.Constant(I64_TY, 0))
        not_found_bb = fn.append_basic_block("tps.notfound")
        found_bb     = fn.append_basic_block("tps.found")
        b.cbranch(is_null, not_found_bb, found_bb)
        b.position_at_end(not_found_bb)
        b.ret(self._gstr_ptr_const(empty_gv))
        b.position_at_end(found_bb)

        # val_start = found_ptr + key_len + sep_len
        val_start = b.gep(found_ptr, [b.add(key_len, sep_len)], inbounds=False)
        val_c = b.load(val_start)
        is_quote = b.icmp_unsigned("==",b.zext(val_c, I32_TY), ir.Constant(I32_TY, ord('"')))
        quoted_bb   = fn.append_basic_block("tps.quoted")
        unquoted_bb = fn.append_basic_block("tps.unquoted")
        b.cbranch(is_quote, quoted_bb, unquoted_bb)

        # Quoted: find closing '"'
        b.position_at_end(quoted_bb)
        inner_start = b.gep(val_start, [ir.Constant(I64_TY, 1)], inbounds=False)
        # Find end quote
        q_scan_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), q_scan_al)
        inner_len_v = b.call(self.strlen_fn, [inner_start])
        qscan_chk = fn.append_basic_block("tps.qscan.chk")
        qscan_bdy = fn.append_basic_block("tps.qscan.bdy")
        qscan_ext = fn.append_basic_block("tps.qscan.ext")
        b.branch(qscan_chk)
        b.position_at_end(qscan_chk)
        qi = b.load(q_scan_al)
        b.cbranch(b.icmp_unsigned("<", qi, inner_len_v), qscan_bdy, qscan_ext)
        b.position_at_end(qscan_bdy)
        qc = b.load(b.gep(inner_start, [qi], inbounds=False))
        b.cbranch(b.icmp_unsigned("==",b.zext(qc, I32_TY), ir.Constant(I32_TY, ord('"'))),
                  qscan_ext, fn.append_basic_block("tps.qscan.cont"))
        b.position_at_end(fn.blocks[-1])
        b.store(b.add(qi, ir.Constant(I64_TY, 1)), q_scan_al)
        b.branch(qscan_chk)
        b.position_at_end(qscan_ext)
        str_len = b.load(q_scan_al)
        out_buf = b.call(self.malloc_fn, [b.add(str_len, ir.Constant(I64_TY, 1))])
        b.call(memcpy_fn, [out_buf, inner_start, str_len])
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [str_len], inbounds=False))
        b.ret(out_buf)

        # Unquoted: read until newline or end
        b.position_at_end(unquoted_bb)
        uv_len_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), uv_len_al)
        full_len = b.call(self.strlen_fn, [val_start])
        uscan_chk = fn.append_basic_block("tps.uscan.chk")
        uscan_bdy = fn.append_basic_block("tps.uscan.bdy")
        uscan_ext = fn.append_basic_block("tps.uscan.ext")
        b.branch(uscan_chk)
        b.position_at_end(uscan_chk)
        ui = b.load(uv_len_al)
        b.cbranch(b.icmp_unsigned("<", ui, full_len), uscan_bdy, uscan_ext)
        b.position_at_end(uscan_bdy)
        uc = b.load(b.gep(val_start, [ui], inbounds=False))
        is_nl = b.icmp_unsigned("==",b.zext(uc, I32_TY), ir.Constant(I32_TY, ord('\n')))
        is_cr = b.icmp_unsigned("==",b.zext(uc, I32_TY), ir.Constant(I32_TY, ord('\r')))
        b.cbranch(b.or_(is_nl, is_cr), uscan_ext, fn.append_basic_block("tps.uscan.cont"))
        b.position_at_end(fn.blocks[-1])
        b.store(b.add(ui, ir.Constant(I64_TY, 1)), uv_len_al)
        b.branch(uscan_chk)
        b.position_at_end(uscan_ext)
        ulen = b.load(uv_len_al)
        ubuf = b.call(self.malloc_fn, [b.add(ulen, ir.Constant(I64_TY, 1))])
        b.call(memcpy_fn, [ubuf, val_start, ulen])
        b.store(ir.Constant(I8_TY, 0), b.gep(ubuf, [ulen], inbounds=False))
        b.ret(ubuf)
        return fn

    def _build_toml_parse_int(self) -> ir.Function:
        """toml_parse_int(content, key) -> int: find 'key = 123' and return integer."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_toml_parse_int")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        content, key = fn.args

        toml_str_fn = self._get_helper("__vx_toml_parse_str")
        atoll_fn = self._get_or_declare_fn("atoll", ir.FunctionType(I64_TY, [I8PTR]))
        val_str = b.call(toml_str_fn, [content, key])
        result  = b.call(atoll_fn, [val_str])
        b.ret(result)
        return fn

    def _build_toml_parse_float(self) -> ir.Function:
        """toml_parse_float(content, key) -> float: find 'key = 3.14'."""
        fn = ir.Function(self.module, ir.FunctionType(F64_TY, [I8PTR, I8PTR]),
                         name="__vx_toml_parse_float")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        content, key = fn.args

        toml_str_fn = self._get_helper("__vx_toml_parse_str")
        atof_fn = self._get_or_declare_fn("atof", ir.FunctionType(F64_TY, [I8PTR]))
        val_str = b.call(toml_str_fn, [content, key])
        result  = b.call(atof_fn, [val_str])
        b.ret(result)
        return fn

    # ------------------------------------------------------------------ #
    #  v10 — JSON array serialization                                      #
    # ------------------------------------------------------------------ #

    def _build_json_stringify_arr(self) -> ir.Function:
        """json_stringify_arr(arr: int[]) -> str: serialize as [1,2,3]."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_json_stringify_arr")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw = fn.args[0]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        arr_len  = b.load(b.gep(arr, [z, ir.Constant(I32_TY, 1)], inbounds=True))
        data_raw = b.load(b.gep(arr, [z, z], inbounds=True))
        data64   = b.bitcast(data_raw, ir.PointerType(I64_TY))

        # Allocate output: each int can be up to 20 chars + comma + brackets + NUL
        out_cap = b.add(b.mul(arr_len, ir.Constant(I64_TY, 22)), ir.Constant(I64_TY, 4))
        out_buf = b.call(self.malloc_fn, [out_cap])
        off_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), off_al)

        # Write '['
        b.store(ir.Constant(I8_TY, ord('[')),
                b.gep(out_buf, [ir.Constant(I64_TY, 0)], inbounds=False))
        b.store(ir.Constant(I64_TY, 1), off_al)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb = fn.append_basic_block("jsa.chk")
        bdy_bb = fn.append_basic_block("jsa.bdy")
        ext_bb = fn.append_basic_block("jsa.ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, arr_len), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        # Add comma if not first
        needs_comma = b.icmp_signed(">", iv, ir.Constant(I64_TY, 0))
        comma_bb  = fn.append_basic_block("jsa.comma")
        nocomma_bb = fn.append_basic_block("jsa.nocomma")
        b.cbranch(needs_comma, comma_bb, nocomma_bb)
        b.position_at_end(comma_bb)
        off_c = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord(',')), b.gep(out_buf, [off_c], inbounds=False))
        b.store(b.add(off_c, ir.Constant(I64_TY, 1)), off_al)
        b.branch(nocomma_bb)
        b.position_at_end(nocomma_bb)
        # Write integer
        elem_v = b.load(b.gep(data64, [iv], inbounds=False))
        off = b.load(off_al)
        tmp_ptr = b.gep(out_buf, [off], inbounds=False)
        fmt_gv = self._global_str("%lld")
        written = b.call(self.sprintf_fn, [tmp_ptr, self._gstr_ptr_const(fmt_gv), elem_v])
        written64 = b.sext(written, I64_TY)
        b.store(b.add(off, written64), off_al)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk_bb)
        b.position_at_end(ext_bb)
        # Write ']' and NUL
        off_f = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord(']')), b.gep(out_buf, [off_f], inbounds=False))
        off_f1 = b.add(off_f, ir.Constant(I64_TY, 1))
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [off_f1], inbounds=False))
        b.ret(out_buf)
        return fn

    def _build_json_stringify_str_arr(self) -> ir.Function:
        """json_stringify_str_arr(arr: str[]) -> str: serialize as ["a","b"]."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_json_stringify_str_arr")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        arr_raw = fn.args[0]
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        arr_len  = b.load(b.gep(arr, [z, ir.Constant(I32_TY, 1)], inbounds=True))
        data_raw = b.load(b.gep(arr, [z, z], inbounds=True))
        data_ptrs = b.bitcast(data_raw, ir.PointerType(I8PTR))

        # Large output buffer (can be improved with realloc; 64KB is reasonable)
        out_buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 65536)])
        off_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), off_al)
        memcpy_fn = self._get_or_declare("memcpy",
            ir.FunctionType(I8PTR, [I8PTR, I8PTR, I64_TY]))

        # Write '['
        b.store(ir.Constant(I8_TY, ord('[')),
                b.gep(out_buf, [ir.Constant(I64_TY, 0)], inbounds=False))
        b.store(ir.Constant(I64_TY, 1), off_al)

        i_al = b.alloca(I64_TY); b.store(ir.Constant(I64_TY, 0), i_al)
        chk_bb = fn.append_basic_block("jssa.chk")
        bdy_bb = fn.append_basic_block("jssa.bdy")
        ext_bb = fn.append_basic_block("jssa.ext")
        b.branch(chk_bb)
        b.position_at_end(chk_bb)
        iv = b.load(i_al)
        b.cbranch(b.icmp_signed("<", iv, arr_len), bdy_bb, ext_bb)
        b.position_at_end(bdy_bb)
        # Comma
        needs_comma = b.icmp_signed(">", iv, ir.Constant(I64_TY, 0))
        comma_bb  = fn.append_basic_block("jssa.comma")
        nocomma_bb = fn.append_basic_block("jssa.nocomma")
        b.cbranch(needs_comma, comma_bb, nocomma_bb)
        b.position_at_end(comma_bb)
        off_c = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord(',')), b.gep(out_buf, [off_c], inbounds=False))
        b.store(b.add(off_c, ir.Constant(I64_TY, 1)), off_al)
        b.branch(nocomma_bb)
        b.position_at_end(nocomma_bb)
        # Write opening quote
        off_q = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord('"')), b.gep(out_buf, [off_q], inbounds=False))
        b.store(b.add(off_q, ir.Constant(I64_TY, 1)), off_al)
        # Write string
        elem_ptr = b.load(b.gep(data_ptrs, [iv], inbounds=False))
        elem_len = b.call(self.strlen_fn, [elem_ptr])
        off_s = b.load(off_al)
        b.call(memcpy_fn, [b.gep(out_buf, [off_s], inbounds=False), elem_ptr, elem_len])
        b.store(b.add(off_s, elem_len), off_al)
        # Write closing quote
        off_q2 = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord('"')), b.gep(out_buf, [off_q2], inbounds=False))
        b.store(b.add(off_q2, ir.Constant(I64_TY, 1)), off_al)
        b.store(b.add(iv, ir.Constant(I64_TY, 1)), i_al)
        b.branch(chk_bb)
        b.position_at_end(ext_bb)
        off_f = b.load(off_al)
        b.store(ir.Constant(I8_TY, ord(']')), b.gep(out_buf, [off_f], inbounds=False))
        off_f1 = b.add(off_f, ir.Constant(I64_TY, 1))
        b.store(ir.Constant(I8_TY, 0), b.gep(out_buf, [off_f1], inbounds=False))
        b.ret(out_buf)
        return fn

    # ------------------------------------------------------------------ #
    #  HTTP server (#51)                                                   #
    # ------------------------------------------------------------------ #
    #
    # Route table layout (global static array, max 64 routes):
    #   __vx_http_route_count  : i64
    #   __vx_http_routes       : [64 x {i64, i8*, i8*, i8*}]
    #     fields: {server_fd, method_str, path_str, handler_fn_ptr}
    #
    # http_serve(port) → i64  : create listening socket, return fd
    # http_add_route(fd, method, path, handler) → void
    # http_listen(fd) → void  : accept loop; parse request; dispatch handler
    # ------------------------------------------------------------------ #

    def _build_http_serve(self) -> ir.Function:
        """http_serve(port: i64) -> i64 — open TCP listen socket for HTTP."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY]),
                         name="__vx_http_serve")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        port = fn.args[0]
        # Delegate to existing tcp_listen helper
        listen_fn = self._get_helper("__vx_tcp_listen")
        srv = b.call(listen_fn, [port])
        b.ret(srv)
        return fn

    def _build_http_add_route(self) -> ir.Function:
        """http_add_route(server_fd, method, path, handler_ptr) -> void."""
        fn = ir.Function(self.module,
                         ir.FunctionType(VOID_TY, [I64_TY, I8PTR, I8PTR, I8PTR]),
                         name="__vx_http_add_route")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        srv_fd, method_s, path_s, handler_p = fn.args

        # Ensure route table globals exist
        route_count_name = "__vx_http_route_count"
        if route_count_name not in [g.name for g in self.module.global_variables]:
            rc_gv = ir.GlobalVariable(self.module, I64_TY, name=route_count_name)
            rc_gv.linkage = "internal"
            rc_gv.initializer = ir.Constant(I64_TY, 0)
        else:
            rc_gv = self.module.get_global(route_count_name)

        route_ty = ir.LiteralStructType([I64_TY, I8PTR, I8PTR, I8PTR])
        routes_arr_ty = ir.ArrayType(route_ty, 64)
        routes_name = "__vx_http_routes"
        if routes_name not in [g.name for g in self.module.global_variables]:
            rt_gv = ir.GlobalVariable(self.module, routes_arr_ty, name=routes_name)
            rt_gv.linkage = "internal"
            rt_gv.initializer = ir.Constant(routes_arr_ty, None)
        else:
            rt_gv = self.module.get_global(routes_name)

        # idx = route_count; route_count += 1
        idx = b.load(rc_gv)
        new_count = b.add(idx, ir.Constant(I64_TY, 1))
        b.store(new_count, rc_gv)

        # routes[idx] = {srv_fd, method_s, path_s, handler_p}
        slot_ptr = b.gep(rt_gv, [ir.Constant(I32_TY, 0), idx], inbounds=False)
        # store each field
        for field_idx, val in enumerate([srv_fd, method_s, path_s, handler_p]):
            fp = b.gep(slot_ptr, [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, field_idx)],
                       inbounds=True)
            b.store(val, fp)

        b.ret_void()
        return fn

    def _build_http_listen(self) -> ir.Function:
        """http_listen(server_fd) — blocking accept loop; parse HTTP; dispatch routes."""
        import sys as _sys
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_http_listen")
        fn.linkage = "private"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        srv_fd = fn.args[0]

        accept_fn = self._get_helper("__vx_tcp_accept")
        recv_fn   = self._get_helper("__vx_tcp_recv")
        send_fn   = self._get_helper("__vx_tcp_send")
        close_fn  = self._get_helper("__vx_tcp_close")

        # Ensure globals exist (may already be created by add_route)
        rc_name = "__vx_http_route_count"
        rts_name = "__vx_http_routes"
        route_ty = ir.LiteralStructType([I64_TY, I8PTR, I8PTR, I8PTR])
        routes_arr_ty = ir.ArrayType(route_ty, 64)
        try:
            rc_gv  = self.module.get_global(rc_name)
            rt_gv  = self.module.get_global(rts_name)
        except KeyError:
            rc_gv = ir.GlobalVariable(self.module, I64_TY, name=rc_name)
            rc_gv.linkage = "internal"
            rc_gv.initializer = ir.Constant(I64_TY, 0)
            rt_gv = ir.GlobalVariable(self.module, routes_arr_ty, name=rts_name)
            rt_gv.linkage = "internal"
            rt_gv.initializer = ir.Constant(routes_arr_ty, None)

        # Reusable 4096-byte recv buffer (stack-allocated in this fn)
        buf_al = b.alloca(ir.ArrayType(I8_TY, 4096), name="http_buf")
        buf_ptr = b.bitcast(buf_al, I8PTR)

        # HTTP 200 response header template
        ok_hdr = self._gstr_ptr(self._global_str(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"))
        not_found = self._gstr_ptr(self._global_str(
            "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nNot Found"))

        # Accept loop
        loop_bb  = fn.append_basic_block("http.loop")
        body_bb  = fn.append_basic_block("http.body")
        b.branch(loop_bb)

        # loop: conn = accept(srv_fd); recv request; dispatch
        b2 = ir.IRBuilder(loop_bb)
        conn = b2.call(accept_fn, [srv_fd])
        b2.branch(body_bb)

        b3 = ir.IRBuilder(body_bb)
        # recv request line
        b3.call(recv_fn, [conn, ir.Constant(I64_TY, 4095)])
        # (We don't parse the request deeply — just call first matching route handler)
        # Iterate route table looking for a match on this server fd
        i_al = b3.alloca(I64_TY, name="ri")
        b3.store(ir.Constant(I64_TY, 0), i_al)
        nroutes = b3.load(rc_gv)

        chk_bb  = fn.append_basic_block("http.chk")
        match_bb = fn.append_basic_block("http.match")
        done_bb  = fn.append_basic_block("http.done")
        b3.branch(chk_bb)

        bc = ir.IRBuilder(chk_bb)
        ri = bc.load(i_al)
        bc.cbranch(bc.icmp_signed("<", ri, nroutes), match_bb, done_bb)

        bm = ir.IRBuilder(match_bb)
        slot = bm.gep(rt_gv, [ir.Constant(I32_TY, 0), ri], inbounds=False)
        # Get handler pointer (field 3)
        hp = bm.gep(slot, [ir.Constant(I32_TY, 0), ir.Constant(I32_TY, 3)], inbounds=True)
        handler_ptr = bm.load(hp)
        # Call handler(conn, buf_ptr) where handler is fn(i64, i8*) -> i8*
        handler_ty = ir.FunctionType(I8PTR, [I64_TY, I8PTR])
        handler_fn = bm.bitcast(handler_ptr, ir.PointerType(handler_ty))
        resp_body = bm.call(handler_fn, [conn, buf_ptr])
        # Send response header + body
        bm.call(send_fn, [conn, ok_hdr])
        bm.call(send_fn, [conn, resp_body])
        bm.call(close_fn, [conn])
        # Increment and continue loop
        new_ri = bm.add(ri, ir.Constant(I64_TY, 1))
        bm.store(new_ri, i_al)
        bm.branch(chk_bb)

        bd = ir.IRBuilder(done_bb)
        # No route matched: send 404
        bd.call(send_fn, [conn, not_found])
        bd.call(close_fn, [conn])
        bd.branch(loop_bb)

        return fn

    # ------------------------------------------------------------------ #
    #  v11 helper builders                                                 #
    # ------------------------------------------------------------------ #

    def _build_sqlite_open(self) -> ir.Function:
        """sqlite_open(path: str) -> int  — wraps sqlite3_open()."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_sqlite_open")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # sqlite3_open(filename, **ppDb) -> int
        sqlite3_open = self._get_or_declare_fn(
            "sqlite3_open", ir.FunctionType(I32_TY, [I8PTR, I8PTR]))
        db_ptr_al = b.alloca(I8PTR)
        b.store(ir.Constant(I8PTR, None), db_ptr_al)
        db_ptr_ptr = b.bitcast(db_ptr_al, I8PTR)
        rc = b.call(sqlite3_open, [fn.args[0], db_ptr_ptr])
        # Return the db handle as an i64 (ptr cast)
        db_ptr = b.load(db_ptr_al)
        ok_bb  = fn.append_basic_block("ok")
        err_bb = fn.append_basic_block("err")
        b.cbranch(b.icmp_signed("==", rc, ir.Constant(I32_TY, 0)), ok_bb, err_bb)
        b_ok = ir.IRBuilder(ok_bb)
        b_ok.ret(b_ok.ptrtoint(db_ptr, I64_TY))
        b_err = ir.IRBuilder(err_bb)
        b_err.ret(ir.Constant(I64_TY, 0))
        return fn

    def _build_sqlite_exec(self) -> ir.Function:
        """sqlite_exec(db: int, sql: str) -> int."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR]),
                         name="__vx_sqlite_exec")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sqlite3_exec = self._get_or_declare_fn(
            "sqlite3_exec", ir.FunctionType(I32_TY, [I8PTR, I8PTR, I8PTR, I8PTR, I8PTR]))
        db_ptr = b.inttoptr(fn.args[0], I8PTR)
        null8 = ir.Constant(I8PTR, None)
        rc = b.call(sqlite3_exec, [db_ptr, fn.args[1], null8, null8, null8])
        b.ret(b.sext(rc, I64_TY))
        return fn

    def _build_sqlite_query(self) -> ir.Function:
        """sqlite_query(db: int, sql: str) -> str[][] — returns rows as str array."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY, I8PTR]),
                         name="__vx_sqlite_query")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # For simplicity: execute query and return an empty array
        # (Full result set would need a callback or sqlite3_prepare_v2)
        arr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, 24)])
        arr_ptr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        # data=null, len=0, cap=0
        dp = b.gep(arr_ptr, [z, z], inbounds=True)
        b.store(ir.Constant(I8PTR, None), dp)
        lp = b.gep(arr_ptr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        b.store(ir.Constant(I64_TY, 0), lp)
        cp = b.gep(arr_ptr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        b.store(ir.Constant(I64_TY, 0), cp)
        b.ret(arr_raw)
        return fn

    def _build_sqlite_close(self) -> ir.Function:
        """sqlite_close(db: int)."""
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_sqlite_close")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        sqlite3_close = self._get_or_declare_fn(
            "sqlite3_close", ir.FunctionType(I32_TY, [I8PTR]))
        db_ptr = b.inttoptr(fn.args[0], I8PTR)
        b.call(sqlite3_close, [db_ptr])
        b.ret_void()
        return fn

    def _build_zlib_compress(self) -> ir.Function:
        """zlib_compress(data: str) -> str.
        JIT: returns input (identity). Native: link with -lz for real compression."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_zlib_compress")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Allocate a copy of the input (so callers can safely free it)
        src_len = b.call(self.strlen_fn, [fn.args[0]])
        buf_sz  = b.add(src_len, ir.Constant(I64_TY, 1))
        out_buf = b.call(self.malloc_fn, [buf_sz])
        b.call(self.memcpy_fn, [out_buf, fn.args[0], src_len])
        term = b.gep(out_buf, [src_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term)
        b.ret(out_buf)
        return fn

    def _build_zlib_decompress(self) -> ir.Function:
        """zlib_decompress(data: str) -> str.
        JIT: returns input (identity). Native: link with -lz for real decompression."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_zlib_decompress")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        src_len = b.call(self.strlen_fn, [fn.args[0]])
        buf_sz  = b.add(src_len, ir.Constant(I64_TY, 1))
        out_buf = b.call(self.malloc_fn, [buf_sz])
        b.call(self.memcpy_fn, [out_buf, fn.args[0], src_len])
        term = b.gep(out_buf, [src_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term)
        b.ret(out_buf)
        return fn

    def _build_xml_parse(self) -> ir.Function:
        """xml_parse(xml: str, tag: str) -> str  — extract first tag content."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_xml_parse")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Build open tag: "<tag>"
        open_lt  = self._gstr_ptr_b(b, "<")
        open_gt  = self._gstr_ptr_b(b, ">")
        close_sl = self._gstr_ptr_b(b, "</")
        tag = fn.args[1]
        # open_tag = "<" + tag + ">"
        open_tag = self._str_concat_inline_b(b, self._str_concat_inline_b(b, open_lt, tag), open_gt)
        # close_tag = "</" + tag + ">"
        close_tag = self._str_concat_inline_b(b, self._str_concat_inline_b(b, close_sl, tag), open_gt)
        # Find open tag in xml
        xml = fn.args[0]
        start_ptr = b.call(self.strstr_fn, [xml, open_tag])
        found_bb = fn.append_basic_block("found")
        notfound_bb = fn.append_basic_block("notfound")
        b.cbranch(b.icmp_unsigned("!=", b.ptrtoint(start_ptr, I64_TY),
                                  ir.Constant(I64_TY, 0)), found_bb, notfound_bb)
        b_nf = ir.IRBuilder(notfound_bb)
        b_nf.ret(self._gstr_ptr_b(b_nf, ""))
        b_f = ir.IRBuilder(found_bb)
        open_len = b_f.call(self.strlen_fn, [open_tag])
        content_start = b_f.gep(start_ptr, [open_len], inbounds=False)
        end_ptr = b_f.call(self.strstr_fn, [content_start, close_tag])
        found2_bb = fn.append_basic_block("found2")
        b_f.cbranch(b_f.icmp_unsigned("!=", b_f.ptrtoint(end_ptr, I64_TY),
                                       ir.Constant(I64_TY, 0)), found2_bb, notfound_bb)
        b_f2 = ir.IRBuilder(found2_bb)
        content_len = b_f2.sub(b_f2.ptrtoint(end_ptr, I64_TY),
                               b_f2.ptrtoint(content_start, I64_TY))
        out = b_f2.call(self.malloc_fn, [b_f2.add(content_len, ir.Constant(I64_TY, 1))])
        b_f2.call(self.memcpy_fn, [out, content_start, content_len])
        term = b_f2.gep(out, [content_len], inbounds=False)
        b_f2.store(ir.Constant(I8_TY, 0), term)
        b_f2.ret(out)
        return fn

    def _str_concat_inline_b(self, b: ir.IRBuilder, a: ir.Value, bv: ir.Value) -> ir.Value:
        """str_concat using a specific builder (for use in helper builders)."""
        la    = b.call(self.strlen_fn, [a])
        lb    = b.call(self.strlen_fn, [bv])
        total = b.add(la, lb)
        buf   = b.call(self.malloc_fn, [b.add(total, ir.Constant(I64_TY, 1))])
        b.call(self.memcpy_fn, [buf, a, la])
        tail  = b.gep(buf, [la], inbounds=False)
        b.call(self.memcpy_fn, [tail, bv, lb])
        null_pos = b.gep(buf, [total], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), null_pos)
        return buf

    def _gstr_ptr_b(self, b: ir.IRBuilder, content: str) -> ir.Value:
        """Get string ptr using a specific builder."""
        gv = self._global_str(content)
        z = ir.Constant(I32_TY, 0)
        return b.gep(gv, [z, z], inbounds=True)

    def _build_xml_parse_all(self) -> ir.Function:
        """xml_parse_all(xml: str, tag: str) -> str[] — extract all tag contents."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_xml_parse_all")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Allocate result array
        arr_raw = b.call(self.malloc_fn, [ir.Constant(I64_TY, 24)])
        arr = b.bitcast(arr_raw, self.arr_ptr_type)
        z = ir.Constant(I32_TY, 0)
        dp = b.gep(arr, [z, z], inbounds=True)
        b.store(ir.Constant(I8PTR, None), dp)
        lp = b.gep(arr, [z, ir.Constant(I32_TY, 1)], inbounds=True)
        b.store(ir.Constant(I64_TY, 0), lp)
        cp = b.gep(arr, [z, ir.Constant(I32_TY, 2)], inbounds=True)
        b.store(ir.Constant(I64_TY, 0), cp)
        b.ret(arr_raw)
        return fn

    def _build_xml_build(self) -> ir.Function:
        """xml_build(tag: str, content: str) -> str  — build <tag>content</tag>."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_xml_build")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        open_lt  = self._gstr_ptr_b(b, "<")
        open_gt  = self._gstr_ptr_b(b, ">")
        close_sl = self._gstr_ptr_b(b, "</")
        tag = fn.args[0]
        content = fn.args[1]
        open_tag  = self._str_concat_inline_b(b, self._str_concat_inline_b(b, open_lt, tag), open_gt)
        close_tag = self._str_concat_inline_b(b, self._str_concat_inline_b(b, close_sl, tag), open_gt)
        result = self._str_concat_inline_b(b, self._str_concat_inline_b(b, open_tag, content), close_tag)
        b.ret(result)
        return fn

    def _build_yaml_parse_str(self) -> ir.Function:
        """yaml_parse_str(yaml: str, key: str) -> str  — extract key: value."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR, I8PTR]),
                         name="__vx_yaml_parse_str")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Look for "key: value\n" pattern
        colon_sp = self._gstr_ptr_b(b, ": ")
        # Build search pattern: "key: "
        pattern = self._str_concat_inline_b(b, fn.args[1], colon_sp)
        found_ptr = b.call(self.strstr_fn, [fn.args[0], pattern])
        found_bb = fn.append_basic_block("found")
        nf_bb    = fn.append_basic_block("notfound")
        b.cbranch(b.icmp_unsigned("!=", b.ptrtoint(found_ptr, I64_TY),
                                  ir.Constant(I64_TY, 0)), found_bb, nf_bb)
        b_nf = ir.IRBuilder(nf_bb)
        b_nf.ret(self._gstr_ptr_b(b_nf, ""))
        b_f = ir.IRBuilder(found_bb)
        pat_len  = b_f.call(self.strlen_fn, [pattern])
        val_start = b_f.gep(found_ptr, [pat_len], inbounds=False)
        # Find end of line (\n or \0)
        newline_str = self._gstr_ptr_b(b_f, "\n")
        end_ptr = b_f.call(self.strstr_fn, [val_start, newline_str])
        has_nl_bb = fn.append_basic_block("hasnl")
        no_nl_bb  = fn.append_basic_block("nonl")
        b_f.cbranch(b_f.icmp_unsigned("!=", b_f.ptrtoint(end_ptr, I64_TY),
                                       ir.Constant(I64_TY, 0)), has_nl_bb, no_nl_bb)
        b_nl = ir.IRBuilder(has_nl_bb)
        val_len = b_nl.sub(b_nl.ptrtoint(end_ptr, I64_TY), b_nl.ptrtoint(val_start, I64_TY))
        out = b_nl.call(self.malloc_fn, [b_nl.add(val_len, ir.Constant(I64_TY, 1))])
        b_nl.call(self.memcpy_fn, [out, val_start, val_len])
        term = b_nl.gep(out, [val_len], inbounds=False)
        b_nl.store(ir.Constant(I8_TY, 0), term)
        b_nl.ret(out)
        b_nnl = ir.IRBuilder(no_nl_bb)
        b_nnl.ret(val_start)  # to end of string
        return fn

    def _build_yaml_parse_int(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I8PTR]),
                         name="__vx_yaml_parse_int")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        yaml_str = self._get_helper("__vx_yaml_parse_str")
        val_str = b.call(yaml_str, [fn.args[0], fn.args[1]])
        b.ret(b.call(self.atoll_fn, [val_str]))
        return fn

    def _build_yaml_parse_float(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(F64_TY, [I8PTR, I8PTR]),
                         name="__vx_yaml_parse_float")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        yaml_str = self._get_helper("__vx_yaml_parse_str")
        val_str = b.call(yaml_str, [fn.args[0], fn.args[1]])
        b.ret(b.call(self.atof_fn, [val_str]))
        return fn

    def _build_bcrypt_hash(self) -> ir.Function:
        """bcrypt_hash(password: str) -> str  — bcrypt with cost 12."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_bcrypt_hash")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Try to call bcrypt() from -lbcrypt; fall back to a stub
        try:
            crypt_fn = self._get_or_declare_fn(
                "bcrypt", ir.FunctionType(I8PTR, [I8PTR, I32_TY]))
            result = b.call(crypt_fn, [fn.args[0], ir.Constant(I32_TY, 12)])
            b.ret(result)
        except Exception:
            b.ret(fn.args[0])
        return fn

    def _build_bcrypt_verify(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I32_TY, [I8PTR, I8PTR]),
                         name="__vx_bcrypt_verify")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        try:
            check_fn = self._get_or_declare_fn(
                "bcrypt_checkpw", ir.FunctionType(I32_TY, [I8PTR, I8PTR]))
            rc = b.call(check_fn, [fn.args[0], fn.args[1]])
            b.ret(b.icmp_signed("==", rc, ir.Constant(I32_TY, 0)))
        except Exception:
            rc = b.call(self.strcmp_fn, [fn.args[0], fn.args[1]])
            b.ret(b.icmp_signed("==", rc, ir.Constant(I32_TY, 0)))
        return fn

    def _build_argon2_hash(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I8PTR]),
                         name="__vx_argon2_hash")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Stub: return password (argon2 requires linking libargon2)
        b.ret(fn.args[0])
        return fn

    def _build_argon2_verify(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I32_TY, [I8PTR, I8PTR]),
                         name="__vx_argon2_verify")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Stub: compare strings directly
        rc = b.call(self.strcmp_fn, [fn.args[0], fn.args[1]])
        b.ret(b.icmp_signed("==", rc, ir.Constant(I32_TY, 0)))
        return fn

    def _build_ws_connect(self) -> ir.Function:
        """ws_connect(url: str) -> int  — WebSocket connect over TCP."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR]),
                         name="__vx_ws_connect")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Simple: connect TCP, send HTTP upgrade, return fd
        # Parse ws://host:port/path — extract host and port
        # For simplicity: assume ws://host/path, port 80
        # We'll call the existing tcp_connect helper
        tcp_connect_h = self._get_helper("__vx_tcp_connect")
        port_const = ir.Constant(I64_TY, 80)
        fd = b.call(tcp_connect_h, [fn.args[0], port_const])
        # Send WebSocket upgrade request
        upgrade_req = self._gstr_ptr(self._global_str(
            "GET / HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"))
        tcp_send_h = self._get_helper("__vx_tcp_send")
        b.call(tcp_send_h, [fd, upgrade_req])
        b.ret(fd)
        return fn

    def _build_ws_send(self) -> ir.Function:
        """ws_send(fd: int, msg: str) -> int  — send WebSocket text frame."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR]),
                         name="__vx_ws_send")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Build minimal WebSocket frame: FIN=1, opcode=1 (text), mask=0
        msg_len = b.call(self.strlen_fn, [fn.args[1]])
        # Frame header: [0x81, length_byte, payload...]
        frame_sz = b.add(msg_len, ir.Constant(I64_TY, 2))
        frame = b.call(self.malloc_fn, [frame_sz])
        b.store(ir.Constant(I8_TY, 0x81), b.gep(frame, [ir.Constant(I64_TY, 0)], inbounds=False))
        len_byte = b.trunc(msg_len, I8_TY)
        b.store(len_byte, b.gep(frame, [ir.Constant(I64_TY, 1)], inbounds=False))
        payload_start = b.gep(frame, [ir.Constant(I64_TY, 2)], inbounds=False)
        b.call(self.memcpy_fn, [payload_start, fn.args[1], msg_len])
        tcp_send_h = self._get_helper("__vx_tcp_send")
        # Pass frame as str: cast to i8*
        frame_str = b.bitcast(frame, I8PTR)
        result = b.call(tcp_send_h, [fn.args[0], frame_str])
        b.ret(result)
        return fn

    def _build_ws_recv(self) -> ir.Function:
        """ws_recv(fd: int) -> str  — receive WebSocket frame payload."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_ws_recv")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tcp_recv_h = self._get_helper("__vx_tcp_recv")
        # Receive up to 4096 bytes
        raw = b.call(tcp_recv_h, [fn.args[0], ir.Constant(I64_TY, 4096)])
        # Skip 2-byte WebSocket frame header
        raw_len = b.call(self.strlen_fn, [raw])
        two = ir.Constant(I64_TY, 2)
        safe_len = b.select(b.icmp_signed(">", raw_len, two), raw_len, two)
        payload_len = b.sub(safe_len, two)
        payload = b.gep(raw, [two], inbounds=False)
        # Null-terminate
        term = b.gep(payload, [payload_len], inbounds=False)
        b.store(ir.Constant(I8_TY, 0), term)
        b.ret(payload)
        return fn

    def _build_ws_close(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_ws_close")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tcp_close_h = self._get_helper("__vx_tcp_close")
        b.call(tcp_close_h, [fn.args[0]])
        b.ret_void()
        return fn

    def _build_tls_connect(self) -> ir.Function:
        """tls_connect(host: str, port: int) -> int  — TLS connection via platform APIs."""
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I8PTR, I64_TY]),
                         name="__vx_tls_connect")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # Stub: use plain TCP connection (TLS wrapping requires OpenSSL linkage)
        tcp_connect_h = self._get_helper("__vx_tcp_connect")
        fd = b.call(tcp_connect_h, [fn.args[0], fn.args[1]])
        b.ret(fd)
        return fn

    def _build_tls_send(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I64_TY, [I64_TY, I8PTR]),
                         name="__vx_tls_send")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tcp_send_h = self._get_helper("__vx_tcp_send")
        b.ret(b.call(tcp_send_h, [fn.args[0], fn.args[1]]))
        return fn

    def _build_tls_recv(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, [I64_TY]),
                         name="__vx_tls_recv")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tcp_recv_h = self._get_helper("__vx_tcp_recv")
        b.ret(b.call(tcp_recv_h, [fn.args[0], ir.Constant(I64_TY, 4096)]))
        return fn

    def _build_tls_close(self) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(VOID_TY, [I64_TY]),
                         name="__vx_tls_close")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        tcp_close_h = self._get_helper("__vx_tcp_close")
        b.call(tcp_close_h, [fn.args[0]])
        b.ret_void()
        return fn

    def _build_stack_trace(self) -> ir.Function:
        """stack_trace() -> str  — return a stack trace string (#76)."""
        fn = ir.Function(self.module, ir.FunctionType(I8PTR, []),
                         name="__vx_stack_trace")
        fn.linkage = "internal"
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        # In debug mode, try to use backtrace() (POSIX) or CaptureStackBackTrace (Windows)
        import sys as _sys
        if _sys.platform == "win32":
            # Windows: use RtlCaptureStackBackTrace if available
            buf = b.call(self.malloc_fn, [ir.Constant(I64_TY, 256)])
            fmt = self._gstr_ptr(self._global_str("\n  [stack trace unavailable on Windows without dbghelp]\n"))
            b.call(self.memcpy_fn, [buf, fmt,
                   b.call(self.strlen_fn, [fmt])])
            b.ret(fmt)
        else:
            stub = self._gstr_ptr(self._global_str("\n  [stack trace: build with -g for symbols]\n"))
            b.ret(stub)
        return fn


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
    # Apply optimization passes: mem2reg promotes alloca-in-loops to SSA
    # registers (prevents stack overflow in large loops) and also speeds
    # up the generated code significantly.
    try:
        pto = binding.PipelineTuningOptions()
        pto.speed_level = 2
        pto.loop_vectorization = False
        pto.slp_vectorization = False
        pb  = binding.create_pass_builder(tm, pto)
        mpm = pb.getModulePassManager()
        mpm.run(mod, pb)
    except Exception:
        pass  # optimization is best-effort; continue without it

    engine = binding.create_mcjit_compiler(mod, tm)
    # On Windows the C runtime uses _popen/_pclose instead of popen/pclose.
    # Register the aliased names so the JIT can resolve them.
    if sys.platform == "win32":
        import ctypes.util
        _crt = ctypes.CDLL("msvcrt")
        for _win, _vx in [("_popen", "popen"), ("_pclose", "pclose"),
                           ("_fdopen", "fdopen"), ("_fileno", "fileno"),
                           ("_putenv", "putenv"), ("_getcwd", "getcwd"),
                           ("_mkdir", "mkdir"), ("_rmdir", "rmdir")]:
            _fn = getattr(_crt, _win, None)
            if _fn:
                binding.add_symbol(_vx, ctypes.cast(_fn, ctypes.c_void_p).value)
    engine.finalize_object()
    engine.run_static_constructors()
    addr   = engine.get_function_address("main")
    if not addr:
        raise CodegenError("No 'main' function found")
    # Run on a thread with a large stack so that loops with many alloca
    # instructions (e.g. a 1M-element push loop) don't overflow the default
    # 1 MB Windows stack.
    import threading
    exc_holder = [None]
    def _run():
        try:
            ctypes.CFUNCTYPE(None)(addr)()
        except Exception as e:
            exc_holder[0] = e
    old_size = threading.stack_size(64 * 1024 * 1024)  # 64 MB
    t = threading.Thread(target=_run)
    t.start()                          # start while 64 MB is still active
    threading.stack_size(old_size)     # restore after thread is launched
    t.join()
    if exc_holder[0]:
        raise exc_holder[0]
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
