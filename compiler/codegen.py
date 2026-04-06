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

        # Interface vtable registry
        # name → {methods:[str,...], method_sigs:{name:MethodSig}, vtable_ll, fat_ll}
        self._interfaces: dict[str, dict] = {}

        # Impl vtable data: "{struct}__{iface}" → {vtable_gv, impl_fns:[fn,...]}
        self._impls_data: dict[str, dict] = {}

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
            if node.name in ("PI", "E"): return "float"
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

        # 1. Struct definitions
        for d in program.declarations:
            if isinstance(d, StructDecl): self._define_struct(d)

        # 2. Interface definitions (must be before forward-declaring fns that use them)
        for d in program.declarations:
            if isinstance(d, InterfaceDecl):
                self._define_interface(d)

        # 3. Enum definitions (global i64 constants)
        for d in program.declarations:
            if isinstance(d, EnumDecl):
                for i, variant in enumerate(d.variants):
                    gname = f"{d.name}.{variant}"
                    gv = ir.GlobalVariable(self.module, I64_TY, name=gname)
                    gv.linkage = "internal"
                    gv.global_constant = True
                    gv.initializer = ir.Constant(I64_TY, i)
                    self._globals[gname] = {"ptr": gv, "vx_type": "int"}

        # 4. Global variables
        for d in program.declarations:
            if isinstance(d, (GlobalLet, GlobalConst)):
                self._compile_global(d)

        # 5. Forward-declare all non-generic user functions
        for d in program.declarations:
            if isinstance(d, FnDecl) and not d.type_params:
                self._declare_fn(d)

        # 6. Forward-declare impl methods + create vtable globals
        for d in program.declarations:
            if isinstance(d, ImplDecl):
                self._declare_impl(d)

        # 7. Build the vtable init function (needs impl fns already declared)
        self._build_vtable_init_fn()

        # 8. Compile non-generic function bodies
        for d in program.declarations:
            if isinstance(d, FnDecl) and not d.type_params:
                self._compile_fn(d)

        # 9. Compile impl method bodies
        for d in program.declarations:
            if isinstance(d, ImplDecl):
                self._compile_impl(d)

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
        # Register default param values for this function
        self._fn_defaults[d.name] = [p.default for p in d.params]

    def _compile_fn(self, d: FnDecl):
        info = self._functions[d.name]
        fn   = info["fn"]
        self.current_fn = fn

        entry = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._push_scope()

        # Inject vtable initializer at the start of main
        if d.name == "main" and "__vx_vtable_init" in self._helper_fns:
            self.builder.call(self._helper_fns["__vx_vtable_init"], [])

        for arg, param in zip(fn.args, d.params):
            resolved = self._resolve_type(param.type_name)
            al = self.builder.alloca(self._vx_to_llvm(resolved), name=param.name)
            self.builder.store(arg, al)
            self._declare(param.name, al, resolved)

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

        if   isinstance(node, LetStmt):         self._compile_let(node)
        elif isinstance(node, TupleUnpack):      self._compile_tuple_unpack(node)
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
        elif isinstance(node, TryCatch):         self._compile_try_catch(node)
        elif isinstance(node, ForEnumerate):     self._compile_for_enumerate(node)
        elif isinstance(node, (EnumDecl, ImportStmt, TypeAlias, NamespaceHint,
                               InterfaceDecl, ImplDecl)):
            pass  # handled in compile() pass or before codegen
        else:
            raise CodegenError(f"Unknown stmt: {type(node).__name__}")

    def _compile_let(self, node: LetStmt):
        val, vt  = self._compile_expr(node.value)
        declared = self._resolve_type(node.type_annotation or vt)
        ll_ty    = self._vx_to_llvm(declared)

        if declared == "float" and vt == "int":
            val = self.builder.sitofp(val, F64_TY); vt = "float"

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
            case_body_b = fn.append_basic_block("match.case.body")
            next_b = fn.append_basic_block("match.case.next")

            loaded = self.builder.load(val_al)

            combined_cond = None
            for pat in case.patterns:
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
            if node.name == "E":
                return ir.Constant(F64_TY, 2.718281828459045), "float"
            info = self._lookup(node.name)
            if info is None:
                raise CodegenError(f"Undefined variable '{node.name}'")
            vx_t = self._resolve_type(info["vx_type"])
            return self.builder.load(info["ptr"], name=node.name), vx_t

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
            # Dict helpers
            "__vx_dict_new":             self._build_dict_new,
            "__vx_dict_set":             self._build_dict_set,
            "__vx_dict_get":             self._build_dict_get,
            "__vx_dict_has":             self._build_dict_has,
            "__vx_dict_remove":          self._build_dict_remove,
            "__vx_dict_len":             self._build_dict_len,
            "__vx_dict_keys":            self._build_dict_keys,
            # v5 helpers
            "__vx_time_format":          self._build_time_format,
            "__vx_input":                self._build_input,
            "__vx_os_list_dir":          self._build_os_list_dir,
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

        # `x in container`
        if op == "in":
            return self._compile_in(node)

        lv, lt = self._compile_expr(node.left)
        rv, rt = self._compile_expr(node.right)

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

        # --- Generic function monomorphization ---
        if name in self.analysis.generic_fns:
            return self._compile_generic_call(node)

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
        """Compile a lambda expression. Returns (fn_ptr as i8*, fn_type_str)."""
        lname = f"__lambda_{self._lambda_count}"
        self._lambda_count += 1

        param_vx_types = [p.type_name for p in node.params]
        ret_vx = node.ret_type or "void"
        fn_type_str = f"fn({','.join(param_vx_types)})->{ret_vx}"

        # Build a concrete FnDecl and compile it
        import copy as _copy
        decl = FnDecl(lname, node.params, node.ret_type, node.body)
        from compiler.analyzer import FnSig as _FnSig
        self.analysis.fn_sigs[lname] = _FnSig(
            [(p.name, p.type_name) for p in node.params], ret_vx
        )
        self._fn_defaults[lname] = [None] * len(node.params)

        # Compile in saved context
        saved_builder  = self.builder
        saved_fn       = self.current_fn
        saved_scopes   = self._scope_stack
        self._scope_stack = [{}]
        self._declare_fn(decl)
        self._compile_fn(decl)
        self.builder      = saved_builder
        self.current_fn   = saved_fn
        self._scope_stack = saved_scopes

        # Return pointer to the function cast to i8*
        fn_ir = self._functions[lname]["fn"]
        fn_ptr = self.builder.bitcast(fn_ir, I8PTR)
        return fn_ptr, fn_type_str

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
