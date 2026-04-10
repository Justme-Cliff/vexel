"""
Vexel semantic analyzer.

Two-pass analysis over the AST:
  Pass 1 — register all top-level declarations (functions, structs,
            globals, enums, interfaces, impls).
  Pass 2 — walk function bodies, check for undefined names, and infer
            expression types.

Errors accumulate in ``AnalysisResult.errors``; the compiler does not
raise on the first problem so multiple errors can be reported at once.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from compiler.ast_nodes import *


VX_INT   = "int"
VX_FLOAT = "float"
VX_BOOL  = "bool"
VX_STR   = "str"
VX_VOID  = "void"
VX_NULL  = "null"

# Built-in functions that the codegen handles without a FnDecl.
BUILTINS: dict[str, tuple[list[str], str]] = {
    # name → ([param_types...], return_type)
    "len":          (["any"],                     VX_INT),
    "sqrt":         (["float"],                   VX_FLOAT),
    "abs":          (["any"],                     VX_INT),     # overloaded
    "min":          (["any", "any"],              VX_INT),     # overloaded
    "max":          (["any", "any"],              VX_INT),     # overloaded
    "pow":          (["float", "float"],          VX_FLOAT),
    "floor":        (["float"],                   VX_FLOAT),
    "ceil":         (["float"],                   VX_FLOAT),
    "int":          (["any"],                     VX_INT),
    "float":        (["any"],                     VX_FLOAT),
    "str":          (["any"],                     VX_STR),
    "bool":         (["any"],                     VX_BOOL),
    # v3 additions
    "exit":         (["int"],                     VX_VOID),
    "read_file":    (["str"],                     VX_STR),
    "write_file":   (["str", "str"],              VX_VOID),
    "append_file":  (["str", "str"],              VX_VOID),
    "file_exists":  (["str"],                     VX_BOOL),
    "rand":         ([],                          VX_FLOAT),
    "rand_int":     (["int", "int"],              VX_INT),
    "sin":          (["float"],                   VX_FLOAT),
    "cos":          (["float"],                   VX_FLOAT),
    "tan":          (["float"],                   VX_FLOAT),
    "log":          (["float"],                   VX_FLOAT),
    "log2":         (["float"],                   VX_FLOAT),
    # v4 additions
    "round":        (["float"],                   VX_INT),
    "clamp":        (["any", "any", "any"],       VX_INT),     # overloaded
    "lerp":         (["float", "float", "float"], VX_FLOAT),
    "atan2":        (["float", "float"],          VX_FLOAT),
    "throw":        (["str"],                     VX_VOID),
    "os_cwd":       ([],                          VX_STR),
    "os_mkdir":     (["str"],                     VX_BOOL),
    "os_delete":    (["str"],                     VX_BOOL),
    # v5 additions
    "os_list_dir":  (["str"],                     "str[]"),
    "parse_int":    (["str"],                     VX_INT),
    "parse_float":  (["str"],                     VX_FLOAT),
    "time_now":     ([],                          VX_INT),
    "time_format":  (["int"],                     VX_STR),
    "input":        (["str"],                     VX_STR),
    # v7 additions — system
    "argv":         ([],                          "str[]"),
    "env_get":      (["str"],                     VX_STR),
    "env_set":      (["str", "str"],              VX_VOID),
    "env_unset":    (["str"],                     VX_VOID),
    "shell":        (["str"],                     VX_STR),
    "run":          (["str", "str[]"],            VX_INT),
    # v7 — string extras
    "str_find":         (["str", "str"],              VX_INT),
    "str_slice":        (["str", "int", "int"],       VX_STR),
    "str_repeat":       (["str", "int"],              VX_STR),
    "str_join":         (["str[]", "str"],            VX_STR),
    "str_char_at":      (["str", "int"],              VX_INT),
    "str_len_utf8":     (["str"],                     VX_INT),
    "char_to_int":      (["str"],                     VX_INT),
    "int_to_char":      (["int"],                     VX_STR),
    # B1/B8 fix — these were implemented in codegen but not registered here
    "str_split":        (["str", "str"],              "str[]"),
    "str_trim":         (["str"],                     VX_STR),
    "str_contains":     (["str", "str"],              VX_BOOL),
    "char_at":          (["str", "int"],              VX_STR),
    "str_upper":        (["str"],                     VX_STR),
    "str_lower":        (["str"],                     VX_STR),
    "str_starts_with":  (["str", "str"],              VX_BOOL),
    "str_ends_with":    (["str", "str"],              VX_BOOL),
    "str_replace":      (["str", "str", "str"],       VX_STR),
    # v8 — new stdlib
    "rand_seed":        (["int"],                     VX_VOID),
    "csv_write":        (["str[][]", "str"],          VX_STR),
    "popcount":         (["int"],                     VX_INT),
    "clz":              (["int"],                     VX_INT),
    "ctz":              (["int"],                     VX_INT),
    "bit_reverse":      (["int"],                     VX_INT),
    "log_info":         (["str"],                     VX_VOID),
    "log_warn":         (["str"],                     VX_VOID),
    "log_error":        (["str"],                     VX_VOID),
    "log_debug":        (["str"],                     VX_VOID),
    "print_color":      (["str", "str"],              VX_VOID),
    "print_bold":       (["str"],                     VX_VOID),
    "dict_values":      (["any"],                     "any[]"),
    "dict_items":       (["any"],                     "str[]"),
    # v7 — array extras
    "array_sort":   (["any[]"],                   VX_VOID),
    "array_sort_by":  (["any[]", "any"],          VX_VOID),
    "array_map":    (["any[]", "any"],            "any[]"),
    "array_filter": (["any[]", "any"],            "any[]"),
    "array_reduce": (["any[]", "any", "any"],     "any"),
    "array_index_of": (["any[]", "any"],          VX_INT),
    "array_join":   (["str[]", "str"],            VX_STR),
    # v7 — math
    "pi":           ([],                          VX_FLOAT),
    "tau":          ([],                          VX_FLOAT),
    "inf":          ([],                          VX_FLOAT),
    "nan":          ([],                          VX_FLOAT),
    "is_nan":       (["float"],                   VX_BOOL),
    "is_inf":       (["float"],                   VX_BOOL),
    "hypot":        (["float", "float"],          VX_FLOAT),
    "log10":        (["float"],                   VX_FLOAT),
    "exp":          (["float"],                   VX_FLOAT),
    "trunc":        (["float"],                   VX_FLOAT),
    "sign":         (["float"],                   VX_INT),
    # v7 — base64 / uuid / hash
    "base64_encode": (["str"],                    VX_STR),
    "base64_decode": (["str"],                    VX_STR),
    "uuid_v4":      ([],                          VX_STR),
    "sha256":       (["str"],                     VX_STR),
    "md5":          (["str"],                     VX_STR),
    # v7 — regex
    "regex_match":  (["str", "str"],              VX_STR),
    "regex_test":   (["str", "str"],              VX_BOOL),
    "regex_find_all": (["str", "str"],            "str[]"),
    "regex_replace": (["str", "str", "str"],      VX_STR),
    # v7 — csv
    "csv_parse":    (["str"],                     "str[][]"),
    "csv_stringify":  (["str[][]"],               VX_STR),
    # v7 — threading (basic)
    "thread_spawn": (["any"],                     VX_INT),
    "thread_join":  (["int"],                     VX_VOID),
    "mutex_new":    ([],                          VX_INT),
    "mutex_lock":   (["int"],                     VX_VOID),
    "mutex_unlock": (["int"],                     VX_VOID),
    "mutex_try_lock": (["int"],                   VX_BOOL),
    # v7 — atomics
    "atomic_new":   (["int"],                     VX_INT),
    "atomic_load":  (["int"],                     VX_INT),
    "atomic_store": (["int", "int"],              VX_VOID),
    "atomic_add":   (["int", "int"],              VX_INT),
    "atomic_sub":   (["int", "int"],              VX_INT),
    "atomic_cas":   (["int", "int", "int"],       VX_BOOL),
    # v7 — file I/O expansion
    "file_open":    (["str", "str"],              VX_INT),
    "file_close":   (["int"],                     VX_VOID),
    "file_read_bytes": (["int", "int"],           VX_STR),
    "file_write_bytes": (["int", "str"],          VX_INT),
    "file_seek":    (["int", "int"],              VX_VOID),
    "file_tell":    (["int"],                     VX_INT),
    "file_size":    (["str"],                     VX_INT),
    # v7 — test helpers
    "assert_eq":    (["any", "any"],              VX_VOID),
    "assert_neq":   (["any", "any"],              VX_VOID),
    "assert_true":  (["bool"],                    VX_VOID),
    "assert_false": (["bool"],                    VX_VOID),
    # v7 — sqlite
    "db_open":      (["str"],                     VX_INT),
    "db_exec":      (["int", "str"],              VX_VOID),
    "db_query":     (["int", "str"],              "str[][]"),
    "db_close":     (["int"],                     VX_VOID),
    # v8 — date/time
    "datetime_now":      ([],                     VX_STR),
    "datetime_format":   (["str"],                VX_STR),
    "datetime_timestamp":([],                     VX_INT),
    "datetime_from_ts":  (["int"],                VX_STR),
    "sleep_ms":          (["int"],                VX_VOID),
    # v8 — signal handling
    "signal_handle":     (["int", "any"],         VX_VOID),
    "SIGINT":            ([],                     VX_INT),
    "SIGTERM":           ([],                     VX_INT),
    # v8 — process spawning
    "process_spawn":     (["str", "str[]"],       VX_INT),
    "process_wait":      (["int"],                VX_INT),
    "process_kill":      (["int"],                VX_VOID),
    # v8 — progress bars / terminal UI
    "progress_new":      (["int"],                VX_INT),
    "progress_update":   (["int", "int"],         VX_VOID),
    "progress_finish":   (["int", "str"],         VX_VOID),
    "term_clear":        ([],                     VX_VOID),
    "term_move":         (["int", "int"],         VX_VOID),
    "term_width":        ([],                     VX_INT),
    # v8 — channels (thread messaging)
    "channel_new":       ([],                     VX_INT),
    "channel_send":      (["int", "int"],         VX_VOID),
    "channel_recv":      (["int"],                VX_INT),
    "channel_try_recv":  (["int"],                VX_INT),
    "channel_close":     (["int"],                VX_VOID),
    # v8 — rwlocks
    "rwlock_new":        ([],                     VX_INT),
    "rwlock_read_lock":  (["int"],                VX_VOID),
    "rwlock_read_unlock":(["int"],                VX_VOID),
    "rwlock_write_lock": (["int"],                VX_VOID),
    "rwlock_write_unlock":(["int"],               VX_VOID),
    # v8 — thread pool
    "thread_pool_new":   (["int"],                VX_INT),
    "thread_pool_submit":(["int", "any"],         VX_VOID),
    "thread_pool_wait":  (["int"],                VX_VOID),
    "thread_pool_destroy":(["int"],               VX_VOID),
    # v8 — benchmarking
    "benchmark":         (["any", "int"],         VX_STR),
    "bench_ns":          (["any"],                VX_INT),
    # v8 — JSON parsing (basic)
    "json_parse_int":    (["str", "str"],         VX_INT),
    "json_parse_str":    (["str", "str"],         VX_STR),
    "json_parse_float":  (["str", "str"],         VX_FLOAT),
    "json_parse_bool":   (["str", "str"],         VX_BOOL),
    # v8 — .env config
    "env_load":          (["str"],                VX_VOID),
    # v8 — set operations
    "set_new":           ([],                     VX_INT),
    "set_add":           (["int", "int"],         VX_VOID),
    "set_contains":      (["int", "int"],         VX_BOOL),
    "set_remove":        (["int", "int"],         VX_VOID),
    "set_size":          (["int"],                VX_INT),
    "set_to_array":      (["int"],                "int[]"),
    "set_union":         (["int", "int"],         VX_INT),
    "set_intersect":     (["int", "int"],         VX_INT),
    # v8 — UUID extras
    "uuid_v1":           ([],                     VX_STR),
    # v8 — bounds checking helpers (debug mode)
    "bounds_check":      (["int", "int"],         VX_VOID),
    # v8 — condition variables
    "condvar_new":       ([],                     VX_INT),
    "condvar_wait":      (["int", "int"],         VX_VOID),
    "condvar_signal":    (["int"],                VX_VOID),
    "condvar_broadcast": (["int"],                VX_VOID),
    # v9 — TCP/UDP sockets (#54)
    "tcp_connect":       (["str", "int"],         VX_INT),
    "tcp_send":          (["int", "str"],         VX_INT),
    "tcp_recv":          (["int", "int"],         VX_STR),
    "tcp_close":         (["int"],                VX_VOID),
    "tcp_listen":        (["int"],                VX_INT),
    "tcp_accept":        (["int"],                VX_INT),
    "udp_socket":        ([],                     VX_INT),
    "udp_send_to":       (["int", "str", "int", "str"], VX_INT),
    "udp_recv_from":     (["int", "int"],         VX_STR),
    # v9 — file watching (#58)
    "file_watch":        (["str", "any"],         VX_VOID),
    # v9 — named pipes / IPC (#64)
    "pipe_open":         (["str"],                VX_INT),
    "pipe_write":        (["int", "str"],         VX_INT),
    "pipe_read":         (["int", "int"],         VX_STR),
    "pipe_close":        (["int"],                VX_VOID),
    # v9 — SHA-256 (real implementation)
    "sha256_real":       (["str"],                VX_STR),
    # v9 — select/multiplex (#70)
    "chan_select":       (["int[]"],              VX_INT),
}


@dataclass
class FnSig:
    params:      list[tuple[str, str]]
    return_type: str
    variadic:    bool = False          # last param is variadic
    type_params: list[str] = field(default_factory=list)  # generic type params


@dataclass
class AnalysisResult:
    errors:        list[str]
    fn_sigs:       dict[str, FnSig]
    struct_fields: dict[str, list[tuple[str, str]]]
    global_types:  dict[str, str]
    type_aliases:  dict[str, str]      = field(default_factory=dict)
    namespaces:    set[str]            = field(default_factory=set)
    generic_fns:   dict[str, FnDecl]  = field(default_factory=dict)
    interfaces:    dict[str, InterfaceDecl] = field(default_factory=dict)
    impls:         dict[tuple, ImplDecl]    = field(default_factory=dict)


class Analyzer:
    def __init__(self):
        self._errors:        list[str]                        = []
        self._fn_sigs:       dict[str, FnSig]                 = {}
        self._struct_fields: dict[str, list[tuple[str, str]]] = {}
        self._global_types:  dict[str, str]                   = {}
        self._type_aliases:  dict[str, str]                   = {}
        self._namespaces:    set[str]                         = set()
        self._generic_fns:   dict[str, "FnDecl"]              = {}
        self._interfaces:    dict[str, "InterfaceDecl"]       = {}
        self._impls:         dict[tuple, "ImplDecl"]          = {}
        self._scopes:        list[dict[str, str]]             = [{}]
        self._loop_depth:    int                              = 0
        self._lambda_count:  int                              = 0

    # ------------------------------------------------------------------ #

    def analyze(self, program: Program) -> AnalysisResult:
        # Inject built-ins
        for name, (params, ret) in BUILTINS.items():
            self._fn_sigs[name] = FnSig([(f"a{i}", t) for i, t in enumerate(params)], ret)

        # Pass 1 — register all top-level declarations
        for decl in program.declarations:
            if isinstance(decl, FnDecl):
                if decl.type_params:
                    self._generic_fns[decl.name] = decl
                    # Register with placeholder sig so calls don't error
                    self._fn_sigs[decl.name] = FnSig(
                        [(p.name, p.type_name) for p in decl.params],
                        decl.return_type or VX_VOID,
                        variadic=any(p.variadic for p in decl.params),
                        type_params=decl.type_params,
                    )
                else:
                    self._register_fn(decl)
            elif isinstance(decl, StructDecl):
                self._register_struct(decl)
                # Register methods defined inside the struct body
                for m in getattr(decl, 'methods', []):
                    mn = f"{decl.name}__{m.name}"
                    self._fn_sigs[mn] = FnSig(
                        [(p.name, p.type_name) for p in m.params],
                        m.return_type or VX_VOID,
                    )
            elif isinstance(decl, (GlobalLet, GlobalConst)):
                self._register_global(decl)
            elif isinstance(decl, EnumDecl):
                for i, variant in enumerate(decl.variants):
                    key = f"{decl.name}.{variant}"
                    self._global_types[key] = VX_INT
                    self._scopes[0][key]    = VX_INT
            elif isinstance(decl, TypeAlias):
                self._type_aliases[decl.name] = decl.target
            elif isinstance(decl, NamespaceHint):
                self._namespaces.add(decl.alias)
            elif isinstance(decl, InterfaceDecl):
                self._interfaces[decl.name] = decl
            elif isinstance(decl, ImplDecl):
                key = (decl.struct_name, decl.interface_name)
                self._impls[key] = decl
                # Register each impl method so calls inside bodies resolve
                for m in decl.methods:
                    fn_key = f"{decl.struct_name}__{m.name}__impl_{decl.interface_name}"
                    params = [(p.name, p.type_name) for p in m.params]
                    self._fn_sigs[fn_key] = FnSig(params, m.return_type or VX_VOID)
            elif isinstance(decl, (PubDecl, PrivDecl)):
                # Recurse into visibility wrappers
                inner = decl.inner
                if isinstance(inner, FnDecl):
                    if inner.type_params:
                        self._generic_fns[inner.name] = inner
                        self._fn_sigs[inner.name] = FnSig(
                            [(p.name, p.type_name) for p in inner.params],
                            inner.return_type or VX_VOID,
                        )
                    else:
                        self._register_fn(inner)
                elif isinstance(inner, StructDecl):
                    self._register_struct(inner)
                    # Register struct methods
                    for m in getattr(inner, 'methods', []):
                        mn = f"{inner.name}__{m.name}"
                        self._fn_sigs[mn] = FnSig(
                            [(p.name, p.type_name) for p in m.params],
                            m.return_type or VX_VOID,
                        )
            elif isinstance(decl, ExternFnDecl):
                params = [(p.name, p.type_name) for p in decl.params]
                self._fn_sigs[decl.name] = FnSig(params, decl.return_type or VX_VOID)
            elif isinstance(decl, TestDecl):
                test_fn_name = f"__test__{decl.name.replace(' ', '_')}"
                self._fn_sigs[test_fn_name] = FnSig([], VX_VOID)
            elif isinstance(decl, ComptimeDecl):
                # Treat as a global constant
                t = self._infer_expr(decl.value)
                self._global_types[decl.name] = t
                self._scopes[0][decl.name] = t
            elif isinstance(decl, EnumDeclADT):
                # Register ADT variants as struct-like types
                for variant in decl.variants:
                    vkey = f"{decl.name}.{variant.name}"
                    self._global_types[vkey] = decl.name
                    self._scopes[0][vkey] = decl.name
                    if variant.fields:
                        self._struct_fields[variant.name] = [(f.name, f.type_name) for f in variant.fields]

        # Pass 2 — analyze function bodies
        for decl in program.declarations:
            if isinstance(decl, FnDecl) and not decl.type_params:
                self._analyze_fn(decl)
            elif isinstance(decl, ImplDecl):
                for m in decl.methods:
                    self._analyze_fn(m)

        return AnalysisResult(
            self._errors, self._fn_sigs,
            self._struct_fields, self._global_types,
            self._type_aliases, self._namespaces,
            self._generic_fns,
            self._interfaces, self._impls,
        )

    # ------------------------------------------------------------------ #
    #  Registration                                                        #
    # ------------------------------------------------------------------ #

    def _register_fn(self, d: FnDecl):
        params   = [(p.name, p.type_name) for p in d.params]
        variadic = any(p.variadic for p in d.params)
        self._fn_sigs[d.name] = FnSig(params, d.return_type or VX_VOID,
                                      variadic=variadic)

    def _register_struct(self, d: StructDecl):
        self._struct_fields[d.name] = [(f.name, f.type_name) for f in d.fields]

    def _register_global(self, d):
        t = d.type_annotation or self._infer_expr(d.value)
        self._global_types[d.name] = t
        self._scopes[0][d.name]    = t

    # ------------------------------------------------------------------ #
    #  Scopes                                                              #
    # ------------------------------------------------------------------ #

    def _push(self): self._scopes.append({})
    def _pop(self):  self._scopes.pop()

    def _declare(self, name: str, t: str):
        self._scopes[-1][name] = t

    def _lookup(self, name: str) -> Optional[str]:
        for s in reversed(self._scopes):
            if name in s: return s[name]
        return None

    def _err(self, msg: str, node=None):
        line = getattr(node, 'line', 0) if node else 0
        prefix = f"Line {line}: " if line else ""
        self._errors.append(prefix + msg)

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(a) > len(b): a, b = b, a
        row = list(range(len(a) + 1))
        for j, cb in enumerate(b):
            new_row = [j + 1]
            for i, ca in enumerate(a):
                new_row.append(min(row[i+1]+1, new_row[i]+1,
                                   row[i] + (0 if ca == cb else 1)))
            row = new_row
        return row[-1]

    def _suggest(self, name: str, candidates: list[str], max_dist: int = 2) -> str | None:
        """Return closest candidate within max_dist edits, or None."""
        best, best_d = None, max_dist + 1
        for c in candidates:
            d = self._edit_distance(name, c)
            if d < best_d:
                best, best_d = c, d
        return best if best_d <= max_dist else None

    # ------------------------------------------------------------------ #
    #  Functions                                                           #
    # ------------------------------------------------------------------ #

    def _analyze_fn(self, d: FnDecl):
        self._push()
        for p in d.params:
            self._declare(p.name, p.type_name)
        for stmt in d.body:
            self._analyze_stmt(stmt, d.return_type or VX_VOID)
        self._pop()

    # ------------------------------------------------------------------ #
    #  Statements                                                          #
    # ------------------------------------------------------------------ #

    def _analyze_stmt(self, node: Node, ret_type: str):
        if isinstance(node, LetStmt):
            t = self._infer_expr(node.value)
            self._declare(node.name, node.type_annotation or t)

        elif isinstance(node, TupleUnpack):
            t = self._infer_expr(node.value)
            # t is like "(int,float)" — extract element types
            elem_types = self._parse_tuple_type(t)
            for i, name in enumerate(node.names):
                ann = node.annotations[i] if i < len(node.annotations) else None
                et  = ann or (elem_types[i] if i < len(elem_types) else VX_INT)
                self._declare(name, et)

        elif isinstance(node, AssignStmt):
            if isinstance(node.target, Identifier):
                if self._lookup(node.target.name) is None:
                    self._err(f"Undefined variable '{node.target.name}'")
            self._infer_expr(node.value)

        elif isinstance(node, IndexAssignStmt):
            self._infer_expr(node.obj)
            self._infer_expr(node.index)
            self._infer_expr(node.value)

        elif isinstance(node, ReturnStmt):
            if node.value: self._infer_expr(node.value)

        elif isinstance(node, PrintStmt):
            for v in node.values: self._infer_expr(v)

        elif isinstance(node, IfStmt):
            self._infer_expr(node.condition)
            self._push()
            for s in node.then_body: self._analyze_stmt(s, ret_type)
            self._pop()
            if node.else_body:
                self._push()
                for s in node.else_body: self._analyze_stmt(s, ret_type)
                self._pop()

        elif isinstance(node, ForStmt):
            self._infer_expr(node.start); self._infer_expr(node.end)
            self._push(); self._loop_depth += 1
            self._declare(node.var, VX_INT)
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, ForEach):
            arr_type = self._infer_expr(node.iterable)
            elem_type = arr_type[:-2] if arr_type.endswith("[]") else VX_INT
            self._push(); self._loop_depth += 1
            self._declare(node.var, elem_type)
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, WhileStmt):
            self._infer_expr(node.condition)
            self._push(); self._loop_depth += 1
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, BreakStmt):
            if self._loop_depth == 0:
                self._err("'break' outside loop")

        elif isinstance(node, ContinueStmt):
            if self._loop_depth == 0:
                self._err("'continue' outside loop")

        elif isinstance(node, ExprStmt):
            self._infer_expr(node.expr)

        elif isinstance(node, MatchStmt):
            vt = self._infer_expr(node.value)
            for case in node.cases:
                has_tp = any(isinstance(p, TypePattern) for p in case.patterns)
                if has_tp:
                    # TypePattern: push scope, bind fields, analyze body
                    self._push()
                    for pat in case.patterns:
                        if isinstance(pat, TypePattern):
                            fields = self._struct_fields.get(pat.type_name, [])
                            for i, bind in enumerate(pat.bindings):
                                ft = fields[i][1] if i < len(fields) else VX_INT
                                self._declare(bind, ft)
                    for s in case.body:
                        self._analyze_stmt(s, ret_type)
                    self._pop()
                else:
                    for pat in case.patterns:
                        self._infer_expr(pat)
                    self._push()
                    for s in case.body:
                        self._analyze_stmt(s, ret_type)
                    self._pop()
            if node.default_body:
                self._push()
                for s in node.default_body:
                    self._analyze_stmt(s, ret_type)
                self._pop()

        elif isinstance(node, AssertStmt):
            self._infer_expr(node.condition)
            if node.message:
                self._infer_expr(node.message)

        elif isinstance(node, TryCatch):
            self._push()
            for s in node.try_body:
                self._analyze_stmt(s, ret_type)
            self._pop()
            self._push()
            self._declare(node.catch_var, VX_STR)
            for s in node.catch_body:
                self._analyze_stmt(s, ret_type)
            self._pop()

        elif isinstance(node, ForEnumerate):
            arr_type  = self._infer_expr(node.iterable)
            elem_type = arr_type[:-2] if arr_type.endswith("[]") else VX_STR
            self._push(); self._loop_depth += 1
            self._declare(node.idx_var, VX_INT)
            self._declare(node.val_var, elem_type)
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()

        elif isinstance(node, DoWhileStmt):
            self._push(); self._loop_depth += 1
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._loop_depth -= 1; self._pop()
            self._infer_expr(node.condition)

        elif isinstance(node, LabeledStmt):
            self._analyze_stmt(node.stmt, ret_type)

        elif isinstance(node, (BreakLabel, ContinueLabel)):
            pass  # label validation is optional for now

        elif isinstance(node, DeferStmt):
            self._infer_expr(node.expr)

        elif isinstance(node, YieldStmt):
            if node.value: self._infer_expr(node.value)

        elif isinstance(node, ThrowStmt):
            self._infer_expr(node.value)

        elif isinstance(node, RaiseStmt):
            self._infer_expr(node.value)

        elif isinstance(node, UnsafeBlock):
            self._push()
            for s in node.body: self._analyze_stmt(s, ret_type)
            self._pop()

        elif isinstance(node, TryCatchFinally):
            self._push()
            for s in node.try_body: self._analyze_stmt(s, ret_type)
            self._pop()
            for clause in node.catches:
                self._push()
                self._declare(clause.var, VX_STR)
                for s in clause.body: self._analyze_stmt(s, ret_type)
                self._pop()
            if node.finally_body:
                self._push()
                for s in node.finally_body: self._analyze_stmt(s, ret_type)
                self._pop()

        elif isinstance(node, StructDestructure):
            t = self._infer_expr(node.value)
            fields = self._struct_fields.get(t, [])
            field_map = {fn: ft for fn, ft in fields}
            for i, fname in enumerate(node.fields):
                alias = node.aliases[i] if i < len(node.aliases) else None
                bind_name = alias if alias else fname
                ft = field_map.get(fname, VX_INT)
                self._declare(bind_name, ft)

        elif isinstance(node, ArrayDestructure):
            arr_t = self._infer_expr(node.value)
            elem_t = arr_t[:-2] if arr_t.endswith("[]") else VX_INT
            for name in node.names:
                self._declare(name, elem_t)
            if node.rest_name:
                self._declare(node.rest_name, arr_t)

        elif isinstance(node, MatchCaseGuard):
            # handled inside MatchStmt analysis — shouldn't arrive here alone
            pass

        elif isinstance(node, (TestDecl, ExternFnDecl, ComptimeDecl,
                               EnumDeclADT, PubDecl, PrivDecl)):
            pass  # handled in pass 1 or structure pass

        elif isinstance(node, (EnumDecl, ImportStmt, TypeAlias, NamespaceHint,
                               InterfaceDecl, ImplDecl)):
            pass  # handled in pass 1 or before analysis

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _parse_tuple_type(self, t: str) -> list[str]:
        """Extract element types from tuple type string like '(int,float)'."""
        if t.startswith("(") and t.endswith(")"):
            inner = t[1:-1]
            return self._split_types(inner)
        return []

    def _split_types(self, s: str) -> list[str]:
        """Split type list respecting nesting."""
        parts, depth, cur = [], 0, []
        for c in s:
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

    # ------------------------------------------------------------------ #
    #  Expression type inference                                           #
    # ------------------------------------------------------------------ #

    def _infer_expr(self, node: Node) -> str:
        from compiler.ast_nodes import RangeExpr as _RangeExpr, PipeExpr as _PipeExpr
        if isinstance(node, _RangeExpr):        return "int[]"   # range produces ints
        if isinstance(node, _PipeExpr):         return self._infer_expr(node.func)
        if isinstance(node, TypePattern):       return VX_BOOL
        if isinstance(node, IntLiteral):        return VX_INT
        if isinstance(node, CharLiteral):       return VX_INT   # char is an int code point
        if isinstance(node, NullCoalesceExpr):
            lt = self._infer_expr(node.left)
            return lt[:-1] if lt.endswith("?") else lt
        if isinstance(node, OptionalChainExpr):
            # Return the field type with ? appended
            ot = self._infer_expr(node.obj)
            base = ot[:-1] if ot.endswith("?") else ot
            fields = self._struct_fields.get(base, [])
            for fn, ft in fields:
                if fn == node.field: return ft + "?"
            return VX_INT
        if isinstance(node, SliceExpr):
            ot = self._infer_expr(node.obj)
            return ot   # slice of str→str, arr→arr
        if isinstance(node, ListComp):
            # B3 fix: push scope and declare loop variable before inferring body
            self._push()
            from compiler.ast_nodes import RangeExpr as _RangeExpr
            if isinstance(node.iterable, _RangeExpr):
                elem_t = VX_INT
            else:
                it_t = self._infer_expr(node.iterable)
                elem_t = it_t[:-2] if it_t.endswith("[]") else VX_INT
            self._declare(node.var, elem_t)
            et = self._infer_expr(node.expr)
            self._pop()
            return et + "[]"
        if isinstance(node, AwaitExpr):
            return self._infer_expr(node.expr)
        if isinstance(node, NamedArg):
            return self._infer_expr(node.value)
        if isinstance(node, IntLiteral):    return VX_INT
        if isinstance(node, FloatLiteral):  return VX_FLOAT
        if isinstance(node, BoolLiteral):   return VX_BOOL
        if isinstance(node, StringLiteral): return VX_STR
        if isinstance(node, NullLiteral):   return VX_NULL

        if isinstance(node, ArrayLiteral):
            if not node.elements: return "int[]"
            et = self._infer_expr(node.elements[0])
            return f"{et}[]"

        if isinstance(node, DictLiteral):
            if not node.pairs: return "dict[str,int]"
            kt = self._infer_expr(node.pairs[0][0])
            vt = self._infer_expr(node.pairs[0][1])
            return f"dict[{kt},{vt}]"

        if isinstance(node, TupleLiteral):
            types = [self._infer_expr(e) for e in node.elements]
            return "(" + ",".join(types) + ")"

        if isinstance(node, LambdaExpr):
            param_types = ",".join(p.type_name for p in node.params)
            ret = node.ret_type or VX_VOID
            # Register the lambda as an anonymous function
            lname = f"__lambda_{self._lambda_count}"
            self._lambda_count += 1
            self._fn_sigs[lname] = FnSig(
                [(p.name, p.type_name) for p in node.params], ret
            )
            # Analyze the body
            self._push()
            for p in node.params:
                self._declare(p.name, p.type_name)
            for s in node.body:
                self._analyze_stmt(s, ret)
            self._pop()
            return f"fn({param_types})->{ret}"

        if isinstance(node, Identifier):
            # Built-in constants
            if node.name in ("PI", "TAU", "E", "INF", "NAN"):
                return VX_FLOAT
            # Namespace name used as value? Return special marker
            if node.name in self._namespaces:
                return f"__namespace__{node.name}"
            t = self._lookup(node.name)
            if t is None:
                # Collect all visible names for suggestion
                all_names: list[str] = []
                for scope in self._scopes:
                    all_names.extend(scope.keys())
                all_names.extend(self._fn_sigs.keys())
                suggestion = self._suggest(node.name, all_names)
                hint = f" — did you mean '{suggestion}'?" if suggestion else ""
                self._err(f"Undefined variable '{node.name}'{hint}", node)
                return VX_INT
            return t

        if isinstance(node, BinOp):
            lt = self._infer_expr(node.left)
            rt = self._infer_expr(node.right)
            if node.op in ("==","!=","<",">","<=",">=","and","or","in"):
                return VX_BOOL
            # str + str → str
            if node.op == "+" and lt == VX_STR: return VX_STR
            if lt == VX_FLOAT or rt == VX_FLOAT: return VX_FLOAT
            return lt

        if isinstance(node, UnaryOp):
            t = self._infer_expr(node.operand)
            return VX_BOOL if node.op == "not" else t

        if isinstance(node, Call):
            sig = self._fn_sigs.get(node.func)
            if sig is None:
                # Check if node.func is a variable with a function type
                var_t = self._lookup(node.func)
                if var_t and var_t.startswith("fn("):
                    # Extract return type from "fn(int,float)->str"
                    arrow_idx = var_t.rindex("->")
                    for a in node.args: self._infer_expr(a)
                    return var_t[arrow_idx+2:]
                all_fns = list(self._fn_sigs.keys())
                suggestion = self._suggest(node.func, all_fns)
                hint = f" — did you mean '{suggestion}'?" if suggestion else ""
                self._err(f"Undefined function '{node.func}'{hint}", node)
                return VX_VOID
            # For overloaded builtins, infer return type from args
            if node.func in ("abs", "min", "max", "clamp"):
                if node.args:
                    at = self._infer_expr(node.args[0])
                    return VX_FLOAT if at == VX_FLOAT else VX_INT
            # For generic functions, substitute T
            if sig.type_params and node.args:
                at = self._infer_expr(node.args[0])
                # Infer return type by substituting type param
                ret = sig.return_type
                if sig.type_params:
                    tp = sig.type_params[0]
                    # Infer T from first arg
                    inferred_t = at[:-2] if at.endswith("[]") else at
                    ret = ret.replace(tp, inferred_t) if tp in ret else ret
                return ret
            for a in node.args: self._infer_expr(a)
            return sig.return_type

        if isinstance(node, MethodCall):
            obj_t = self._infer_expr(node.obj)
            for a in node.args: self._infer_expr(a)

            # Interface method dispatch
            if obj_t in self._interfaces:
                iface = self._interfaces[obj_t]
                for m in iface.methods:
                    if m.name == node.method:
                        return m.return_type or VX_VOID
                self._err(f"Interface '{obj_t}' has no method '{node.method}'", node)
                return VX_VOID

            # Namespace call: ns.func(args)
            if obj_t.startswith("__namespace__"):
                ns = obj_t[len("__namespace__"):]
                fn_name = f"{ns}__{node.method}"
                sig = self._fn_sigs.get(fn_name)
                if sig:
                    return sig.return_type
                self._err(f"Namespace '{ns}' has no function '{node.method}'")
                return VX_VOID

            if obj_t.startswith("dict["):
                return {"has": VX_BOOL, "remove": VX_VOID,
                        "len": VX_INT, "keys": "str[]"}.get(node.method, VX_VOID)
            if obj_t == VX_STR:
                str_methods = {
                    "len":         VX_INT,
                    "upper":       VX_STR,
                    "lower":       VX_STR,
                    "trim":        VX_STR,
                    "contains":    VX_BOOL,
                    "starts_with": VX_BOOL,
                    "ends_with":   VX_BOOL,
                    "replace":     VX_STR,
                    "split":       "str[]",
                }
                return str_methods.get(node.method, VX_STR)
            if obj_t.endswith("[]"):
                elem_t = obj_t[:-2]
                arr_methods = {
                    "len":      VX_INT,
                    "push":     VX_VOID,
                    "pop":      elem_t,
                    "contains": VX_BOOL,
                    "reverse":  VX_VOID,
                }
                return arr_methods.get(node.method, VX_VOID)
            # Nullable type: strip ? and recurse
            if obj_t.endswith("?"):
                inner_t = obj_t[:-1]
                # Try struct methods
                for fn_name, ft in self._struct_fields.get(inner_t, []):
                    if fn_name == node.method: return ft
            return VX_VOID

        if isinstance(node, TernaryExpr):
            self._infer_expr(node.condition)
            then_t = self._infer_expr(node.then_val)
            self._infer_expr(node.else_val)
            return then_t

        if isinstance(node, FieldAccess):
            # Check if this is an enum access: Color.Red -> look up "Color.Red"
            if isinstance(node.obj, Identifier):
                dotted = f"{node.obj.name}.{node.field}"
                t = self._lookup(dotted)
                if t is not None:
                    return t
            ot = self._infer_expr(node.obj)
            for fn, ft in self._struct_fields.get(ot, []):
                if fn == node.field: return ft
            # Nullable struct field
            if ot.endswith("?"):
                for fn, ft in self._struct_fields.get(ot[:-1], []):
                    if fn == node.field: return ft
            self._err(f"Type '{ot}' has no field '{node.field}'")
            return VX_INT

        if isinstance(node, NewExpr):
            if node.type_name not in self._struct_fields:
                self._err(f"Unknown struct '{node.type_name}'")
            return node.type_name

        if isinstance(node, IndexExpr):
            ot = self._infer_expr(node.obj)
            self._infer_expr(node.index)
            if ot.startswith("dict["):
                inner = ot[5:-1]
                return inner[inner.index(',')+1:]
            if ot.endswith("[]"): return ot[:-2]
            if ot == VX_STR: return VX_STR
            return VX_INT

        return VX_INT
