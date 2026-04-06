/*
 * Vexel Runtime Library
 *
 * NOTE: __chkstk stub — LLVM with the MSVC target triple inserts calls to
 * __chkstk for stack-frame probing when linking with MinGW gcc. Our programs
 * don't have frames > 4 KB, so a no-op stub is safe here. Remove if you ever
 * switch to clang/MSVC for linking.
 */
void __chkstk(void) {}

/*
 * ---------------------
 * Provides the garbage collector, string utilities, and math helpers
 * that get linked into every native Vexel binary.
 *
 * Compile:
 *   gcc -O2 -c runtime.c -o runtime.o
 * Then link with the Vexel object:
 *   gcc vexel_out.o runtime.o -o my_program -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  GC object header                                                    */
/* ------------------------------------------------------------------ */

typedef struct VxObjHeader {
    uint64_t          size;     /* payload bytes                      */
    int               marked;   /* 1 = reachable during mark phase    */
    struct VxObjHeader *next;   /* intrusive linked list of all objs  */
} VxObjHeader;

static VxObjHeader *gc_head       = NULL;
static uint64_t     gc_allocated  = 0;
static uint64_t     gc_threshold  = 1024 * 1024;  /* 1 MB trigger    */

/* ------------------------------------------------------------------ */
/*  GC roots (explicit registration — no stack scanning)              */
/* ------------------------------------------------------------------ */

#define VX_MAX_ROOTS 4096
static void **gc_roots[VX_MAX_ROOTS];
static int    gc_root_count = 0;

void vx_gc_root_push(void **root) {
    if (gc_root_count < VX_MAX_ROOTS)
        gc_roots[gc_root_count++] = root;
}

void vx_gc_root_pop(void) {
    if (gc_root_count > 0) gc_root_count--;
}

/* ------------------------------------------------------------------ */
/*  Allocation                                                          */
/* ------------------------------------------------------------------ */

void *vx_alloc(uint64_t size) {
    if (gc_allocated > gc_threshold)
        vx_gc_collect();

    VxObjHeader *header = (VxObjHeader *)malloc(sizeof(VxObjHeader) + size);
    if (!header) {
        fprintf(stderr, "vexel: out of memory\n");
        exit(1);
    }
    header->size   = size;
    header->marked = 0;
    header->next   = gc_head;
    gc_head        = header;
    gc_allocated  += sizeof(VxObjHeader) + size;
    return (void *)(header + 1);
}

/* ------------------------------------------------------------------ */
/*  Mark & Sweep                                                        */
/* ------------------------------------------------------------------ */

static void vx_mark(void *ptr) {
    if (!ptr) return;
    VxObjHeader *h = ((VxObjHeader *)ptr) - 1;
    if (h->marked) return;
    h->marked = 1;
    /* Conservative interior scan: treat every aligned word that looks
       like a heap pointer as a potential reference.                    */
    uintptr_t *words = (uintptr_t *)(h + 1);
    uint64_t   count = h->size / sizeof(uintptr_t);
    VxObjHeader *cur = gc_head;
    while (cur) {
        uintptr_t base = (uintptr_t)(cur + 1);
        uintptr_t end  = base + cur->size;
        for (uint64_t i = 0; i < count; i++) {
            if (words[i] >= base && words[i] < end) {
                vx_mark((void *)words[i]);
            }
        }
        cur = cur->next;
    }
}

void vx_gc_collect(void) {
    /* Mark phase */
    for (int i = 0; i < gc_root_count; i++)
        vx_mark(*gc_roots[i]);

    /* Sweep phase */
    VxObjHeader **curr = &gc_head;
    while (*curr) {
        VxObjHeader *h = *curr;
        if (!h->marked) {
            *curr = h->next;
            gc_allocated -= sizeof(VxObjHeader) + h->size;
            free(h);
        } else {
            h->marked = 0;
            curr = &h->next;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  String helpers                                                      */
/* ------------------------------------------------------------------ */

char *vx_str_concat(const char *a, const char *b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char  *s  = (char *)vx_alloc(la + lb + 1);
    memcpy(s, a, la);
    memcpy(s + la, b, lb);
    s[la + lb] = '\0';
    return s;
}

int vx_str_eq(const char *a, const char *b) {
    return strcmp(a, b) == 0;
}

int64_t vx_str_len(const char *s) {
    return (int64_t)strlen(s);
}

/* ------------------------------------------------------------------ */
/*  Math helpers (wrappers so Vexel code doesn't need libm directly)  */
/* ------------------------------------------------------------------ */

#include <math.h>

double vx_sqrt(double x)  { return sqrt(x);  }
double vx_pow(double b, double e) { return pow(b, e); }
double vx_abs_f(double x) { return fabs(x);  }
int64_t vx_abs_i(int64_t x) { return x < 0 ? -x : x; }
double vx_sin(double x)   { return sin(x);   }
double vx_cos(double x)   { return cos(x);   }
double vx_floor(double x) { return floor(x); }
double vx_ceil(double x)  { return ceil(x);  }

/* ------------------------------------------------------------------ */
/*  Type-to-string conversions                                          */
/* ------------------------------------------------------------------ */

char *vx_int_to_str(int64_t n) {
    char *buf = (char *)vx_alloc(32);
    snprintf(buf, 32, "%lld", (long long)n);
    return buf;
}

char *vx_float_to_str(double f) {
    char *buf = (char *)vx_alloc(32);
    snprintf(buf, 32, "%g", f);
    return buf;
}

char *vx_bool_to_str(int b) {
    return b ? "true" : "false";
}

/* ------------------------------------------------------------------ */
/*  OS helpers (v4)                                                     */
/* ------------------------------------------------------------------ */

#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#define vx_platform_mkdir(p)  _mkdir(p)
#else
#include <unistd.h>
#include <sys/stat.h>
#define vx_platform_mkdir(p)  mkdir((p), 0755)
#endif

/* These are called from generated code via the 'getcwd' / 'mkdir' / 'remove'
   extern declarations in codegen.py, so no separate vx_ wrapper is needed.
   This file exists mainly to document platform compat. */

/* Double-check that getcwd is available (it's in stdlib.h / unistd.h). */
#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#endif
