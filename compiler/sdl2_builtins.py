"""
Vexel SDL2 Built-in Declarations
---------------------------------
When --sdl2 flag is passed, the compiler injects these extern function
declarations into the LLVM module so Vexel code can call SDL2 helpers
from vx_sdl2.c without writing any FFI boilerplate.

Each entry:
  name  →  (param_vx_types, return_vx_type)
"""

SDL2_BUILTINS: dict[str, tuple[list[str], str]] = {
    "sdl2_init":       (["str", "int", "int"],  "int"),
    "sdl2_quit":       ([],                      "void"),
    "sdl2_clear":      (["int", "int", "int"],   "void"),
    "sdl2_set_color":  (["int", "int", "int", "int"], "void"),
    "sdl2_draw_rect":  (["int", "int", "int", "int"], "void"),
    "sdl2_draw_point": (["int", "int"],           "void"),
    "sdl2_present":    ([],                       "void"),
    "sdl2_poll":       ([],                       "int"),
    "sdl2_ticks":      ([],                       "int"),
    "sdl2_delay":        (["int"],                "void"),
    "sdl2_poll_events":    ([],                    "int"),
    "sdl2_key_space":      ([],                    "int"),
    "sdl2_key_space_held": ([],                    "int"),
}

# Map the simple names above to the actual C symbols in vx_sdl2.c
SDL2_C_NAMES: dict[str, str] = {
    "sdl2_init":       "vx_sdl2_init",
    "sdl2_quit":       "vx_sdl2_quit",
    "sdl2_clear":      "vx_sdl2_clear",
    "sdl2_set_color":  "vx_sdl2_set_color",
    "sdl2_draw_rect":  "vx_sdl2_draw_rect",
    "sdl2_draw_point": "vx_sdl2_draw_point",
    "sdl2_present":    "vx_sdl2_present",
    "sdl2_poll":       "vx_sdl2_poll",
    "sdl2_ticks":      "vx_sdl2_ticks",
    "sdl2_delay":        "vx_sdl2_delay",
    "sdl2_poll_events":    "vx_sdl2_poll_events",
    "sdl2_key_space":      "vx_sdl2_key_space",
    "sdl2_key_space_held": "vx_sdl2_key_space_held",
}
