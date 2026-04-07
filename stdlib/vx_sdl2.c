/*
 * Vexel SDL2 Bindings
 * -------------------
 * Thin C wrapper that exposes SDL2 to Vexel programs.
 * Every function here is callable from Vexel via extern declarations
 * in the compiler's SDL2 built-in module.
 *
 * Setup (Windows — run setup_sdl2.py first):
 *   gcc -O2 -c vx_sdl2.c -I./sdl2/include -o vx_sdl2.o
 *   gcc vexel_out.o runtime.o vx_sdl2.o -L./sdl2/lib -lSDL2 -o game.exe
 */

#include <stdint.h>
#include <stdio.h>

#ifdef VX_SDL2_ENABLED
#  include <SDL2/SDL.h>
#else
/* Stub types when SDL2 is not installed — lets the file compile cleanly */
typedef void SDL_Window;
typedef void SDL_Renderer;
typedef struct { int x, y, w, h; } SDL_Rect;
#endif

/* ------------------------------------------------------------------ */
/*  Window / Renderer                                                   */
/* ------------------------------------------------------------------ */

SDL_Window   *vx_sdl2_window   = NULL;
SDL_Renderer *vx_sdl2_renderer = NULL;

int vx_sdl2_init(const char *title, int64_t width, int64_t height) {
#ifdef VX_SDL2_ENABLED
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 0;
    }
    vx_sdl2_window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        (int)width, (int)height,
        SDL_WINDOW_SHOWN
    );
    if (!vx_sdl2_window) {
        fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        return 0;
    }
    vx_sdl2_renderer = SDL_CreateRenderer(
        vx_sdl2_window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!vx_sdl2_renderer) {
        fprintf(stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError());
        return 0;
    }
    return 1;
#else
    fprintf(stderr, "SDL2 not installed. Run: python setup_sdl2.py\n");
    return 0;
#endif
}

void vx_sdl2_quit(void) {
#ifdef VX_SDL2_ENABLED
    if (vx_sdl2_renderer) SDL_DestroyRenderer(vx_sdl2_renderer);
    if (vx_sdl2_window)   SDL_DestroyWindow(vx_sdl2_window);
    SDL_Quit();
#endif
}

/* ------------------------------------------------------------------ */
/*  Drawing                                                             */
/* ------------------------------------------------------------------ */

void vx_sdl2_clear(int64_t r, int64_t g, int64_t b) {
#ifdef VX_SDL2_ENABLED
    SDL_SetRenderDrawColor(vx_sdl2_renderer, (Uint8)r, (Uint8)g, (Uint8)b, 255);
    SDL_RenderClear(vx_sdl2_renderer);
#endif
}

void vx_sdl2_set_color(int64_t r, int64_t g, int64_t b, int64_t a) {
#ifdef VX_SDL2_ENABLED
    SDL_SetRenderDrawColor(vx_sdl2_renderer, (Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a);
#endif
}

void vx_sdl2_draw_rect(int64_t x, int64_t y, int64_t w, int64_t h) {
#ifdef VX_SDL2_ENABLED
    SDL_Rect rect = { (int)x, (int)y, (int)w, (int)h };
    SDL_RenderFillRect(vx_sdl2_renderer, &rect);
#endif
}

void vx_sdl2_draw_point(int64_t x, int64_t y) {
#ifdef VX_SDL2_ENABLED
    SDL_RenderDrawPoint(vx_sdl2_renderer, (int)x, (int)y);
#endif
}

void vx_sdl2_present(void) {
#ifdef VX_SDL2_ENABLED
    SDL_RenderPresent(vx_sdl2_renderer);
#endif
}

/* ------------------------------------------------------------------ */
/*  Events — returns 0 if quit was requested, 1 to keep running       */
/* ------------------------------------------------------------------ */

int vx_sdl2_poll(void) {
#ifdef VX_SDL2_ENABLED
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) return 0;
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) return 0;
    }
    return 1;
#else
    return 0;
#endif
}

/* Returns milliseconds since SDL was initialized */
int64_t vx_sdl2_ticks(void) {
#ifdef VX_SDL2_ENABLED
    return (int64_t)SDL_GetTicks();
#else
    return 0;
#endif
}

void vx_sdl2_delay(int64_t ms) {
#ifdef VX_SDL2_ENABLED
    SDL_Delay((Uint32)ms);
#endif
}

/* ------------------------------------------------------------------ */
/*  Keyboard input                                                      */
/* ------------------------------------------------------------------ */

static int vx_space_pressed = 0;
static int vx_should_quit   = 0;

/*
 * Call once per frame.  Drains the event queue, records SPACE and quit.
 * Returns 1 = keep running, 0 = quit was requested.
 */
int64_t vx_sdl2_poll_events(void) {
    vx_space_pressed = 0;
#ifdef VX_SDL2_ENABLED
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            vx_should_quit = 1;
        }
        if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE) vx_should_quit = 1;
            if (e.key.keysym.sym == SDLK_SPACE)  vx_space_pressed = 1;
        }
    }
#endif
    return vx_should_quit ? 0 : 1;
}

/* Returns 1 if SPACE was pressed this frame, 0 otherwise. */
int64_t vx_sdl2_key_space(void) {
    return (int64_t)vx_space_pressed;
}
