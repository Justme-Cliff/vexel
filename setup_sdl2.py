"""
SDL2 Setup for Vexel on Windows
---------------------------------
Downloads SDL2 development libraries and places them in stdlib/sdl2/.
Run once before building SDL2-enabled Vexel programs.

Usage:
    python setup_sdl2.py
"""

import urllib.request
import zipfile
import shutil
import os

SDL2_VERSION = "2.30.3"
SDL2_URL = (
    f"https://github.com/libsdl-org/SDL/releases/download/"
    f"release-{SDL2_VERSION}/"
    f"SDL2-devel-{SDL2_VERSION}-mingw.zip"
)

DEST = os.path.join(os.path.dirname(__file__), "stdlib", "sdl2")
ZIP_PATH = "sdl2_tmp.zip"


def main():
    print(f"Downloading SDL2 {SDL2_VERSION} for MinGW/Windows...")
    urllib.request.urlretrieve(SDL2_URL, ZIP_PATH)
    print("Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall("sdl2_tmp")

    # Find the x86_64-w64-mingw32 directory inside the archive
    base = os.path.join("sdl2_tmp", f"SDL2-{SDL2_VERSION}", "x86_64-w64-mingw32")

    os.makedirs(DEST, exist_ok=True)
    for sub in ("include", "lib", "bin"):
        src = os.path.join(base, sub)
        dst = os.path.join(DEST, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Also copy SDL2.dll next to main.py for easy running
    dll_src = os.path.join(DEST, "bin", "SDL2.dll")
    dll_dst = os.path.join(os.path.dirname(__file__), "SDL2.dll")
    shutil.copy2(dll_src, dll_dst)

    # Clean up
    shutil.rmtree("sdl2_tmp")
    os.remove(ZIP_PATH)

    print(f"\nSDL2 installed to {DEST}")
    print("SDL2.dll copied next to main.py")
    print("\nYou can now compile SDL2 programs:")
    print("  python main.py compile examples/window.vx --sdl2")


if __name__ == "__main__":
    main()
