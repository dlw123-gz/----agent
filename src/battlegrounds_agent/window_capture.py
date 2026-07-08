from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import ImageGrab


@dataclass(frozen=True, slots=True)
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)


def list_windows(title_contains: str | None = None) -> list[WindowInfo]:
    user32 = ctypes.windll.user32
    results: list[WindowInfo] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if not title:
            return True
        if title_contains and title_contains.lower() not in title.lower():
            return True
        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        info = WindowInfo(
            hwnd=int(hwnd),
            title=title,
            left=int(rect.left),
            top=int(rect.top),
            right=int(rect.right),
            bottom=int(rect.bottom),
        )
        if info.width > 0 and info.height > 0:
            results.append(info)
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return results


def find_window(title_contains: str = "炉石传说") -> WindowInfo:
    windows = list_windows(title_contains)
    if not windows:
        raise RuntimeError(f"No visible window title contains {title_contains!r}")
    return max(windows, key=lambda item: item.width * item.height)


def capture_window(title_contains: str = "炉石传说", output_path: str | Path | None = None) -> Path:
    window = find_window(title_contains)
    image = ImageGrab.grab(bbox=(window.left, window.top, window.right, window.bottom), all_screens=True)
    path = Path(output_path) if output_path else Path.cwd() / "hearthstone_window.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def capture_screen(output_path: str | Path | None = None) -> Path:
    image = ImageGrab.grab(all_screens=True)
    path = Path(output_path) if output_path else Path.cwd() / "screen.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Capture a visible game window by title.")
    parser.add_argument("--title", default="炉石传说", help="Substring of the target window title.")
    parser.add_argument("--output", default="work/current_screenshot.png", help="Output screenshot path.")
    parser.add_argument("--list", action="store_true", help="List matching windows instead of capturing.")
    parser.add_argument("--screen", action="store_true", help="Capture the full screen instead of a window.")
    args = parser.parse_args()

    if args.list:
        import json

        windows = [asdict(window) for window in list_windows(args.title)]
        print(json.dumps(windows, ensure_ascii=False, indent=2))
        return

    if args.screen:
        print(capture_screen(args.output))
    else:
        print(capture_window(args.title, args.output))

def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
