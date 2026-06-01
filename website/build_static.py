"""Vercel build hook: collect static files and verify app CSS is present."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_CSS = ROOT / "static" / "css" / "base" / "home.css"
OUT_CSS = ROOT / "staticfiles" / "css" / "base" / "home.css"


def main() -> None:
    if not SRC_CSS.is_file():
        raise SystemExit(f"Missing source CSS: {SRC_CSS}")

    subprocess.check_call(
        [sys.executable, "manage.py", "collectstatic", "--noinput"],
        cwd=ROOT,
    )

    if not OUT_CSS.is_file():
        raise SystemExit(f"collectstatic did not produce {OUT_CSS}")

    print(f"Static assets OK: {OUT_CSS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
