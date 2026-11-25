

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256_MANIFEST.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def included_files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path != OUTPUT
            and "__pycache__" not in path.parts
            and path.suffix.lower() != ".pyc"
        ),
        key=lambda path: path.relative_to(ROOT).as_posix().lower(),
    )


def main() -> None:
    rows = ["sha256\trelative_path"]
    for path in included_files():
        relative = path.relative_to(ROOT).as_posix()
        rows.append(f"{sha256(path)}\t{relative}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"manifest={OUTPUT}")
    print(f"file_count={len(rows) - 1}")


if __name__ == "__main__":
    main()
