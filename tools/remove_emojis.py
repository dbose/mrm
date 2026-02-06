#!/usr/bin/env python3
"""Remove common Unicode emoji and ASCII emoticons from text files.

Creates a `.bak` backup for each file modified.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXTS = {'.md', '.rst', '.py', '.yml', '.yaml', '.txt', '.json', '.ini', '.sh', '.cfg', '.toml'}

# Broad emoji ranges plus common symbols (keeps regex simple and safe)
EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001F5FF'
    '\U0001F600-\U0001F64F'
    '\U0001F680-\U0001F6FF'
    '\U0001F700-\U0001F77F'
    '\U0001F780-\U0001F7FF'
    '\U0001F900-\U0001F9FF'
    '\u2600-\u26FF'
    '\u2700-\u27BF'
    '\uFE0F'  # variation selector
    ']+',
    flags=re.UNICODE,
)

# Common ASCII emoticons and heart
EMOTICON_RE = re.compile(r"(:-\)|:\)|||||;\)|;-\)|:\(|:-\(|||:\|||\\|\\)")


def should_process(path: Path) -> bool:
    return path.suffix.lower() in EXTS


def process_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return False

    new = EMOJI_RE.sub('', text)
    new = EMOTICON_RE.sub('', new)

    if new != text:
        bak = path.with_name(path.name + '.bak')
        bak.write_text(text, encoding='utf-8')
        path.write_text(new, encoding='utf-8')
        print(f"Updated: {path.relative_to(ROOT)} (backup -> {bak.name})")
        return True
    return False


def main():
    changed = 0
    for p in ROOT.rglob('*'):
        if p.is_file() and should_process(p):
            if process_file(p):
                changed += 1

    print(f"Done. Files changed: {changed}")


if __name__ == '__main__':
    main()
