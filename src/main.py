from __future__ import annotations

import argparse

from .analyzer import analyze_text
from .storage import create_entry, get_last_entries


def handle_add(args: argparse.Namespace) -> None:
    text = " ".join(args.text)
    tags = analyze_text(text)
    entry = create_entry(text, tags)

    print(f"\n✔ Saved entry #{entry.id}")
    print(f"  Mood : {entry.tags.get('mood')}")
    print(f"  Energy: {entry.tags.get('energy')}\n")


def handle_summary(args: argparse.Namespace) -> None:
    entries = get_last_entries(3)
    if not entries:
        print("No journal entries found yet. Try adding one with `add`.")
        return

    print("\nLast 3 journal entries:\n")
    for e in entries:
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {e.text}")
        print(f"  → Mood: {e.tags.get('mood')}, Energy: {e.tags.get('energy')}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Powered Mood Journal")
    subparsers = parser.add_subparsers(dest="command")

    # add command
    add_cmd = subparsers.add_parser("add", help="Add a new journal entry")
    add_cmd.add_argument("text", nargs="+", help="Text of the journal entry")
    add_cmd.set_defaults(func=handle_add)

    # summary command
    summary_cmd = subparsers.add_parser("summary", help="Show last 3 entries")
    summary_cmd.set_defaults(func=handle_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
