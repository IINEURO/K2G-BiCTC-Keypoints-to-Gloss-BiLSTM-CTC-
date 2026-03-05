#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bisignlangtrans.data.ce_csl import write_manifest_and_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CE-CSL manifest and vocabulary")
    parser.add_argument("--raw-root", type=str, default="data/raw/CE-CSL")
    parser.add_argument("--manifest", type=str, default="data/meta/manifest.jsonl")
    parser.add_argument("--vocab", type=str, default="data/meta/vocab.json")
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--keep-punct", action="store_true", help="Keep punctuation tokens in gloss")
    parser.add_argument(
        "--no-unk",
        action="store_true",
        help="Do not add <UNK> token to vocab (not recommended when min-freq > 1)",
    )
    args = parser.parse_args()

    drop_punct = not args.keep_punct
    write_manifest_and_vocab(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        vocab_path=args.vocab,
        min_freq=args.min_freq,
        drop_punct=drop_punct,
        use_unk=(not args.no_unk),
    )

    print(f"[prepare] raw_root={Path(args.raw_root).resolve()}")
    print(f"[prepare] manifest={Path(args.manifest).resolve()}")
    print(f"[prepare] vocab={Path(args.vocab).resolve()}")


if __name__ == "__main__":
    main()
