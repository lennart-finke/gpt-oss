#!/usr/bin/env python3
"""
Read the saved embedding matrix and write a CSV with token and L2 norm per row.
Sorted by norm (descending by default).
"""

import argparse
import csv
from pathlib import Path

import torch
from gpt_oss.tokenizer import get_tokenizer


def is_ascii_only(text: str) -> bool:
    """Check if a string contains only ASCII characters."""
    return all(ord(char) < 128 for char in text)


def format_token_for_display(token_str: str, tokenizer, token_id: int) -> str:
    """Format token string for display, handling unprintable unicode characters."""
    # Check if the string contains the replacement character ()
    if "" in token_str:
        # Get the raw bytes from the tokenizer's vocabulary
        try:
            # Get the raw bytes for this token ID
            raw_bytes = tokenizer.convert_ids_to_tokens(
                token_id, skip_special_tokens=False
            )
            if isinstance(raw_bytes, bytes):
                hex_repr = " ".join(f"{b:02x}" for b in raw_bytes)
                return f"[BYTES: {hex_repr}]"
            else:
                # If it's not bytes, try to get the raw representation
                hex_repr = " ".join(f"{ord(c):02x}" for c in raw_bytes)
                return f"[BYTES: {hex_repr}]"
        except Exception:
            # Fallback: show the raw string with escape sequences
            return repr(token_str)

    # If no replacement characters, return the string as is
    return token_str


def compute_and_write_norms(
    embeddings_path: Path,
    output_csv_path: Path,
    ascending: bool = False,
    non_ascii_only: bool = False,
) -> None:
    # Load tensor on CPU
    embeddings = torch.load(embeddings_path, map_location="cpu")
    if embeddings.dtype in (torch.bfloat16, torch.float16):
        embeddings_f32 = embeddings.to(torch.float32)
    else:
        embeddings_f32 = embeddings

    # Compute L2 norms along embedding dim
    norms = torch.linalg.norm(embeddings_f32, dim=1)

    # Build tokenizer to map ids->tokens
    tokenizer = get_tokenizer()

    vocab_size = embeddings.shape[0]

    # Collect rows: (token_id, token_str, norm)
    rows = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])

        # Filter for non-ASCII tokens if requested
        if non_ascii_only and is_ascii_only(token_str):
            continue

        # Format token for display, handling unprintable unicode
        formatted_token = format_token_for_display(token_str, tokenizer, token_id)

        rows.append((token_id, formatted_token, float(norms[token_id].item())))

    # Sort by norm
    rows.sort(key=lambda r: r[2], reverse=not ascending)

    # Write CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "l2_norm"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token L2 norms from embedding matrix and write CSV"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/gpt_oss_20b_embeddings.pt"),
        help="Path to the saved embedding tensor (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/token_norms.csv"),
        help="Path to write the output CSV",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order of norm (default: descending)",
    )
    parser.add_argument(
        "--non-ascii-only",
        action="store_true",
        help="Filter to include only tokens that are not exclusively ASCII",
    )
    args = parser.parse_args()

    compute_and_write_norms(
        args.embeddings, args.output, args.ascending, args.non_ascii_only
    )
    print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()
