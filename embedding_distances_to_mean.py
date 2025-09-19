#!/usr/bin/env python3
"""
Read the saved embedding matrix and write a CSV with token and distance to mean per row.
Sorted by distance (descending by default).
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
    if "" in token_str:
        try:
            raw_bytes = tokenizer.convert_ids_to_tokens(
                token_id, skip_special_tokens=False
            )
            if isinstance(raw_bytes, bytes):
                hex_repr = " ".join(f"{b:02x}" for b in raw_bytes)
                return f"[BYTES: {hex_repr}]"
            else:
                hex_repr = " ".join(f"{ord(c):02x}" for c in raw_bytes)
                return f"[BYTES: {hex_repr}]"
        except Exception:
            return repr(token_str)
    return token_str


def compute_and_write_distances(
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

    # Compute mean embedding
    mean_embedding = embeddings_f32.mean(dim=0, keepdim=True)

    # Compute distances to mean
    distances = torch.linalg.norm(embeddings_f32 - mean_embedding, dim=1)

    # Build tokenizer to map ids->tokens
    tokenizer = get_tokenizer()
    vocab_size = embeddings.shape[0]

    # Collect rows: (token_id, token_str, distance)
    rows = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])

        # Filter for non-ASCII tokens if requested
        if non_ascii_only and is_ascii_only(token_str):
            continue

        # Format token for display, handling unprintable unicode
        formatted_token = format_token_for_display(token_str, tokenizer, token_id)

        rows.append((token_id, formatted_token, float(distances[token_id].item())))

    # Sort by distance
    rows.sort(key=lambda r: r[2], reverse=not ascending)

    # Write CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "distance_to_mean"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token distances to mean embedding and write CSV"
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
        default=Path("data/token_distances_to_mean.csv"),
        help="Path to write the output CSV",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order of distance (default: descending)",
    )
    parser.add_argument(
        "--non-ascii-only",
        action="store_true",
        help="Filter to include only tokens that are not exclusively ASCII",
    )
    args = parser.parse_args()

    compute_and_write_distances(
        args.embeddings, args.output, args.ascending, args.non_ascii_only
    )
    print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()
