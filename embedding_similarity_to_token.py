#!/usr/bin/env python3
"""
Read the saved embedding matrix and compute similarity to token 47989.
Plot histogram of similarities.
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from gpt_oss.tokenizer import get_tokenizer


def compute_similarities_and_plot(
    embeddings_path: Path,
    target_token_id: int = 162877,
    output_plot_path: Path = Path("figures/similarity_histogram.png"),
) -> None:
    # Load tensor on CPU
    embeddings = torch.load(embeddings_path, map_location="cpu")
    if embeddings.dtype in (torch.bfloat16, torch.float16):
        embeddings_f32 = embeddings.to(torch.float32)
    else:
        embeddings_f32 = embeddings

    # Get target token embedding
    target_embedding = embeddings_f32[target_token_id]

    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        embeddings_f32, target_embedding.unsqueeze(0), dim=1
    )

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(similarities.numpy(), bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Cosine Similarity to Token 47989")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title(f"Distribution of Similarities to Token {target_token_id}")
    plt.grid(True, alpha=0.3)

    # Add vertical line for target token's self-similarity
    target_similarity = similarities[target_token_id].item()
    plt.axvline(
        target_similarity,
        color="red",
        linestyle="--",
        label=f"Token {target_token_id} self-similarity: {target_similarity:.3f}",
    )
    plt.legend()

    # Save plot
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print top similar tokens
    tokenizer = get_tokenizer()
    top_indices = torch.topk(similarities, 10).indices

    print(f"Top 10 tokens most similar to token {target_token_id}:")
    for i, idx in enumerate(top_indices):
        token_str = tokenizer.decode([idx.item()])
        sim = similarities[idx].item()
        print(f"{i+1:2d}. Token {idx:5d} ('{token_str}'): {sim:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarities to target token and plot histogram"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/gpt_oss_20b_embeddings.pt"),
        help="Path to the saved embedding tensor (.pt)",
    )
    parser.add_argument(
        "--target-token",
        type=int,
        default=72472,
        help="Target token ID to compute similarities against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/similarity_histogram.png"),
        help="Path to save the histogram plot",
    )
    args = parser.parse_args()

    compute_similarities_and_plot(args.embeddings, args.target_token, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
