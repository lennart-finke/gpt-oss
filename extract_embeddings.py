#!/usr/bin/env python3
"""
Script to extract the embedding matrix from GPT-OSS 20B model and save it as a torch tensor.
"""

import json
import torch
from safetensors import safe_open
import os
from pathlib import Path


def load_embedding_matrix(model_path, config_path, dtypes_path):
    """
    Load the embedding matrix from the GPT-OSS model.

    Args:
        model_path: Path to the model.safetensors file
        config_path: Path to the config.json file
        dtypes_path: Path to the dtypes.json file

    Returns:
        torch.Tensor: The embedding matrix
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    with open(dtypes_path, "r") as f:
        dtypes = json.load(f)

    print("Model configuration:")
    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Embedding dtype: {dtypes['embedding.weight']}")

    # Load the embedding matrix from safetensors
    with safe_open(model_path, framework="pt", device="cpu") as f:
        # The embedding matrix is stored as 'embedding.weight'
        embedding_tensor = f.get_tensor("embedding.weight")

    print(f"Loaded embedding matrix shape: {embedding_tensor.shape}")
    print(f"Embedding matrix dtype: {embedding_tensor.dtype}")

    return embedding_tensor


def main():
    # Paths to the model files
    model_dir = Path("gpt-oss-20b/original")
    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    dtypes_path = model_dir / "dtypes.json"

    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not dtypes_path.exists():
        raise FileNotFoundError(f"Dtypes file not found: {dtypes_path}")

    print("Loading embedding matrix from GPT-OSS 20B model...")

    # Load the embedding matrix
    embedding_matrix = load_embedding_matrix(model_path, config_path, dtypes_path)

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the embedding matrix as a torch tensor
    output_path = "data/gpt_oss_20b_embeddings.pt"
    torch.save(embedding_matrix, output_path)

    print(f"Embedding matrix saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    # Verify the saved tensor
    loaded_tensor = torch.load(output_path)
    print(f"Verified loaded tensor shape: {loaded_tensor.shape}")
    print(f"Verified loaded tensor dtype: {loaded_tensor.dtype}")

    # Print some statistics
    print("\nEmbedding matrix statistics:")
    print(f"  Mean: {embedding_matrix.mean().item():.6f}")
    print(f"  Std: {embedding_matrix.std().item():.6f}")
    print(f"  Min: {embedding_matrix.min().item():.6f}")
    print(f"  Max: {embedding_matrix.max().item():.6f}")


if __name__ == "__main__":
    main()
