#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
import torch
import umap
from sklearn.decomposition import PCA


def create_visualizations(
    embeddings_path: Path,
    output_dir: Path,
    token_norms_path: Path = None,
    sample_size: int = 10000,
) -> None:
    # Load embeddings
    embeddings = torch.load(embeddings_path, map_location="cpu")
    if embeddings.dtype in (torch.bfloat16, torch.float16):
        embeddings = embeddings.to(torch.float32)

    # Load token norms and select tokens with highest norms
    if token_norms_path and token_norms_path.exists():
        token_norms_df = pd.read_csv(token_norms_path)
        token_norms_df = token_norms_df.sort_values("l2_norm", ascending=False)
        top_tokens_df = token_norms_df.head(sample_size)

        indices = torch.tensor(top_tokens_df["token_id"].values, dtype=torch.long)
        sampled_embeddings = embeddings[indices]
        tokens = top_tokens_df["token"].tolist()
    else:
        # Random sampling fallback
        sample_size = min(sample_size, embeddings.shape[0])
        indices = torch.randperm(embeddings.shape[0])[:sample_size]
        sampled_embeddings = embeddings[indices]
        tokens = [f"token_{i}" for i in indices.tolist()]

    # Compute UMAP and PCA
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(sampled_embeddings.numpy())

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(sampled_embeddings.numpy())

    # Create plots
    umap_df = pd.DataFrame(
        {
            "UMAP 1": umap_coords[:, 0],
            "UMAP 2": umap_coords[:, 1],
            "Token": tokens,
            "Token ID": indices.tolist(),
        }
    )

    fig_umap = px.scatter(
        umap_df,
        x="UMAP 1",
        y="UMAP 2",
        hover_data=["Token", "Token ID"],
        title=f"UMAP Visualization ({len(indices)} tokens)",
    )
    fig_umap.update_traces(marker=dict(size=3, opacity=0.7, color="black"))
    fig_umap.update_layout(
        template="simple_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    pca_df = pd.DataFrame(
        {
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "Token": tokens,
            "Token ID": indices.tolist(),
        }
    )

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_data=["Token", "Token ID"],
        title=f"PCA Visualization ({len(indices)} tokens, {pca.explained_variance_ratio_.sum():.3f} variance)",
    )
    fig_pca.update_traces(marker=dict(size=3, opacity=0.7, color="black"))
    fig_pca.update_layout(
        template="simple_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Save HTML and PNG files
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_umap.write_html(output_dir / "umap_embeddings.html")
    fig_umap.write_image(output_dir / "umap_embeddings.png", scale=2)
    fig_pca.write_html(output_dir / "pca_embeddings.html")
    fig_pca.write_image(output_dir / "pca_embeddings.png", scale=2)

    print(f"Wrote UMAP plot: {output_dir / 'umap_embeddings.html'}")
    print(f"Wrote PCA plot: {output_dir / 'pca_embeddings.html'}")


def main():
    parser = argparse.ArgumentParser(
        description="Create UMAP and PCA 2D visualizations of token embeddings"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/simplestories_35M_embeddings.pt"),
        help="Path to the saved embedding tensor (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to write the output HTML files",
    )
    parser.add_argument(
        "--token-norms",
        type=Path,
        default=Path("data/token_norms.csv"),
        help="Path to the token norms CSV file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of top tokens by L2 norm to visualize",
    )
    args = parser.parse_args()

    create_visualizations(
        args.embeddings, args.output_dir, args.token_norms, args.sample_size
    )


if __name__ == "__main__":
    main()
