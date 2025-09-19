#!/usr/bin/env python3

import csv
import plotly.express as px
import plotly.graph_objects as go
import os

os.makedirs("figures", exist_ok=True)

# Load data
data = []
with open("data/token_norms.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append((int(row["token_id"]), row["token"], float(row["l2_norm"])))

# Extract norms for histogram
norms = [row[2] for row in data]

# Histogram
fig = px.histogram(
    x=norms,
    nbins=50,
    title="Distribution of Token Embedding L2 Norms",
    labels={"x": "L2 Norm", "y": "Number of Tokens"},
)
fig.update_traces(marker=dict(color="black"))
fig.update_layout(
    template="simple_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
)
# log y axis
fig.update_yaxes(type="log")
fig.write_html(
    "figures/token_norms_histogram.html", include_plotlyjs="cdn", full_html=False
)
fig.write_image("figures/token_norms_histogram.png", scale=2)

# Top tokens plot
top_n = 25
top_tokens = data[:top_n]

fig = go.Figure(
    data=go.Bar(
        x=[row[2] for row in top_tokens],
        y=[
            f"{row[0]}: {row[1][:20]}{'...' if len(row[1]) > 20 else ''}"
            for row in top_tokens
        ],
        orientation="h",
        text=[f"{row[2]:.1f}" for row in top_tokens],
        textposition="auto",
        marker=dict(color="black"),
    )
)

fig.update_layout(
    title=f"Top {top_n} Tokens by L2 Norm",
    xaxis_title="L2 Norm",
    yaxis_title="Token",
    height=600,
    template="simple_white",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.write_html(
    "figures/top_tokens_by_norm.html", include_plotlyjs="cdn", full_html=False
)
fig.write_image("figures/top_tokens_by_norm.png", scale=2)

print(f"Plotted {len(data)} tokens")
print(f"Mean L2 norm: {sum(norms)/len(norms):.3f}")
print(f"Max L2 norm: {max(norms):.3f}")
