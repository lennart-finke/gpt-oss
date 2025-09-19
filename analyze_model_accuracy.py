#!/usr/bin/env python3
import json
import sys
import csv
import numpy as np
import plotly.graph_objects as go
import os
from scipy.stats import spearmanr


def calculate_accuracy(languages):
    """Calculate accuracy where correct means Chinese or Japanese."""
    if not languages:
        return 0.0

    correct_count = sum(1 for lang in languages if lang in ["Chinese", "Japanese"])
    total_count = len(languages)

    return correct_count / total_count


def analyze_overlap(log_files):
    """Analyze accuracy between Claude and other models."""
    # Handle single file or list of files
    if isinstance(log_files, str):
        log_files = [log_files]

    # Load and combine data from all log files
    all_results = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            data = json.load(f)
            all_results.extend(data["results"])

    # Create combined data structure
    combined_data = {"results": all_results}

    # Group results by token_id and model
    token_results = {}
    token_info = {}  # Store token strings for CSV output

    for result in combined_data["results"]:
        if not result["success"]:
            continue

        token_id = result["token_id"]
        token = result["token"]
        model = result["model"]
        language = result["original_language"]

        # Strip quotes and underscore characters from the beginning of token
        # Remove surrounding quotes if present
        if token.startswith("'") and token.endswith("'"):
            token = token[1:-1]

        # Strip underscore characters from the beginning
        if token.startswith("_") or token.startswith("＿") or token.startswith("￣"):
            continue

        # Store token string
        token_info[token_id] = token

        if token_id not in token_results:
            token_results[token_id] = {}
        if model not in token_results[token_id]:
            token_results[token_id][model] = []

        token_results[token_id][model].append(language)

    # Compute accuracy for each token
    results = []
    for token_id, model_results in token_results.items():
        if "claude-sonnet-4-20250514" not in model_results:
            continue

        claude_languages = model_results["claude-sonnet-4-20250514"]
        claude_accuracy = calculate_accuracy(claude_languages)

        token_result = {
            "token_id": token_id,
            "token": token_info.get(token_id, ""),
            "claude_accuracy": claude_accuracy,
            "accuracy_vs_gpt5": None,
            "accuracy_vs_oss120b": None,
            "accuracy_vs_oss20b": None,
            "accuracy_vs_gpt5_mini": None,
            "accuracy_vs_gpt5_nano": None,
        }

        for model, languages in model_results.items():
            if model == "claude-sonnet-4-20250514":
                continue

            other_accuracy = calculate_accuracy(languages)

            if model == "gpt-5-2025-08-07":
                token_result["accuracy_vs_gpt5"] = other_accuracy
            elif model == "openrouter/openai/gpt-oss-120b":
                token_result["accuracy_vs_oss120b"] = other_accuracy
            elif model == "openrouter/openai/gpt-oss-20b":
                token_result["accuracy_vs_oss20b"] = other_accuracy
            elif model == "gpt-5-mini-2025-08-07":
                token_result["accuracy_vs_gpt5_mini"] = other_accuracy
            elif model == "gpt-5-nano-2025-08-07":
                token_result["accuracy_vs_gpt5_nano"] = other_accuracy

        results.append(token_result)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python analyze_model_overlap.py <log_file.json> [additional_log_file.json]"
        )
        sys.exit(1)

    log_files = sys.argv[1:]
    results = analyze_overlap(log_files)

    # Create CSV output
    import os
    from datetime import datetime

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"logs/model_overlap_analysis_{timestamp}.csv"

    # Write CSV
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "token",
            "accuracy_vs_gpt5",
            "accuracy_vs_gpt5_mini",
            "accuracy_vs_gpt5_nano",
            "accuracy_vs_oss120b",
            "accuracy_vs_oss20b",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "token": result["token"],
                    "accuracy_vs_gpt5": round(result["accuracy_vs_gpt5"], 2)
                    if result["accuracy_vs_gpt5"] is not None
                    else "",
                    "accuracy_vs_gpt5_mini": round(result["accuracy_vs_gpt5_mini"], 2)
                    if result["accuracy_vs_gpt5_mini"] is not None
                    else "",
                    "accuracy_vs_gpt5_nano": round(result["accuracy_vs_gpt5_nano"], 2)
                    if result["accuracy_vs_gpt5_nano"] is not None
                    else "",
                    "accuracy_vs_oss120b": round(result["accuracy_vs_oss120b"], 2)
                    if result["accuracy_vs_oss120b"] is not None
                    else "",
                    "accuracy_vs_oss20b": round(result["accuracy_vs_oss20b"], 2)
                    if result["accuracy_vs_oss20b"] is not None
                    else "",
                }
            )

    print(f"Results saved to {csv_filename}")

    # Create symbolic CSV
    symbolic_csv_filename = f"logs/model_overlap_analysis_symbolic_{timestamp}.csv"

    def accuracy_to_symbol(accuracy):
        """Convert accuracy value to symbol."""
        if accuracy is None or accuracy == "":
            return ""
        elif accuracy == 0.0:
            return "✗"  # Cross symbol for 0.0
        elif 0.0 < accuracy < 0.5:
            return "?"  # Question mark for between 0 and 0.5
        elif 0.5 <= accuracy < 1.0:
            return "!"  # Exclamation mark for 0.5 to 1
        elif accuracy == 1.0:
            return "✓"  # Unicode tick for 1.0
        else:
            return "?"

    # Write symbolic CSV
    with open(symbolic_csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "token",
            "accuracy_vs_gpt5",
            "accuracy_vs_gpt5_mini",
            "accuracy_vs_gpt5_nano",
            "accuracy_vs_oss120b",
            "accuracy_vs_oss20b",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "token": result["token"],
                    "accuracy_vs_gpt5": accuracy_to_symbol(result["accuracy_vs_gpt5"]),
                    "accuracy_vs_gpt5_mini": accuracy_to_symbol(
                        result["accuracy_vs_gpt5_mini"]
                    ),
                    "accuracy_vs_gpt5_nano": accuracy_to_symbol(
                        result["accuracy_vs_gpt5_nano"]
                    ),
                    "accuracy_vs_oss120b": accuracy_to_symbol(
                        result["accuracy_vs_oss120b"]
                    ),
                    "accuracy_vs_oss20b": accuracy_to_symbol(
                        result["accuracy_vs_oss20b"]
                    ),
                }
            )

    print(f"Symbolic results saved to {symbolic_csv_filename}")

    # Load GitHub counts - fail if not available
    github_counts = {}
    github_file = "logs/github_counts.json"
    if not os.path.exists(github_file):
        raise FileNotFoundError(f"GitHub counts file not found: {github_file}")

    with open(github_file, "r") as f:
        github_data = json.load(f)
        for token_id, data in github_data.items():
            if "github_count" not in data:
                raise KeyError(f"Missing 'github_count' field for token_id {token_id}")
            github_counts[token_id] = data["github_count"]
    print(f"Loaded GitHub counts for {len(github_counts)} tokens")

    # Create scatter plot of GitHub counts vs GPT model accuracy
    if not github_counts:
        raise ValueError("No GitHub counts loaded")

    # Calculate sum of correct guesses by all GPT models per token
    plot_data = []
    for result in results:
        token = result["token"]
        # Find matching token_id from GitHub data
        token_id = None
        for tid, gdata in github_data.items():
            # Normalize tokens by removing quotes for comparison
            github_token = gdata["token"]
            if github_token.startswith("'") and github_token.endswith("'"):
                github_token = github_token[1:-1]

            if github_token == token:
                token_id = tid
                break

        if not token_id:
            print(f"Warning: No matching token_id found for token: {token} - skipping")
            continue

        if token_id not in github_counts:
            print(
                f"Warning: Token_id {token_id} not found in GitHub counts for token: {token} - skipping"
            )
            continue

        # Sum up all GPT model accuracies (convert symbols to numbers)
        gpt_sum = 0
        gpt_models = [
            "accuracy_vs_gpt5",
            "accuracy_vs_gpt5_mini",
            "accuracy_vs_gpt5_nano",
            "accuracy_vs_oss120b",
            "accuracy_vs_oss20b",
        ]

        for model in gpt_models:
            if result[model] is not None:
                # Convert accuracy to numeric (0-1 scale)
                if isinstance(result[model], (int, float)):
                    gpt_sum += result[model]
                else:
                    print(
                        f"Warning: Non-numeric accuracy value for {model}: {result[model]} - skipping token {token}"
                    )
                    continue

        plot_data.append(
            {
                "token": token,
                "github_count": github_counts[token_id],
                "gpt_accuracy_sum": gpt_sum,
            }
        )

    if not plot_data:
        print("Warning: No plot data generated - no matching tokens found")
        # Skip the plotting section
        plot_data = None

    # Create scatter plot of accuracies vs log-counts
    if plot_data:
        github_values = [d["github_count"] for d in plot_data]
        gpt_values = [d["gpt_accuracy_sum"] for d in plot_data]

        if not github_values or not gpt_values:
            print("Warning: Empty data for plotting - skipping plot")
        else:
            # Calculate Spearman correlation with log-transformed counts
            log_counts = [np.log10(count + 1) for count in github_values]
            correlation, p_value = spearmanr(log_counts, gpt_values)

            # Create scatter plot with original values but log x-axis
            fig = go.Figure(
                data=go.Scatter(
                    x=github_values,
                    y=gpt_values,
                    mode="markers",
                    marker=dict(color="black", size=8, opacity=0.7),
                    text=[d["token"] for d in plot_data],
                    hovertemplate="<b>%{text}</b><br>GitHub Count: %{x}<br>GPT Accuracy Sum: %{y:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=f"GitHub Count vs GPT Model Accuracy Sum<br><sub>Spearman ρ = {correlation:.3f} (p = {p_value:.2e})</sub>",
                xaxis_title="GitHub Count",
                yaxis_title="Sum of GPT Model Accuracies",
                xaxis=dict(type="log"),
                template="simple_white",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
            )

            # Ensure figures directory exists
            os.makedirs("figures", exist_ok=True)

            # Save plot - fail if save fails
            try:
                fig.write_html(
                    "figures/github_vs_gpt_accuracy.html",
                    include_plotlyjs="cdn",
                    full_html=False,
                )
                fig.write_image("figures/github_vs_gpt_accuracy.png", scale=2)
            except Exception as e:
                raise RuntimeError(f"Failed to save plot: {e}")

            print(f"Scatter plot saved with {len(plot_data)} data points")
            print(
                f"Spearman correlation (log-transformed counts): ρ = {correlation:.3f}, p-value = {p_value:.2e}"
            )
            print(f"GitHub count range: {min(github_values)} to {max(github_values)}")
            print(f"Accuracy range: {min(gpt_values):.2f} to {max(gpt_values):.2f}")

    # Print summary statistics
    claude_accuracies = [r["claude_accuracy"] for r in results]
    gpt5_accuracies = [
        r["accuracy_vs_gpt5"] for r in results if r["accuracy_vs_gpt5"] is not None
    ]
    oss120b_accuracies = [
        r["accuracy_vs_oss120b"]
        for r in results
        if r["accuracy_vs_oss120b"] is not None
    ]
    oss20b_accuracies = [
        r["accuracy_vs_oss20b"] for r in results if r["accuracy_vs_oss20b"] is not None
    ]
    gpt5_mini_accuracies = [
        r["accuracy_vs_gpt5_mini"]
        for r in results
        if r["accuracy_vs_gpt5_mini"] is not None
    ]
    gpt5_nano_accuracies = [
        r["accuracy_vs_gpt5_nano"]
        for r in results
        if r["accuracy_vs_gpt5_nano"] is not None
    ]

    print("\nSummary Statistics (Accuracy where correct = Chinese or Japanese):")
    if claude_accuracies:
        print(
            f"Claude: Mean={sum(claude_accuracies)/len(claude_accuracies):.3f}, Min={min(claude_accuracies):.3f}, Max={max(claude_accuracies):.3f}"
        )
    if gpt5_accuracies:
        print(
            f"GPT-5: Mean={sum(gpt5_accuracies)/len(gpt5_accuracies):.3f}, Min={min(gpt5_accuracies):.3f}, Max={max(gpt5_accuracies):.3f}"
        )
    if oss120b_accuracies:
        print(
            f"GPT-OSS-120B: Mean={sum(oss120b_accuracies)/len(oss120b_accuracies):.3f}, Min={min(oss120b_accuracies):.3f}, Max={max(oss120b_accuracies):.3f}"
        )
    if oss20b_accuracies:
        print(
            f"GPT-OSS-20B: Mean={sum(oss20b_accuracies)/len(oss20b_accuracies):.3f}, Min={min(oss20b_accuracies):.3f}, Max={max(oss20b_accuracies):.3f}"
        )
    if gpt5_mini_accuracies:
        print(
            f"GPT-5-Mini: Mean={sum(gpt5_mini_accuracies)/len(gpt5_mini_accuracies):.3f}, Min={min(gpt5_mini_accuracies):.3f}, Max={max(gpt5_mini_accuracies):.3f}"
        )
    if gpt5_nano_accuracies:
        print(
            f"GPT-5-Nano: Mean={sum(gpt5_nano_accuracies)/len(gpt5_nano_accuracies):.3f}, Min={min(gpt5_nano_accuracies):.3f}, Max={max(gpt5_nano_accuracies):.3f}"
        )
