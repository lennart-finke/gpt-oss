# What does GPT-OSS tell us about OpenAI's training data?

This is the companion repo to [this](https://fi-le.net/oss) article about analyzing the GPT training pipeline using the weights of gpt-oss.

## Usage
- `extract_embeddings.py` extracts the embedding matrix of the gpt-oss model.
- `embedding_norms_to_csv.py` computes the L2 norms of the extracted embeddings, with an option to only consider non-ascii tokens.
- `embedding_distances_to_mean.py`  computes the embedding distances to the average.
- `plot_token_norms.py` plots the calculated norms.
- `find_chinese_tokens.py` and `chinese_token_ids.py` for isolating Chinese tokens.
- `get_github_counts.py` for getting occurances of token texts in Github using the search API.
- `token_translation.py` for evaluating the completions of various models given different tokens.
- `analyze_model_accuracy.py` for analyzing the results of the previous script.
