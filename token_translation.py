from litellm import acompletion
import asyncio
from datetime import datetime
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Constants
PROMPT = """Translate the following token to English in 1-3 words and identify the original language. 
Format your response exactly as: "English: [translation] | Language: [original language]"

Token: {token}"""

MODELS = {
    "claude-4.0": "claude-sonnet-4-20250514",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
    "gpt-oss-20b": "openrouter/openai/gpt-oss-20b"
}


async def get_translation(token_id: int, token: str, model: str):
    """Get translation for a token from a specific model."""
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(token=token)}],
        )

        if response.choices[0].message.content is None:
            return await get_translation(token_id, token, model)

        response_content = response.choices[0].message.content

        # Parse the response to extract English translation and language
        try:
            if "English:" in response_content and "Language:" in response_content:
                parts = response_content.split("|")
                english_part = parts[0].split("English:")[1].strip()
                language_part = parts[1].split("Language:")[1].strip()
            else:
                # Fallback parsing if format is different
                english_part = response_content
                language_part = "Unknown"
        except Exception:
            english_part = response_content
            language_part = "Unknown"

        return {
            "token_id": token_id,
            "token": token,
            "model": model,
            "english_translation": english_part,
            "original_language": language_part,
            "raw_response": response_content,
            "success": True,
        }
    except Exception as e:
        return {
            "token_id": token_id,
            "token": token,
            "model": model,
            "english_translation": None,
            "original_language": None,
            "raw_response": str(e),
            "success": False,
            "error": str(e),
        }


async def collect_translations(token_ids, num_tokens=50, repeats=1):
    """Collect translations for specified tokens from all models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Load token data
    print("Loading token data...")
    df = pd.read_csv("data/token_norms.csv")

    # Use specific token IDs
    test_tokens = df[df["token_id"].isin(token_ids)]

    # Filter out tokens that start with specific characters
    test_tokens = test_tokens[
        ~test_tokens["token"].str.startswith("_")
        & ~test_tokens["token"].str.startswith("＿")
        & ~test_tokens["token"].str.startswith("￣")
    ]

    print(
        f"Testing {len(test_tokens)} tokens across {len(MODELS)} models with {repeats} repeats each..."
    )

    all_results = []

    # Process each token and model combination with repeats
    tasks = []
    for _, row in test_tokens.iterrows():
        token_id = row["token_id"]
        token = row["token"]
        for model_name, model_id in MODELS.items():
            for repeat in range(repeats):
                tasks.append(get_translation(token_id, token, model_id))

    batch_size = 30
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size} ({len(batch)} tasks)"
        )
        batch_results = await asyncio.gather(*batch)
        all_results.extend(batch_results)

    # Organize results by token
    results_by_token = {}
    for result in all_results:
        token_id = result["token_id"]
        if token_id not in results_by_token:
            results_by_token[token_id] = []
        results_by_token[token_id].append(result)

    # Save the results
    data = {
        "timestamp": timestamp,
        "results": all_results,
        "metadata": {
            "models": MODELS,
            "prompt": PROMPT,
            "num_tokens_tested": len(test_tokens),
            "token_ids_tested": test_tokens["token_id"].tolist(),
            "repeats": repeats,
        },
    }

    filename = f"logs/token_translations_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    # Also save as CSV for easier analysis
    csv_filename = f"logs/token_translations_{timestamp}.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_filename, index=False)

    print(f"Data collected and saved to {filename} and {csv_filename}")
    return filename, csv_filename


async def main():
    print("Running token translation test...")

    # Specify token IDs to test
    token_ids = [
        89721,
        129320,
        170421,
        177625,
        185118,
        104937,
        147298,
        160540,
        154809,
        134370,
        115319,
        66799,
        122712,
        49649,
        79695,
        184805,
        199943,
        139863,
        102670,
        114900,
        113720,
        133011,
        104170,
        76500,
        185143,
        69642,
        146082,
        117448,
        88200,
        81699,
        135234,
        170835,
        188700,
        113480,
        199612,
        167732,
        75194,
        149168,
        101877,
        92219,
        153443,
        123560,
        81700,
        101224,
        167551,
        80495,
        69142,
        182867,
        162657,
        39813,
        49860,
        171886,
        82634,
        182584,
        125946,
        140893,
    ]

    filename, csv_filename = await collect_translations(
        token_ids=token_ids, num_tokens=40, repeats=4
    )
    print(f"Test complete. Results saved to {filename} and {csv_filename}")


if __name__ == "__main__":
    asyncio.run(main())
