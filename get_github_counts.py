import requests
import os
import dotenv
import json
import csv
import time
import sys

dotenv.load_dotenv()


def search_github_token(token, headers):
    """Search GitHub for a token with backoff"""
    url = f"https://api.github.com/search/code?q={token} +in:file"

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()["total_count"]
            elif response.status_code == 403:
                # Rate limited, wait and retry
                if attempt < 2:
                    time.sleep(5**attempt)
                    continue
                else:
                    print(f"Rate limited for token: {token}")
                    return None
            else:
                print(f"Error {response.status_code} for token: {token}")
                return None

        except Exception as e:
            print(f"Exception for token {token}: {e}")
            if attempt < 2:
                time.sleep(1)
            else:
                return None

    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python get_github_counts.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Load unique tokens from CSV
    unique_tokens = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_id = row["token_id"]
            token = row["token"]
            if token_id not in unique_tokens:
                unique_tokens[token_id] = token

    print(f"Found {len(unique_tokens)} unique tokens")

    headers = {"Authorization": f'Token {os.getenv("GITHUB_TOKEN")}'}

    # Search each token
    results = {}
    for i, (token_id, token) in enumerate(unique_tokens.items()):
        print(f"Searching {i+1}/{len(unique_tokens)}: {token}")
        count = search_github_token(token, headers)
        if count is not None:
            results[token_id] = {"token": token, "github_count": count}
        time.sleep(10)  # Small delay between requests

    # Save results
    output_file = f"github_counts_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Successfully processed {len(results)} tokens")


if __name__ == "__main__":
    main()
