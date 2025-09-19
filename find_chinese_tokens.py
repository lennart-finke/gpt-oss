#!/usr/bin/env python3
"""
Script to find token IDs that contain Chinese characters from token_norms_non_ascii.csv
"""

import csv
import re


def contains_chinese(text):
    """
    Check if a string contains Chinese characters.
    Chinese characters are in the Unicode ranges:
    - CJK Unified Ideographs: U+4E00-U+9FFF
    - CJK Extension A: U+3400-U+4DBF
    - CJK Extension B: U+20000-U+2A6DF
    - CJK Extension C: U+2A700-U+2B73F
    - CJK Extension D: U+2B740-U+2B81F
    - CJK Extension E: U+2B820-U+2CEAF
    - CJK Extension F: U+2CEB0-U+2EBEF
    - CJK Compatibility Ideographs: U+F900-U+FAFF
    - CJK Unified Ideographs Extension A: U+3400-U+4DBF
    """
    # Pattern for Chinese characters
    chinese_pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    return bool(chinese_pattern.search(text))


def find_chinese_tokens(csv_file_path):
    """
    Read the CSV file and find all token IDs that contain Chinese characters.
    """
    chinese_token_ids = []

    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                token_id = row["token_id"]
                token = row["token"]

                if contains_chinese(token):
                    chinese_token_ids.append((token_id, token))

        return chinese_token_ids

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def main():
    csv_file_path = "data/token_norms_non_ascii.csv"

    print("Searching for tokens containing Chinese characters...")
    print("=" * 60)

    chinese_tokens = find_chinese_tokens(csv_file_path)

    if chinese_tokens:
        print(f"Found {len(chinese_tokens)} tokens with Chinese characters:")
        print()

        for token_id, token in chinese_tokens:
            print(f"Token ID: {token_id}, Token: '{token}'")
    else:
        print("No tokens with Chinese characters found.")

    print("=" * 60)
    print(f"Total tokens with Chinese characters: {len(chinese_tokens)}")


if __name__ == "__main__":
    main()
