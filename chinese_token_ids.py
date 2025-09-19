#!/usr/bin/env python3
"""
Script to extract just the token IDs that contain Chinese characters from token_norms_non_ascii.csv
"""


def contains_chinese(text):
    """Check if text contains Chinese characters"""
    for char in text:
        # Check for Chinese character ranges
        if "\u4e00" <= char <= "\u9fff":  # CJK Unified Ideographs
            return True
        if "\u3400" <= char <= "\u4dbf":  # CJK Extension A
            return True
        if "\uf900" <= char <= "\ufaff":  # CJK Compatibility Ideographs
            return True
    return False


def main():
    chinese_token_ids = []

    try:
        with open("data/token_norms_non_ascii.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                token_id = parts[0]
                token = parts[1]

                if contains_chinese(token):
                    chinese_token_ids.append(token_id)

        print("Token IDs containing Chinese characters:")
        print([int(id) for id in chinese_token_ids])

        print(f"\nTotal: {len(chinese_token_ids)} tokens with Chinese characters")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
