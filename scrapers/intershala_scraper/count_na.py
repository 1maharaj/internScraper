import json
import sys
from collections import defaultdict


def count_na_per_key(filepath: str) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON file must contain an array of objects.")
        sys.exit(1)

    total = len(data)
    na_counts = defaultdict(int)

    for item in data:
        for key, value in item.items():
            if value == "N/A":
                na_counts[key] += 1

    print(f"Total records: {total}\n")
    print(f"{'Key':<30} {'N/A Count':>10} {'% of total':>12}")
    print("-" * 55)

    for key, count in sorted(na_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        print(f"{key:<30} {count:>10} {pct:>11.1f}%")

    if not na_counts:
        print("No 'N/A' values found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_na.py <path_to_json_file>")
        sys.exit(1)

    count_na_per_key(sys.argv[1])
