#!/usr/bin/env python3
"""
Calculate average scores for Scene Manager evaluation results

Usage:
  python evaluation-framework/calculate_average_scores.py \
    --input /path/to/evaluated.jsonl \
    [--limit N]

Outputs average scores for four metrics:
  - scene_understanding
  - turn_speaker_discipline
  - role_introduction_judgment
  - overall_assessment

Arguments:
  --limit N: Optional, only calculate average scores for the first N valid records
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate average scores for Scene Manager evaluation results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the evaluation results JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional, only calculate average scores for the first N valid records (default: process all records)",
    )
    return parser.parse_args()


def extract_score(record: Dict, key: str) -> float | None:
    """Extract the score for a specified metric from a record"""
    try:
        evaluation = record.get("evaluation", {})
        scores = evaluation.get("scores", {})
        score_obj = scores.get(key, {})
        val = score_obj.get("score", None)
        if isinstance(val, (int, float)):
            return float(val)
    except Exception:
        return None
    return None


def calculate_averages(input_path: str, limit: int | None = None) -> Dict:
    """Calculate average scores

    Args:
        input_path: Path to JSONL file
        limit: Optional, only process the first N valid records (None means process all records)
    """
    metrics = [
        "scene_understanding",
        "turn_speaker_discipline",
        "role_introduction_judgment",
        "overall_assessment",
    ]

    scores_by_metric: Dict[str, List[float]] = {key: [] for key in metrics}
    valid_records = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # If limit is set and reached, stop processing
            if limit is not None and valid_records >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} cannot be parsed as JSON: {e}", file=sys.stderr)
                continue

            # Check if evaluation field exists
            if "evaluation" not in record or not record.get("evaluation"):
                print(f"Warning: Line {line_num} is missing evaluation field, skipping", file=sys.stderr)
                continue

            # Extract scores for each metric
            record_valid = False
            for metric in metrics:
                score = extract_score(record, metric)
                if score is not None:
                    scores_by_metric[metric].append(score)
                    record_valid = True

            if record_valid:
                valid_records += 1

    # Calculate average, standard deviation, minimum and maximum values
    averages = {}
    for metric in metrics:
        scores = scores_by_metric[metric]
        if scores:
            avg = sum(scores) / len(scores)
            # Calculate standard deviation
            if len(scores) > 1:
                variance = sum((x - avg) ** 2 for x in scores) / len(scores)
                std = math.sqrt(variance)
            else:
                std = 0.0
            min_val = min(scores)
            max_val = max(scores)
            # Format as string: average ± std (min:max)
            averages[metric] = f"{avg:.2f} ± {std:.2f} (min:{min_val:.1f} max:{max_val:.1f})"
        else:
            averages[metric] = None

    return {
        "averages": averages,
        "sample_count": valid_records
    }


def main():
    args = parse_args()

    try:
        result = calculate_averages(args.input, limit=args.limit)

        # Output results
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
