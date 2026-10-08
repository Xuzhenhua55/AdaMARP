#!/usr/bin/env python3
"""
Calculate average scores for Actor Model evaluation results

Usage:
  python evaluation-framework/calculate_average_scores_for_actor_models.py \
    --input /path/to/evaluated.jsonl \
    [--limit N]

Outputs average scores for multiple metrics:
  - Character Consistency (5 metrics)
  - Environmental Grounding (2 metrics)
  - Interpersonal Interaction (2 metrics)
  - Narrative Progression (2 metrics)
  - Instruction Compliance (1 metric)

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
        description="Calculate average scores for Actor Model evaluation results"
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


def extract_score(record: Dict, path: List[str]) -> float | None:
    """Extract the score for a specified metric from a record (supports nested paths)

    Args:
        record: JSON record
        path: Path list, e.g., ["character_consistency", "internal_coherence"] or ["instruction_compliance"]

    Returns:
        Score value (float) or None
    """
    try:
        evaluation = record.get("evaluation", {})
        scores = evaluation.get("scores", {})

        # Navigate to the target location
        current = scores
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None

        # At the final location, should be a dict containing a "score" key, or directly a score value
        if isinstance(current, dict):
            val = current.get("score")
            if isinstance(val, (int, float)):
                return float(val)
        elif isinstance(current, (int, float)):
            return float(current)
    except Exception:
        return None
    return None


def calculate_averages(input_path: str, limit: int | None = None) -> Dict:
    """Calculate average scores

    Args:
        input_path: Path to JSONL file
        limit: Optional, only process the first N valid records (None means process all records)
    """
    # Define all metrics and their paths (corresponds to metric_paths in actor_model_evaluation.py)
    metric_paths = {
        "CC_internal_coherence": ["character_consistency", "internal_coherence"],
        "CC_speaking_style": ["character_consistency", "speaking_style_fidelity"],
        "CC_language_fluency": ["character_consistency", "language_fluency"],
        "CC_identity_fidelity": ["character_consistency", "identity_fidelity"],
        "CC_motivation_stability": ["character_consistency", "motivation_stability"],
        "EG_awareness": ["environmental_grounding", "environmental_awareness"],
        "EG_utilization": ["environmental_grounding", "environmental_utilization"],
        "II_contextual_responsiveness": ["interpersonal_interaction", "contextual_responsiveness"],
        "II_relationship_awareness": ["interpersonal_interaction", "relationship_awareness"],
        "NP_narrative_attractiveness": ["narrative_progression", "narrative_attractiveness"],
        "NP_stability_over_time": ["narrative_progression", "stability_over_time"],
        "Instruction_Compliance": ["instruction_compliance"],
    }

    scores_by_metric: Dict[str, List[float]] = {key: [] for key in metric_paths.keys()}
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
            for metric, path in metric_paths.items():
                score = extract_score(record, path)
                if score is not None:
                    scores_by_metric[metric].append(score)
                    record_valid = True

            if record_valid:
                valid_records += 1

    # Calculate average, standard deviation, minimum and maximum values
    averages = {}
    metric_average_values = []  # Store average values for all metrics (numeric)
    for metric in metric_paths.keys():
        scores = scores_by_metric[metric]
        if scores:
            avg = sum(scores) / len(scores)
            metric_average_values.append(avg)  # Save raw average value
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

    # Calculate the average of all metric averages
    overall_average = None
    if metric_average_values:
        overall_average = sum(metric_average_values) / len(metric_average_values)

    result = {
        "averages": averages,
        "sample_count": valid_records
    }

    # Add overall average
    if overall_average is not None:
        result["average"] = round(overall_average, 2)

    return result


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
