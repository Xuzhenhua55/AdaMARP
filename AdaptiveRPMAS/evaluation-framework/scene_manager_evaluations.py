"""Score Scene Manager transcripts on four 0–10 dimensions.

Dimensions: scene understanding, turn/speaker discipline, role introduction
judgment, and overall assessment. Uses the shared model adapter from main.py.
Run scene_manager_evaluations.sh --help for command-line options.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

# Add parent directory to path to import main
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from main import create_chat_model, resolve_model_credentials

RUBRIC = """
You are evaluating a Scene Manager in a narrative, turn-based roleplay system.

IMPORTANT:
You are judging SYSTEM / ORCHESTRATION DECISIONS ONLY.
Do NOT evaluate prose quality, dialogue quality, creativity, emotional impact,
or character acting. Those are explicitly out of scope.

Your task is to assess how well the system manages scenes, turns, and roles
as a narrative orchestrator.

Scoring should be STRICT and CONSERVATIVE:
- 5 is an average / merely acceptable performance.
- 7–8 requires consistently good judgment with only minor issues.
- 9–10 is reserved for exceptional, disciplined orchestration with no
  meaningful errors or lapses.
- Scores below 5 indicate clear weaknesses or recurring problems.
Do NOT inflate scores.

=== SCORING AXES ===

1) Scene Understanding (0–10)
Evaluate whether the system correctly understands and manages the scene.

Consider:
- Correctly distinguishing major scene transitions vs. minor in-scene shifts.
- Avoiding premature scene switches when dialogue only *discusses* or *proposes*
  a possible change.
- Detecting when a scene has naturally concluded and should end.
- Tracking the current scene’s theme, stakes, tension, and goals instead of
  pushing arbitrary progression.
- When switch_scene is used:
  - The new_scene must be clearly described.
  - The transition must be causally justified by prior events.
  - The timing must feel narratively earned, not forced.

Scoring guidance:
- 0–3: Fundamental misunderstanding of scene state or repeated premature switches.
- 4–5: Basic understanding, but notable misreads or blunt transitions.
- 6–7: Generally correct with minor timing or clarity issues.
- 8–10: Precise, patient, and context-aware scene control with strong causal logic.

This is the CORE axis. Errors here should heavily impact the score.

---

2A) Turn & Speaker Selection Discipline (0–10)
Evaluate how well the system manages turn order and speaker selection.

Strictly assess:
- No same speaker chosen in consecutive turns.
- Avoidance of long NPC-only exchanges.
- Proactive inclusion of the user in the dialogue rotation.
- Prevention of any single role monopolizing the scene.
- User is not excluded for excessive rounds (≈3–4 turns).
- The selected speaker must be a valid, existing role.
- The stated reason for selecting the next speaker must be coherent,
  contextually appropriate, and justified.

Scoring guidance:
- 0–3: Clear violations (speaker repetition, user sidelined, invalid roles).
- 4–5: Minimal compliance, but passive or uneven turn management.
- 6–7: Mostly disciplined with small lapses.
- 8–10: Actively balanced, intentional, and fair turn orchestration.

Focus on interaction fairness and structural discipline, NOT dramatic effect.

---

2B) Role Introduction & Utilization Judgment (0–10)
Evaluate decisions to introduce (or not introduce) new roles.

Assess:
- New roles are added only when interaction is clearly required.
- add_role occurs at an appropriate narrative moment.
- The role’s profile, motivation, and function serve scene dynamics or plot movement.
- The provided reason for adding the role is specific and situationally grounded.
- Missed opportunities where a role *should* have been introduced are penalized.
- Avoidance of redundant, inert, or decorative roles.

Scoring guidance:
- 0–3: Arbitrary, unnecessary, or poorly justified role additions.
- 4–5: Technically acceptable but narratively weak or cautious to a fault.
- 6–7: Generally sound judgment with occasional hesitation or excess.
- 8–10: Precise, economical, and clearly purposeful role management.

---

=== OVERALL ASSESSMENT (WITH SCORE) ===
Provide a holistic evaluation of the system’s orchestration quality.

Important constraints:
- This is NOT a simple average of the above scores.
- The overall score should reflect how well the three axes work together
  to maintain coherent pacing, user agency, and narrative control.
- Significant failure in any single axis should cap the overall score,
  even if other axes perform well.
- Do NOT introduce new evaluation criteria.
- Do NOT override or “forgive” clear failures identified above.

Assess:
- Whether scene control, turn discipline, and role usage reinforce each other
  or work at cross-purposes.
- Whether the system maintains momentum without sacrificing user agency.
- Whether orchestration decisions feel intentional rather than reactive.

Scoring guidance:
- 0–3: Systemic breakdown or compounding orchestration failures.
- 4–5: Functional but disjointed or inconsistent control.
- 6–7: Coherent overall flow with noticeable but contained issues.
- 8–10: Highly disciplined, deliberate, and well-integrated orchestration.

---

=== OUTPUT FORMAT ===
Return your evaluation in the following JSON format:

{
  "scene_understanding": {
    "score": 0-10,
    "reasoning": "Concise, bullet-style justification tied directly to the criteria."
  },
  "turn_speaker_discipline": {
    "score": 0-10,
    "reasoning": "Concise, bullet-style justification tied directly to the criteria."
  },
  "role_introduction_judgment": {
    "score": 0-10,
    "reasoning": "Concise, bullet-style justification tied directly to the criteria."
  },
  "overall_assessment": {
    "score": 0-10,
    "reasoning": "1–2 sentences explaining how the three axes interact systemically."
  }
}

REMINDERS:
- Judge SYSTEM DECISIONS, not story quality.
- Penalize premature scene switches, turn violations, unjustified role additions,
  and loss of user agency.
- High scores must be earned through consistent, disciplined orchestration.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based evaluation for Scene Manager transcripts.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more log file paths or glob patterns (json/jsonl). Directories are expanded to *.json and *.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-5.2",
        help="LLM judge model spec (e.g., openai:gpt-5.2, anthropic:claude-3-5-sonnet-20241022, local:...).",
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "adaptiveRPMAS-config-en.json"),
        help="Path to config containing API keys/base URLs.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Override API key (else falls back to config or env based on provider).",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Override base URL for OpenAI-compatible endpoints (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature (default 0.0 for stable scoring).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path to append evaluation results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process from input files (processes first N records).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent evaluation tasks (default: 1, sequential processing).",
    )
    return parser.parse_args()


def expand_log_paths(raw_paths: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for path in raw_paths:
        if os.path.isdir(path):
            expanded.extend(glob.glob(os.path.join(path, "*.json")))
            expanded.extend(glob.glob(os.path.join(path, "*.jsonl")))
        else:
            globbed = glob.glob(path)
            expanded.extend(globbed if globbed else [path])
    return sorted(list(dict.fromkeys(expanded)))


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dialogue_context(log_data: Dict) -> str:
    main_char = log_data.get("roles", {}).get("main_character", {})
    main_name = main_char.get("name", "").strip()
    main_profile = main_char.get("profile", "").strip()
    main_motivation = main_char.get("motivation", "").strip()

    main_char_text = ""
    if main_name:
        main_char_text = f"{main_name}:{main_profile}"
        if main_motivation:
            main_char_text += f"\n{main_name}'s motivation:{main_motivation}"

    other_roles = log_data.get("roles", {}).get("other_characters", []) or []
    other_char_texts = []
    for role in other_roles:
        role_name = role.get("name", "").strip()
        role_profile = role.get("profile", "").strip()
        role_motivation = role.get("motivation", "").strip()
        if role_name:
            role_text = f"{role_name}:{role_profile}"
            if role_motivation:
                role_text += f"\n{role_name}'s motivation:{role_motivation}"
            other_char_texts.append(role_text)

    other_profiles = "\n".join(other_char_texts)

    dialogue_lines = []
    for msg in log_data.get("messages", []):
        role = msg.get("role") or msg.get("name") or "未知角色"
        content = (msg.get("content") or "").strip()
        dialogue_lines.append(f"{role}：{content}")
    dialogue_text = "\n".join(dialogue_lines)

    sections = [
        "===Main Character===",
        main_char_text,
        "===Information about the other Characters===",
        other_profiles,
        "===Dialogue History===",
        dialogue_text,
    ]
    return "\n".join([s for s in sections if s])


def build_messages(context_text: str) -> List:
    return [
        SystemMessage(content=RUBRIC),
        HumanMessage(content=context_text),
    ]


def read_log_records(log_path: str) -> List[Tuple[Dict, str]]:
    """Read a log file that may contain a single JSON object or JSONL with one object per line.

    Returns:
        List of tuples: (parsed_dict, original_line) for JSONL, or (parsed_dict, None) for JSON.
    """
    records: List[Tuple[Dict, str]] = []

    # Try JSONL first (most common case - each line is a session)
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Try parsing as JSONL (each line is a JSON object)
    jsonl_success = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append((obj, line))
                jsonl_success = True
        except json.JSONDecodeError:
            continue

    # If JSONL parsing succeeded, return results
    if jsonl_success:
        return records

    # Fallback to single JSON object (for backward compatibility)
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return [(loaded, None)]
        if isinstance(loaded, list):
            return [(item, None) for item in loaded if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    if not records:
        raise ValueError(f"Could not parse any JSON objects from {log_path}")
    return records


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code blocks or plain text.

    Handles cases where LLM wraps JSON in ```json ... ``` blocks.
    """
    content = content.strip()

    # Try to extract JSON from markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    # Match ```json ... ``` or ``` ... ```
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block found, try to find JSON object boundaries
    # Look for first { and try to extract from there
    first_brace = content.find('{')
    if first_brace >= 0:
        # Try to find the matching closing brace
        brace_count = 0
        for i in range(first_brace, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[first_brace:i+1]
        # If no closing brace found, return from first brace to end (might be truncated)
        return content[first_brace:]

    # If no JSON structure found, return original content
    return content


def evaluate_one(model, log_data: Dict) -> Tuple[Dict, str, Dict]:
    context_text = build_dialogue_context(log_data)
    messages = build_messages(context_text)

    # ChatModelWrapper.invoke() expects a state dict with "messages" key
    state = {"messages": messages}
    response = model.invoke(state)
    content = getattr(response, "content", "") or ""

    parsed = {}
    try:
        # First try parsing as-is
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        try:
            cleaned_content = extract_json_from_markdown(content)
            parsed = json.loads(cleaned_content)
        except json.JSONDecodeError:
            parsed = {"parse_error": "LLM response was not valid JSON", "raw": content}

    # Extract token usage information
    # ChatModelWrapper.invoke() returns an object with .usage attribute
    usage_info = {}
    if hasattr(response, "usage") and response.usage:
        usage_info = response.usage
        # Convert to dict if it's an object
        if not isinstance(usage_info, dict):
            usage_info = {
                "prompt_tokens": getattr(usage_info, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_info, "completion_tokens", 0),
                "total_tokens": getattr(usage_info, "total_tokens", 0),
            }
    elif hasattr(response, "response_metadata") and response.response_metadata:
        usage_info = response.response_metadata.get("token_usage", {})

    return parsed, content, usage_info


def extract_score(parsed: Dict, key: str) -> float | None:
    try:
        val = parsed.get(key, {}).get("score", None)
        if isinstance(val, (int, float)):
            return float(val)
    except Exception:
        return None
    return None


def validate_scores(parsed: Dict) -> bool:
    """Validate that all required scores are successfully parsed.

    Returns:
        True if all required scores are valid, False otherwise.
    """
    # Check if there's a parse error
    if "parse_error" in parsed:
        return False

    # Required score keys
    required_keys = [
        "scene_understanding",
        "turn_speaker_discipline",
        "role_introduction_judgment",
        "overall_assessment",
    ]

    # Check that all required scores exist and are valid numbers
    for key in required_keys:
        score = extract_score(parsed, key)
        if score is None:
            return False

    return True


def load_existing_records(output_path: str) -> Dict[str, Dict]:
    """Load all records from the output file if it exists, indexed by UUID."""
    existing_records = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "uuid" in record and record["uuid"]:
                            existing_records[record["uuid"]] = record
                    except json.JSONDecodeError:
                        continue
        except Exception:
            # If file exists but can't be read, return empty dict
            pass
    return existing_records


# Thread-safe file writing lock
_file_write_lock = threading.Lock()

def append_jsonl(path: str, record: Dict) -> None:
    """Thread-safe JSONL append operation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _file_write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_one_record(
    log_obj: Dict,
    judge,
    args,
    existing_records_by_uuid: Dict[str, Dict],
) -> Tuple[Dict | None, Dict, Dict]:
    """Process a single log record for evaluation.

    Returns:
        Tuple of (record, parsed_scores, token_usage)
        Returns (None, {}, {}) if the record should be skipped.
    """
    uuid_value = log_obj.get("uuid")

    # Check UUID to avoid duplicate evaluations
    if uuid_value and uuid_value in existing_records_by_uuid:
        # UUID already exists in output file, skip evaluation and use existing record
        record = existing_records_by_uuid[uuid_value]
        existing_eval = record.get("evaluation", {})
        parsed = existing_eval.get("scores", {})

        # Validate scores - skip if parsing failed
        if not validate_scores(parsed):
            uuid_str = uuid_value or "unknown"
            raw_response = existing_eval.get("raw_response", "")
            print(f"Skipping record with UUID {uuid_str}: score parsing failed", file=sys.stderr)
            print(f"Raw output for UUID {uuid_str}:", file=sys.stderr)
            print(f"{raw_response}", file=sys.stderr)
            print(f"Parsed content: {json.dumps(parsed, ensure_ascii=False, indent=2)}", file=sys.stderr)
            return None, {}, {}

        # Extract token usage from existing evaluation if available
        usage_info = existing_eval.get("token_usage", {})
        if usage_info:
            prompt_tokens = usage_info.get("prompt_tokens", 0) or usage_info.get("input_tokens", 0) or 0
            completion_tokens = usage_info.get("completion_tokens", 0) or usage_info.get("output_tokens", 0) or 0
            total_tokens = usage_info.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        else:
            usage_info = {}

        return record, parsed, usage_info

    # Need to evaluate
    parsed, raw, usage_info = evaluate_one(judge, log_obj)

    # Validate scores - skip if parsing failed
    if not validate_scores(parsed):
        uuid_str = log_obj.get("uuid", "unknown")
        print(f"Skipping record with UUID {uuid_str}: score parsing failed", file=sys.stderr)
        print(f"Raw output for UUID {uuid_str}:", file=sys.stderr)
        print(f"{raw}", file=sys.stderr)
        print(f"Parsed content: {json.dumps(parsed, ensure_ascii=False, indent=2)}", file=sys.stderr)
        return None, {}, {}

    # Build output record: preserve original data and add evaluation results
    record = log_obj.copy()

    # Add evaluation results
    record["evaluation"] = {
        "judge_model": args.model,
        "scores": parsed,
        "raw_response": raw,
    }

    # Add token usage to evaluation if available
    if usage_info:
        record["evaluation"]["token_usage"] = usage_info

    return record, parsed, usage_info


def main():
    args = parse_args()
    log_paths = expand_log_paths(args.logs)
    if not log_paths:
        raise FileNotFoundError("No log files found for the given --logs patterns.")

    config = load_config(args.config)
    api_key, base_url = resolve_model_credentials(args.model, config, args.api_key, args.base_url)
    judge = create_chat_model(
        args.model,
        generation_config={
            "temperature": args.temperature,
            "max_new_tokens": 10240,  # Allow room for scores and justifications
        },
        api_key=api_key,
        base_url=base_url,
    )

    # Thread-safe shared data structures
    all_scores = {
        "scene_understanding": [],
        "turn_speaker_discipline": [],
        "role_introduction_judgment": [],
        "overall_assessment": [],
    }
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    stats_lock = threading.Lock()

    # Load existing records from output file to avoid duplicate evaluations
    existing_records_by_uuid = {}
    if args.output:
        existing_records_by_uuid = load_existing_records(args.output)

    # Collect all records to process
    all_records: List[Tuple[Dict, str]] = []
    for path in log_paths:
        records = read_log_records(path)
        all_records.extend(records)
        if args.limit is not None and len(all_records) >= args.limit:
            all_records = all_records[:args.limit]
            break

    # Apply limit if specified
    if args.limit is not None:
        all_records = all_records[:args.limit]

    # Process records concurrently or sequentially
    if args.concurrency > 1:
        # Concurrent processing
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(
                    process_one_record,
                    log_obj,
                    judge,
                    args,
                    existing_records_by_uuid,
                ): (log_obj, original_line)
                for log_obj, original_line in all_records
            }

            # Process completed tasks
            for future in as_completed(future_to_record):
                log_obj, original_line = future_to_record[future]
                try:
                    record, parsed, usage_info = future.result()

                    # Skip if record was None (validation failed)
                    if record is None:
                        continue

                    # Print result
                    print(json.dumps(record, ensure_ascii=False, indent=2))

                    # Write to output file (thread-safe)
                    if args.output:
                        append_jsonl(args.output, record)

                    # Accumulate statistics (thread-safe)
                    with stats_lock:
                        # Accumulate scores
                        for key in all_scores.keys():
                            val = extract_score(parsed, key)
                            if val is not None:
                                all_scores[key].append(val)

                        # Accumulate token usage
                        if usage_info:
                            total_token_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                            total_token_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
                            total_token_usage["total_tokens"] += usage_info.get("total_tokens", 0)

                except Exception as exc:
                    uuid_str = log_obj.get("uuid", "unknown")
                    print(f"Record with UUID {uuid_str} generated an exception: {exc}", file=sys.stderr)
    else:
        # Sequential processing (original behavior)
        for log_obj, original_line in all_records:
            record, parsed, usage_info = process_one_record(
                log_obj,
                judge,
                args,
                existing_records_by_uuid,
            )

            # Skip if record was None (validation failed)
            if record is None:
                continue

            # Print result
            print(json.dumps(record, ensure_ascii=False, indent=2))

            # Write to output file
            if args.output:
                append_jsonl(args.output, record)

            # Accumulate statistics
            for key in all_scores.keys():
                val = extract_score(parsed, key)
                if val is not None:
                    all_scores[key].append(val)

            # Accumulate token usage
            if usage_info:
                total_token_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += usage_info.get("total_tokens", 0)

    # Print averages and token usage summary
    avg = {}
    for key, vals in all_scores.items():
        if vals:
            avg[key] = sum(vals) / len(vals)

    summary = {}
    if avg:
        summary["average_scores"] = avg
        sample_count = len(next(iter(all_scores.values())) or [])
        summary["sample_count"] = sample_count

    # Add token usage summary
    if total_token_usage["total_tokens"] > 0:
        summary["total_token_usage"] = total_token_usage

    if summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
