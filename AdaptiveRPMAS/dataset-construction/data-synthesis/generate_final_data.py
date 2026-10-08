#!/usr/bin/env python3
"""
Generate role-playing training and evaluation data from samples_deduplicated jsonl files.
- Training data: format {"messages": [{"role": "system", "content": "..."}, ...]}
- Evaluation data: format {"messages": [...], "main_character": {...}, "other_characters": [...]}
- Evaluation samples: last 5 samples from each category
- Training samples: remaining samples (no overlap with evaluation)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import uuid

def build_system_prompt(main_profile: Dict[str, Any], other_characters: List[Dict[str, Any]]) -> str:
    """Build the system prompt from main profile and other characters."""

    # Main Character section
    main_section = f"""===Main Character===
Name:{main_profile.get('name', '')}
Identity&Appearance:{main_profile.get('identity_appearance', '')}
Personality&Psychology:{main_profile.get('personality_psychology', '')}
Speaking Style:{main_profile.get('speaking_style', '')}
Abilities&Interests&Achievements:{main_profile.get('abilities_interests_achievements', '')}
Social Historical Context:{main_profile.get('social_historical_context', '')}
Personal History Arc:{main_profile.get('personal_history_arc', '')}
Relationships:{main_profile.get('relationships', '')}
Motivation:{main_profile.get('motivation', '')}
"""

    # Other Characters section
    other_section = "\n===Information about the other Characters===\n"
    for char in other_characters:
        char_name = char.get('name', '')
        # Remove (user) marker if present in the name
        char_name_clean = char_name.replace("(user)", "").replace(" (user)", "").strip()
        if not char_name_clean:
            char_name_clean = char_name  # Fallback to original if stripping removes everything
        char_profile = char.get('profile', '')
        char_motivation = char.get('motivation', '')
        other_section += f"{char_name_clean}:{char_profile}\n{char_name_clean}'s motivation:{char_motivation}\n"

    # Instructions section
    instructions_section = """
===Instructions for roleplaying===
- Your output should include **thought**, **speech**, **action**, and **environmental changes**.
- Use **[thought]** for inner thoughts that others cannot see.
- Use **(action)** for visible actions.
- Use **<environmental changes>** to describe changes in the surroundings.
- These elements can be freely interwoven in any order to support narrative flow.
- Only generate the designated character's own thoughts, speech, and actions.
- Do not write dialogue, thoughts, or actions for other characters.
- The character may reference others only within their own thoughts or spoken lines.
- Stay in character and keep responses concise.
- Limit spoken dialogue to 1–2 sentences per turn, each on a single line.
"""

    return main_section + other_section + instructions_section

def _normalize_name(name: str) -> str:
    """
    Normalize a character name for comparison.
    - Lowercase
    - Strip spaces
    - Remove common decorations like '(user)'
    - Remove leading titles like 'captain '
    """
    if not name:
        return ""
    n = name.strip().lower()
    # remove explicit user markers
    n = n.replace("(user)", "").replace(" (user)", "").strip()
    # strip a few common titles
    for prefix in ["captain ", "dr. ", "dr ", "professor ", "mr. ", "mr ", "mrs. ", "mrs ", "ms. ", "ms "]:
        if n.startswith(prefix):
            n = n[len(prefix):].strip()
            break
    return n


def _is_main_character_role(speaker_role: str, main_character_name: str) -> bool:
    """
    Decide whether a given speaker role corresponds to the main character.
    Uses relaxed matching so that e.g. 'Elara Vance' matches 'Captain Elara Vance'.
    """
    speaker_norm = _normalize_name(speaker_role)
    main_norm = _normalize_name(main_character_name)

    if not speaker_norm or not main_norm:
        return False

    if speaker_norm == main_norm:
        return True

    # substring match to handle cases like 'Elara Vance' vs 'Captain Elara Vance'
    if speaker_norm in main_norm or main_norm in speaker_norm:
        return True

    return False


def filter_and_convert_messages(
    messages: List[Dict[str, str]],
    main_character_name: str,
) -> List[Dict[str, str]]:
    """
    Convert raw messages to the final training format:
    - Main character utterances -> role 'assistant'
    - Other characters & scene_manager -> role 'user'
    - content -> 'role_name:original_content'
    """
    filtered_messages: List[Dict[str, str]] = []

    for msg in messages:
        speaker = msg.get("role", "")
        content = msg.get("content", "")

        # Skip empty contents
        if not content:
            continue

        if _is_main_character_role(speaker, main_character_name):
            role = "assistant"
        else:
            # All non-main-character speakers (including scene_manager) are treated as user
            role = "user"

        content_with_name = f"{speaker}:{content}"

        filtered_messages.append(
            {
                "role": role,
                "content": content_with_name,
            }
        )

    return filtered_messages

def convert_to_evaluation_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sample data to evaluation format."""
    main_profile = data.get('main_profile', {})
    other_characters = data.get('other_characters', [])
    messages = data.get('messages', [])
    relations = data.get('relations', '')

    # Build main_character profile by combining all fields
    profile_parts = []
    if main_profile.get('identity_appearance'):
        profile_parts.append(main_profile.get('identity_appearance'))
    if main_profile.get('personality_psychology'):
        profile_parts.append(main_profile.get('personality_psychology'))
    if main_profile.get('speaking_style'):
        profile_parts.append(main_profile.get('speaking_style'))
    if main_profile.get('abilities_interests_achievements'):
        profile_parts.append(main_profile.get('abilities_interests_achievements'))
    if main_profile.get('social_historical_context'):
        profile_parts.append(main_profile.get('social_historical_context'))
    if main_profile.get('personal_history_arc'):
        profile_parts.append(main_profile.get('personal_history_arc'))
    if main_profile.get('relationships'):
        profile_parts.append(main_profile.get('relationships'))

    main_character = {
        "name": main_profile.get('name', ''),
        "profile": '\n'.join(profile_parts),
        "motivation": main_profile.get('motivation', '')
    }

    # Build other_characters
    eval_other_characters = []
    for char in other_characters:
        char_name = char.get('name', '')
        eval_other_characters.append({
            "name": char_name,
            "profile": char.get('profile', ''),
            "motivation": char.get('motivation', '')
        })

    # Build evaluation format
    # Only keep the first message for evaluation
    eval_sample: Dict[str, Any] = {
        "id": uuid.uuid4().hex,  # UUID as index/id for this evaluation sample
        "messages": messages[:1] if messages else [],  # Keep only the first message
        "main_character": main_character,
        "other_characters": eval_other_characters,
    }

    if relations:
        eval_sample["relations"] = relations

    return eval_sample

def process_jsonl_file(input_file: str, eval_samples_per_file: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single jsonl file and return training samples and evaluation samples.
    Last eval_samples_per_file samples go to evaluation, rest go to training.
    """
    all_raw_data = []

    # Read all data first
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                all_raw_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num} of {input_file}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num} of {input_file}: {e}")
                continue

    # Split: last eval_samples_per_file for evaluation, rest for training
    eval_raw_data = all_raw_data[-eval_samples_per_file:]
    training_raw_data = all_raw_data[:-eval_samples_per_file]

    # Process evaluation samples
    eval_samples = []
    for i, raw_data in enumerate(eval_raw_data):
        try:
            eval_sample = convert_to_evaluation_format(raw_data)
            eval_samples.append(eval_sample)
        except Exception as e:
            print(f"Error converting evaluation sample {i+1} from {input_file}: {e}")
            continue

    # Process training samples
    training_samples = []
    for raw_data in training_raw_data:
        try:
            # Extract data
            main_profile = raw_data.get('main_profile', {})
            other_characters = raw_data.get('other_characters', [])
            messages = raw_data.get('messages', [])

            # Build system prompt
            system_prompt = build_system_prompt(main_profile, other_characters)

            # Filter and convert messages
            main_character_name = main_profile.get('name', '')
            filtered_messages = filter_and_convert_messages(
                messages, main_character_name
            )

            # Skip if no valid messages after filtering
            if not filtered_messages:
                continue

            # Build the training sample
            training_sample: Dict[str, Any] = {
                "id": uuid.uuid4().hex,  # UUID as index/id for this training sample
                "messages": [
                    {"role": "system", "content": system_prompt}
                ] + filtered_messages,
            }

            training_samples.append(training_sample)

        except Exception as e:
            print(f"Error converting training sample from {input_file}: {e}")
            continue

    return training_samples, eval_samples

def main():
    """Main function to process all jsonl files."""
    # Base path
    BASE_DIR = Path(__file__).resolve().parents[2]
    input_dir = str(BASE_DIR / "dataset-construction/data-synthesis/synthesis-data")
    training_dir = str(BASE_DIR / "dataset_analysis/final_data/training_data")
    eval_dir = str(BASE_DIR / "dataset_analysis/final_data/evaluation_data")
    eval_samples_per_file = 5

    # Create output directories
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Find all jsonl files
    input_path = Path(input_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))

    # If samples_deduplicated is empty, try samples directory
    if not jsonl_files:
        input_dir = str(BASE_DIR / "dataset-construction/samples")
        input_path = Path(input_dir)
        jsonl_files = list(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No jsonl files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} jsonl files to process")

    # Process each file and collect all samples
    all_training_samples = []
    all_eval_samples = []

    for jsonl_file in jsonl_files:
        training_samples, eval_samples = process_jsonl_file(str(jsonl_file), eval_samples_per_file)

        # Write individual training file
        if training_samples:
            training_output_file = os.path.join(training_dir, jsonl_file.name)
            with open(training_output_file, 'w', encoding='utf-8') as f:
                for sample in training_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Processed {jsonl_file.name}: {len(training_samples)} training samples, {len(eval_samples)} evaluation samples")

        # Write individual evaluation file
        if eval_samples:
            eval_output_file = os.path.join(eval_dir, jsonl_file.name)
            with open(eval_output_file, 'w', encoding='utf-8') as f:
                for sample in eval_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        all_training_samples.extend(training_samples)
        all_eval_samples.extend(eval_samples)

    # Write merged training file
    merged_training_file = os.path.join(training_dir, "AdaRPSet-Synthesis.jsonl")
    with open(merged_training_file, 'w', encoding='utf-8') as f:
        for sample in all_training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Write merged evaluation file
    merged_eval_file = os.path.join(eval_dir, "AdaBenchEvaluate.jsonl")
    with open(merged_eval_file, 'w', encoding='utf-8') as f:
        for sample in all_eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nAll files processed.")
    print(f"Training data saved to {training_dir}")
    print(f"  - Merged training file: {merged_training_file} with {len(all_training_samples)} total samples")
    print(f"Evaluation data saved to {eval_dir}")
    print(f"  - Merged evaluation file: {merged_eval_file} with {len(all_eval_samples)} total samples")
    print(f"\nNote: Evaluation samples (last {eval_samples_per_file} per file) and training samples (remaining) do not overlap.")

if __name__ == "__main__":
    main()
