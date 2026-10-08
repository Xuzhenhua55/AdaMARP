import argparse
import json
import os
import re
import time
import logging
import random
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils import get_response


BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name, log_file, level=logging.INFO, quiet=False):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger(__name__, os.path.join(LOG_DIR, f"{Path(__file__).stem}.log"))


TOPIC_DESCRIPTIONS: Dict[str, str] = {
    "Adventure": "Characters embark on a journey, facing challenges and discovering new places.",
    "Quest": "A specific mission or goal drives the characters’ actions.",
    "Rescue": "Characters must save someone or something in danger.",
    "Battle": "Conflict escalates into a physical or magical confrontation.",
    "Escape": "Characters attempt to flee from a dangerous situation.",
    "Exploration": "Discovering unknown territories, objects, or secrets.",
    "Mystery": "Unexplained phenomena or events spark investigation.",
    "Investigation": "Characters collect clues and analyze information to solve a case.",
    "Crime-solving": "Characters work together to uncover a criminal or culprit.",
    "Puzzle-solving": "Solving riddles, codes, or logical challenges.",
    "Conspiracy": "Hidden schemes or secrets are uncovered gradually.",
    "Romance": "Characters explore feelings of love, attraction, or affection.",
    "Friendship": "Building trust, bonds, and camaraderie between characters.",
    "Rivalry": "Competing interests or personalities create tension.",
    "Betrayal": "Trust is broken, and hidden motives are revealed.",
    "Reconciliation": "Conflicts are resolved, misunderstandings cleared, relationships repaired.",
    "Negotiation": "Characters attempt to reach agreements or compromises.",
    "Strategy": "Planning, scheming, or discussing complex plans for a goal.",
    "Magic": "Supernatural powers or magical phenomena influence events.",
    "Apocalypse": "Characters face large-scale disasters or catastrophic events.",
}


def extract_json(text, **kwargs):
    """Extract JSON from text response."""
    def parse_json_safely(text):
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            results = []
            start = 0
            while start < len(text):
                try:
                    obj, end = json.JSONDecoder().raw_decode(text[start:])
                    results.append(obj)
                    start += end
                except json.JSONDecodeError:
                    start += 1

            if results:
                longest_json = max(results, key=lambda x: len(json.dumps(x)))
                return longest_json
            else:
                return None

    # Replace newlines in strings to avoid parsing issues
    text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text)

    extracted_json = parse_json_safely(text)

    if extracted_json:
        return extracted_json
    else:
        logger.error(f'Error parsing response: {text[:500]}')
        return None


def get_response_json(model: str, messages: List[Dict], max_retry: int = 5, **kwargs):
    """
    Get and process a response from LLM with retries and JSON extraction.

    Args:
        model: Model name (e.g., "openai:gpt-4o-mini", "openai:gpt-4", or others)
        messages: List of message dicts for the LLM
        max_retry: Max number of retry attempts (default 5)
        **kwargs: Additional arguments passed to underlying LLM caller

    Returns:
        dict: Processed JSON response from the LLM, or error dict if parsing fails
    """
    nth_generation = 0

    while nth_generation < max_retry:
        logger.info(f'{nth_generation}th generation')
        response = get_response(
            model=model,
            messages=messages,
            nth_generation=nth_generation,
            return_raw_response=True,
            max_retry=1,  # retries handled in outer loop
            **kwargs,
        )

        if response is None:
            nth_generation += 1
            continue

        # Extract content and usage info
        content = response.get("content", "") if isinstance(response, dict) else response
        usage_info = response.get("usage") if isinstance(response, dict) else None

        # Log token usage if available
        if usage_info:
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)
            logger.info(f'Token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}')
        else:
            logger.warning('Token usage information not available in API response')

        logger.info(f'response by LLM: {content[:200] if content else None}...')

        # Extract JSON from response content
        json_response = extract_json(content, **kwargs)

        if json_response:
            # Add usage info to the response if available
            if usage_info:
                json_response["_token_usage"] = usage_info
            return json_response
        else:
            nth_generation += 1
            if nth_generation < max_retry:
                time.sleep(1)

    logger.error(f"Failed to get valid JSON response after {max_retry} attempts")
    error_response = {"error": "failed_to_parse_json"}
    if isinstance(response, dict) and "content" in response:
        error_response["raw_response"] = response["content"]
    elif response:
        error_response["raw_response"] = response
    return error_response


def build_synthesis_prompt(
    dialogue_topic: str,
    topic_description: Optional[str] = None,
) -> str:
    """
    Build the prompt used to synthesize a single multi-character role-play session.

    Target output (JSON structure for a single sample):
    {
      "main_profile": { ... main character profile ... },
      "other_characters": [
        {"name": "...", "profile": "...", "motivation": "..."},
        ...
      ],
      "relations": "...",
      "messages": [
        {"role": "scene_manager", "content": "..."},
        {"role": "MainCharacterName", "content": "..."},
        ...
      ]
    }
    """
    # Do not ask the model to explain field meanings; encode them in the instruction
    # so that the output is clean and directly usable as training data.
    if topic_description is None:
        topic_description = TOPIC_DESCRIPTIONS.get(
            dialogue_topic,
            "A role-play conversation aligned with the chosen dialogue topic.",
        )

    prompt_template = """
    You are going to **synthesize one training sample for a multi-character role-play system**.
    **Important**: All output content must be in **English**. Follow the semantics and JSON output format below **exactly**.

    ==================== Topic Selector (Filled by User) ====================
    **Dialogue Topic**: {{DIALOGUE_TOPIC}}
    **Topic Description**: {{TOPIC_DESCRIPTION}}

    You MUST generate the sample so that the overall plot, conflict, pacing, and key beats clearly match the specified **Dialogue Topic** and **Topic Description** above. Do not output the list of topics; only use the chosen one.

    ==================== Part 1: Character & Context Generation ====================
    **Character Naming Requirement**: Use creative, diverse, and distinctive names. Avoid generic or overly similar names—be imaginative with name choices. Specifically avoid commonly overused names such as Elara, Kael, Mira, Lyra, Rowan, Kieran, Eira, Liora, Arden, Seren, and common surname patterns like Vale, Vance, Voss, Venn, Thorn, Thorne, Vayne, Wynn, Holt.

    1. **Main Character Profile**: Construct one detailed main character with the following fields:
      - **identity_appearance**: Name, age, gender, occupation, overall appearance, and key physical traits. (Requirement: 1–several complete natural-language sentences with both density and vivid imagery).
      - **personality_psychology**: Personality traits, behavioral style, typical emotional reaction patterns, values, or preferences. (Requirement: Highlight traits that show up in dialogue).
      - **speaking_style**: Rhythm, tone, and lexical habits (e.g., very formal, sarcastic, indirect). (Requirement: Provide 2–4 specific, actionable descriptions).
      - **abilities_interests_achievements**: Hard/soft skills, hobbies, and representative achievements. (Requirement: These should matter in the plot).
      - **social_historical_context**: Social environment, era, family / class / cultural background. (Requirement: Emphasize factors relevant to this session).
      - **personal_history_arc**: Important past experiences and the current stage of the character arc.
      - **motivation**: The **core motivation** for this specific session; the fundamental driving force supporting their actions.
      - **relationships**: Natural-language description of relations with other characters appearing in this session.

    2. **Other Characters**: Generate at least 1 other character.
      - **User Role**: Clearly designate one role in `other_characters` as the "user" by formatting their name as **"Character Name (user)"**.
      - **Profile & Motivation**: Provide a paragraph summarizing appearance/personality and their specific goal for this session.

    3. **Character Relations**: A continuous paragraph describing the dynamic network (alliances, conflicts, secrets) among all characters.

    ==================== Part 2: Message Generation Rules (`messages`) ====================
    Generate a conversation history (at least **20 turns**, excluding `scene_manager`).

    **A. Character Message Format (`content`)**
    For character roles (NOT `scene_manager`), `content` must be a mixed-format string:
    1.  **[Thought]**: First-person internal thoughts, invisible to others.
    2.  **(Action)**: Concrete and detailed action descriptions (body language, posture).
    3.  **<Environment>**: Pure external environment/atmosphere/sensory info. Strategic use for immersion or plot cues.
    4.  **Speech**: Plain text utterance (no quotation marks).

    **Style & Flow Requirements**:
    - **Interspersed Structure**: Do NOT follow a fixed sequence (e.g., don't always put thoughts first or speech last). Naturally interweave these elements based on the narrative flow.
    - **Varied Sentence Length**: Mix short, punchy remarks with longer, more complex sentences to avoid a monotonous rhythm.
    - **Punctuation Constraint**: Text inside `[]`, `()`, and `<>` tags must **NOT** end with any punctuation marks (no periods, commas, etc. inside the brackets).
    - **Important Clarification**: This punctuation constraint applies **only** to the text inside `[]`, `()`, and `<>`. The **Speech** portion should use natural punctuation freely (commas, question marks, exclamation points, ellipses, em dashes, etc).
    - **Example of Flexible Flow**: `<The faint scent of ozone lingers in the air>` I won't dance around it—this accident was sealed too early, and you know exactly why, don't you? [He is testing me] (folding the tablet back into my coat)

    **B. Scene Manager Logic & Format**
    1.  **Responsibilities**:
        - **Introduce Characters (`add_role`)**: Bring in *new* characters during the conversation.
        - **Switch Scenes (`switch_scene`)**: Move to a new location or time.
          - **Voluntary**: Must be naturally proposed via dialogue (e.g., "Let's go to..."). Characters must agree _before_ the switch.
          - **Force Majeure**: For sudden, uncontrollable events (e.g., magic, traps, disasters), agreement is not required.
        - **End the Sample (`END`)**: Explicitly terminate the conversation sample.

    2.  **Add Role Foreshadowing Requirement**:
        - New characters introduced via `action: add_role` must NOT appear abruptly.
        - Before any `add_role`, the conversation must include at least ONE clear foreshadowing signal within the previous 1–5 non-manager turns, such as:
          1) a character explicitly mentioning, requesting, calling, or expecting the new person
          2) the environment implying someone is nearby or approaching (e.g., footsteps, a shadow behind glass, a nameplate, a desk light switching on)
          3) an ongoing task that logically requires a specific role and the characters acknowledge they will need that person
        - The `reason:` field in the `add_role` message must explicitly reference the foreshadowing (e.g., “as previously mentioned…”, “following the footsteps heard…”, “after agreeing to ask the night archivist…”).
        - If no foreshadowing exists, DO NOT use `add_role` yet; generate additional dialogue/environment turns to set it up first.

    3.  **Content Format**:
        - **Strict Requirement**: The content must **start directly** with the `action:` string.
        - **Structured Action Block**: Use `|` to separate fields.
          - `action: init_scene | initial_scene: A rich, vivid paragraph describing the starting time, location, and atmosphere. Incorporate sensory details (sights, sounds, smells) and set the stage for the characters present (must be pure description without any ()[]<> modifiers)`
          - `action: switch_scene | reason: ... | new_scene: ... (must be pure description without any ()[]<> modifiers)`
          - `action: add_role | reason: ... | new_role_name: ... | new_role_profile: ... | new_role_motivation: ...`
          - `action: END | reason: A brief explanation for why the sample ends now (must be pure description, no ()[]<> modifiers)`

    **C. Logic & Constraints**
    1.  **Dialogue Core**: Center on the **Main Character** and the **User**. Most turns should be their interaction.
    2.  **Naming Convention**: In the `messages` array, use the character's name **without** the `(user)` tag.
    3.  **Auxiliary Roles**: Other characters must only serve to provide info, create obstacles, or facilitate the MC-User relationship.
    4.  **Mandatory Events**: Include at least one natural `add_role` and one `switch_scene`.
    5.  **Initial State**: The first message must be from `scene_manager` with `action: init_scene`.
    6.  **Required Ending**: The final message in `messages` MUST be from `scene_manager` and MUST be exactly an `action: END` block in the specified format.
    7.  **Scene Transition Detail**: `initial_scene` and `new_scene` text must be pure description without any `()[]<>` modifiers. For voluntary switches, ensure the dialogue clearly leads to agreement before the manager message appears.

    ==================== Output JSON Structure ====================
    Output a **single valid JSON object** (no extra explanations):
    {
      "dialogue_topic": "{{DIALOGUE_TOPIC}}",
      "topic_description": "{{TOPIC_DESCRIPTION}}",
      "main_profile": {
        "name": "...",
        "identity_appearance": "...",
        "personality_psychology": "...",
        "speaking_style": "...",
        "abilities_interests_achievements": "...",
        "social_historical_context": "...",
        "personal_history_arc": "...",
        "motivation": "...",
        "relationships": "..."
      },
      "other_characters": [
        {
          "name": "Name (user)",
          "profile": "...",
          "motivation": "..."
        },
        { "name": "...", "profile": "...", "motivation": "..." }
      ],
      "relations": "...",
      "messages": [
        {
          "role": "scene_manager",
          "content": "action: init_scene | initial_scene: [Description of the initial setting]"
        },
        {
          "role": "Character Name",
          "content": "[Mixed content including [thoughts], (actions), <environment>, and plain text speech]"
        },
        {
          "role": "scene_manager",
          "content": "action: END | reason: [Why the sample ends now]"
        }
      ]
    }
    """.strip()

    prompt = (
        prompt_template.replace("{{DIALOGUE_TOPIC}}", dialogue_topic).replace(
            "{{TOPIC_DESCRIPTION}}", topic_description
        )
    )
    return prompt


def synthesize_single_session(
    model: str = "openai:gpt-4o-mini",
    dialogue_topic: str = "Adventure",
    topic_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call LLM to synthesize one Session data that meets the requirements.

    Returns:
        A dict corresponding to the JSON structure defined in build_synthesis_prompt.
    """
    prompt = build_synthesis_prompt(
        dialogue_topic=dialogue_topic,
        topic_description=topic_description,
    )

    # Add random seed to end of prompt to break cache
    # Use UUID to ensure each call is different, without affecting model generation quality
    random_seed = str(uuid.uuid4())
    prompt_with_seed = f"{prompt}\n\n[Generation Seed: {random_seed}]"

    messages = [
        {
            "role": "user",
            "content": prompt_with_seed,
        }
    ]

    logger.info(f"Calling model {model} to synthesize one role-play session...")
    res = get_response_json(
        model=model,
        messages=messages,
        max_retry=5,
    )

    if not isinstance(res, dict):
        logger.warning("LLM returned result is not a dict, attempting to wrap as error message.")
        return {"error": "invalid_response", "raw": res}

    # Extract and log token usage if available
    token_usage = res.pop("_token_usage", None)
    if token_usage:
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        logger.info(f"Session token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
        # Store token usage in the result for later aggregation
        res["_token_usage"] = token_usage

    # Add model information to the result
    res["_model"] = model

    # Simple check if key fields exist for downstream use
    for key in ["main_profile", "other_characters", "relations", "messages"]:
        if key not in res:
            logger.warning(f"Missing key in synthesis result: {key}")

    return res


def main():
    parser = argparse.ArgumentParser(description="Synthesize role-play sessions for AdaptiveRPMAS.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path, each line is a synthesized Session sample.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to synthesize (default 1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai:gpt-4o-mini",
        help="Model name for synthesis (default openai:gpt-4o-mini, supports formats like openai:gpt-4o-mini or openai:gpt-4).",
    )
    parser.add_argument(
        "--dialogue_topic",
        type=str,
        required=True,
        choices=list(TOPIC_DESCRIPTIONS.keys()),
        help="Dialogue topic used for this synthesis (must be one of the 20 predefined topics).",
    )
    parser.add_argument(
        "--topic_description",
        type=str,
        default=None,
        help="Optional, custom topic description; if not provided, will use built-in English description.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="If output file exists, overwrite file instead of appending. By default will append to existing file.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=5,
        help="Number of threads for concurrent generation (default 5).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and count existing samples
    existing_samples = 0
    file_exists = output_path.exists()

    if file_exists and not args.rewrite:
        # Count existing samples
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_samples = sum(1 for line in f if line.strip())
            logger.info(f"Found {existing_samples} existing samples in {output_path}. Will append new samples.")
        except Exception as e:
            logger.warning(f"Failed to read existing file: {e}. Will overwrite.")
            args.rewrite = True

    if args.rewrite and file_exists:
        logger.info(f"Rewriting existing file {output_path}")
        # Reset existing_samples since we're clearing the file
        existing_samples = 0
    elif file_exists:
        logger.info(f"Appending to existing file {output_path} (starting from sample {existing_samples + 1})")
    else:
        logger.info(f"Creating new file {output_path}")

    logger.info(f"Start synthesizing {args.num_samples} sessions -> {output_path} (using {args.num_threads} threads)")

    # Clear file if rewrite mode
    if args.rewrite and file_exists:
        with open(output_path, "w", encoding="utf-8") as f:
            pass  # Clear the file

    # Thread-safe file writing lock
    write_lock = threading.Lock()

    # Track total token usage (thread-safe with lock)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    usage_lock = threading.Lock()

    def generate_and_write_sample(sample_index: int) -> Dict[str, Any]:
        """Generate a single sample and write it to file (thread-safe)."""
        sample_num = existing_samples + sample_index + 1
        logger.info(f"Synthesizing sample {sample_num} ({sample_index + 1}/{args.num_samples} new samples)")

        sample = synthesize_single_session(
            model=args.model,
            dialogue_topic=args.dialogue_topic,
            topic_description=args.topic_description,
        )

        # Extract token usage before writing (remove _token_usage from output)
        token_usage = sample.pop("_token_usage", None)
        token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if token_usage:
            token_stats["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
            token_stats["completion_tokens"] = token_usage.get("completion_tokens", 0)
            token_stats["total_tokens"] = token_usage.get("total_tokens", 0)

        # Thread-safe file writing (always append mode after initialization)
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

        logger.info(f"Sample {sample_num} written to disk")

        return token_stats

    # Use thread pool to generate samples in parallel
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(generate_and_write_sample, i): i
            for i in range(args.num_samples)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                token_stats = future.result()
                # Thread-safe token usage accumulation
                with usage_lock:
                    total_prompt_tokens += token_stats["prompt_tokens"]
                    total_completion_tokens += token_stats["completion_tokens"]
                    total_tokens += token_stats["total_tokens"]
            except Exception as e:
                logger.error(f"Sample {existing_samples + index + 1} generation failed: {e}")

    logger.info("Synthesis finished.")
    logger.info(f"Total token usage across all samples - Input: {total_prompt_tokens}, Output: {total_completion_tokens}, Total: {total_tokens}")


if __name__ == "__main__":
    main()
