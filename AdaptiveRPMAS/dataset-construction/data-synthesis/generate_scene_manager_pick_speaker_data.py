import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage

# Allow importing helpers from repo root (main.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import create_chat_model, resolve_model_credentials  # type: ignore

# Base path
BASE_DIR = REPO_ROOT
INPUT_PATH = BASE_DIR / "dataset_analysis/final_data/training_data/AdaRPSet-Synthesis.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset_analysis/final_data/training_data/AdaSMSet.jsonl"

# Default number of worker threads for parallel processing
NUM_WORKERS = int(os.getenv("SCENE_MANAGER_NUM_WORKERS", "32"))

# Global variables for prompt version and output path (set by main)
PROMPT_VERSION = "basic"  # "basic" or "enhance"
OUTPUT_PATH = DEFAULT_OUTPUT_PATH

# Reference to prompts.py lines 61-82
ASSISTANT_INSTRUCTION = """===Instructions for Scene Manager===
You must choose ONE action per turn, following this priority order:
1. If the user asked to stop or the story is complete, set action="end".
2. If there is a MAJOR scene change AND characters have explicitly agreed to move there OR have already moved there through unavoidable circumstances, set action="switch_scene" and provide new_scene.
3. If adding a new role would significantly enrich and advance the plot, OR if the user or a role explicitly wants to interact with a character not in the current role list, set action="add_role" and provide new_role_name, new_role_profile, new_role_motivation.
4. Otherwise, set action="pick_speaker" and provide speaker (must be one of existing roles or "user").

CRITICAL RULES for pick_speaker:
- ROTATE SPEAKERS: Never pick the same speaker twice in a row. After someone speaks, pick a different role next turn.
- INCLUDE USER IN ROTATION: The user is part of the rotation. Avoid long stretches of roles talking only to each other; bring the user back in regularly.
- **AVOID MONOPOLY: Do not let a single role dominate. Prefer cycling through available roles so everyone gets turns.**

Notes:
- The role whose name includes "(user)" is the real user. If you choose the real user, set speaker to the actual role name (e.g., "Role_Name (user)"), not just "user".

Return ONLY a JSON object with fields:
- action: "switch_scene" | "add_role" | "pick_speaker" | "end"
- reason: (REQUIRED) A concise explanation for why you chose this action—clarify your rationale for picking this speaker, adding this role, switching scenes, or ending the story.
- new_scene: (when action is switch_scene)
- new_role_name, new_role_profile, new_role_motivation: (when action is add_role)
- speaker: (when action is pick_speaker; must be one of existing roles or "user")
Example:
{
    "action": "...",
    "reason": "...",
    ...other fields...
}
"""


def parse_scene_manager_action(raw_content: str) -> Dict[str, Any]:
    """Convert scene_manager textual content into structured action JSON."""
    text = raw_content.strip()
    if text.startswith("scene_manager:"):
        text = text[len("scene_manager:") :].strip()

    parts = [p.strip() for p in text.split("|") if p.strip()]
    kv: Dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        kv[key.strip()] = val.strip()

    action = kv.get("action", "").lower()
    reason = kv.get("reason", "").strip()

    if action == "init_scene":
        return {
            "action": "init_scene",
            "reason": "",
            "initial_scene": kv.get("initial_scene", ""),
        }
    if action == "switch_scene":
        return {
            "action": "switch_scene",
            "reason": reason or "initialize or switch scene",
            "new_scene": kv.get("new_scene", ""),
        }
    if action == "add_role":
        return {
            "action": "add_role",
            "reason": reason or "add a new role",
            "new_role_name": kv.get("new_role_name", ""),
            "new_role_profile": kv.get("new_role_profile", ""),
            "new_role_motivation": kv.get("new_role_motivation", ""),
        }
    if action in {"end", "end_scene", "finish"}:
        return {"action": "end", "reason": reason or "end scene"}

    # Fallback to raw action, keeping any known fields.
    result: Dict[str, Any] = {"action": kv.get("action", ""), "reason": reason}
    if "speaker" in kv:
        result["speaker"] = kv["speaker"]
    return result


def extract_speaker_name(role_field: str, content: str) -> str:
    """Resolve speaker name from message role/content."""
    if ":" in content:
        candidate = content.split(":", 1)[0].strip()
        if candidate:
            return candidate
    return role_field


def build_basic_prompt(
    system_text: str, history: List[Dict[str, str]], speaker_name: str
) -> str:
    """Build basic version of the prompt (simple and concise)."""
    return (
        "You are generating a brief reason for why the next speaker is chosen.\n"
        "Given the system prompt and recent history and the pending speaker name \n "
        f"System:\n{system_text}\n\n"
        f"History:\n{json.dumps(history, ensure_ascii=False)}\n\n"
        f"Pending speaker: {speaker_name}\n"
        f"Return ONLY a sentence (max ~25 words) explaining why this speaker speaks now.\n"
        "IMPORTANT: Avoid using fixed patterns like '[Role_NAME] is chosen to speak next' or '[Role_NAME] speaks next'. "
        "Use varied sentence beginnings and natural phrasing. For example:\n"
        "- Start with the character's action or response\n"
        "- Begin with the scene context or situation\n"
        "- Lead with character dynamics or relationships\n"
        "- Use narrative flow or story progression\n"
    )


def build_enhance_prompt(
    system_text: str, history: List[Dict[str, str]], speaker_name: str
) -> str:
    """Build enhanced version of the prompt (comprehensive and detailed)."""
    return (
        "You are generating a comprehensive and well-reasoned explanation for why the next speaker is chosen in this role-playing scenario.\n\n"
        "Your analysis should synthesize multiple contextual factors:\n"
        "1. **Scene Information**: Consider the initial scene setting, any scene transitions that have occurred, "
        "and scene-related details mentioned in the conversation history (including information within angle brackets < >).\n"
        "2. **Character Information and Relationships**: Analyze the roles involved, including:\n"
        "   - The main character (protagonist)\n"
        "   - The user character\n"
        "   - Any newly introduced characters\n"
        "   - The relationships and dynamics between these characters\n"
        "3. **Current Scene Atmosphere**: Assess the overall mood, tension, and emotional tone of the current scene.\n"
        "4. **Conversation Flow**: Consider the natural progression of dialogue and who should logically speak next.\n\n"
        "Based on these factors, provide an insightful reason that explains why this specific speaker "
        "is chosen at this moment in the narrative.\n\n"
        f"System Prompt:\n{system_text}\n\n"
        f"Conversation History:\n{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
        f"Pending Speaker: {speaker_name}\n\n"
        "Return ONLY a single sentence that provides a clear, contextual explanation for why this speaker is chosen now. "
        "CRITICAL: Avoid using fixed patterns like '[Role_NAME] is chosen to speak next' or '[Role_NAME] speaks next'. "
        "Use varied and natural sentence structures—avoid repetitive phrasing. You may express the reason in different ways, such as:\n"
        "- Starting with the speaker's action, response, or reaction"
        "- Beginning with the scene context or situation"
        "- Leading with character dynamics or relationships"
        "- Focusing on narrative flow or story progression"
        "- Using descriptive or contextual openings"
        "The reason should reflect your comprehensive analysis of the scene, characters, relationships, and current narrative atmosphere. "
        "Vary your sentence structure and phrasing to make each explanation unique and contextually appropriate. "
        "Each explanation should feel natural and context-driven, not formulaic.\n"
    )


def craft_reason_via_llm(
    system_text: str, history: List[Dict[str, str]], speaker_name: str, prompt_version: str = "basic"
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Use an LLM to craft a short reason. Controlled by USE_LLM_REASON (default on).
    Falls back to None on failure.
    Returns: (reason, usage_info) tuple
    """
    use_llm = os.getenv("USE_LLM_REASON", "1") != "0"
    if not use_llm:
        return None, None

    try:
        model_spec = os.getenv("LLM_REASON_MODEL", "gpt-4o-mini")
        # print(f"Using model: {model_spec} (prompt_version={prompt_version})")
        api_key, base_url = resolve_model_credentials(model_spec, {}, None, None)
        model = create_chat_model(
            model_spec=model_spec,
            generation_config={"temperature": 0.2},
            api_key=api_key,
            base_url=base_url,
        )

        # Choose prompt version
        if prompt_version == "enhance":
            prompt = build_enhance_prompt(system_text, history, speaker_name)
        else:  # default to basic
            prompt = build_basic_prompt(system_text, history, speaker_name)

        # Our chat models expect a state dict with a \"messages\" list, not a raw list.
        state = {"messages": [HumanMessage(content=prompt)]}
        resp = model.invoke(state)  # type: ignore[arg-type]
        print(f"Response: {getattr(resp, 'content', '')}")

        # Extract usage info from response
        usage_info = None
        if hasattr(resp, "response_metadata") and resp.response_metadata:
            usage_info = resp.response_metadata.get("token_usage")
        elif hasattr(resp, "usage"):
            usage_info = resp.usage

        reason = getattr(resp, "content", "").strip()
        return reason if reason else None, usage_info
    except Exception as e:
        print(f"Error crafting reason via LLM: {e}")
        return None, None


def craft_reason(system_text: str, history: List[Dict[str, str]], speaker_name: str, prompt_version: str = "basic") -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Heuristic reason; uses LLM when USE_LLM_REASON=1 and succeeds, otherwise falls back.
    Returns: (reason, usage_info) tuple
    """
    llm_reason, usage_info = craft_reason_via_llm(system_text, history, speaker_name, prompt_version)
    reason = llm_reason or f"{speaker_name} continues"
    return reason, usage_info


def build_pick_speaker_action(
    system_text: str, history: List[Dict[str, str]], role_field: str, content: str, prompt_version: str = "basic"
) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    """Create a pick_speaker action for the given role/content.
    Returns: (action_dict, usage_info) tuple
    """
    speaker_name = extract_speaker_name(role_field, content)
    # Remove (user) marker if present, but keep the original name otherwise
    # Use the speaker name as-is, without forcing conversion to "user"
    speaker = speaker_name.replace("(user)", "").replace(" (user)", "").strip()
    if not speaker:
        speaker = speaker_name  # Fallback to original if stripping removes everything

    # Remove roleplaying instructions before crafting reason
    cleaned_system_text = remove_roleplaying_instructions(system_text)
    reason, usage_info = craft_reason(cleaned_system_text, history, speaker_name, prompt_version)
    action_dict = {"action": "pick_speaker", "reason": reason, "speaker": speaker}
    return action_dict, usage_info


def remove_roleplaying_instructions(system_text: str) -> str:
    """Remove the roleplaying instructions section from system prompt."""
    marker = "===Instructions for roleplaying==="
    idx = system_text.find(marker)
    if idx == -1:
        return system_text
    # Remove everything from the marker onwards, including trailing whitespace
    return system_text[:idx].rstrip()


def add_scene_manager_prefix(system_text: str) -> str:
    """Add scene manager prefix before ===Main Character=== if not already present."""
    prefix = "You are a concise, reliable scene manager. You are the Scene Manager for a role-playing story.\n\n"
    marker = "===Main Character==="
    prefix_text = "You are a concise, reliable scene manager. You are the Scene Manager for a role-playing story."

    # Check if prefix is already present at the start
    if system_text.startswith(prefix_text):
        # Prefix already exists, ensure proper formatting
        if system_text.startswith(prefix):
            return system_text
        # Fix formatting if needed
        after_prefix = system_text[len(prefix_text):].lstrip()
        if not after_prefix.startswith("\n\n"):
            return prefix + after_prefix
        return system_text

    # Check if marker exists and add prefix before it
    idx = system_text.find(marker)
    if idx != -1:
        # If marker is at the start, add prefix before it
        if idx == 0:
            return prefix + system_text
        # If there's content before marker, insert prefix before marker
        before_marker = system_text[:idx].rstrip()
        if not before_marker.endswith("role-playing story."):
            return system_text[:idx] + prefix + system_text[idx:]
    else:
        # If marker doesn't exist, add prefix at the start
        return prefix + system_text

    return system_text


def normalise_history(history: List[Dict[str, str]]) -> str:
    """Dump prior messages to JSON for use as user content."""
    return json.dumps(history, ensure_ascii=False)


def process_sample(sample: Dict[str, Any], prompt_version: str = "basic") -> List[Dict[str, Any]]:
    """
    Build a single interleaved conversation:
    user messages (original turns) + assistant scene_manager actions + assistant pick_speaker between roles.
    Returns sample with token_usage field if any LLM calls were made.
    """
    messages: List[Dict[str, str]] = sample["messages"]
    sample_id = sample.get("id")  # Preserve UUID/index field from input if present
    if not messages or messages[0].get("role") != "system":
        return []

    system_msg = messages[0]
    # Remove roleplaying instructions before merging with scene manager instructions
    cleaned_system_content = remove_roleplaying_instructions(system_msg["content"])
    # Add scene manager prefix before ===Main Character=== if not already present
    cleaned_system_content = add_scene_manager_prefix(cleaned_system_content)
    merged_system = f"{cleaned_system_content}\n\n{ASSISTANT_INSTRUCTION}"
    convo: List[Dict[str, str]] = [{"role": "system", "content": merged_system}]

    # Track token usage for this sample
    token_usage_list: List[Dict[str, Any]] = []

    for idx in range(1, len(messages)):
        current = messages[idx]
        is_scene_mgr = current["role"] == "user" and current["content"].lstrip().startswith("scene_manager:")

        if is_scene_mgr:
            action_obj = parse_scene_manager_action(current["content"])
            convo.append({"role": "assistant", "content": json.dumps(action_obj, ensure_ascii=False)})
        else:
            convo.append({"role": "user", "content": current["content"]})

        next_idx = idx + 1
        if next_idx < len(messages):
            nxt = messages[next_idx]
            if not (
                nxt["role"] == "user"
                and nxt["content"].lstrip().startswith("scene_manager:")
            ):
                history_slice = messages[1:next_idx]  # prior turns
                # Use cleaned system content (roleplaying instructions already removed above)
                pick_action, usage_info = build_pick_speaker_action(
                    cleaned_system_content, history_slice, nxt["role"], nxt["content"], prompt_version
                )
                convo.append(
                    {"role": "assistant", "content": json.dumps(pick_action, ensure_ascii=False)}
                )
                # Record token usage if available
                if usage_info:
                    model_spec = os.getenv("LLM_REASON_MODEL", "gpt-4o-mini")
                    token_usage_list.append({
                        "model_spec": model_spec,
                        "usage": usage_info,
                    })

    # Build result with token usage if any
    result: Dict[str, Any] = {"id": sample_id, "messages": convo}
    if token_usage_list:
        # Aggregate token usage
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in token_usage_list:
            usage = entry.get("usage", {})
            # Handle different token usage formats
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += tokens

        result["token_usage"] = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "calls": len(token_usage_list),
        }

    return [result]


def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scene manager pick speaker training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        choices=["basic", "enhance"],
        default=os.getenv("PROMPT_VERSION", "basic"),
        help="Prompt version to use: 'basic' (simple) or 'enhance' (comprehensive)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output file path (default: based on prompt version)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Input file path (default: AdaRPSet-Synthesis.jsonl)",
    )

    args = parser.parse_args()

    # Set global prompt version
    global PROMPT_VERSION, OUTPUT_PATH
    PROMPT_VERSION = args.prompt_version

    # Set input path
    input_path = Path(args.input_path) if args.input_path else INPUT_PATH

    # Set output path
    if args.output_path:
        OUTPUT_PATH = Path(args.output_path)
    else:
        OUTPUT_PATH = DEFAULT_OUTPUT_PATH

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using prompt version: {PROMPT_VERSION}")
    print(f"Input: {input_path}")
    print(f"Output: {OUTPUT_PATH}")

    # Collect IDs that have already been processed (if output file exists),
    # so repeated runs will skip samples whose ID has already been generated.
    seen_ids = set()
    if OUTPUT_PATH.exists():
        print(f"Reading existing output file to collect processed IDs: {OUTPUT_PATH}")
        with OUTPUT_PATH.open("r", encoding="utf-8") as f_existing:
            for line in f_existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("id")
                # Support both string and numeric IDs
                if sid is not None:
                    # Convert to string for consistent comparison
                    seen_ids.add(str(sid))
        print(f"Found {len(seen_ids)} already processed IDs")

    # Load all input samples
    print(f"Loading input samples from: {input_path}")
    all_samples = list(stream_jsonl(input_path))
    print(f"Loaded {len(all_samples)} input samples")

    # Filter samples by ID: if a sample has an id that has already been seen,
    # skip it; otherwise remember the id and keep the sample.
    samples_to_process: List[Dict[str, Any]] = []
    skipped_count = 0
    for sample in all_samples:
        sid = sample.get("id")
        if sid is not None:
            # Convert to string for consistent comparison
            sid_str = str(sid)
            if sid_str in seen_ids:
                # This ID has already been generated before; skip.
                skipped_count += 1
                continue
            seen_ids.add(sid_str)  # Mark as seen to avoid duplicates in input
        samples_to_process.append(sample)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples that were already processed")
    print(f"Will process {len(samples_to_process)} new samples")

    count = 0
    # Track IDs written in this run to avoid duplicates (in case of concurrent processing)
    written_ids_this_run = set()

    # Append to existing file (if any), so previous results are kept and
    # we only add new, non-duplicate ID samples.
    with OUTPUT_PATH.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Use partial to pass prompt_version to process_sample
            process_func = partial(process_sample, prompt_version=PROMPT_VERSION)
            for rows in executor.map(process_func, samples_to_process):
                if not rows:
                    continue
                for row in rows:
                    row_id = row.get("id")
                    if row_id is not None:
                        row_id_str = str(row_id)
                        # Double-check: skip if this ID was already written in this run
                        if row_id_str in written_ids_this_run:
                            print(f"Warning: Skipping duplicate ID in current run: {row_id_str}")
                            continue
                        written_ids_this_run.add(row_id_str)

                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    count += 1

    print(
        f"Generated {count} new scene_manager samples -> {OUTPUT_PATH} "
        f"(workers={NUM_WORKERS}, prompt_version={PROMPT_VERSION})"
    )


if __name__ == "__main__":
    main()
