# Transforming extracted book data to the training format and the test set
# The training data follows the sharegpt format for sft, such as :
# [
#   [
#     {
#       "from": "Human",
#       "value": "Hello, how are you?"
#     },
#     {
#       "from": "Assistant",
#       "value": "I'm fine, thank you!"
#     }
#   ]
# ]

import os
import random
import argparse
import json
from typing import Dict, Any, List

parser = argparse.ArgumentParser(description='Convert data format')
parser.add_argument('--dir', type=str, default='data', help='input_dir path for both input and output')
args = parser.parse_args()

input_dir = args.dir + '/processed-data/'

# Get a list of all files in the input_dir
files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

def stable_shuffle(files):
    """
    Performs a deterministic shuffle of file names using a custom hash function.

    This function provides consistent ordering of files across different runs,
    unlike Python's built-in hash() which is randomized per process.

    Args:
        files: List of filenames (strings) to be shuffled

    Returns:
        List of filenames in a deterministically shuffled order

    Note:
        Uses a simple polynomial rolling hash function (similar to Java's String.hashCode())
        with a multiplier of 31 to generate consistent hash values.
    """
    def string_hash(s):
        """
        Computes a deterministic 32-bit hash value for a string.

        Args:
            s: Input string to hash

        Returns:
            32-bit integer hash value
        """
        h = 0
        # Multiply by 31 at each step to distribute bits well
        # Use bitwise AND with 0xFFFFFFFF to keep within 32 bits
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return h

    # Generate (filename, hash) pairs for stable sorting
    files_with_hash = [(filename, string_hash(filename))
                       for filename in files]

    # Sort files based on their hash values for deterministic ordering
    files_with_hash.sort(key=lambda x: x[1])

    # Extract and return just the filenames in their new order
    return [file_hash_pair[0] for file_hash_pair in files_with_hash]

# Shuffle the files in a stable way
files = stable_shuffle(files)

# Print first 10 files and their total count
print(f"First 10 files: {files[:10]}")
print(f"Total number of files: {len(files)}")

dedup = False

# =============================
# Helpers for prompt generation
# =============================

def build_system_prompt(main_profile: Dict[str, Any], other_characters: List[Dict[str, str]]) -> str:
    """
    Build the system prompt for a character using the richer profile fields.
    Mirrors the structure used by the data synthesis pipeline.
    """
    def _get(profile: Dict[str, Any], key: str) -> str:
        value = profile.get(key, "")
        return value if isinstance(value, str) else ""

    main_section = f"""===Main Character===
Name:{_get(main_profile, 'name')}
Identity&Appearance:{_get(main_profile, 'identity_appearance')}
Personality&Psychology:{_get(main_profile, 'personality_psychology')}
Speaking Style:{_get(main_profile, 'speaking_style')}
Abilities&Interests&Achievements:{_get(main_profile, 'abilities_interests_achievements')}
Social Historical Context:{_get(main_profile, 'social_historical_context')}
Personal History Arc:{_get(main_profile, 'personal_history_arc')}
Relationships:{_get(main_profile, 'relationships')}
Motivation:{_get(main_profile, 'motivation')}
"""

    other_section = "\n===Information about the other Characters===\n"
    for char in other_characters:
        name = char.get("name", "")
        desc = char.get("short_description", "")
        other_section += f"{name}:{desc}\n{name}'s motivation:\n"

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

def process(file):
    """
    This function process a book's extracted data file. Most importantly, it converts the data into GCA training data, which follow the sharegpt format for sft.

    This function processes a JSON file containing book data and converts it into a format suitable
    for training LLMs. It handles:
    - Preparing Characters' System Prompts

    Args:
        file (str): Name of the JSON file containing extracted book data

    Returns:
        tuple: Contains:
            - train_chat_data (list): Chat-formatted training data
            - held_out_plots (list): The plots held-out for testing purposes, from which we sample test samples
            - character_profiles (dict): Character profile information
    """
    # Load and parse the input file
    with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    book_name = os.path.splitext(file)[0]

    # Extract character information
    character_datasets = data['character_datasets']
    character_profiles = {name: data['character_datasets'][name]['profile']
                         for name in character_datasets.keys()}

    characters = character_datasets.keys()

    train_chat_data = []

    # Possible keys for message content
    message_keys = ['message', 'thought', 'description']

    # Process data
    for i_p, plot in enumerate(data['plots']):
        # Clean up plot data
        # plot.pop('text', None)

        # Standardize key_characters format
        plot_key_characters = []
        for c in plot['key_characters']:
            if 'name' in c:
                plot_key_characters.append(c)
            elif 'character' in c:
                c['name'] = c.pop('character')
                plot_key_characters.append(c)

        plot['key_characters'] = plot_key_characters

        # Process each conversation in the plot
        for i_c, conversation in enumerate(plot['conversation']):
            from collections import Counter
            # Count utterances per character
            utterance_counts = Counter(utterance['character']
                                    for utterance in conversation['dialogues'])

            # Sort characters by frequency of speech
            speaking_characters_w_env = sorted(utterance_counts,
                                             key=lambda x: utterance_counts[x],
                                             reverse=True)

            plot_characters = [c['name'] for c in plot['key_characters']]

            # Standardize conversation character format
            conversation_key_characters = []
            for c in conversation['key_characters']:
                if 'name' in c:
                    conversation_key_characters.append(c)
                elif 'character' in c:
                    c['name'] = c.pop('character')
                    conversation_key_characters.append(c)

            conversation['key_characters'] = conversation_key_characters

            # Dialogues are assumed to already have 'character' and 'message'

            # Skip conversations with no valid characters
            if len([c for c in utterance_counts]) == 0:
                continue

            # Build scenario string (use scenario only, no summary)
            scenario_str = conversation['scenario']

            # Select characters for processing
            if dedup:
                # Select single character, either most frequent or random
                if random.random() < 0.5:
                    selected_character = speaking_characters_w_env[0]
                else:
                    count = 0
                    while True:
                        selected_character = random.choice(conversation['dialogues'])['character']
                        if selected_character in speaking_characters_w_env:
                            break
                        count += 1

                selected_characters = [selected_character]
            else:
                selected_characters = speaking_characters_w_env

            # Collect character profiles for this conversation
            tmp_character_profiles = {}
            for character in speaking_characters_w_env:
                character_profile = character_profiles.get(character, {})
                if not isinstance(character_profile, dict):
                    character_profile = {"short_description": str(character_profile)}

                if character in plot_characters:
                    character_info = [c for c in plot['key_characters']
                                   if c.get('name', '') == character][0]
                    if 'description' in character_info and 'short_description' not in character_profile:
                        character_profile['short_description'] = character_info.get('description', '').strip('\n')

                character_profile = {k: v for k, v in character_profile.items() if isinstance(v, str)}
                if character_profile:
                    tmp_character_profiles[character] = character_profile

            # Process each selected character
            for character in selected_characters:
                chat = []

                # Build character profile and system prompt
                main_profile = tmp_character_profiles.get(character, {"name": character})
                if "name" not in main_profile:
                    main_profile["name"] = character

                other_characters = []
                for other in speaking_characters_w_env:
                    if other == character:
                        continue
                    other_profile = tmp_character_profiles.get(other, {})
                    if not isinstance(other_profile, dict):
                        continue
                    other_characters.append({
                        "name": other,
                        "short_description": other_profile.get("short_description", "")
                    })

                system_prompt = build_system_prompt(
                    main_profile=main_profile,
                    other_characters=other_characters
                )

                # Build chat sequence
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"scene_manager:action=init_scene | initial_scene={scenario_str}"},
                ]

                for i_u, utterance in enumerate(conversation['dialogues']):
                    message = None
                    for key in message_keys:
                        if key in utterance:
                            message = utterance[key]
                            break
                    if message is None:
                        continue

                    speaker = utterance['character']

                    role = "assistant" if speaker == character else "user"
                    content = f"{speaker}:{message}"
                    messages.append({"role": role, "content": content})

                # Skip if no assistant messages
                if not any(m for m in messages if m["role"] == "assistant"):
                    continue

                train_chat_data.append({"messages": messages})
    return train_chat_data


# Process each file
n_books = len(files)

train_chat_data = []

for i_b, file in enumerate(files):
    train_chat_data_book = process(file)
    train_chat_data += train_chat_data_book

# print key statistics of the datasets
print(f"n_books: {n_books}")
print(f"train_chat_data: {len(train_chat_data)}")


random.shuffle(train_chat_data)

os.makedirs(args.dir + '/train', exist_ok=True)

output_file = args.dir + '/train/train.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in train_chat_data:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"Training data saved to {output_file}")
