# Standard library imports
import argparse
import json
import os
import shutil
from pathlib import Path
import re
import traceback
import copy
from collections import Counter
from typing import List, Tuple
# Third-party imports
import jsonlines
from tqdm import tqdm
import tiktoken
import re
import difflib
# Local imports
import utils as utils_module
from utils import config, cached, get_response, setup_logger, get_response_json, print_json, encode, decode, num_tokens_from_string

def parse_args():
    parser = argparse.ArgumentParser(description='Construct CoSER-style dataset from source books')
    parser.add_argument('--input', type=str, required=True,
                      help='Input jsonl file path containing books data')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Output directory path (default: data')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers (default: 1)')
    parser.add_argument('--model', type=str, default="gpt-4o",
                      help='Model to use for data construction (default: gpt-4o)')
    parser.add_argument('--candidate_model', type=str, default="gpt-4o",
                      help='Another candidate model to use for data construction when the main model fails (default: gpt-4o)')
    parser.add_argument('--regenerate', action='store_true',
                      help='Force regenerate data even if results already exist (default: False)')
    parser.add_argument('--reset_cache_log', action='store_true',
                      help='If set, clear the log directory and local cache before running')
    parser.add_argument('--max_books', type=int, default=None,
                      help='Maximum number of books to process (default: None, process all books)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    return args

args = parse_args()

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "log")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
DEFAULT_CACHE_FILE = os.path.join(BASE_DIR, ".cache.pkl")


def _clear_directory(path: str):
    """Remove a directory and recreate it empty."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[warn] Failed to clear {path}: {e}")


# Clear logs and cache if requested
if args.reset_cache_log:
    _clear_directory(LOG_DIR)
    _clear_directory(CACHE_DIR)
    if os.path.exists(DEFAULT_CACHE_FILE):
        try:
            os.remove(DEFAULT_CACHE_FILE)
        except Exception as e:
            print(f"[warn] Failed to remove {DEFAULT_CACHE_FILE}: {e}")
    # Reinitialize utils logger because its handlers were created before cleanup
    try:
        utils_module.reset_logger(LOG_DIR)
    except Exception as e:
        print(f"[warn] Failed to reset utils logger: {e}")
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

# Setup logger in dedicated log directory
log_path = os.path.join(LOG_DIR, f"{Path(__file__).stem}.log")
logger = setup_logger(__name__, log_path)

def find_index(lst, key):
    """
    Find the index of a key in a list, returning -1 if not found.

    Args:
        lst: The list to search in
        key: The key to search for

    Returns:
        int: The index of the key if found, -1 if not found
    """
    try:
        return lst.index(key)
    except ValueError:
        return -1

@cached
def create_chunk_generator(book, chunk_size):
    """
    Generates chunks of text from a book while respecting token limits and chapter boundaries.

    Args:
        book (dict): A dictionary containing book information with 'content' and other fields
        chunk_size (int): Roughly the number of tokens per chunk

    Returns:
        list: A list of text chunks from the book, where each chunk is:
            - Limited to chunk_size if no chapters are detected
            - Between chunk_size/2 and 2*chunk_size if chapters are detected
            - Cleaned of copyright notices in the first chunk
            - Cleaned of excessive tabs if present

    The function handles books in two ways:
    1. For books without chapter markers: Splits into fixed-size chunks of chunk_size
    2. For books with chapters: Attempts to keep chapters together while staying within token limits
    """
    # Check and clean excessive tabs that may interfere with text processing
    def has_excessive_tabs(content, threshold=0.05):
        tab_count = content.count('\t')
        return (tab_count / len(content)) > threshold

    if has_excessive_tabs(book['content']):
        book['content'] = book['content'].replace('\t', '')

    # Try to split book into chapters using split_book utility
    from split import split_book
    chapters = split_book(book)

    results = []

    # Case 1: No chapters detected - split into fixed-size chunks
    if not chapters:
        # Convert text to tokens for more precise chunking
        tokens = encode(book['content'])
        start_index = 0

        while start_index < len(tokens):
            # Handle the last chunk which may be smaller than chunk_size
            if len(tokens) - start_index <= chunk_size:
                chunk = decode(tokens[start_index:])
                results.append(chunk)
                break

            # Create chunk of specified size and decode back to text
            chunk_tokens = tokens[start_index:start_index + chunk_size]
            chunk = decode(chunk_tokens)
            results.append(chunk)

            start_index += len(chunk_tokens)

    # Case 2: Chapters detected - try to keep chapters together while respecting size limits
    else:
        current_chunk = []
        current_tokens = 0

        for chapter in chapters:
            # Add chapter to current chunk
            current_chunk.append(chapter['content'])
            current_tokens += len(encode(chapter['content']))

            # Check if current chunk has reached minimum size (chunk_size/2)
            if current_tokens >= chunk_size // 2:
                # If chunk is within acceptable size range (between chunk_size/2 and 2*chunk_size)
                if current_tokens <= 2 * chunk_size:
                    results.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                # If chunk is too large, split it into smaller pieces
                else:
                    chunk_text = ''.join(current_chunk)
                    chunk_tokens = encode(chunk_text)
                    for i in range(0, len(chunk_tokens), chunk_size):
                        # Special handling for final piece to avoid tiny chunks
                        if i + 2 * chunk_size >= len(chunk_tokens):
                            results.append(decode(chunk_tokens[i:]))
                            break
                        else:
                            results.append(decode(chunk_tokens[i:i + chunk_size]))
                    current_chunk = []
                    current_tokens = 0

        # Add any remaining content as final chunk
        if current_chunk:
            results.append(''.join(current_chunk))

    # Clean copyright notices from the first chunk to avoid irrelevant text
    lines = results[0].split('\n')
    filtered_lines = []
    copyright_words = ['rights', 'reserved', 'reproduced', 'copyright', 'reproduce', 'permission']

    # Remove lines that are likely copyright notices (short lines with multiple copyright-related words)
    for line in lines:
        words = line.split()
        if len(words) < 50 and sum(word.lower() in copyright_words for word in words) > 1:
            continue
        filtered_lines.append(line)

    results[0] = '\n'.join(filtered_lines)

    return results


def ngram_jaccard_similarity(text1, text2, n=3):
    """Calculate the Jaccard similarity between two texts using n-grams.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        n (int, optional): Size of n-grams. Defaults to 3.

    Returns:
        float: Jaccard similarity score between 0 and 1, where 1 means identical texts
              and 0 means completely different texts.
    """
    def ngrams(tokens, n):
        """Generate n-grams from a sequence of tokens.

        Args:
            tokens (list): List of tokens
            n (int): Size of n-grams
        Returns:
            list: List of n-gram tuples
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets.

        Args:
            set1 (set): First set
            set2 (set): Second set
        Returns:
            float: Jaccard similarity score
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Tokenize the input texts into sequences of tokens
    tokens1 = encode(text1)
    tokens2 = encode(text2)

    # Generate sets of n-grams from the token sequences
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))

    # Calculate and return the Jaccard similarity between the n-gram sets
    return jaccard_similarity(ngrams1, ngrams2)

@cached
def find_best_match_passage(candidates, target, n=3, threshold=0.3):
    """Find the best matching passage from a list of candidates compared to a target text. These texts are generally LLM-synthesized summaries. Hence, we focus on their semantic similarity.

    Uses n-gram Jaccard similarity to compare texts and find the closest match.

    Args:
        candidates (list): List of candidate passages to search through
        target (str or dict): Target text to match against
        n (int, optional): Size of n-grams to use for comparison. Defaults to 3.
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.3.

    Returns:
        int: Index of best matching passage if score >= threshold, -1 if no good match found
    """
    best_match = None  # Index of current best matching passage
    best_score = 0     # Highest similarity score found so far

    # Handle case where inputs are dictionaries by converting to strings
    if isinstance(candidates, list) and isinstance(target, dict) and isinstance(candidates[0], dict):
        target = str(target)
        candidates = [str(c) for c in candidates]

    # Compare target against each candidate passage
    for i, candidate in enumerate(candidates):
        score = ngram_jaccard_similarity(target, candidate, n)
        if score >= best_score:
            best_score = score
            best_match = i

    # Return best match if it meets threshold, otherwise return -1
    if best_score >= threshold:
        logger.debug(f"Best match: \nInput: {target}\nOutput: {candidates[best_match]}\nScore: {best_score}")
        return best_match
    else:
        return -1


@cached
def find_best_match_sentence(chunk, target, threshold=0.6):
    """Find the best matching sentence from a chunk of text or list of sentences compared to a target sentence. These sentences are generally exact sentences from the book, so we focus on their string similarity.

    Uses SequenceMatcher to calculate string similarity ratios between sentences to find the closest match.

    Args:
        chunk (str or list): Input text chunk or list of sentences to search through
        target (str): Target sentence to match against
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.6.

    Returns:
        str or None: Best matching sentence if score >= threshold, None if no good match found
                    or if target is None/invalid
    """
    # Return None for invalid target inputs
    if target == 'None' or target is None:
        return None

    # Split chunk into sentences if it's a string, otherwise use as-is if it's a list
    if isinstance(chunk, str):
        # Split on sentence endings followed by whitespace
        sentences = re.split(r'(?<=[.!?。！？\n])\s*', chunk)
    else: # chunk is a list
        assert isinstance(chunk, list)
        sentences = chunk

    # Initialize variables to track best match
    best_match = 0
    best_score = 0

    # Compare target against each sentence
    for i, sentence in enumerate(sentences):
        # Calculate similarity ratio between target and current sentence
        score = difflib.SequenceMatcher(None, target, sentence).ratio()

        # Update best match if current score is higher
        if score > best_score:
            best_score = score
            best_match = sentence

    # Log the matching results
    logger.debug(f"Best match: \nInput: {target}\nOutput: {best_match}\nScore: {best_score}")

    # Return best match if it meets threshold, otherwise return None
    if best_score >= threshold:
        return best_match
    else:
        return None

def find_sentence_with_llm(chunk, target_sentence, book, context_hint=None):
    """
    Use LLM to find the best matching sentence in the chunk when automatic matching fails.
    This is a fallback mechanism when find_best_match_sentence returns None.

    Args:
        chunk (str): The text chunk to search in
        target_sentence (str): The target sentence to find in the chunk
        book (dict): Book metadata (for context in LLM prompt)
        context_hint (str, optional): Additional context hint (e.g., "first sentence of plot" or "last sentence of plot")

    Returns:
        str or None: The matching sentence if found, None if still not found
    """
    if target_sentence == 'None' or target_sentence is None:
        return None

    try:
        context_desc = ""
        if context_hint:
            context_desc = f" ({context_hint})"

        prompt = f"""You are helping to find the text span (a full sentence or a shorter fragment) in a book excerpt that is **semantically closest** to a given target sentence/fragment, and your answer **must be copied exactly from the excerpt**.

Book: {book.get('title', 'Unknown')}
Target sentence{context_desc}: "{target_sentence}"

Below is the book text chunk:
---
{chunk}
---

Your task:
1. In the given chunk, identify the single text span (which may be a complete sentence or only a fragment) whose **meaning is closest** to the target sentence/fragment. The wording does not need to be identical, but it should be semantically the best match.
2. When you output the result, you **must copy that span verbatim from the chunk**, character by character, without rewriting, paraphrasing, adding, or removing anything.

Important:
1. Return ONLY that one span, copied exactly as it appears in the chunk
2. If no good match is found, return "None"
3. Do not add any explanation or formatting

Matching sentence:"""

        # Use get_response directly with a simple extraction function
        def extract_sentence(text):
            if not text:
                return None
            text = text.strip()
            # Remove any markdown formatting if present
            text = re.sub(r'^```.*?\n', '', text, flags=re.DOTALL)
            text = re.sub(r'\n```.*?$', '', text, flags=re.DOTALL)
            text = text.strip()
            if text.lower() == 'none' or len(text) < 3:
                return None
            return text

        response = get_response(
            model=args.model,
            messages=[{"role": "user", "content": prompt}]
        )

        if response:
            matched_sentence = extract_sentence(response)
            if matched_sentence and matched_sentence != "None":
                # Log full sentences instead of truncated previews
                logger.info(f"LLM found match for target: '{target_sentence}' -> '{matched_sentence}'")
                return matched_sentence

        # Log full target sentence when no match is found
        logger.warning(f"LLM could not find match for target sentence: '{target_sentence}'")
        return None

    except Exception as e:
        logger.error(f"Error in find_sentence_with_llm: {e}, {traceback.format_exc()}")
        return None

def extract_from_chunk(book, i_c, chunk, truncated_plots=None):
    """
    Extract and process plot information from a chunk of book text.

    This function analyzes a chunk of text to identify chapter beginnings, plots, conversations,
    and other narrative elements. It uses an LLM to generate structured information about the text.

    Args:
        book (dict): Dictionary containing book metadata including title and author
        i_c (int): Chunk index
        chunk (str): Text content of the current chunk to analyze
        truncated_plots (list, optional): List of incomplete plots from previous chunk that need to be finished

    Returns:
        tuple: Contains:
            - chapter_beginnings (list): List of identified chapter starts
            - plots (list): Extracted plot information including summaries, characters, conversations
            - remaining_chunk (str): Unused portion of chunk to process in next iteration

    The function generates a detailed prompt for the LLM that requests:
    1. Chapter beginning identification
    2. Plot extraction and analysis
    3. Conversation reconstruction
    4. Character motivation analysis
    5. Next chunk starting point determination
    """
    logger.info(f"Extracting plots from chunk for book: {book['title']}")

    # Create deep copy of truncated plots and remove text field to avoid redundancy
    import copy
    if truncated_plots:
        truncated_plots = copy.deepcopy(truncated_plots)
        for plot in truncated_plots:
            plot.pop('text')

    # Construct the prompt for the LLM
    prompt = f"""
    Based on the provided book chunk, complete the following tasks:
    1. Recognize chapter beginnings if they exist in the chunk. Prefer chapter titles or explicit section headers (e.g., lines starting with "Chapter", roman numerals, numeric labels like "#1", "#2", or similar markers) as the beginning; if no clear title exists, use the first meaningful fragment of that chapter.
    2. Identify the important plots in this chunk. Identify the beginning and ending of each plot by its first and last sentence. Determine the chapter title that the plot belongs to. Set "state" as "truncated" if the plot is truncated in this chunk, or "finished" otherwise. You will be provided with the truncated plots from the previous chunk, and you **must** extend the conversations with the current chunk while keeping the **scenario** unchanged.
    3. Summarize each important plot. For each plot, generate its summary, score its prominence from 1 to 100, and list the key characters and their roles, thoughts and actions in it.
    4. Extract conversations for each plot.
        First, state the **scenario** (the static context *before* the conversation starts, e.g., location, time, present characters). **Do NOT** include dynamic events that happen *during* the conversation in the scenario.
        Then, list the key characters with their names, descriptions and thoughts (motivations) at this point.
        Finally, extract the conversations among them based on the following requirements:
        i) Ensure the conversations are faithful to the plot and characters. They should be based on the original conversations in the text as much as possible.
        ii) The conversations should be complete, covering the key dialogues and information. Each conversation should contain at least 10 utterances.
        iii) [CRITICAL] Message Structure & Definitions:
            Each "message" must be a single string mixing the following elements. **Strictly distinguish between Action and Environment**:
            - **Thoughts []**: Internal thoughts NOT visible to others.
                *   **PERSPECTIVE**: **MUST be written in the First-Person Perspective ("I", "me", "my")**.
                *   **CONTENT**: Interpret the subtext as a specific, detailed inner monologue.
                *   **PROHIBITED**: Do NOT use abstract emotion labels (e.g., `[Amused]`). Do NOT use Third-Person descriptions.
            - **Speech**: The spoken words (no double quotes needed).
            - **Actions ()**: **ANYTHING** stemming from the character.
                *   **RICHNESS REQUIREMENT**: **Do NOT simplify actions into 1-2 words**. You **MUST** preserve the narrative detail, adverbs, and mannerisms from the text (e.g., `(flings himself onto the divan, burying his face in the cushions)`).
            - **Environmental Info <>**: **ONLY** external, non-character events or sensory details. **CRITICAL FOR IMMERSION**.
                *   **Requirement**: Actively extract descriptions of the setting, light, sound, smell, or "heavy silence" found in the narrative surrounding the dialogue.
                *   **Content**: Weather, background noises (e.g., "birds chirping", "clock ticking"), changes in lighting ("shadows lengthen"), or atmospheric moods ("the room feels oppressive").
        iv) [CRITICAL] Source Fidelity & Immersion Strategy:
            - **Narrative Distribution**: If the text contains descriptive paragraphs between lines of dialogue (e.g., describing the garden, the silence, or the sunset), **you must incorporate these details** into the adjacent "Action" or "Environment" tags. Do not ignore them.
            - **Richness**: Avoid dry logs. If the text says "The bees were buzzing in the blue throat of the honeysuckle," include `<Bees buzz in the blue throat of the honeysuckle>`.
    5. Identify the optimal starting point for the subsequent chunk. If the last storyline has been extracted as an truncated plot, set next_chunk_start as None. Otherwise, set next_chunk_start as the first sentence of the last storyline.
    ===Output Format===
    Please provide the output in the following JSON format:
    {{
        "chapter_beginnings": [
            {{
                "beginning_sentence": "Exactly the first fragment of this chapter. If this chapter starts with a title or heading (e.g., 'Chapter I. The Beginning', '#1', '#2'), copy that title in full as it appears; otherwise, copy the first meaningful fragment from the chapter opening."
            }}
        ],
        "plots": [
            // Extend the truncated plots from previous chunk, if any
            {{
                ...
            }},
            // New plots in this chunk
            {{
                "chapter_title": "The chapter title that the plot belongs to. Output None if not found.",
                "first_sentence": "Exactly the first sentence of the plot in this **chunk**.",
                "last_sentence": "Exactly the last sentence of the plot in this **chunk**. If the plot is truncated in this chunk, provide the last sentence of this chunk. ",
                "prominence": "Whether this plot is recognized to fans of this book, from 1 to 100.",
                "summary": "The summary of the plot. Just summarize, do not extend unrelated discussions.",
                "key_characters": [
                    {{
                        "name": "Character name",
                        "description": "The description of the character before this plot (~20 words).",
                        "experience": "The summary of the character's role, thoughts and behaviors towards this plot, and any significant character development relevant to the plot (~30 words)."
                    }}
                ],
                "conversation": [{{
                    "scenario": "The static setting at the start (location, time, initial mood). DO NOT include events that happen *during* the dialogue here.",
                    "topic": "The topic of the conversation (~10 words)",
                    "key_characters": [
                        {{
                            "name": "Character name",
                            "motivation": "The thought of the character before starting the conversation, including their attitudes, feelings, motivations, goals, information to convey or topics to be discussed"
                        }}
                    ],
                    "dialogues": [
                        {{
                            "character": "Character name",
                            "message": "The complete utterance string. MUST mix speech with [First-person Thoughts], (Detailed Actions), and <Rich Environmental Info>. Example: '<The heavy scent of lilacs hangs in the humid air> (turns his head slowly, catching the light) [I must tread carefully; he knows too much.] I suppose you are right.'"
                        }}
                    ]
                }}],
                "state": "finished" or "truncated"
            }}
        ],
        "next_chunk_start": "The first fragment of the next chunk (it does not need to be a full sentence)."
    }}
    ===Requirements===
    1. Adhere strictly to the specified output JSON format.
    2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
    3. In the OUTPUT, use characters' full names, omitting any titles.
    4. Maintain Story Fidelity: The plot must accurately reflect the book's content. Avoid introducing plots that are out of context. If the plot contains multiple conversations, prioritize the original dialogue from the book. In the absence of explicit conversations, create dialogue that aligns closely with the plot details.
    5. [CRITICAL] For "chapter_beginnings.beginning_sentence" and "next_chunk_start", you MUST copy the sentence **verbatim from the given chunk**, without adding, deleting, or modifying any characters (no paraphrasing, no added quotes, no extra spaces).
    ===Input===
    ==Book title==
    {book['title']}
    ==Author==
    {book['author']}
    ==Chunk of Book Content==
    {chunk}
    ==Truncated plot from previous chunk (to be finished)==
    {json.dumps(truncated_plots, ensure_ascii=False, indent=2) if truncated_plots else "None"}
    """

    # Example format for character utterances in conversations
    # "[My father's words fill me with awe, but I still feel uneasy.]
    # (Nods seriously, but with a slight frown remaining)
    # I understand, Father. Responsibility is important. But… is killing really necessary?
    # (A flash of compassion in his eyes)
    # If someone has done something wrong, can't we give them a chance to make amends?"

    logger.debug(prompt)

    def parse_response(response, chunk, book, **kwargs):
        """
        Parse and validate the LLM response, extracting structured plot information.

        Args:
            response: Raw LLM response to parse
            chunk: Original text chunk for reference
            book: Book metadata
            **kwargs: Additional keyword arguments

        Returns:
            tuple or bool: (chapter_beginnings, plots, remaining_chunk) if successful, False if failed
        """
        if not response:
            return False

        try:
            # Handle different response formats
            # Sometimes response is just a list of plots
            if isinstance(response, dict) and 'first_sentence' in response:
                response = [response]

            # Normalize response to dict format
            if isinstance(response, list):
                if len(response) > 0 and isinstance(response[0], dict) and 'first_sentence' in response[0]:
                    response = {
                        'chapter_beginnings': [],
                        'plots': response,
                        'next_chunk_start': None
                    }
                else:
                    # Empty list or unexpected format - create empty structure
                    response = {
                        'chapter_beginnings': [],
                        'plots': [],
                        'next_chunk_start': None
                    }
            elif not isinstance(response, dict):
                # Unexpected type - create empty structure
                logger.warning(f"Unexpected response type: {type(response)}, creating empty structure")
                response = {
                    'chapter_beginnings': [],
                    'plots': [],
                    'next_chunk_start': None
                }

            # Ensure response has all required fields with defaults
            response.setdefault('chapter_beginnings', [])
            response.setdefault('plots', [])
            response.setdefault('next_chunk_start', None)

            chapter_beginnings = response['chapter_beginnings']
            plots = []

            # Process next chunk starting point
            if response['next_chunk_start']:
                # Store original value before matching
                original_next_chunk_start = response['next_chunk_start']
                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                # Fallback to LLM if automatic matching failed
                if response['next_chunk_start'] is None:
                    logger.warning(f"Automatic matching failed for next_chunk_start, using LLM fallback")
                    response['next_chunk_start'] = find_sentence_with_llm(chunk, original_next_chunk_start, book, "next chunk starting point")

                if response['next_chunk_start']:
                    try:
                        # Calculate remaining chunk (20% of original)
                        remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                        remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                        # Ensure remaining chunk starts with complete sentence
                        remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:]
                    except ValueError as e:
                        # Log full next_chunk_start sentence for debugging
                        logger.error(f"Error finding next_chunk_start in chunk: {e}. Sentence: '{response['next_chunk_start']}'")
                        remaining_chunk = ''
                else:
                    remaining_chunk = ''
            else:
                remaining_chunk = ''

            # Process each plot from the response
            for unprocessed_plot in response['plots']:

                chapter_title = unprocessed_plot['chapter_title']

                # Store original values before matching (for LLM fallback)
                original_first_sentence = unprocessed_plot['first_sentence']
                original_last_sentence = unprocessed_plot['last_sentence']

                # Track whether we ended up using LLM to locate sentences
                first_from_llm = False
                last_from_llm = False

                # First try automatic matching against chunk
                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'])
                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'])

                # Fallback to LLM if automatic matching failed
                if unprocessed_plot['first_sentence'] is None:
                    # Log full plot summary for easier debugging
                    logger.warning(f"Automatic matching failed for first_sentence, using LLM fallback. Plot summary: {unprocessed_plot.get('summary', '')}")
                    unprocessed_plot['first_sentence'] = find_sentence_with_llm(chunk, original_first_sentence, book, "first sentence of plot")
                    if unprocessed_plot['first_sentence'] is not None:
                        first_from_llm = True

                if unprocessed_plot['last_sentence'] is None:
                    logger.warning(f"Automatic matching failed for last_sentence, using LLM fallback. Plot summary: {unprocessed_plot.get('summary', '')}")
                    unprocessed_plot['last_sentence'] = find_sentence_with_llm(chunk, original_last_sentence, book, "last sentence of plot")
                    if unprocessed_plot['last_sentence'] is not None:
                        last_from_llm = True

                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                # Skip this plot if we still can't find the sentences after LLM fallback
                if first_sentence is None or last_sentence is None:
                    logger.error(f"Skipping plot because sentence matching failed even after LLM fallback. First: {first_sentence is None}, Last: {last_sentence is None}")
                    continue

                # Extract original text for this plot
                # If sentences were obtained via LLM (which is instructed to copy verbatim),
                # trust them directly instead of re-matching with a strict threshold.
                if first_from_llm:
                    first_sentence_in_chunk = first_sentence
                else:
                    first_sentence_in_chunk = find_best_match_sentence(chunk, first_sentence, threshold=0.9)

                if last_from_llm:
                    last_sentence_in_chunk = last_sentence
                else:
                    last_sentence_in_chunk = find_best_match_sentence(chunk, last_sentence, threshold=0.9)

                if first_sentence_in_chunk is None or last_sentence_in_chunk is None:
                    # Detailed debug logging to inspect mismatch cases
                    logger.error(
                        "Cannot find sentences in chunk even after LLM fallback. "
                        f"First found: {first_sentence_in_chunk is not None}, "
                        f"Last found: {last_sentence_in_chunk is not None}\n"
                        f"[DEBUG] first_sentence (LLM): {first_sentence}\n"
                        f"[DEBUG] last_sentence (LLM): {last_sentence}\n"
                        f"[DEBUG] first_sentence_in_chunk: {first_sentence_in_chunk}\n"
                        f"[DEBUG] last_sentence_in_chunk: {last_sentence_in_chunk}\n"
                        f"[DEBUG] chunk excerpt: {chunk}\n"
                    )
                    continue

                try:
                    original_text = chunk[chunk.index(first_sentence_in_chunk):chunk.index(last_sentence_in_chunk) + len(last_sentence_in_chunk)]
                except ValueError as e:
                    logger.error(f"Error extracting text from chunk: {e}. First sentence: '{first_sentence_in_chunk[:50]}...', Last sentence: '{last_sentence_in_chunk[:50]}...'")
                    continue

                # If conversation or key_characters is None, set it to an empty list
                if unprocessed_plot['conversation'] is None:
                    unprocessed_plot['conversation'] = []
                if unprocessed_plot['key_characters'] is None:
                    unprocessed_plot['key_characters'] = []

                # Create structured plot object
                plot = {
                    'text': original_text,
                    'summary': unprocessed_plot['summary'],
                    'prominence': unprocessed_plot['prominence'],
                    'key_characters': unprocessed_plot['key_characters'],
                    'chapter': chapter_title,
                    'conversation': unprocessed_plot['conversation'],
                    'state': unprocessed_plot['state']
                }

                plots.append(plot)

            # Log processed response
            print_json(response)
            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

            return chapter_beginnings, plots, remaining_chunk

        except Exception as e:
            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
            return False

    from utils import get_response_json, extract_json

    # Get and parse LLM response
    response = get_response_json([extract_json, parse_response], model=args.model, messages=[{"role": "user", "content": prompt}], book=book, chunk=chunk, fix_truncated_json=True)

    return response

def extract(book, chunk_size=8192):
    """Process a book by splitting it into chunks and extracting structured information.

    This function processes a book by:
    1. Splitting the book text into chunks of specified size
    2. Extracting chapter beginnings, plots and conversations from each chunk
    3. Handling truncated plots that span multiple chunks by merging them
    4. Saving the extracted results to a JSON file

    Args:
        book (dict): Book data containing 'title', 'author', and 'content'
        chunk_size (int, optional): Roughly the number of tokens per chunk. Defaults to 8192.

    Returns:
        dict: Extracted results containing:
            - chapter_beginnings: List of chapter names (start locations)
            - plots: List of extracted plots with conversations
            - fail_to_parse_responses: List of chunks that failed parsing
    """
    # Set up save path and skip if already processed
    save_dir = f'{args.output_dir}/extracted-data'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'
    if os.path.exists(save_path) and not args.regenerate:
        return

    # Set up cache path
    from utils import set_cache_path
    set_cache_path(f'.cache/{book["title"]}.pkl')

    # Prepare chunks and log workload size
    chunks = create_chunk_generator(book, chunk_size)
    token_counts = [len(encode(chunk)) for chunk in chunks]
    total_tokens = sum(token_counts)
    logger.info(f"Planned workload: {len(chunks)} chunks (~{total_tokens} tokens)")

    # Initialize results structure
    results = {
        'chapter_beginnings': [],
        'plots': [],
    }

    # Track state between chunks
    remaining_chunk = ''  # Text carried over from previous chunk
    truncated_plots = []  # Plots that continue into next chunk

    # Process each chunk
    for i, chunk in enumerate(chunks):

        print(f"Processing chunk {i} with {token_counts[i]} tokens")

        # Extract information from current chunk
        response = extract_from_chunk(book, i, remaining_chunk + chunk, truncated_plots)

        fail_to_parse_responses = []

        # Handle the response
        if response:
            if isinstance(response, tuple) and len(response) == 3:
                # Successful extraction
                chapter_beginnings, plots, remaining_chunk = response
            else:
                # Failed extraction
                chapter_beginnings, plots, remaining_chunk = [], [], ''
                fail_to_parse_responses.append(response['fail_to_parse_response'])
        else:
            # No response
            chapter_beginnings, plots, remaining_chunk = [], [], ''

        # Merge truncated plots from previous chunk with current plots
        for u_plot in truncated_plots:
            # Find matching plot in current chunk (by summary similarity)
            idx = find_best_match_passage([p['summary'] for p in plots], u_plot['summary'])

            if idx != -1:
                # Found matching plot - merge them
                plots[idx]['text'] = u_plot['text'] + plots[idx]['text']

                # Merge conversations
                old_conversations = u_plot['conversation']
                new_conversations = plots[idx]['conversation']
                merged_conversations = []

                # Check each previous conversation
                for prev_conv in old_conversations:
                    idx_c = find_best_match_passage([s['scenario'] for s in new_conversations], prev_conv['scenario'])

                    if idx_c != -1:
                        # Use new conversation if scenarios match
                        merged_conversations.append(new_conversations[idx_c])
                    else:
                        # Keep old conversation if no match
                        merged_conversations.append(prev_conv)

                # Add any new conversations not already merged
                merged_conversations += [ c for c in new_conversations if c not in merged_conversations ]
                plots[idx]['conversation'] = merged_conversations
            else:
                # No matching plot found - mark as finished
                u_plot['state'] = 'finished'
                results['plots'].append(u_plot)

        # Separate/Update finished and truncated plots
        finished_plots = [plot for plot in plots if plot['state'] == 'finished']
        truncated_plots = [plot for plot in plots if plot['state'] == 'truncated']

        # Add to results
        results['chapter_beginnings'].extend(chapter_beginnings)
        results['plots'].extend(finished_plots)

    # Finish any remaining truncated plots
    for u_plot in truncated_plots:
        u_plot['state'] = 'finished'
        results['plots'].append(u_plot)

    results['fail_to_parse_responses'] = fail_to_parse_responses

    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

count_nth_generation = {i: 0 for i in range(7)}

def restore_from_cache(book):
    """
    As we typically encounter issues during the extraction process, this function restores some previously extracted plot data from cache.

    This function loads cached responses from extraction LLMs, processes them into the regular data format, and merges them with the results of extract().

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        None: Results are saved to disk, no return value
    """
    # Load existing extracted results

    save_dir = f'{args.output_dir}/extracted-data'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    save_path = f'{save_dir}/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
       return

    # Load cached API responses
    import pickle
    with open(f'.cache/cache_{book["title"]}.pkl', 'rb') as f:
        cache = pickle.load(f)

    # Get only the get_response cache entries
    keys = [ k for k in cache.keys() if k[0] == 'get_response' ]

    global count_nth_generation

    fail_prompts = []
    responses = {}

    # Generate chunks from book content
    chunk_generator = create_chunk_generator(book, chunk_size=8192)
    chunks = [chunk for chunk in chunk_generator]

    # Process each cached response
    for key, value in cache.items():
        if key[0] == 'get_response':
            # Extract kwargs from cache key
            dict_string = key[-1][11:-1]
            import ast
            parsed_list = ast.literal_eval(dict_string)
            restored_kwargs = dict(parsed_list)

            # Only process responses for plot extraction prompts
            if restored_kwargs['model'] == 'claude-3-5-sonnet-20240620' and restored_kwargs['messages'][0]['content'].startswith("\nBased on the provided book chunk, complete the following tasks:\n\n1. Recognize chapter beginnings if"):
                # Verify book title matches
                if not restored_kwargs['book']['title'] == book['title']:
                    logger.info(f"Warning: {restored_kwargs['book']['title']} != {book['title']}")
                    continue

                # Track generation attempts
                nth_generation = restored_kwargs['nth_generation']
                count_nth_generation[nth_generation] += 1

                # Store response
                prompt = restored_kwargs['messages'][0]['content']
                responses.setdefault(prompt, {})
                responses[prompt][nth_generation] = value

                # Track failed prompts (those that needed max retries)
                if nth_generation == 5:
                    fail_prompts.append(prompt)

    fetched_plots = []

    # Process failed prompts to extract any valid plots
    for prompt in fail_prompts:
        for nth_generation in range(6):
            if nth_generation in responses[prompt]:
                response = responses[prompt][nth_generation]

                # Check if response contains all required fields
                required_fields = ["chapter_beginnings", "plots", "chapter_title", "first_sentence", "last_sentence", "summary", "key_characters", "name", "description", "dialogues", "message"]
                if all(field in str(response) for field in required_fields):
                    # Extract JSON from potentially truncated response
                    from utils import extract_json
                    response = extract_json(response, post_fix_truncated_json=True)

                    if response is None:
                        continue

                    # Helper function to parse response and extract plots
                    def parse_response(response, chunk, book, **kwargs):
                        if not response:
                            return False

                        try:
                            # Handle different response formats
                            # Sometimes response is just a list of plots
                            if isinstance(response, dict) and 'first_sentence' in response:
                                response = [response]

                            # Normalize response to dict format
                            if isinstance(response, list):
                                if len(response) > 0 and isinstance(response[0], dict) and 'first_sentence' in response[0]:
                                    response = {
                                        'chapter_beginnings': [],
                                        'plots': response,
                                        'next_chunk_start': None
                                    }
                                else:
                                    # Empty list or unexpected format - create empty structure
                                    response = {
                                        'chapter_beginnings': [],
                                        'plots': [],
                                        'next_chunk_start': None
                                    }
                            elif not isinstance(response, dict):
                                # Unexpected type - create empty structure
                                logger.warning(f"[restore_from_cache] Unexpected response type: {type(response)}, creating empty structure")
                                response = {
                                    'chapter_beginnings': [],
                                    'plots': [],
                                    'next_chunk_start': None
                                }

                            # Ensure response has all required fields with defaults
                            response.setdefault('chapter_beginnings', [])
                            response.setdefault('plots', [])
                            response.setdefault('next_chunk_start', None)

                            chapter_beginnings = response['chapter_beginnings']
                            plots = []

                            # Handle remaining chunk logic
                            if response['next_chunk_start']:
                                # Store original value before matching
                                original_next_chunk_start = response['next_chunk_start']
                                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                                # Fallback to LLM if automatic matching failed
                                if response['next_chunk_start'] is None:
                                    logger.warning(f"[restore_from_cache] Automatic matching failed for next_chunk_start, using LLM fallback")
                                    response['next_chunk_start'] = find_sentence_with_llm(chunk, original_next_chunk_start, book, "next chunk starting point")
                                    # Verify the LLM result is actually in the chunk
                                    if response['next_chunk_start']:
                                        response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'], threshold=0.9)

                                if response['next_chunk_start']:
                                    try:
                                        remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                                        # Keep up to 20% content of the chunk
                                        remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                                        # Ensure remaining chunk starts with a sentence
                                        remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:]
                                    except ValueError as e:
                                        # Log full next_chunk_start for restore_from_cache debugging
                                        logger.error(f"[restore_from_cache] Error finding next_chunk_start in chunk: {e}. Sentence: '{response['next_chunk_start']}'")
                                        remaining_chunk = ''
                                else:
                                    remaining_chunk = ''
                            else:
                                remaining_chunk = ''

                            # Process each plot in the response
                            for unprocessed_plot in response['plots']:
                                chapter_title = unprocessed_plot['chapter_title']

                                # Store original values before matching (for LLM fallback)
                                original_first_sentence = unprocessed_plot['first_sentence']
                                original_last_sentence = unprocessed_plot['last_sentence']

                                # Match first and last sentences
                                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'], threshold=0.6)
                                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'], threshold=0.6)

                                # Fallback to LLM if automatic matching failed
                                if unprocessed_plot['first_sentence'] is None:
                                    logger.warning(f"[restore_from_cache] Automatic matching failed for first_sentence, using LLM fallback. Plot summary: {unprocessed_plot.get('summary', '')[:100]}...")
                                    unprocessed_plot['first_sentence'] = find_sentence_with_llm(chunk, original_first_sentence, book, "first sentence of plot")

                                if unprocessed_plot['last_sentence'] is None:
                                    logger.warning(f"[restore_from_cache] Automatic matching failed for last_sentence, using LLM fallback. Plot summary: {unprocessed_plot.get('summary', '')[:100]}...")
                                    unprocessed_plot['last_sentence'] = find_sentence_with_llm(chunk, original_last_sentence, book, "last sentence of plot")

                                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                                # Extract original text if sentences found (skip if still None after LLM fallback)
                                if first_sentence and last_sentence:
                                    # Use find_best_match_sentence to ensure the sentences are actually in the chunk
                                    first_sentence_in_chunk = find_best_match_sentence(chunk, first_sentence, threshold=0.9)
                                    last_sentence_in_chunk = find_best_match_sentence(chunk, last_sentence, threshold=0.9)

                                    if first_sentence_in_chunk and last_sentence_in_chunk:
                                        try:
                                            original_text = chunk[chunk.index(first_sentence_in_chunk):chunk.index(last_sentence_in_chunk) + len(last_sentence_in_chunk)]
                                        except ValueError as e:
                                            # Log full sentences when extraction fails
                                            logger.error(f"[restore_from_cache] Error extracting text from chunk: {e}. First: '{first_sentence_in_chunk}', Last: '{last_sentence_in_chunk}'")
                                            original_text = ''
                                    else:
                                        logger.error(f"[restore_from_cache] Cannot find sentences in chunk. First found: {first_sentence_in_chunk is not None}, Last found: {last_sentence_in_chunk is not None}")
                                        original_text = ''
                                else:
                                    logger.error(f"[restore_from_cache] Skipping plot because sentence matching failed even after LLM fallback. First: {first_sentence is None}, Last: {last_sentence is None}")
                                    original_text = ''

                                # Create plot object
                                plot = {
                                    'text': original_text,
                                    'summary': unprocessed_plot['summary'],
                                    'prominence': unprocessed_plot['prominence'],
                                    'key_characters': unprocessed_plot['key_characters'],
                                    'chapter': chapter_title,
                                    'conversation': unprocessed_plot['conversation'],
                                    'state': unprocessed_plot['state']
                                }

                                plots.append(plot)

                            print_json(response)
                            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

                            return chapter_beginnings, plots, remaining_chunk

                        except Exception as e:
                            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
                            return False

                    # Extract chunk from prompt
                    chunk = prompt.split('==Truncated plot from previous chunk (to be finished)==')[0].split('==Chunk of Book Content==')[-1].strip(' \n')

                    # Parse response to get plots
                    res = parse_response(response, chunk, book)

                    if res :
                        chapter_beginnings, plots, remaining_chunk = res
                    else:
                        continue

                    # Process extracted plots
                    for plot in plots:
                        plot['state'] = 'finished'
                        plot['i_chunk'] = -1
                        # Find which chunk this plot belongs to
                        for i_chunk, another_chunk in enumerate(chunks):
                            if another_chunk.strip(' \n').endswith(chunk[-100:]):
                                plot['i_chunk'] = i_chunk
                                break

                    fetched_plots.extend(plots)
                    break

    # Find chunk indices for original plots
    for plot in results['plots']:
        plot['i_chunk'] = -1
        for i_chunk, chunk in enumerate(chunks):
            if plot['text'][-100:] in chunk:
                plot['i_chunk'] = i_chunk
                break

    # Merge and sort all plots
    logger.info(f'Number of Original Plots: {len(results["plots"])}, Fetched New Plots: {len(fetched_plots)}, Total Plots: {len(results["plots"]) + len(fetched_plots)}')

    new_plots = results['plots'] + fetched_plots
    new_plots = sorted(new_plots, key=lambda x: x['i_chunk'])

    results['plots'] = new_plots

    # Save restored results (together with the original results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return

def assemble(book):
    """
    Assemble the extracted plot data for a book into the final structured data.

    This function:
    1. Processes extracted plots and conversations
    2. Enhances conversation scenarios and character motivations using LLM
    3. Normalizes and standardizes character names
    4. Generates character profiles and datasets
    5. Saves the final assembled data

    Args:
        book (dict): Book data containing title, author and content

    Returns:
        None: Results are saved to disk at {args.output_dir}/processed-data/{book_title}.json
    """
    # Set up caching for this book
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    os.makedirs(f'{args.output_dir}/processed-data', exist_ok=True)

    save_path = f'{args.output_dir}/processed-data/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
        return

    logger.info(f"Assembling book: {book['title']}")

    # Load extracted plot data
    with open(f'{args.output_dir}/extracted-data/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    plots = results['plots']

    # Detect language from first plot text
    if len(plots) > 0:
        from utils import lang_detect
        language = lang_detect(plots[0]['text'][:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    # Normalize conversation format and enhance scenarios/motivations
    for plot in plots:
        # Convert single conversation dict to list format
        if isinstance(plot['conversation'], dict) and 'scenario' in plot['conversation']:
            plot['conversation'] = [plot['conversation']]

        if plot['conversation'] is None:
            continue
        for conversation in plot['conversation']:
            # Prepare input for conversation enhancement
            input_conversation = {
                'plot_summary': plot['summary'],
                'character_information': plot['key_characters'],
                **conversation
            }

            # Get character names for this conversation
            conv_key_characters = [
                _.get('name', _.get('character', ''))
                for _ in conversation['key_characters']
                if 'name' in _ or 'character' in _
            ]

            # Generate prompt for enhancing scenario and motivations
            prompt = f"""
Given a conversation from {book['title']}, enhance the scene setup and characters' motivations to create a comprehensive foundation for dramatic performance, i.e., to provide necessary background for actors to act out the conversation:

1. Review the provided conversation and contextual details thoroughly.
2. Expand the 'scenario' with rich situational context that actors need to convincingly perform the scene. Focus on essential background information, while excluding future details to be portrayed in the conversation.
3. Enhance each character's 'motivation' section with their complete mental and emotional state, including their feelings, ideas, objectives, topics they want to discuss, and information they want to convey. Align with their established character and role in the plot.

===Output Format===
Please provide the output in the following JSON format:
{{
    "scenario": "A detailed scene-setting description that provides actors with essential context and atmosphere (< 200 words). Include all necessary background information while excluding future information to be revealed in the conversation.",
    "key_characters": [
        {{
            "name": "Character name",
            "motivation": "The character's complete mental and emotional state before the conversation (< 100 words). Including their feelings, motivations, objectives, and information they want to convey or discuss."
        }}
    ],
}}

===Requirements===
1. Adhere strictly to the specified output JSON format.
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. In the OUTPUT, keep the character names unchanged.
4. Output in the same language as the input.
5. Ensure the key_characters include exactly the same characters as the input, including {conv_key_characters}.

===Input Conversation and Background===
{json.dumps(input_conversation, ensure_ascii=False, indent=2)}
"""

            # Helper function to validate enhanced conversation response
            from utils import extract_json
            def parse_response(response, characters, **kwargs):
                try:
                    assert 'scenario' in response
                    assert 'key_characters' in response
                    key_characters = {_['name']: _['motivation'] for _ in response['key_characters']}
                    for character in characters:
                        assert character in key_characters
                    return response
                except:
                    return False

            # Get enhanced conversation from LLM
            response = get_response_json(
                [extract_json, parse_response],
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                characters=conv_key_characters,
                max_retry=5
            )

            # Normalize character name field
            try:
                for chara in response['key_characters']:
                    if 'name' not in chara and 'character' in chara:
                        chara['name'] = chara.pop('character')
            except:
                continue

            # Update conversation with enhanced content
            conversation['scenario'] = response['scenario']
            enhanced_motivations = {chara['name']: chara['motivation'] for chara in response['key_characters']}
            for chara in conversation['key_characters']:
                if 'name' not in chara and 'character' in chara:
                    chara['name'] = chara.pop('character')
                chara['motivation'] = enhanced_motivations[chara['name']]

    # Filter out plots with invalid character data
    plots = [
        p for p in plots
        if all(['name' in c for c in p['key_characters']])
        and len(p['key_characters']) == len(set([c['name'] for c in p['key_characters']]))
    ]

    # Split plots into train/test sets
    split_index = int(len(plots) * 0.9)

    # Collect all character names
    character_names = set()
    for plot in plots:
        # Add names from plot key characters
        for character in plot['key_characters']:
            character_names.add(character['name'])

        # Normalize conversation format
        if isinstance(plot['conversation'], dict) and all([k in plot['conversation'] for k in ['key_characters', 'dialogues']]):
            plot['conversation'] = [plot['conversation']]

        # Process each conversation
        for conversation in plot['conversation']:
            # Add names from conversation key characters
            for key_character in conversation['key_characters']:
                try:
                    character_names.add(key_character['name'])
                except:
                    logger.error(f"Error: {key_character}")
                    if 'character' in key_character:
                        key_character['name'] = key_character.pop('character')

            # Add names from dialogue utterances
            for utterance in conversation['dialogues']:
                try:
                    character_names.add(utterance['character'])
                except:
                    logger.error(f"Error: {utterance}")
                    if 'name' in utterance:
                        utterance['character'] = utterance.pop('name')

    # Sort character names
    character_names = sorted(character_names)

    # Generate prompt for standardizing character names
    prompt = """Given a list of character names, titles, or form of address, your task is to: i) generate a list of named characters with their official names (in {language}); ii) For each name in the given list, align it with the official character name if it refers to a named character, or denote it as "impersonal" otherwise.

===Output Format===
Please provide the output in the following JSON format:
{{
    "named_characters": [
        The list of named characters with their official names. Each character should appear only once.
    ],
    "to_official_name": {{
        "The name in the list": "The official name of the character, or 'impersonal' if it does not refer to a named character."
    }}
}}
===Input===
{character_names}
"""

    prompt = prompt.replace('{character_names}', str(character_names))
    prompt = prompt.replace('{language}', language)

    # Helper function to validate name standardization response
    def parse_response(response, **kwargs):
        if 'named_characters' in response:
            return response
        else:
            return False

    # Get standardized names from LLM
    from utils import extract_json
    response = get_response_json(
        [extract_json, parse_response],
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_retry=5
    )

    # Extract official names, falling back to candidate model if needed
    try:
        official_names = response['named_characters']
    except:
        response = get_response_json(
            [extract_json, parse_response],
            model=args.candidate_model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=5
        )
        official_names = response['named_characters']

    # Normalize list values in name mapping
    for k, v in response['to_official_name'].items():
        if isinstance(v, list):
            response['to_official_name'][k] = v[0]

    # Add missing official names
    official_names += [
        n for n in set(response['to_official_name'].values()) - set(official_names)
        if n.lower() not in ['impersonal', 'environment']
    ]

    to_official_name = response['to_official_name']

    # Handle names missing from the mapping
    missing_names = []
    for name in character_names:
        if name not in to_official_name:
            print(f"Warning: {name} not included in to_official_name")
            missing_names.append(name)

    print(f"Missing names: {missing_names}")

    # Find closest matches for missing names
    for name in missing_names:
        closest_name = find_best_match_passage(official_names, name)
        print(f"Closest name: {closest_name}")
        if closest_name != -1:
            to_official_name[name] = official_names[closest_name]
        else:
            to_official_name[name] = "impersonal"

    # Initialize character datasets
    character_datasets = {
        character: {
            "plots": [],
            "conversations": [],
            "utterances": []
        } for character in official_names
    }

    # Populate character datasets
    for i_p, plot in enumerate(plots):
        # Process plot key characters
        for character in plot['key_characters']:
            if to_official_name[character['name']] != "impersonal":
                character['name'] = to_official_name[character['name']]
                character_datasets[character['name']]['plots'].append((i_p, character))

        # Process conversations
        for i_c, conversation in enumerate(plot['conversation']):
            # Process conversation key characters
            for character in conversation['key_characters']:
                if to_official_name[character['name']] != "impersonal":
                    character['name'] = to_official_name[character['name']]
                    character_datasets[character['name']]['conversations'].append((i_p, i_c, character))

            # Process utterances
            for i_u, utterance in enumerate(conversation['dialogues']):
                if to_official_name[utterance['character']] != "impersonal":
                    utterance['character'] = to_official_name[utterance['character']]
                    character_datasets[utterance['character']]['utterances'].append((i_p, i_c, i_u, utterance))

    # Generate character profiles prompt template
    prompt = """Please generate a structured character profile for {character_name} from "{book_title}" in JSON format. The profile should be based on either your existing knowledge of the character or the provided information, without fabricating or inferring any inaccurate or uncertain details.

**IMPORTANT CONSTRAINT**: If this character is not a major character in the book (i.e., neither your existing knowledge nor the provided character data can effectively support extracting the corresponding fields), you must avoid hallucination at all costs. Even if some fields remain empty strings, do not fabricate or infer any information that is not clearly supported by the available evidence. It is better to leave fields empty than to generate inaccurate or speculative content.

You will be provided with summaries and dialogues of some key plots in the book as reference:

{character_data}

The profile must be output as a valid JSON object with the following fields (all fields should be in {language}):

1. **name**: The character's name (should be "{character_name}").
2. **short_description**: A concise, condensed summary of the character in 1-2 sentences, capturing their essence and significance.
3. **identity_appearance**: Name, age, gender, occupation, overall appearance, and key physical traits. (Requirement: 1–several complete natural-language sentences with both density and vivid imagery).
4. **personality_psychology**: Personality traits, behavioral style, typical emotional reaction patterns, values, or preferences. (Requirement: Highlight traits that show up in dialogue).
5. **speaking_style**: Rhythm, tone, and lexical habits (e.g., very formal, sarcastic, indirect). (Requirement: Provide 2–4 specific, actionable descriptions).
6. **abilities_interests_achievements**: Hard/soft skills, hobbies, and representative achievements. (Requirement: These should matter in the plot).
7. **social_historical_context**: Social environment, era, family / class / cultural background. (Requirement: Emphasize factors relevant to this character's story).
8. **personal_history_arc**: Important past experiences and the current stage of the character arc.
9. **relationships**: Natural-language description of relations with other characters appearing in the work.

Output format example:
{
  "name": "Character Name",
  "short_description": "A brief summary of the character in 1-2 sentences.",
  "identity_appearance": "Detailed description of name, age, gender, occupation, and physical traits.",
  "personality_psychology": "Description of personality traits, behavioral style, emotional patterns, values, and preferences.",
  "speaking_style": "Description of rhythm, tone, and lexical habits with 2-4 specific examples.",
  "abilities_interests_achievements": "Description of skills, hobbies, and achievements relevant to the plot.",
  "social_historical_context": "Description of social environment, era, family, class, and cultural background.",
  "personal_history_arc": "Description of important past experiences and current stage of character development.",
  "relationships": "Description of relations with other characters in the work."
}

Output only a valid JSON object, with no additional text before or after the JSON.
"""

    # Generate character profiles
    for character_name, character_data in character_datasets.items():
        # Get plots involving this character
        involved_plots = sorted(set(
            [p[0] for p in character_data['plots']] +
            [c[0] for c in character_data['conversations']] +
            [u[0] for u in character_data['utterances']]
        ))

        # Filter to training set plots only
        involved_plots = [i_p for i_p in involved_plots if i_p < split_index]

        # Collect plot information
        plot_infos = []
        for i_p in involved_plots:
            plot = plots[i_p]

            plot_info = {
                "plot": plot['summary'],
            }

            # Add character-specific information
            for key_character_info in plot['key_characters']:
                if key_character_info['name'] == character_name:
                    plot_info["character_experience"] = key_character_info

            # Add relevant conversations
            plot_info["conversation"] = []
            for conversation in plot['conversation']:
                if character_name in conversation['key_characters'] or character_name in [u['character'] for u in conversation['dialogues']]:
                    plot_info["conversation"].append(conversation)

            plot_infos.append(plot_info)

        # Check and truncate character_data if it exceeds maximum context length
        # Maximum context length is 128000 tokens, reserve 8000 for output and safety margin
        MAX_CONTEXT_TOKENS = 120000

        # Calculate base prompt tokens (without character_data)
        base_prompt = prompt.replace("{character_name}", character_name).replace("{book_title}", book["title"]).replace("{character_data}", "").replace("{language}", language)
        base_prompt_tokens = num_tokens_from_string(base_prompt)
        available_tokens = MAX_CONTEXT_TOKENS - base_prompt_tokens

        # Convert plot_infos to JSON and check token count
        character_data_json = json.dumps(plot_infos, ensure_ascii=False, indent=2)
        character_data_tokens = num_tokens_from_string(character_data_json)
        original_plot_count = len(plot_infos)

        # If exceeds limit, truncate plot_infos
        if character_data_tokens > available_tokens:
            logger.warning(f"Character data for {character_name} exceeds token limit ({character_data_tokens} > {available_tokens}). Truncating...")
            # Try to keep as many plots as possible while staying under the limit
            truncated_plot_infos = []
            for plot_info in plot_infos:
                test_plot_infos = truncated_plot_infos + [plot_info]
                test_json = json.dumps(test_plot_infos, ensure_ascii=False, indent=2)
                test_tokens = num_tokens_from_string(test_json)
                if test_tokens <= available_tokens:
                    truncated_plot_infos.append(plot_info)
                else:
                    # If adding this plot would exceed, try to include a reduced version
                    # First try removing some conversations from this plot
                    reduced_plot_info = copy.deepcopy(plot_info)
                    if 'conversation' in reduced_plot_info and len(reduced_plot_info['conversation']) > 0:
                        # Keep only the first half of conversations
                        reduced_plot_info['conversation'] = reduced_plot_info['conversation'][:len(reduced_plot_info['conversation'])//2]
                        test_plot_infos = truncated_plot_infos + [reduced_plot_info]
                        test_json = json.dumps(test_plot_infos, ensure_ascii=False, indent=2)
                        test_tokens = num_tokens_from_string(test_json)
                        if test_tokens <= available_tokens:
                            truncated_plot_infos.append(reduced_plot_info)
                    break

            plot_infos = truncated_plot_infos
            character_data_json = json.dumps(plot_infos, ensure_ascii=False, indent=2)
            final_tokens = num_tokens_from_string(character_data_json)
            logger.info(f"Truncated character data for {character_name}: {character_data_tokens} -> {final_tokens} tokens (kept {len(plot_infos)}/{original_plot_count} plots)")

        character_prompt = prompt.replace("{character_name}", character_name).replace("{book_title}", book["title"]).replace("{character_data}", character_data_json).replace("{language}", language)

        print(character_prompt)

        # Get profile from LLM with retries
        nth_generation = 0
        while True:
            if nth_generation > 0:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}],
                    nth_generation=nth_generation
                )
            else:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}]
                )

            try:
                profile_json_str = profile.strip()

                # Remove markdown code blocks if present
                if profile_json_str.startswith("```json"):
                    profile_json_str = profile_json_str[7:].strip()
                elif profile_json_str.startswith("```"):
                    profile_json_str = profile_json_str[3:].strip()
                if profile_json_str.endswith("```"):
                    profile_json_str = profile_json_str[:-3].strip()

                # Skip if it's an apology message
                if profile_json_str.startswith('I apologize') or profile_json_str.startswith('Sorry'):
                    profile_json_str = ''

                # Simple JSON parsing
                if profile_json_str:
                    profile_dict = json.loads(profile_json_str)
                    character_datasets[character_name]['profile'] = profile_dict
                else:
                    character_datasets[character_name]['profile'] = {}
                break
            except (json.JSONDecodeError, ValueError) as e:
                nth_generation += 1
                if nth_generation > 5:
                    character_datasets[character_name]['profile'] = {}
                    break
                continue

    # Update data format for readability
    for character_name, character_data in character_datasets.items():
        # Flatten plot data
        for i, plot in enumerate(character_data['plots']):
            character_data['plots'][i] = plot[-1]
            character_data['plots'][i]['i_p'] = plot[0]

        # Flatten conversation data
        for i, conversation in enumerate(character_data['conversations']):
            character_data['conversations'][i] = conversation[-1]
            character_data['conversations'][i]['i_p'] = conversation[0]
            character_data['conversations'][i]['i_c'] = conversation[1]

        # Flatten utterance data
        for i, utterance in enumerate(character_data['utterances']):
            character_data['utterances'][i] = utterance[-1]
            character_data['utterances'][i]['i_p'] = utterance[0]
            character_data['utterances'][i]['i_c'] = utterance[1]
            character_data['utterances'][i]['i_u'] = utterance[2]

    # Save final results
    results['character_datasets'] = character_datasets
    results['split_plot_index'] = split_index

    results.pop("chapter_beginnings")
    results.pop("fail_to_parse_responses")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



if __name__ == '__main__':

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input data
    with jsonlines.open(args.input, mode='r') as reader:
        books_data = list(reader)

    # Clean book titles
    for book in books_data:
        book['title'] = book['title'].replace('/', '-').replace(':', '_').replace('.', ' ')

    # Apply max_books limit if specified
    if args.max_books is not None and args.max_books > 0:
        original_count = len(books_data)
        books_data = books_data[:args.max_books]
        logger.info(f"Limited processing to {len(books_data)} books (from {original_count} total)")

    logger.info(f"Processing {len(books_data)} books")

    def process_book(book):
        try:
            extract(book)
            restore_from_cache(book)
            result = assemble(book)
            logger.info(f"Successfully processed book: {book.get('title', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error processing book {book.get('title', 'Unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Starting parallel processing with {args.num_workers} workers")

        # Process books in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            processed_books = list(tqdm(
                executor.map(process_book, books_data),
                total=len(books_data),
                desc="Processing books"
            ))
    else:
        processed_books = []
        for book in tqdm(books_data):
            processed_book = process_book(book)
            processed_books.append(processed_book)
