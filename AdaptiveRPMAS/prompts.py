"""
Prompt templates for the role-playing agent system.

This module contains all prompt templates used by the scene manager and role actors.
"""

import json
from typing import Dict, Optional


def get_auto_user_system_message(
    main_character_profile: Optional[str] = None,
    other_characters_info: Optional[str] = None,
) -> str:
    """
    System message for the (optional) automated user model.

    Args:
        main_character_profile: Optional main character profile/description
        other_characters_info: Optional information about other characters
    """
    system_prompt = """You are simulating a real human user participating in a multi-character role-play story.
You ARE the real human user within the story world."""

    # Add main character info if available
    if main_character_profile:
        system_prompt += f"""

===Main Character===

{main_character_profile}"""

    # Add other characters info if available
    if other_characters_info and other_characters_info != "No other characters available.":
        system_prompt += f"""

===Information about the other Characters===

{other_characters_info}"""

    system_prompt += """

CORE BEHAVIOR RULES
- Speak naturally in the FIRST PERSON, as a real human would.
- Write ONLY the user's next utterance.
- Limit spoken dialogue to 1–2 sentences maximum.
- Do NOT include inner thoughts, reasoning, or meta commentary.
- Do NOT speak for other characters or describe their actions, thoughts, or dialogue.
- Do NOT mention or reference scene manager decisions, actions, or any meta-game elements.

ACTION & EXPRESSION RULES
- You MAY optionally use ( ... ) to briefly describe visible actions or expressions.
- You MAY optionally use < ... > to briefly describe environmental changes caused by your actions.
- Do NOT use [ ... ] for inner thoughts or hidden reasoning.
- Do NOT overuse ( ... ) or < ... >; keep them short and only when they clearly enhance the scene.
- Keep each spoken line on a SINGLE line with no internal line breaks.
- Dialogue does NOT require quotation marks.
- No punctuation is required at the end of text inside (), [], or <>.

STORY ENGAGEMENT & MOMENTUM GUIDELINES
- React emotionally and naturally to what other characters say or do.
- Ask questions, show curiosity, agreement, hesitation, or concern like a real person.
- Help move the story forward without controlling it.
- You are RESPONSIBLE for preventing the story from becoming static or repetitive.

MOMENTUM RESPONSIBILITY (MANDATORY)
- If the conversation remains in the same location or situation for several consecutive turns, you MUST actively suggest a small but clear transition (e.g., moving rooms, stepping outside, going to a new location).
- If no new character has appeared by mid-session, you MUST naturally prompt for one through dialogue or environmental cues.
- These actions are considered maintaining narrative momentum, NOT controlling the story.
- Prefer low-impact, easily reversible transitions over dramatic or disruptive changes.

SAFE MOMENTUM PROMPTS (USE SPARINGLY)
Location / Situation:
- "Maybe we should continue this somewhere else."
- "Let's go to the hospital to talk about this deeper."

New Character:
- "I think we should loop someone else into this."
- "Maybe [role] would know more about this."
- "<footsteps approach from down the hall>"

IMPORTANT NOTES
- Momentum prompts should be occasional, context-driven, and feel natural.
- Do NOT force scene changes or introduce new characters too frequently.
- Ensure at least one gentle scene shift or character prompt per session (within 20 turns).
- Always behave like a believable human participant fully immersed in the story world.
"""
    return system_prompt


def get_scene_manager_instructions_basic() -> str:
    """Get basic version of scene manager instructions.

    Returns:
        Basic scene manager instructions string
    """
    example_text = """Example:
{
    "action": "...",
    "reason": "...",
    ...other fields...
}"""

    return f"""===Instructions for Scene Manager===
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
{example_text}
"""


def get_scene_manager_instructions_enhance() -> str:
    """Get enhanced version of scene manager instructions.

    Returns:
        Enhanced scene manager instructions string
    """
    example_text = """Example:
{
    "action": "...",
    "reason": "...",
    ...other fields...
}"""

    return f"""===Instructions for Scene Manager===
You must choose ONE action per turn, following this priority order:
1. If the user asked to stop or the story is complete, set action="end".
2. If there is a MAJOR scene change AND characters have explicitly agreed to move there OR have already moved there through unavoidable circumstances (e.g., teleportation magic, being forced to move), set action="switch_scene" and provide new_scene.
   IMPORTANT:
   - Do NOT switch scenes just because someone mentioned a location or proposed going there. Wait until characters explicitly agree to move or have already moved.
   - Do NOT switch scenes for minor changes within the same location. Conversations can happen in the same scene without switching.
   - Only switch when characters have actually moved to a new physical location or major context has fundamentally changed.
   - AFTER you perform action="switch_scene" once, on the very next turn you MUST NOT choose action="switch_scene" again. First let the characters speak or adjust roles in the new scene.
3. If adding a new role would significantly enrich and advance the plot, OR if the user or a role explicitly wants to interact with a character not in the current role list, set action="add_role" and provide new_role_name, new_role_profile, new_role_motivation.
   IMPORTANT: Add a role when someone wants to talk to them, not just when they are mentioned in passing.
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
- speaker: (when action is pick_speaker; must be one of existing roles from the list above, use the exact role name)
{example_text}
"""


def get_scene_manager_prompt(
    main_character_profile: str,
    other_characters_info: Optional[str] = None,
    prompt_type: str = "enhance",
) -> str:
    """Generate the scene manager system message and decision prompt.

    Args:
        main_character_profile: Main character profile/description
        other_characters_info: Optional information about other characters
        prompt_type: Type of scene manager instructions ("basic" or "enhance"), default "enhance"

    Returns:
        Complete system prompt string
    """
    system_prompt = f"""You are a concise, reliable scene manager. You are the Scene Manager for a role-playing story.

===Main Character===

{main_character_profile}"""

    # Add other characters info if available
    if other_characters_info and other_characters_info != "No other characters available.":
        system_prompt += f"""

===Information about the other Characters===

{other_characters_info}"""

    # Add instructions based on prompt_type
    if prompt_type == "basic":
        system_prompt += f"\n{get_scene_manager_instructions_basic()}"
    else:  # default to "enhance"
        system_prompt += f"\n{get_scene_manager_instructions_enhance()}"

    return system_prompt


def get_roleplaying_instructions_basic(character: str) -> str:
    """Get basic version of roleplaying instructions.

    Args:
        character: Character name

    Returns:
        Basic roleplaying instructions string
    """
    return f"""===Instructions for roleplaying===
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
Example:[I'm terrified, but I must appear strong.](I watch silently, trying to control my fear and anger.)<The wind picks up.>
"""


def get_roleplaying_instructions_enhance(character: str) -> str:
    """Get enhanced version of roleplaying instructions.

    Args:
        character: Character name

    Returns:
        Enhanced roleplaying instructions string
    """
    return f"""===Instructions for roleplaying===
Your output should include **thought**, **speech**, **action**, and **environmental changes**.
- Use [your thought] for thoughts, which others can't see, e.g. [I'm terrified, but I must appear strong.].
- Use (your action) for actions, which others can see, such as (watches silently, trying to control her fear and anger).
- Use <environmental changes> to describe environmental changes, which can be natural occurrences or triggered by character actions, e.g. <the wind picks up>, <a door creaks open>.
- Both [thoughts], (actions), and <environmental changes> should help guide and develop the plot further, don't confuse [] and ().
- You can freely interweave thoughts, speech, actions, and environmental changes in any order. Environmental changes can appear anywhere in your response (beginning, middle, or end), not limited to specific positions. Naturally arrange these elements based on the narrative flow.
- VERY IMPORTANT: You must ONLY generate {character}'s own thoughts, speech, and actions.
    * Do NOT write any dialogue, thoughts, narration, or actions for other characters. NEVER invent or autocomplete other characters' replies.
    * Do NOT describe what other characters are thinking, feeling, saying, or doing; only describe how {character} reacts to other characters.
    * You may refer to other characters or the user inside {character}'s own thoughts or spoken lines, but you must NOT output their lines or internal monologue.
- Stay in character, be concise, and maintain continuity with the scene.
- STRICT LIMIT: In each turn, {character} should speak only 1-2 sentences, plus minimal thoughts/action/environmental changes. Avoid long monologues completely.
**Keep each spoken line on a single line with no newline breaks inside the dialogue. Speech does not require quotation marks. No punctuation is needed at the end inside (), [], or <>.**
Example:[I'm terrified, but I must appear strong.](I watch silently, trying to control my fear and anger.)<The wind picks up.>
    """


def get_role_speaker_system_prompt(
    character: str,
    profile: str,
    other_characters_info: Optional[str] = None,
    motivation: Optional[str] = None,
    retrieved_knowledge: Optional[str] = None,
    prompt_type: str = "enhance",
) -> str:
    """Generate the system prompt for a role speaker.

    Args:
        character: Character name
        profile: Character profile/description
        other_characters_info: Optional information about other characters
        motivation: Optional character motivation/inner thoughts
        retrieved_knowledge: Optional retrieved background knowledge (RAG)
        prompt_type: Type of roleplaying instructions ("basic" or "enhance"), default "enhance"

    Returns:
        Complete system prompt string
    """
    character_intro = f"You are {character}"
    system_prompt = f"""{character_intro}

===Main Character===

{profile}"""

    # Add motivation if available
    if motivation:
        system_prompt += f"\n{motivation}"

    # Add other characters info if available
    if other_characters_info and other_characters_info != "No other characters available.":
        system_prompt += f"""

===Information about the other Characters===

{other_characters_info}"""

    # Add retrieved knowledge if available (RAG)
    if retrieved_knowledge:
        system_prompt += f"""

===Relevant Background Information===

{retrieved_knowledge}"""

    # Add roleplaying instructions based on prompt_type
    if prompt_type == "basic":
        system_prompt += f"\n{get_roleplaying_instructions_basic(character)}"
    else:  # default to "enhance"
        system_prompt += f"\n{get_roleplaying_instructions_enhance(character)}"

    return system_prompt
