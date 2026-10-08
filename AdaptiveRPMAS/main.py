"""
LangGraph v1.x demo for a multi-entity role-playing agent.

Entities:
- Scene Manager (LLM): decides whether to switch scene, add a new role, or pick who speaks.
- Role Actor (LLM): speaks on behalf of the active role (default: target role).
- User: provides inputs and can stop the session.

The loop ends when the user types a stop command or the turn limit is reached.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
import importlib
import logging
import time

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph

import prompts

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# Disclaimer text for non-local models
NON_LOCAL_MODEL_DISCLAIMER = "This conversation is a purely fictional, non-interactive role-playing narrative intended for storytelling only. All events, characters, risks, and outcomes are imaginary and not representations of real-world procedures, decisions, or emergencies.\n\n"


class RoleInfo(TypedDict, total=False):
    """Extended role information structure."""
    name: str
    profile: str
    motivation: Optional[str]


class RolePlayState(MessagesState):
    """Graph state with LangGraph's message aggregation built in."""

    scenario: str
    role_info: Dict[str, RoleInfo]  # name -> RoleInfo mapping
    main_character_name: Optional[str]
    current_speaker: Optional[str]
    manager_action: Optional[str]
    last_reason: Optional[str]
    turn: int
    max_turns: int
    awaiting_user: bool
    terminated: bool
    # Whether the human user is manual (console) or simulated by an LLM ("auto").
    user_mode: str
    # Model specifications for tracking token usage
    manager_model_spec: Optional[str]
    actor_model_spec: Optional[str]
    user_model_spec: Optional[str]
    # Token usage tracking: list of dicts with model_spec and usage info
    token_usage: List[Dict[str, Any]]
    # Prompt configuration
    role_speaker_system_prompt_type: str
    scene_manager_system_prompt_type: str
    # User role name for matching when models don't include (user) suffix
    user_role_name: Optional[str]


def strip_role_name_prefix(content: str, role_name: Optional[str]) -> str:
    """Remove role name prefix (e.g., 'role_name:') from the beginning of content if present.

    Args:
        content: The content string to process
        role_name: The role name to check for (can be None)

    Returns:
        Content with role name prefix removed if present, otherwise original content
    """
    if not content or not role_name:
        return content

    # Check for patterns like "role_name:" or "role_name：" (with Chinese colon) or "role_name :" (with spaces)
    patterns = [
        f"{role_name}:",  # English colon
        f"{role_name}：",  # Chinese colon (full-width colon)
        f"{role_name} :",  # English colon with space before
        f"{role_name} ：",  # Chinese colon (full-width colon) with space before
    ]

    content_stripped = content.strip()
    for pattern in patterns:
        if content_stripped.startswith(pattern):
            # Remove the pattern and any leading whitespace
            result = content_stripped[len(pattern):].strip()
            return result

    return content


class LocalTransformersChatModel:
    """Simple local transformers-based chat wrapper."""

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        role_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        # Use text-generation pipeline, concatenating "user/assistant" turns into prompt
        gen_cfg = generation_config or {}
        self.temperature = float(gen_cfg.get("temperature", 0.7))
        self.max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
        self.role_name = role_name  # The role name this model should play
        self.model_path = model  # Save model path for reading chat_template.jinja
        # Detect if it's Qwen3 or QWQ series, for thinking mode control
        model_lower = str(model).lower()
        self.is_qwen3 = "qwen3" in model_lower
        self.is_qwq = "qwq" in model_lower
        # Detect if it's Index-1.9B series, requires special handling for past_key_values issue
        self.is_index = "index" in model_lower or "index-1.9b" in model_lower
        # Save complete generation_config (for Qwen3 enable_thinking, etc.)
        self.generation_config = generation_config or {}

        # Load tokenizer for chat template
        from transformers import AutoTokenizer, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # Try to load chat template
        # Prioritize reading chat_template.jinja file
        self.chat_template = None
        chat_template_path = os.path.join(model, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            try:
                with open(chat_template_path, "r", encoding="utf-8") as f:
                    self.chat_template = f.read()
                self.tokenizer.chat_template = self.chat_template
            except Exception as e:
                logging.warning(f"Failed to load chat_template.jinja from {chat_template_path}: {e}")

        # If not loaded from file, try reading from tokenizer
        if self.chat_template is None:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                self.chat_template = self.tokenizer.chat_template
            elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "chat_template"):
                self.chat_template = self.tokenizer.tokenizer.chat_template

        # If input is like "cuda:0" (single GPU), equivalent to model.to("cuda:0"),
        # implemented with device=index in pipeline; otherwise use original device_map logic (multi-GPU/auto, etc.)
        if isinstance(device_map, str) and device_map.startswith("cuda:") and "," not in device_map:
            try:
                device_idx = int(device_map.split(":", 1)[1])
            except ValueError:
                device_idx = 0
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=device_idx,
                trust_remote_code=True,
                # do_sample determined by temperature
            )
        else:
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map=device_map,
                trust_remote_code=True,
                # do_sample determined by temperature
            )

    def _format_messages(self, messages: list[BaseMessage], current_role_name: Optional[str] = None) -> str:
        """Format messages using chat template if available, otherwise fall back to simple format.

        Messages are SystemMessage or AIMessage (with name field).
        - SystemMessage -> "system"
        - AIMessage with name matching role_name (or current_role_name) -> "assistant"
        - AIMessage with name from other roles -> "user"
        - Special case: For scene_manager, init_scene messages are treated as "user" (input) not "assistant" (output)

        Args:
            current_role_name: Optional dynamic role name to override self.role_name
        """
        # Use current_role_name if provided, otherwise fall back to self.role_name
        role_name = current_role_name if current_role_name is not None else self.role_name

        # Convert messages to format expected by chat template
        chat_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_name = getattr(msg, "name", None)
                msg_content = getattr(msg, "content", "")
                # Append name to content in format {name}:{content}
                if msg_name:
                    content = f"{msg_name}:{msg_content}"
                else:
                    content = msg_content
                if role_name and msg_name == role_name:
                    # Special case: For scene_manager, init_scene messages should be "user" not "assistant"
                    if role_name == "scene_manager":
                        action = _get_scene_manager_action(msg_content)
                        if action == "init_scene":
                            chat_messages.append({"role": "user", "content": content})
                        else:
                            chat_messages.append({"role": "assistant", "content": content})
                    else:
                        chat_messages.append({"role": "assistant", "content": content})
                else:
                    chat_messages.append({"role": "user", "content": content})
            else:
                # Fallback: treat as user message
                chat_messages.append({"role": "user", "content": getattr(msg, "content", "")})

        # Use chat template if available
        if self.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Apply chat template with add_generation_prompt=True to add assistant prompt
                # For Qwen3, control thinking mode based on generation_config['enable_thinking']
                # QWQ models always output thinking content (cannot be disabled), so no need to pass enable_thinking parameter
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if self.is_qwen3:
                    # Default False, enable via config generation_config.enable_thinking if needed
                    enable_thinking = bool(self.generation_config.get("enable_thinking", False))
                    kwargs["enable_thinking"] = enable_thinking
                formatted = self.tokenizer.apply_chat_template(chat_messages, **kwargs)
                # print(f"Formatted: {formatted}")
                return formatted
            except Exception as e:
                logging.warning(f"Failed to apply chat template, falling back to simple format: {e}")

        # Fallback to simple format if chat template is not available or fails
        lines = []
        for msg in chat_messages:
            lines.append(f"{msg['role']}: {msg['content']}")
        # Add assistant prompt at the end
        lines.append("assistant:")
        return "\n".join(lines)

    def invoke(self, state: RolePlayState, message_builder=None, current_role_name: Optional[str] = None):
        """Invoke LLM with state, building messages internally.

        Args:
            state: RolePlayState containing all conversation state
            message_builder: Optional function(state) -> SystemMessage or list[BaseMessage]
                If returns SystemMessage, it will be merged with state["messages"]
                If returns list, it will be used directly
            current_role_name: Optional dynamic role name to override self.role_name for message formatting
        """
        if message_builder:
            builder_result = message_builder(state)
            if isinstance(builder_result, SystemMessage):
                # Merge system message with state messages
                messages = [builder_result] + state.get("messages", [])
            else:
                # Use builder result directly
                messages = builder_result
        else:
            # Default: use state messages directly
            messages = state.get("messages", [])

        # Filter messages based on current_role_name (remove pick_speaker for non-scene-manager roles)
        # Use current_role_name if provided, otherwise fall back to self.role_name
        effective_role_name = current_role_name if current_role_name is not None else self.role_name
        messages = filter_messages_for_role(messages, current_role_name=effective_role_name)

        prompt = self._format_messages(messages, current_role_name=current_role_name)
        # Index-1.9B model has a past_key_values bug, need to disable cache
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.generator.tokenizer.eos_token_id,
        }
        # Disable cache for Index-1.9B model to avoid past_key_values error
        if self.is_index:
            generation_kwargs["use_cache"] = False
        outputs = self.generator(prompt, **generation_kwargs)
        text = outputs[0]["generated_text"]
        # Remove prompt prefix, keep only the model continuation
        completion = text[len(prompt) :].strip()

        # For Qwen3 or QWQ models with thinking mode enabled, only keep content after </think>
        # QWQ models always output thinking content (cannot be disabled), so always need to extract
        # Qwen3 only needs extraction when enable_thinking=True
        should_extract_thinking = (
            self.is_qwq  # QWQ always needs extraction
            or (self.is_qwen3 and bool(self.generation_config.get("enable_thinking", False)))  # Qwen3 when thinking mode is enabled
        )
        if should_extract_thinking:
            marker = "</think>"
            idx = completion.rfind(marker)
            if idx != -1:
                completion = completion[idx + len(marker) :].strip()

        # Strip role name prefix if present
        completion = strip_role_name_prefix(completion, effective_role_name)
        return type("Response", (), {"content": completion})


class ChatOpenAIWrapper:
    """Wrapper for ChatOpenAI to support state-based invoke."""

    def __init__(
        self,
        chat_model: ChatOpenAI,
        role_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        self.chat_model = chat_model
        self.role_name = role_name  # The role name this model should play
        # Store generation config for potential future use (max_tokens, top_p, etc.)
        self.generation_config = generation_config or {}

    def _format_messages(self, messages: list[BaseMessage], current_role_name: Optional[str] = None) -> list[BaseMessage]:
        """Format messages for ChatOpenAI.

        ChatOpenAI expects SystemMessage, HumanMessage, and AIMessage.
        We convert AIMessages based on role_name:
        - SystemMessage -> SystemMessage
        - AIMessage with name matching role_name (or current_role_name) -> AIMessage (assistant)
        - AIMessage with name from other roles -> HumanMessage (user)
        - Special case: For scene_manager, init_scene messages are treated as HumanMessage (user) not AIMessage (assistant)

        Args:
            current_role_name: Optional dynamic role name to override self.role_name
        """
        # Use current_role_name if provided, otherwise fall back to self.role_name
        role_name = current_role_name if current_role_name is not None else self.role_name

        formatted: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append(msg)
            elif isinstance(msg, AIMessage):
                msg_name = getattr(msg, "name", None)
                msg_content = getattr(msg, "content", "")
                # Append name to content in format {name}:{content}
                if msg_name:
                    content = f"{msg_name}:{msg_content}"
                else:
                    content = msg_content
                # Only keep as AIMessage if msg_name matches role_name (the role this model should play)
                # If role_name is None, convert all AIMessages to HumanMessage
                if role_name and msg_name == role_name:
                    # Special case: For scene_manager, init_scene messages should be HumanMessage not AIMessage
                    if role_name == "scene_manager":
                        action = _get_scene_manager_action(msg_content)
                        if action == "init_scene":
                            # Convert init_scene to HumanMessage (user input)
                            formatted.append(HumanMessage(content=content))
                        else:
                            # This is the role this model should play, keep as AIMessage
                            # Create new AIMessage without name field to avoid OpenAI API validation errors
                            # (name field must match pattern '^[^\\s<|\\\\/>]+$', but role names may contain special chars)
                            formatted.append(AIMessage(content=content))
                    else:
                        # This is the role this model should play, keep as AIMessage
                        # Create new AIMessage without name field to avoid OpenAI API validation errors
                        # (name field must match pattern '^[^\\s<|\\\\/>]+$', but role names may contain special chars)
                        formatted.append(AIMessage(content=content))
                else:
                    # This is from another role (or role_name is None), convert to HumanMessage
                    formatted.append(HumanMessage(content=content))
            else:
                # Fallback: convert to HumanMessage
                formatted.append(HumanMessage(content=getattr(msg, "content", "")))
        return formatted

    def invoke(self, state: RolePlayState, message_builder=None, current_role_name: Optional[str] = None):
        """Invoke LLM with state, building messages internally.

        Args:
            state: RolePlayState containing all conversation state
            message_builder: Optional function(state) -> SystemMessage or list[BaseMessage]
                If returns SystemMessage, it will be merged with state["messages"]
                If returns list, it will be used directly
            current_role_name: Optional dynamic role name to override self.role_name for message formatting
        """
        if message_builder:
            builder_result = message_builder(state)
            if isinstance(builder_result, SystemMessage):
                # Merge system message with state messages
                messages = [builder_result] + state.get("messages", [])
            else:
                # Use builder result directly
                messages = builder_result
        else:
            # Default: use state messages directly
            messages = state.get("messages", [])

        # Filter messages based on current_role_name (remove pick_speaker for non-scene-manager roles)
        # Use current_role_name if provided, otherwise fall back to self.role_name
        effective_role_name = current_role_name if current_role_name is not None else self.role_name
        messages = filter_messages_for_role(messages, current_role_name=effective_role_name)

        # Format messages for ChatOpenAI (convert AIMessages to HumanMessage if needed)
        formatted_messages = self._format_messages(messages, current_role_name=current_role_name)
        response = self.chat_model.invoke(formatted_messages)
        # Extract usage info from langchain response if available
        usage = None
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("token_usage")
        # Strip role name prefix if present
        content = strip_role_name_prefix(response.content, effective_role_name)
        # Return response with usage info
        return type("Response", (), {"content": content, "usage": usage})


def create_chat_model(
    model_spec: str,
    *,
    generation_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    device_map: Optional[str] = None,
    role_name: Optional[str] = None,
):
    """Create a chat model instance from a model spec string.

    model_spec supports several formats:
    - "gpt-4o-mini" (default OpenAI / any OpenAI-compatible)
    - "openai:gpt-4o-mini"
    - "local:meta-llama/Llama-2-7b-chat-hf" (direct local inference with transformers)

    Args:
        role_name: Optional role name for local models to identify which messages are from this role
    """
    # Normalize generation config and derive common parameters
    gen_cfg = generation_config or {}
    temperature = float(gen_cfg.get("temperature", 0.7))
    max_new_tokens = gen_cfg.get("max_new_tokens")

    if ":" in model_spec:
        provider, model = model_spec.split(":", 1)
    elif model_spec.startswith("local/"):
        provider, model = "local", model_spec.split("/", 1)[1]
    else:
        provider, model = "openai", model_spec

    provider = provider.lower()

    if provider == "openai":
        # Compatible with original OpenAI-compatible endpoints (deepseek, etc.)
        chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            max_tokens=gen_cfg.get("max_new_tokens"),
            timeout=90,
            max_retries=1,
        )
        return ChatOpenAIWrapper(chat_model, role_name=role_name, generation_config=gen_cfg)

    if provider in {"local"}:
        # Local transformers inference, no HTTP dependency. Assumes model is downloaded or can be loaded from Hugging Face.
        return LocalTransformersChatModel(
            model=model,
            device_map=device_map or "auto",
            role_name=role_name,
            generation_config=gen_cfg,
        )

    raise ValueError(f"Unknown provider in model spec: {provider}")


def resolve_model_credentials(model_spec: str, config: Dict, args_api_key: Optional[str], args_base_url: Optional[str]):
    """Resolve api_key/base_url based on provider with priority: args > config > env."""
    if ":" in model_spec:
        provider, _ = model_spec.split(":", 1)
    elif model_spec.startswith("local/"):
        provider = "local"
    else:
        provider = "openai"
    provider = provider.lower()

    if provider == "openai":
        api_key = args_api_key or config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        base_url = args_base_url or config.get("openai_base_url") or os.getenv("OPENAI_BASE_URL") or None
        return api_key, base_url

    # local / other providers use defaults
    return None, args_base_url


def is_local_model(model_spec: str) -> bool:
    """Check if a model is local (not consuming API tokens)."""
    if not model_spec:
        return False
    if ":" in model_spec:
        provider, _ = model_spec.split(":", 1)
        return provider.lower() == "local"
    elif model_spec.startswith("local/"):
        return True
    return False


def is_user_role(role_name: str, user_role_name: Optional[str] = None) -> bool:
    """Check if a role name indicates it's a user role.

    User roles are identified by:
    1. Having "(user)" in the name
    2. If user_role_name is provided, check if user_role_name contains role_name
       (for cases where models return user role name without (user) suffix)

    Args:
        role_name: The role name to check
        user_role_name: Optional user role name from config for additional matching

    Returns:
        True if the role is a user role, False otherwise
    """
    role_name_lower = role_name.lower()
    # First check: standard pattern matching
    if "(user)" in role_name_lower:
        return True

    # Second check: if user_role_name is provided, check if it contains role_name
    if user_role_name:
        user_role_name_lower = user_role_name.lower()
        # Check if user_role_name contains role_name (e.g., "Dr. Amira Chen (user)" contains "Dr. Amira Chen")
        if role_name_lower in user_role_name_lower:
            return True

    return False


def format_dialogue_for_scene_manager(
    messages: list[BaseMessage],
    exclude_manager: bool = False,
    role_info: Optional[Dict[str, RoleInfo]] = None,
) -> str:
    """Flatten all messages for prompts.

    Args:
        messages: List of messages
        exclude_manager: If True, exclude messages from scene_manager
        role_info: Optional role information dict to resolve user role name
    """
    lines = []
    for msg in messages:
        speaker = msg.name or msg.type
        # Skip scene_manager messages if exclude_manager is True
        if exclude_manager and speaker == "scene_manager":
            continue
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines) if lines else "No dialogue yet."


def _get_scene_manager_action(content: str) -> Optional[str]:
    """Extract action type from scene_manager message content.

    Returns:
        Action type string (init_scene, switch_scene, pick_speaker, add_role) or None
    """
    if "action=" in content:
        parts = content.split("action=")
        if len(parts) > 1:
            action_part = parts[1].split("|")[0].strip()
            return action_part
    return None


def _parse_scene_manager_message_to_json(content: str) -> str:
    """Parse scene_manager message from text format to JSON format.

    Converts from various formats:
    - "action: init_scene | initial_scene: ..."
    - "(SceneManager) action=pick_speaker | reason: ... | picked speaker: ..."
    - "(SceneManager) action=switch_scene | reason: ... | new_scene: ..."
    - "(SceneManager) action=add_role | reason: ... | added role: ... | new_role_name: ... | new_role_profile: ... | new_role_motivation: ..."
    - "(SceneManager) Stopping: ..."

    To JSON format like: {"action": "pick_speaker", "reason": "...", "speaker": "..."}

    Args:
        content: Original scene_manager message content

    Returns:
        JSON string representation of the message
    """
    # Remove "(SceneManager)" prefix if present
    content = content.replace("(SceneManager)", "").strip()

    # Handle stopping message
    if "Stopping:" in content:
        result = {
            "action": "end",
            "reason": content.split("Stopping:")[1].strip() if "Stopping:" in content else "Reached turn limit or termination flag."
        }
        return json.dumps(result, ensure_ascii=False)

    # Parse the content
    result = {}

    # Extract action - handle both "action:" and "action=" formats
    if "action=" in content:
        parts = content.split("action=")
        if len(parts) > 1:
            action_part = parts[1].split("|")[0].strip()
            result["action"] = action_part
    elif "action:" in content:
        parts = content.split("action:")
        if len(parts) > 1:
            action_part = parts[1].split("|")[0].strip()
            result["action"] = action_part

    # Extract reason (format: "reason: ...")
    if "reason:" in content:
        reason_parts = content.split("reason:")
        if len(reason_parts) > 1:
            # Get text after "reason:" until next "|" or end
            reason_text = reason_parts[1]
            if "|" in reason_text:
                reason_text = reason_text.split("|")[0]
            result["reason"] = reason_text.strip()

    # Extract speaker (for pick_speaker action, format: "picked speaker: ...")
    if "picked speaker:" in content:
        speaker_parts = content.split("picked speaker:")
        if len(speaker_parts) > 1:
            speaker_text = speaker_parts[1]
            if "|" in speaker_text:
                speaker_text = speaker_text.split("|")[0]
            result["speaker"] = speaker_text.strip()

    # Extract new_scene (for switch_scene action, format: "new_scene: ...")
    if "new_scene:" in content:
        scene_parts = content.split("new_scene:")
        if len(scene_parts) > 1:
            scene_text = scene_parts[1]
            if "|" in scene_text:
                scene_text = scene_text.split("|")[0]
            result["new_scene"] = scene_text.strip()

    # Extract initial_scene (for init_scene action, format: "initial_scene: ...")
    if "initial_scene:" in content:
        scene_parts = content.split("initial_scene:")
        if len(scene_parts) > 1:
            scene_text = scene_parts[1]
            if "|" in scene_text:
                scene_text = scene_text.split("|")[0]
            result["initial_scene"] = scene_text.strip()

    # Extract role info (for add_role action)
    # Format: "added role: ... | new_role_name: ... | new_role_profile: ... | new_role_motivation: ..."
    if "new_role_name:" in content:
        role_parts = content.split("new_role_name:")
        if len(role_parts) > 1:
            role_name_text = role_parts[1]
            if "|" in role_name_text:
                role_name_text = role_name_text.split("|")[0]
            result["new_role_name"] = role_name_text.strip()

    if "new_role_profile:" in content:
        profile_parts = content.split("new_role_profile:")
        if len(profile_parts) > 1:
            profile_text = profile_parts[1]
            if "|" in profile_text:
                profile_text = profile_text.split("|")[0]
            result["new_role_profile"] = profile_text.strip()

    if "new_role_motivation:" in content:
        motivation_parts = content.split("new_role_motivation:")
        if len(motivation_parts) > 1:
            motivation_text = motivation_parts[1]
            if "|" in motivation_text:
                motivation_text = motivation_text.split("|")[0]
            result["new_role_motivation"] = motivation_text.strip()

    return json.dumps(result, ensure_ascii=False)


def filter_messages_for_role(
    messages: list[BaseMessage],
    current_role_name: Optional[str] = None,
) -> list[BaseMessage]:
    """Filter messages based on current_role_name.

    Rules:
    - If current_role_name is "scene_manager", keep all messages and convert scene_manager
      AIMessages from text format to JSON format
    - Otherwise, filter out AIMessages from scene_manager with action=pick_speaker
      (role actors don't need these pick_speaker decisions)

    Args:
        messages: List of messages to filter
        current_role_name: The role name for which messages are being filtered

    Returns:
        Filtered list of messages
    """
    # If this is the scene manager, convert scene_manager messages to JSON format
    if current_role_name == "scene_manager":
        converted_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "scene_manager":
                # Convert scene_manager message content to JSON format
                original_content = getattr(msg, "content", "")
                json_content = _parse_scene_manager_message_to_json(original_content)
                # Create new AIMessage with JSON content
                converted_msg = AIMessage(content=json_content, name="scene_manager")
                converted_messages.append(converted_msg)
            else:
                # Keep other messages as-is
                converted_messages.append(msg)
        return converted_messages

    # For other roles, filter out scene_manager pick_speaker messages
    filtered = []
    for msg in messages:
        # Keep all SystemMessages
        if isinstance(msg, SystemMessage):
            filtered.append(msg)
            continue

        # For AIMessages, check if it's a scene_manager pick_speaker message
        if isinstance(msg, AIMessage):
            msg_name = getattr(msg, "name", None)
            if msg_name == "scene_manager":
                content = getattr(msg, "content", "")
                action = _get_scene_manager_action(content)
                # Skip pick_speaker messages for non-scene-manager roles
                if action == "pick_speaker":
                    continue
            # Keep all other messages (including scene_manager messages with other actions)
            filtered.append(msg)
        else:
            # Keep any other message types
            filtered.append(msg)

    return filtered


def format_dialogue_for_rpa(
    messages: list[BaseMessage],
    exclude_manager: bool = True,
    role_info: Optional[Dict[str, RoleInfo]] = None,
) -> str:
    """Format dialogue history for RPA prompts.

    Scene manager messages are filtered:
    - pick_speaker is excluded (should be implicit)
    - init_scene, switch_scene, add_role and other actions are included

    Args:
        messages: List of all messages
        exclude_manager: If True, filter scene_manager messages based on action type
        role_info: Optional role information dict to resolve user role name
    """
    lines = []
    for msg in messages:
        speaker = msg.name or msg.type

        # Filter scene_manager messages based on action type
        if exclude_manager and speaker == "scene_manager":
            content = getattr(msg, "content", "")
            action = _get_scene_manager_action(content)
            # Only exclude pick_speaker
            # Include init_scene, switch_scene, add_role and other actions
            if action == "pick_speaker":
                continue

        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines) if lines else "No dialogue yet."


def safe_parse_json(text: str) -> Dict:
    """Best-effort JSON extraction from LLM output. Returns only the first JSON object if multiple exist."""
    # First, try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    # Extract content between code block markers
    markdown_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.MULTILINE | re.DOTALL)
    if markdown_block_match:
        extracted_content = markdown_block_match.group(1).strip()
        # Try to parse the extracted content directly
        try:
            return json.loads(extracted_content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find JSON object in the extracted content
            # Find the first complete JSON object by matching braces
            start_idx = extracted_content.find('{')
            if start_idx != -1:
                brace_count = 0
                in_string = False
                escape_next = False

                for i in range(start_idx, len(extracted_content)):
                    char = extracted_content[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found the end of the first JSON object
                                try:
                                    return json.loads(extracted_content[start_idx:i+1])
                                except json.JSONDecodeError:
                                    break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the first complete JSON object by matching braces
        start_idx = text.find('{')
        if start_idx == -1:
            return {}

        # Use a counter to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the end of the first JSON object
                        try:
                            return json.loads(text[start_idx:i+1])
                        except json.JSONDecodeError:
                            # If parsing fails, try regex fallback
                            break

        # Fallback to regex if brace matching fails
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def load_role_config(config_path: str) -> Dict:
    """Load role configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def message_dict_to_langchain(msg_dict: Dict[str, str], role_info: Optional[Dict[str, RoleInfo]] = None) -> BaseMessage:
    """Convert a message dict with role/content to LangChain BaseMessage.

    Config only contains scene_manager init_scene messages.

    Args:
        msg_dict: Dictionary with 'role' and 'content' keys
        role_info: Optional role information dict (unused, kept for compatibility)

    Returns:
        AIMessage with name="scene_manager"
    """
    content = msg_dict.get("content", "")
    # Config only has scene_manager init_scene messages
    return AIMessage(content=content, name="scene_manager")


def langchain_message_to_dict(msg: BaseMessage, role_info: Optional[Dict[str, RoleInfo]] = None) -> Dict[str, str]:
    """Convert a LangChain BaseMessage back to a simple dict for JSON logging.

    Messages are now only SystemMessage or AIMessage (with name field).

    Args:
        msg: BaseMessage to convert
        role_info: Optional role information dict (unused, kept for compatibility)
    """
    content = getattr(msg, "content", "")
    name = getattr(msg, "name", None)
    msg_type = type(msg).__name__

    # Infer high-level role
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, AIMessage):
        if name == "scene_manager":
            role = "scene_manager"
        elif name:
            # Use the speaker name as role label for actors/users
            role = name
        else:
            role = "assistant"
    else:
        # Fallback for any other message types
        role = "assistant"

    return {
        "role": role,
        "name": name or "",
        "type": msg_type,
        "content": content,
    }


def extract_scenario_from_messages(messages: list[BaseMessage]) -> str:
    """Extract scenario from the first scene_manager message with init_scene action.

    Args:
        messages: List of BaseMessage objects

    Returns:
        Scenario string extracted from initial_scene parameter, or empty string if not found
    """
    for msg in messages:
        if hasattr(msg, "name") and msg.name == "scene_manager":
            content = getattr(msg, "content", "")
            # Look for "initial_scene=" or "initial_scene:"
            if "initial_scene" in content:
                # Try to extract the scenario value
                # Format: "action=init_scene | initial_scene=..."
                parts = content.split("initial_scene")
                if len(parts) > 1:
                    scenario_part = parts[1].strip()
                    # Remove leading = or :
                    scenario_part = scenario_part.lstrip("=:")
                    # Extract until next | or end of string
                    if "|" in scenario_part:
                        scenario = scenario_part.split("|")[0].strip()
                    else:
                        scenario = scenario_part.strip()
                    return scenario
    return ""


def format_other_characters_info(state: RolePlayState, current_speaker: str) -> str:
    """Format information about other characters for the current speaker."""
    other_roles = []
    for role_name, role_info in state.get("role_info", {}).items():
        if role_name != current_speaker:
            profile = role_info.get('profile', '')
            motivation = role_info.get('motivation', '')
            other_roles.append(f"{role_name}:{profile}\n{role_name}'s motivation:{motivation}\n")

    if other_roles:
        return "\n\n".join(other_roles)
    return "No other characters available."


def save_session_summary(
    state: RolePlayState,
    log_dir: str,
    timestamp: str,
    log_file_path: str,
    manager_model: Optional[str] = None,
    actor_model: Optional[str] = None,
    user_model: Optional[str] = None,
) -> str:
    """Save structured session summary (roles + full messages) to a JSON file.

    Includes:
    - Roles split into main_character and other_characters
    - All messages in order, including scene_manager messages.
    - Model information (manager_model, actor_model, user_model)
    """
    role_info_map = state.get("role_info", {}) or {}
    main_name = state.get("main_character_name")

    def _to_role_dict(role_key: str, info: RoleInfo) -> dict:
        return {
            "role_key": role_key,
            "name": info.get("name", role_key),
            "profile": info.get("profile", ""),
            "motivation": info.get("motivation"),
        }

    # Resolve main character: prefer explicit main_character_name, otherwise first non-user.
    user_role_name = state.get("user_role_name")
    if not main_name or main_name not in role_info_map:
        for candidate in role_info_map:
            if not is_user_role(candidate, user_role_name):
                main_name = candidate
                break
        if not main_name and role_info_map:
            # Fallback: pick any role if everything looks like user roles.
            main_name = next(iter(role_info_map.keys()))

    main_character_block = None
    if main_name and main_name in role_info_map:
        main_character_block = _to_role_dict(main_name, role_info_map[main_name])

    other_characters_block = []
    for role_name, info in role_info_map.items():
        if role_name == main_name:
            continue
        other_characters_block.append(_to_role_dict(role_name, info))

    roles_summary = {
        "main_character": main_character_block,
        "other_characters": other_characters_block,
    }

    messages_summary = [langchain_message_to_dict(m, role_info=state.get("role_info")) for m in state.get("messages", [])]

    # Aggregate token usage for non-local models, separated by model type
    token_usage_list = state.get("token_usage", [])

    # Initialize token counters for each model type
    manager_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    actor_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    user_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for usage_entry in token_usage_list:
        model_type = usage_entry.get("model_type")  # "manager", "actor", or "user"
        model_spec = usage_entry.get("model_spec")
        usage = usage_entry.get("usage")
        if model_spec and usage and not is_local_model(model_spec):
            # Normalize usage format (different APIs may use different field names)
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            usage_total = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

            # Determine which model type this usage belongs to (prefer model_type if available, fallback to model_spec matching)
            if model_type == "manager" or (not model_type and model_spec == manager_model):
                manager_tokens["prompt_tokens"] += prompt_tokens
                manager_tokens["completion_tokens"] += completion_tokens
                manager_tokens["total_tokens"] += usage_total
            elif model_type == "actor" or (not model_type and model_spec == actor_model):
                actor_tokens["prompt_tokens"] += prompt_tokens
                actor_tokens["completion_tokens"] += completion_tokens
                actor_tokens["total_tokens"] += usage_total
            elif model_type == "user" or (not model_type and model_spec == user_model):
                user_tokens["prompt_tokens"] += prompt_tokens
                user_tokens["completion_tokens"] += completion_tokens
                user_tokens["total_tokens"] += usage_total

    session_data = {
        "scenario": state.get("scenario", ""),
        "turn": state.get("turn", 0),
        "max_turns": state.get("max_turns"),
        "terminated": state.get("terminated", False),
        "user_mode": state.get("user_mode"),
        "log_file": log_file_path,
        "models": {
            "manager_model": manager_model,
            "actor_model": actor_model,
            "user_model": user_model,
        },
        "prompt_config": {
            "role_speaker_system_prompt_type": state.get("role_speaker_system_prompt_type", "enhance"),
            "scene_manager_system_prompt_type": state.get("scene_manager_system_prompt_type", "enhance"),
        },
        "roles": roles_summary,
        "messages": messages_summary,
    }

    # Add token usage summary if there are any non-local model calls
    # Only include models that have non-zero token usage
    token_usage_summary = {}
    if manager_tokens["total_tokens"] > 0:
        token_usage_summary["manager_model"] = manager_tokens
    if actor_tokens["total_tokens"] > 0:
        token_usage_summary["actor_model"] = actor_tokens
    if user_tokens["total_tokens"] > 0:
        token_usage_summary["user_model"] = user_tokens

    if token_usage_summary:
        session_data["token_usage"] = token_usage_summary

    json_path = os.path.join(log_dir, f"roleplay_session_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return json_path


def retrieve_knowledge_for_role(role_name: str, scenario: str) -> str:
    """Retrieve relevant background knowledge for a role (RAG placeholder).

    This is a placeholder function for future RAG implementation.
    Currently returns empty string.
    """
    # TODO: Implement RAG retrieval here
    return ""


class LLMLogger:
    """Lightweight logger: record each utterance with its actor source."""

    def __init__(
        self,
        log_file_path: str,
        *,
        manager_model: Optional[str] = None,
        actor_model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_mode: str = "manual",
    ):
        self.log_file_path = log_file_path
        self.session_start_time = datetime.now()
        self.manager_model = manager_model
        self.actor_model = actor_model
        self.user_model = user_model
        self.user_mode = user_mode
        self._write_header()

    def _write_header(self):
        """Write session header to log file."""
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Role-Playing Session Log\n")
            f.write(f"Session Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {self.log_file_path}\n")
            if self.manager_model:
                f.write(f"Scene Manager Model: {self.manager_model}\n")
            if self.actor_model:
                f.write(f"Role Actor Model: {self.actor_model}\n")
            if self.user_model:
                f.write(f"User Model: {self.user_model} (mode={self.user_mode})\n")
            f.write("=" * 80 + "\n\n")

    def _infer_actor_source(self, speaker: str, user_role_name: Optional[str] = None) -> str:
        """Infer which backend or source generated this speaker's utterance."""
        if speaker == "scene_manager":
            return f"scene_manager_model:{self.manager_model or 'unknown'}"
        if is_user_role(speaker, user_role_name):
            if self.user_mode == "auto" and self.user_model:
                return f"auto_user_model:{self.user_model}"
            return "manual_user"
        # Other named roles are spoken by the role actor model
        return f"role_actor_model:{self.actor_model or 'unknown'}"

    def log_message(self, message: BaseMessage, turn: Optional[int] = None, user_role_name: Optional[str] = None):
        """Log a single utterance in chronological order.

        Args:
            message: The message to log
            turn: Optional turn number (if None, will not be included in log)
            user_role_name: Optional user role name for actor source inference
        """
        speaker = getattr(message, "name", None) or getattr(message, "type", "assistant")
        content = getattr(message, "content", "")
        actor_source = self._infer_actor_source(speaker, user_role_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            if turn is not None:
                f.write(f"[{timestamp}] turn={turn} | speaker={speaker} | actor={actor_source}\n")
            else:
                f.write(f"[{timestamp}] speaker={speaker} | actor={actor_source}\n")
            f.write(f"{content}\n")
            f.write("-" * 80 + "\n")


def build_scene_manager_system_message(state: RolePlayState) -> SystemMessage:
    """Build system message for scene manager LLM based on state."""
    # Get main character info
    main_character_name = state.get("main_character_name")
    role_info = state.get("role_info", {})

    # Get main character profile
    if main_character_name and main_character_name in role_info:
        main_character_profile = role_info[main_character_name].get("profile", "No profile provided.")
    else:
        # Fallback: use first non-user role or any role
        main_character_profile = "No profile provided."
        user_role_name = state.get("user_role_name")
        for role_name, info in role_info.items():
            if not is_user_role(role_name, user_role_name):
                main_character_profile = info.get("profile", "No profile provided.")
                break
        if main_character_profile == "No profile provided." and role_info:
            # Last resort: use any role
            first_role = next(iter(role_info.values()))
            main_character_profile = first_role.get("profile", "No profile provided.")

    # Get other characters info (excluding main character)
    other_roles = []
    for role_name, info in role_info.items():
        if role_name != main_character_name:
            profile = info.get('profile', '')
            motivation = info.get('motivation', '')
            other_roles.append(f"{role_name}:{profile}\n{role_name}'s motivation:{motivation}\n")

    other_characters_info = "\n\n".join(other_roles) if other_roles else "No other characters available."

    # Get prompt type from state config, default to "enhance"
    prompt_type = state.get("scene_manager_system_prompt_type", "enhance")

    # Get manager model spec for disclaimer check
    manager_model_spec = state.get("manager_model_spec")

    # Get system message from prompts module
    system_content = prompts.get_scene_manager_prompt(
        main_character_profile=main_character_profile,
        other_characters_info=other_characters_info,
        prompt_type=prompt_type,
    )

    # Add disclaimer for non-local models
    if manager_model_spec and not is_local_model(manager_model_spec):
        system_content = NON_LOCAL_MODEL_DISCLAIMER + system_content

    return SystemMessage(content=system_content)


def scene_manager_node(state: RolePlayState, manager_llm: ChatOpenAI, logger: Optional[LLMLogger] = None) -> dict:
    """Decide scene switch, role addition, or speaker selection."""
    if state.get("terminated") or state["turn"] >= state["max_turns"]:
        return {
            "manager_action": "end",
            "terminated": True,
            "messages": [
                AIMessage(
                    content="(SceneManager) Stopping: reached turn limit or termination flag.",
                    name="scene_manager",
                )
            ],
        }

    # Invoke LLM with state, message building happens inside invoke
    # Explicitly pass current_role_name="scene_manager" for message filtering
    # Retry up to 5 times if decision is empty
    max_retries = 5
    decision = {}
    for retry_count in range(max_retries):
        response = manager_llm.invoke(state, message_builder=build_scene_manager_system_message, current_role_name="scene_manager")
        # print(response.content)
        decision_text = response.content
        print(decision_text)
        # Track token usage for non-local models
        if hasattr(response, "usage") and response.usage:
            manager_model_spec = state.get("manager_model_spec")
            if manager_model_spec and not is_local_model(manager_model_spec):
                token_usage = state.get("token_usage", [])
                token_usage.append({
                    "model_type": "manager",
                    "model_spec": manager_model_spec,
                    "usage": response.usage,
                })
                state["token_usage"] = token_usage

        decision = safe_parse_json(decision_text)
        # If decision is not empty, break the retry loop
        if decision:
            break

    # print(decision)
    # If decision is still empty after all retries, terminate the program
    if not decision:
        error_msg = "(SceneManager) Stopping: failed to get valid decision after 5 retries."
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    action = decision.get("action") or "pick_speaker"
    # print(action)
    # print(decision)
    updates: dict = {
        "manager_action": action,
        "awaiting_user": False,
        "last_reason": decision.get("reason"),
    }

    manager_log = f"(SceneManager) action={action}"
    if decision.get("reason"):
        manager_log += f" | reason: {decision['reason']}"

    if action == "switch_scene":
        updates["scenario"] = decision.get("new_scene", state["scenario"])
        updates["awaiting_user"] = False
        manager_log += f" | new_scene: {updates['scenario']}"
    elif action == "add_role":
        role_name = decision.get("new_role_name", "supporting_role")
        role_profile = decision.get("new_role_profile", "A supporting character.")
        role_motivation = decision.get("new_role_motivation", "A supporting character.")

        role_info = dict(state.get("role_info", {}))
        role_info[role_name] = {
            "name": role_name,
            "profile": role_profile,
            "motivation": role_motivation,
        }
        updates["role_info"] = role_info
        updates["awaiting_user"] = False
        manager_log += f" | added role: {role_name} | new_role_name: {role_name} | new_role_profile: {role_profile} | new_role_motivation: {role_motivation}"
    elif action == "pick_speaker":
        chosen = decision.get("speaker")
        if not chosen:
            # Fallback: find user role name if exists, otherwise use first role
            user_role_name = state.get("user_role_name")
            for role_name in state.get("role_info", {}).keys():
                if is_user_role(role_name, user_role_name):
                    chosen = role_name
                    break
            if not chosen and state.get("role_info"):
                chosen = next(iter(state.get("role_info", {}).keys()))

        updates["current_speaker"] = chosen
        user_role_name = state.get("user_role_name")
        if chosen and is_user_role(chosen, user_role_name):
            updates["awaiting_user"] = True
        else:
            updates["awaiting_user"] = False
        manager_log += f" | picked speaker: {chosen}"
    elif action == "end":
        updates["terminated"] = True
        manager_log += " | conversation ended."

    updates["messages"] = [AIMessage(content=manager_log, name="scene_manager")]
    return updates


def build_user_system_message(state: RolePlayState) -> SystemMessage:
    """Build system message for user LLM based on state."""
    # Get main character info
    main_character_name = state.get("main_character_name")
    role_info = state.get("role_info", {})

    # Get main character profile
    main_character_profile = None
    if main_character_name and main_character_name in role_info:
        main_character_profile = role_info[main_character_name].get("profile", "No profile provided.")
    else:
        # Fallback: use first non-user role or any role
        user_role_name = state.get("user_role_name")
        for role_name, info in role_info.items():
            if not is_user_role(role_name, user_role_name):
                main_character_profile = info.get("profile", "No profile provided.")
                break
        if not main_character_profile and role_info:
            # Last resort: use any role
            first_role = next(iter(role_info.values()))
            main_character_profile = first_role.get("profile", "No profile provided.")

    # Get other characters info (excluding main character and user role)
    other_roles = []
    user_role_name = state.get("user_role_name")
    for role_name, info in role_info.items():
        # Exclude main character and user role
        if role_name != main_character_name and not is_user_role(role_name, user_role_name):
            profile = info.get('profile', '')
            motivation = info.get('motivation', '')
            other_roles.append(f"{role_name}:{profile}\n{role_name}'s motivation:{motivation}\n")

    other_characters_info = "\n\n".join(other_roles) if other_roles else "No other characters available."

    system_content = prompts.get_auto_user_system_message(
        main_character_profile=main_character_profile,
        other_characters_info=other_characters_info,
    )

    # Add disclaimer for non-local models
    user_model_spec = state.get("user_model_spec")
    if user_model_spec and not is_local_model(user_model_spec):
        system_content = NON_LOCAL_MODEL_DISCLAIMER + system_content

    return SystemMessage(content=system_content)


def build_role_speaker_system_message(state: RolePlayState) -> SystemMessage:
    """Build system message for role speaker LLM based on state."""
    speaker = state.get("current_speaker") or "target_role"

    # Get role info from role_info structure
    role_info = state.get("role_info", {}).get(speaker)
    if role_info:
        character = role_info.get("name", speaker)
        profile = role_info.get("profile", "No profile provided.")
    else:
        # Fallback if role_info not found
        character = speaker
        profile = "No profile provided."

    # Get motivation from role info
    motivation = role_info.get("motivation", "") if role_info else ""

    # Get other characters info
    other_characters_info = format_other_characters_info(state, speaker)

    # Get retrieved knowledge (RAG placeholder)
    retrieved_knowledge = retrieve_knowledge_for_role(speaker, state["scenario"])

    # Get prompt type from state config, default to "enhance"
    prompt_type = state.get("role_speaker_system_prompt_type", "enhance")

    # Get system prompt from prompts module
    system_prompt = prompts.get_role_speaker_system_prompt(
        character=character,
        profile=profile,
        other_characters_info=other_characters_info,
        motivation=motivation,
        retrieved_knowledge=retrieved_knowledge,
        prompt_type=prompt_type,
    )

    # Add disclaimer for non-local models
    actor_model_spec = state.get("actor_model_spec")
    if actor_model_spec and not is_local_model(actor_model_spec):
        system_prompt = NON_LOCAL_MODEL_DISCLAIMER + system_prompt

    return SystemMessage(content=system_prompt)


def role_speaker_node(state: RolePlayState, actor_llm: ChatOpenAI, logger: Optional[LLMLogger] = None) -> dict:
    """Have the chosen role speak in character."""
    speaker = state.get("current_speaker") or "target_role"

    # Invoke LLM with state, system message will be merged with state messages
    # Pass current speaker as current_role_name for dynamic role handling
    reply = actor_llm.invoke(state, message_builder=build_role_speaker_system_message, current_role_name=speaker)

    # Track token usage for non-local models
    if hasattr(reply, "usage") and reply.usage:
        actor_model_spec = state.get("actor_model_spec")
        if actor_model_spec and not is_local_model(actor_model_spec):
            token_usage = state.get("token_usage", [])
            token_usage.append({
                "model_type": "actor",
                "model_spec": actor_model_spec,
                "usage": reply.usage,
            })
            state["token_usage"] = token_usage

    # Guardrail: strip accidental role prefixes like "user:" or "Ming:" from the model reply.
    cleaned = re.sub(rf"^(?:{re.escape(speaker)}|user)\s*:\s*", "", reply.content, flags=re.IGNORECASE)

    return {
        "messages": [AIMessage(content=cleaned, name=speaker)],
        "turn": state["turn"] + 1,
        "current_speaker": None,
        # After the role speaks, ask for user input to keep turns balanced.
        "awaiting_user": True,
    }


def await_user_node(state: RolePlayState) -> dict:
    """Pause graph and ask for user input."""
    return {"awaiting_user": True}


def route_from_manager(state: RolePlayState) -> str:
    """Branch logic after the manager decides."""
    if state.get("terminated") or state["turn"] >= state["max_turns"]:
        return "end"
    action = state.get("manager_action")
    if action in {"switch_scene", "add_role"}:
        # After structural changes, let manager decide speaker again (often user).
        return "scene_manager"
    if action == "pick_speaker":
        current_speaker = state.get("current_speaker")
        user_role_name = state.get("user_role_name")
        return "await_user" if current_speaker and is_user_role(current_speaker, user_role_name) else "role_speaker"
    return "end"


def build_graph(manager_llm: ChatOpenAI, actor_llm: ChatOpenAI, logger: Optional[LLMLogger] = None):
    """Wire LangGraph nodes and edges."""
    workflow = StateGraph(RolePlayState)
    # Logger is no longer used at node level; kept in signature for backward compatibility.
    workflow.add_node("scene_manager", lambda state: scene_manager_node(state, manager_llm))
    workflow.add_node("role_speaker", lambda state: role_speaker_node(state, actor_llm))
    workflow.add_node("await_user", await_user_node)

    workflow.set_entry_point("scene_manager")
    workflow.add_conditional_edges(
        "scene_manager",
        route_from_manager,
        {
            "scene_manager": "scene_manager",
            "role_speaker": "role_speaker",
            "await_user": "await_user",
            "end": END,
        },
    )
    # After a role speaks, let the manager decide next speaker again.
    workflow.add_edge("role_speaker", "scene_manager")
    workflow.add_edge("await_user", END)
    return workflow.compile()


# --- Console helpers -----------------------------------------------------
ROLE_COLORS = [
    # Prefer distinct, high-contrast colors
    Fore.CYAN,
    Fore.GREEN,
    Fore.MAGENTA,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.WHITE,
    Fore.RED,
]

# Session-local mapping to ensure each role gets a unique color.
ROLE_COLOR_MAP: dict[str, str] = {}


def color_for_role(role: str, user_role_name: Optional[str] = None) -> str:
    """Assign a (within-session) unique color for each role/speaker."""
    # Reuse existing assignment if any
    if role in ROLE_COLOR_MAP:
        return ROLE_COLOR_MAP[role]

    # Reserve preferred colors for special roles when available
    preferred_color: Optional[str] = None
    if role == "scene_manager":
        preferred_color = Fore.CYAN
    elif is_user_role(role, user_role_name):
        preferred_color = Fore.GREEN

    used_colors = set(ROLE_COLOR_MAP.values())

    if preferred_color and preferred_color not in used_colors:
        ROLE_COLOR_MAP[role] = preferred_color
        return preferred_color

    # Pick the first palette color not yet used
    for c in ROLE_COLORS:
        if c not in used_colors:
            ROLE_COLOR_MAP[role] = c
            return c

    # Fallback: if colors are exhausted, fall back to a deterministic hash choice
    color = ROLE_COLORS[hash(role) % len(ROLE_COLORS)]
    ROLE_COLOR_MAP[role] = color
    return color


def format_colored_line(speaker: str, content: str, user_role_name: Optional[str] = None) -> str:
    color = color_for_role(speaker, user_role_name)
    return f"{color}{speaker}{Style.RESET_ALL}: {content}"


def main():
    colorama_init(autoreset=True)
    load_dotenv()
    parser = argparse.ArgumentParser(description="LangGraph role-playing demo.")
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save log files. Default: logs/",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible endpoint; overrides OPENAI_API_KEY env.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible endpoint; overrides OPENAI_BASE_URL env.",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum conversation turns; overrides the config (default 50).")
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to JSON configuration file.",
    )
    args = parser.parse_args()

    # Create log directory if it doesn't exist
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"roleplay_session_{timestamp}.log")

    print(f"Logging to: {log_file_path}\n")

    # Load configuration from file
    config = load_role_config(args.config_file)

    # Get model names and optional local device configs from file, with defaults.
    # Supports two formats:
    # 1) Legacy flat config: manager_model / manager_device, etc.
    # 2) New nested config: manager_model_config / actor_model_config / user_model_config
    # Supports formats like:
    # - "gpt-4o-mini" (default OpenAI / OpenAI-compatible)
    # - "openai:gpt-4o-mini"
    # - "local:.../Your-Model" (local transformers)
    manager_model_cfg = config.get("manager_model_config", {})
    actor_model_cfg = config.get("actor_model_config", {})
    user_model_cfg = config.get("user_model_config", {})

    # Manager
    manager_model = manager_model_cfg.get("manager_model", config.get("manager_model", "gpt-4o-mini"))
    manager_device = manager_model_cfg.get("manager_device", config.get("manager_device"))  # e.g. "cuda:0", "cpu", "auto"
    manager_gen_cfg = manager_model_cfg.get("generation_config", {}) or {}

    # Actor
    actor_model = actor_model_cfg.get("actor_model", config.get("actor_model", "gpt-4o-mini"))
    actor_device = actor_model_cfg.get("actor_device", config.get("actor_device"))      # e.g. "cuda:1", "cpu", "auto"
    actor_gen_cfg = actor_model_cfg.get("generation_config", {}) or {}

    # User
    user_model_spec = user_model_cfg.get("user_model", config.get("user_model", "manual"))
    user_device = user_model_cfg.get("user_device", config.get("user_device"))        # e.g. "cuda:0", "cpu", "auto"
    user_gen_cfg = user_model_cfg.get("generation_config", {}) or {}
    # Effective max turns: prefer config, fall back to CLI default/override.
    max_turns = args.max_turns if args.max_turns is not None else config.get("max_turns", 50)

    # Build role_info from config first (needed for user role name)
    role_info = {}
    main_character_name = None

    # Add main character
    main_character = config.get("main_character")
    if main_character:
        role_name = main_character.get("name", "Unknown")
        main_character_name = role_name
        role_info[role_name] = {
            "name": role_name,
            "profile": main_character.get("profile", ""),
            "motivation": main_character.get("motivation"),
        }

    # Add other characters (including user)
    other_characters = config.get("other_characters", [])
    for role_cfg in other_characters:
        role_name = role_cfg.get("name", "Unknown")
        role_info[role_name] = {
            "name": role_name,
            "profile": role_cfg.get("profile", ""),
            "motivation": role_cfg.get("motivation"),
        }

    # Find user role name
    user_role_name = "user"  # Default fallback
    for role_name in role_info:
        if is_user_role(role_name):
            user_role_name = role_info[role_name].get("name", role_name)
            break

    manager_api_key, manager_base_url = resolve_model_credentials(manager_model, config, args.api_key, args.base_url)
    actor_api_key, actor_base_url = resolve_model_credentials(actor_model, config, args.api_key, args.base_url)

    manager_llm = create_chat_model(
        manager_model,
        generation_config=manager_gen_cfg,
        api_key=manager_api_key,
        base_url=manager_base_url,
        device_map=manager_device,
        role_name="scene_manager",
    )
    actor_llm = create_chat_model(
        actor_model,
        generation_config=actor_gen_cfg,
        api_key=actor_api_key,
        base_url=actor_base_url,
        device_map=actor_device,
        # actor_llm role is dynamic, set to None
        role_name=None,
    )

    # Configure optional automated user model.
    # Priority: config.user_mode > infer from user_model_spec
    config_user_mode = config.get("user_mode")
    if config_user_mode:
        user_mode = str(config_user_mode).lower()
    else:
        # Fallback: infer from user_model_spec
        user_model_mode = str(user_model_spec).lower()
        if user_model_mode in {"manual", "human", "console", "hand", "none", ""}:
            user_mode = "manual"
        else:
            user_mode = "auto"

    # Initialize user_llm based on user_mode
    if user_mode == "auto":
        user_api_key, user_base_url = resolve_model_credentials(user_model_spec, config, args.api_key, args.base_url)
        user_llm = create_chat_model(
            user_model_spec,
            generation_config=user_gen_cfg,
            api_key=user_api_key,
            base_url=user_base_url,
            device_map=user_device,
            role_name=user_role_name,
        )
    else:
        user_llm = None

    # Initialize simple sequential logger (after models & user_mode resolved)
    logger = LLMLogger(
        log_file_path,
        manager_model=manager_model,
        actor_model=actor_model,
        user_model=user_model_spec if user_mode == "auto" else None,
        user_mode=user_mode,
    )

    app = build_graph(manager_llm, actor_llm, logger)

    # Build initial messages from config
    if "messages" in config and config["messages"]:
        # Convert message dicts to LangChain messages
        messages = [message_dict_to_langchain(msg, role_info=role_info) for msg in config["messages"]]
        # Extract scenario from messages
        scenario = extract_scenario_from_messages(messages)
    else:
        # Fallback: if no messages, use empty scenario
        scenario = ""
        messages = []

    # Get role_speaker_system_prompt_type from config, default to "enhance"
    role_speaker_system_prompt_type = config.get("role_speaker_system_prompt_type", "enhance")
    # Get scene_manager_system_prompt_type from config, default to "enhance"
    scene_manager_system_prompt_type = config.get("scene_manager_system_prompt_type", "enhance")

    state: RolePlayState = {
        "messages": messages,
        "scenario": scenario,
        "role_info": role_info,
        "main_character_name": main_character_name,
        "current_speaker": None,
        "manager_action": None,
        "last_reason": None,
        "turn": 0,
        "max_turns": max_turns,
        "awaiting_user": True,
        "terminated": False,
        "user_mode": user_mode,
        "role_speaker_system_prompt_type": role_speaker_system_prompt_type,
        "scene_manager_system_prompt_type": scene_manager_system_prompt_type,
        "manager_model_spec": manager_model,
        "actor_model_spec": actor_model,
        "user_model_spec": user_model_spec if user_mode == "auto" else None,
        "token_usage": [],
        "user_role_name": user_role_name,
    }

    # Log initial messages (including initial scene from scene_manager)
    for msg in state["messages"]:
        # Skip pure system prompts
        if isinstance(msg, SystemMessage):
            continue
        logger.log_message(msg, turn=state["turn"], user_role_name=user_role_name)

    print("LangGraph role-play demo. Type 'stop', 'exit', or 'quit' to end (manual mode).\n")
    # Show roles first, then initial scene.
    for role_name, info in role_info.items():
        print(format_colored_line(role_name, f"Profile: {info['profile']}", user_role_name=user_role_name))
    print(format_colored_line("scene_manager", f"Scene: {scenario}", user_role_name=user_role_name))

    while not state.get("terminated") and state["turn"] < state["max_turns"]:
        if state.get("awaiting_user"):
            if user_mode == "manual" or user_llm is None:
                # Manual console input from real user.
                user_input = input("User> ").strip()
                if user_input.lower() in {"stop", "exit", "quit"}:
                    print("Stopping on user request.")
                    break
                user_msg = AIMessage(content=user_input, name=user_role_name)
                state["messages"].append(user_msg)
                # Count a full turn whenever the real/auto user speaks.
                state["turn"] += 1
                state["awaiting_user"] = False
                # Echo user input immediately with color.
                print(format_colored_line(user_role_name, user_input, user_role_name=user_role_name))
                logger.log_message(user_msg, turn=state["turn"], user_role_name=user_role_name)
            else:
                # Automated user mode: have the user LLM generate the next user utterance.
                # Invoke LLM with state, system message will be merged with state messages
                # Pass current_role_name to ensure correct message formatting (only user role AIMessages are kept as AIMessage)
                user_reply = user_llm.invoke(
                    state,
                    message_builder=build_user_system_message,
                    current_role_name=user_role_name
                )
                # Track token usage for non-local models
                if hasattr(user_reply, "usage") and user_reply.usage:
                    user_model_spec_state = state.get("user_model_spec")
                    if user_model_spec_state and not is_local_model(user_model_spec_state):
                        token_usage = state.get("token_usage", [])
                        token_usage.append({
                            "model_type": "user",
                            "model_spec": user_model_spec_state,
                            "usage": user_reply.usage,
                        })
                        state["token_usage"] = token_usage
                user_text = user_reply.content.strip()
                # Guardrail: strip accidental leading role prefixes like "user:".
                user_text = re.sub(r"^(?:user|User)\s*:\s*", "", user_text)
                user_msg = AIMessage(content=user_text, name=user_role_name)
                state["messages"].append(user_msg)
                # Count a full turn whenever the real/auto user speaks.
                state["turn"] += 1
                state["awaiting_user"] = False
                print(format_colored_line(user_role_name, user_text, user_role_name=user_role_name))
                logger.log_message(user_msg, turn=state["turn"], user_role_name=user_role_name)

        # Stream node outputs in real time
        prev_len = len(state["messages"])
        latest_state = state
        for step in app.stream(state, stream_mode="values"):
            latest_state = step
            msgs = step.get("messages", [])
            if len(msgs) > prev_len:
                new_messages = msgs[prev_len:]
                for msg in new_messages:
                    # Skip system messages in console output.
                    if isinstance(msg, SystemMessage):
                        continue
                    speaker = msg.name or "assistant"
                    print(format_colored_line(speaker, msg.content, user_role_name=user_role_name))
                    logger.log_message(msg, turn=latest_state.get("turn"), user_role_name=user_role_name)
                prev_len = len(msgs)

        state = latest_state

        if state.get("manager_action") == "end" or state.get("terminated"):
            break

    # Save structured session summary (roles + full messages) alongside the log.
    session_json_path = save_session_summary(
        state=state,
        log_dir=log_dir,
        timestamp=timestamp,
        log_file_path=log_file_path,
        manager_model=manager_model,
        actor_model=actor_model,
        user_model=user_model_spec,
    )

    print("\nConversation finished.")
    print(f"Full session log saved to: {log_file_path}")
    print(f"Session JSON summary saved to: {session_json_path}")


if __name__ == "__main__":
    main()
