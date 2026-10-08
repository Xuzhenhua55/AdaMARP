#!/usr/bin/env python3
"""
Script to generate evaluation dataset configuration files

Based on base configuration and merged.jsonl file, generate complete configuration files containing all evaluation scenarios
"""

import json
import jsonlines
import uuid


def get_base_config():
    """Get base configuration (manually set)"""
    base_config = {
        "manager_model_config": {
            "manager_model": "openai:gpt-4o-mini",
            "manager_device": "auto",
            "generation_config": {
                "temperature": 0.7,
                "max_new_tokens": 512
            }
        },
        "actor_model_config": {
            "actor_model": "openai:gpt-4o-mini",
            "actor_device": "auto",
            "generation_config": {
                "temperature": 0.7,
                "max_new_tokens": 1024
            }
        },
        "user_model_config": {
            "user_model": "openai:gpt-4o-mini",
            "user_device": "auto",
            "generation_config": {
                "temperature": 0.7,
                "max_new_tokens": 512
            }
        },
        "user_mode": "auto",
        "openai_api_key": "",
        "openai_base_url": "",
        "max_turns": 20,
        "role_speaker_system_prompt_type": "enhance",
        "scene_manager_system_prompt_type": "enhance",
    }

    return base_config


def load_evaluation_scenarios(jsonl_path):
    """Load all evaluation scenarios from jsonl file"""
    scenarios = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            scenarios.append(obj)
    return scenarios


def generate_evaluation_config(base_config, scenarios):
    """Generate configuration file containing all evaluation scenarios"""
    evaluation_configs = []

    for scenario in scenarios:
        # Merge base config and scenario config
        full_config = base_config.copy()

        # Generate unique UUID for each config
        full_config["uuid"] = str(scenario.get("id") or uuid.uuid4())

        # Add scenario-specific configuration
        if "messages" in scenario:
            full_config["messages"] = scenario["messages"]

        if "main_character" in scenario:
            full_config["main_character"] = scenario["main_character"]

        if "other_characters" in scenario:
            full_config["other_characters"] = scenario["other_characters"]

        if "relations" in scenario:
            full_config["relations"] = scenario["relations"]

        evaluation_configs.append(full_config)

    return evaluation_configs


def main():
    """Main function"""
    import argparse
    from pathlib import Path

    # Base path
    BASE_DIR = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description='Generate evaluation dataset configuration files')
    parser.add_argument('--jsonl',
                       default=str(BASE_DIR / 'dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl'),
                       help='Evaluation scenario jsonl file path')
    parser.add_argument('--output',
                       default=str(BASE_DIR / 'evaluation-framework/evaluation-configs.jsonl'),
                       help='Output configuration file path (jsonl format)')

    args = parser.parse_args()

    print("Using hardcoded base configuration")
    base_config = get_base_config()

    print(f"Loading evaluation scenarios: {args.jsonl}")
    scenarios = load_evaluation_scenarios(args.jsonl)
    print(f"Found {len(scenarios)} scenarios")

    print("Generating evaluation configuration files...")
    evaluation_configs = generate_evaluation_config(base_config, scenarios)

    print(f"Saving configuration file: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output, mode='w') as writer:
        for config in evaluation_configs:
            writer.write(config)

    print(f"✓ Successfully generated configuration file with {len(evaluation_configs)} evaluation scenarios")
    print(f"  File path: {args.output}")


if __name__ == "__main__":
    main()
