#!/usr/bin/env python3
"""
Batch generate evaluation configuration files for all models

Generate two configurations for each model: basic and enhance
"""

import json
import jsonlines
import uuid
import os
from pathlib import Path


# Model list extracted from main.sh
MODEL_PATHS = [
    "openai:gpt-4o-mini",
]


def extract_model_name(model_path):
    """Extract model name from model path for use in filenames"""
    # Remove path prefix, use only the last part as model name
    parts = [p for p in model_path.strip().split('/') if p]  # Remove empty strings
    model_name = parts[-1]

    # Replace characters that may not be suitable for filenames
    model_name = model_name.replace('/', '-').replace(' ', '-').replace('_', '-')
    return model_name


def get_base_config(actor_model_path, role_speaker_type, scene_manager_type):
    """Get base configuration (based on model and prompt type)"""
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
            "actor_model": f"{actor_model_path}",
            "actor_device": "auto",
            "generation_config": {
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "enable_thinking": True
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
        "role_speaker_system_prompt_type": role_speaker_type,
        "scene_manager_system_prompt_type": scene_manager_type,
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

    parser = argparse.ArgumentParser(description='Batch generate evaluation configuration files for all models')
    parser.add_argument('--jsonl',
                       default=str(BASE_DIR / 'dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl'),
                       help='Evaluation scenario jsonl file path')
    parser.add_argument('--output-dir',
                       default=str(BASE_DIR / 'evaluation-framework/evaluation-configs'),
                       help='Output configuration directory')
    parser.add_argument('--overwrite',
                       action='store_true',
                       help='Whether to overwrite if output file exists (default: skip existing files)')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation scenarios
    print(f"Loading evaluation scenarios: {args.jsonl}")
    scenarios = load_evaluation_scenarios(args.jsonl)
    print(f"Found {len(scenarios)} scenarios")

    # Generate configs for each model
    total_configs = 0
    skipped_configs = 0
    for model_path in MODEL_PATHS:
        model_name = extract_model_name(model_path)
        print(f"\n{'='*60}")
        print(f"Processing model: {model_path}")
        print(f"Model name: {model_name}")
        print(f"{'='*60}")

        is_local = model_path.startswith("local:")

        if not is_local:
            # Remote/cloud models: keep Basic-Enhance config
            print(f"\nGenerating Basic config...")
            output_file_basic = output_dir / f"{model_name}-Basic-Enhance.jsonl"

            if output_file_basic.exists() and not args.overwrite:
                print(f"⚠ File exists, skipping: {output_file_basic} (use --overwrite to overwrite)")
                skipped_configs += 1
            else:
                base_config_basic = get_base_config(model_path, "basic", "enhance")
                configs_basic = generate_evaluation_config(base_config_basic, scenarios)

                with jsonlines.open(output_file_basic, mode='w') as writer:
                    for config in configs_basic:
                        writer.write(config)

                print(f"✓ Basic config saved: {output_file_basic} ({len(configs_basic)} configs)")
                total_configs += 1
        else:
            print("\nLocal model: skipping Basic config, only generating Enhance-Enhance config")

        # All models generate Enhance-Enhance config
        print(f"\nGenerating Enhance config...")
        output_file_enhance = output_dir / f"{model_name}-Enhance-Enhance.jsonl"

        if output_file_enhance.exists() and not args.overwrite:
            print(f"⚠ File exists, skipping: {output_file_enhance} (use --overwrite to overwrite)")
            skipped_configs += 1
        else:
            base_config_enhance = get_base_config(model_path, "enhance", "enhance")
            configs_enhance = generate_evaluation_config(base_config_enhance, scenarios)

            with jsonlines.open(output_file_enhance, mode='w') as writer:
                for config in configs_enhance:
                    writer.write(config)

            print(f"✓ Enhance config saved: {output_file_enhance} ({len(configs_enhance)} configs)")
            total_configs += 1

    print(f"\n{'='*60}")
    print(f"All completed!")
    print(f"Generated {total_configs} configuration files")
    if skipped_configs > 0:
        print(f"Skipped {skipped_configs} existing files (use --overwrite to overwrite)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
