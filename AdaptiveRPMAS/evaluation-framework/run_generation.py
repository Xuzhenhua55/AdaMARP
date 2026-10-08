"""Generate session JSONL from evaluation configs using the normal demo entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, required=True, help='Configuration JSONL')
    parser.add_argument('--output', type=Path, required=True, help='New session JSONL (will not overwrite)')
    parser.add_argument('--limit', type=int, default=1, help='Number of scenarios; 0 means all')
    parser.add_argument('--max-turns', type=int, default=None)
    parser.add_argument('--model', help='Optional model override for manager, actor and simulated user')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    args = parser.parse_args()
    if args.limit < 0 or (args.max_turns is not None and args.max_turns < 1):
        parser.error('limit must be nonnegative and max-turns must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new path to preserve previous results')
    configs = [json.loads(line) for line in args.configs.read_text().splitlines() if line.strip()]
    if args.limit:
        configs = configs[:args.limit]
    if not configs:
        parser.error('no scenarios found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for index, config in enumerate(configs, 1):
            for role in ('manager', 'actor', 'user'):
                settings = config.setdefault(f'{role}_model_config', {})
                if args.model:
                    settings[f'{role}_model'] = args.model
                if args.temperature is not None:
                    settings.setdefault('generation_config', {})['temperature'] = args.temperature
                if args.max_new_tokens is not None:
                    settings.setdefault('generation_config', {})['max_new_tokens'] = args.max_new_tokens
            config['user_mode'] = 'auto'
            if args.max_turns is not None:
                config['max_turns'] = args.max_turns
            with tempfile.TemporaryDirectory(prefix='adamarp-eval-') as tmp:
                tmp = Path(tmp)
                config_path = tmp / 'config.json'
                config_path.write_text(json.dumps(config))
                os.chmod(config_path, 0o600)
                subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config-file', str(config_path),
                                '--log-dir', str(tmp / 'logs')], check=True)
                sessions = list((tmp / 'logs').glob('*.json'))
                if len(sessions) != 1:
                    raise RuntimeError(f'Expected one session, got {len(sessions)}')
                record = json.loads(sessions[0].read_text())
                record['uuid'] = config.get('uuid') or hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                record.pop('log_file', None)  # Temporary per-call logs are removed after export.
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
            print(f'Generated {index}/{len(configs)} sessions', flush=True)


if __name__ == '__main__':
    main()
