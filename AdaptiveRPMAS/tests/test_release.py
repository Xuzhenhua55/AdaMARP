"""Offline checks for API configuration and the released evaluation data contract."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'evaluation-framework' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_credentials_precedence(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key', 'OPENAI_BASE_URL': 'https://env.invalid'}):
            self.assertEqual(main.resolve_model_credentials('openai:test', {}, None, None),
                             ('env-key', 'https://env.invalid'))
            config = {'openai_api_key': 'config-key', 'openai_base_url': 'https://config.invalid'}
            self.assertEqual(main.resolve_model_credentials('openai:test', config, 'cli-key', 'https://cli.invalid'),
                             ('cli-key', 'https://cli.invalid'))

    def test_empty_base_url_uses_client_default(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": ""}):
            self.assertIsNone(main.resolve_model_credentials("openai:test", {}, None, None)[1])

    def test_model_id_and_token_limit(self):
        with patch.object(main, 'ChatOpenAI') as client:
            main.create_chat_model('openai:vendor/model', generation_config={'temperature': 1, 'max_new_tokens': 123})
            self.assertEqual(client.call_args.kwargs['model'], 'vendor/model')
            self.assertEqual(client.call_args.kwargs['max_tokens'], 123)

    def test_heldout_evaluation_config_contract(self):
        generator = load('generator', 'generate_evaluation_configs.py')
        path = ROOT / 'dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl'
        scenarios = generator.load_evaluation_scenarios(path)
        self.assertEqual(len(scenarios), 100)
        configs = generator.generate_evaluation_config(generator.get_base_config(), scenarios)
        for scenario, config in zip(scenarios, configs):
            self.assertEqual(config['uuid'], str(scenario['id']))
            self.assertTrue(config['main_character']['profile'])
            self.assertTrue(config['messages'])
            self.assertEqual(config['openai_api_key'], '')
            self.assertEqual(config['openai_base_url'], '')

    def test_scene_manager_context_preserves_decisions(self):
        evaluator = load('scene_eval', 'scene_manager_evaluations.py')
        context = evaluator.build_dialogue_context({'roles': {'main_character': {'name': 'A', 'profile': 'Role profile'}},
                 'messages': [{'role': 'scene_manager', 'content': 'action=pick_speaker | speaker=A'}]})
        self.assertIn('Role profile', context)
        self.assertIn('pick_speaker', context)

    def test_training_paths_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'megatron'
            executable.write_text('#!' + sys.executable + '\nimport json, os, sys\nprint(json.dumps({"args": sys.argv[1:], "cache": os.environ["MODELSCOPE_CACHE"]}))\n')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ['PATH'],
                       MEGATRON_LM_PATH=tmp, MODEL_PATH=tmp, CHECKPOINT_PATH=tmp,
                       MODELSCOPE_CACHE=tmp)
            for script in (ROOT / 'training-pipeline').rglob('*.sh'):
                result = subprocess.run(['bash', str(script)], cwd=tmp, env=env,
                                        capture_output=True, text=True, check=True)
                info = json.loads(result.stdout)
                args = info['args']
                dataset = Path(args[args.index('--dataset') + 1])
                self.assertTrue(dataset.is_absolute())
                self.assertTrue(dataset.is_file())
                self.assertEqual(info['cache'], tmp)

    def test_generation_preserves_existing_output(self):
        runner = load('runner', 'run_generation.py')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'existing.jsonl'
            p.write_text('keep me')
            with patch.object(sys, 'argv', ['run_generation', '--configs', 'unused.jsonl', '--output', str(p)]):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(p.read_text(), 'keep me')


if __name__ == '__main__':
    unittest.main()
