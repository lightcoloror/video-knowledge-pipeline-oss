"""Task-local offline acceptance. Reuse the project's socket/model guards."""
import importlib.abc
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1', NO_COLOR='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
sys.modules.setdefault('readline', None)
platform.node = lambda: 'offline-test-host'
import pytest  # noqa: E402 -- establish isolated paths and plugin policy first

blocked = []

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'funasr', 'onnxruntime', 'faster_whisper'}:
            blocked.append('model_import:' + fullname)
            raise ImportError('offline test: model imports denied')

def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
        blocked.append(event)
        raise RuntimeError('offline test: subprocess denied')

sys.addaudithook(audit)
sys.meta_path.insert(0, NoModels())
out = Path(sys.argv[1]).expanduser().resolve()
out.mkdir(exist_ok=False)
code = pytest.main(['-q', '-p', 'no:cacheprovider', '-p', 'evidence_offline_guard',
                    '--basetemp', str(out / 'tmp'), '--junitxml', str(out / 'results.xml'),
                    str(ROOT / 'tests/test_codex_vision_executor.py'),
                    str(ROOT / 'tests/test_codex_vision_session.py'), *sys.argv[2:]])
print(json.dumps({'exit_code': int(code), 'blocked_subprocess_or_models': blocked}))
raise SystemExit(code)
