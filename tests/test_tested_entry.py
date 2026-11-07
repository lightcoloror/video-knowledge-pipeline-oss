"""Offline source/runtime binding checks; synthetic Git metadata, no Git writes."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import tested_entry as entry
from video_knowledge_pipeline.canonical_json import canonical_json_sha256

HEAD = 'a' * 40
OTHER = 'b' * 40


@pytest.fixture(autouse=True)
def isolated_binding_environment(monkeypatch):
    monkeypatch.delenv('VKP_TESTED_VERSION_PATH', raising=False)


def source(tmp_path):
    root = tmp_path / 'source'
    package = root / 'src/video_knowledge_pipeline'
    package.mkdir(parents=True)
    scripts = root / 'scripts'
    scripts.mkdir()
    (root / '.git').mkdir()
    (root / '.git/HEAD').write_text(HEAD + '\n', encoding='utf-8')
    actual = Path(entry.__file__).resolve()
    for name in ('tested_entry.py', 'file_hash.py', 'canonical_json.py'):
        shutil.copyfile(actual.parent / name, package / name)
    (package / '__init__.py').write_text('', encoding='utf-8')
    shutil.copyfile(actual.parents[2] / 'scripts/run-tested-vkp.ps1', scripts / 'run-tested-vkp.ps1')
    (package / 'cli.py').write_text(
        'import json, os, sys\ndef main():\n'
        '    print(json.dumps({"args":sys.argv[1:],"pythonpath":os.environ.get("PYTHONPATH"),"native":True}))\n'
        '    return 0\n', encoding='utf-8')
    (root / 'pyproject.toml').write_text('[project]\nname="synthetic-vkp"\nversion="0"\n'
        '[project.scripts]\nvideo-knowledge="video_knowledge_pipeline.cli:main"\n'
        'video-knowledge-quick-health="video_knowledge_pipeline.quick_health:main"\n'
        'unrelated="other_package.cli:main"\n', encoding='utf-8')
    return root


@pytest.fixture
def fixed_dependencies(monkeypatch):
    rows = [{'name': 'synthetic-dependency', 'version': '1.2', 'location': 'test-only'}]
    snapshot = {'kind': 'metadata', 'distributions': rows, 'sha256': canonical_json_sha256(rows)}
    monkeypatch.setattr(entry, 'dependency_snapshot', lambda: copy.deepcopy(snapshot))
    return snapshot


def bind(root):
    return entry.create_binding(root, python=sys.executable, expected_head=HEAD)


def test_binding_records_actual_head_and_full_source_then_reuses_native_quality_boundary(tmp_path, fixed_dependencies):
    root = source(tmp_path)
    result = bind(root)
    assert result['git_head'] == HEAD
    proof = entry.validate_binding(root)
    assert proof['git_head'] == HEAD and proof['entry_point'] == 'video-knowledge'
    assert proof['callable'] == 'video_knowledge_pipeline.cli:main'
    assert proof['quality_gate_bypassed'] is False
    assert 'candidate_only' not in proof and 'base_head' not in proof
    paths = {row['path'] for row in json.loads(Path(result['binding_path']).read_text(encoding='utf-8'))['source']['files']}
    assert {'scripts/run-tested-vkp.ps1', 'src/video_knowledge_pipeline/tested_entry.py', 'pyproject.toml'} <= paths


@pytest.mark.parametrize('kind', ['new_source', 'changed_source', 'deleted_source', 'wrapper', 'config', 'config_file', 'contract_lock'])
def test_source_manifest_is_derived_and_detects_all_managed_changes(tmp_path, fixed_dependencies, kind):
    root = source(tmp_path)
    bind(root)
    target = root / 'src/video_knowledge_pipeline/cli.py'
    if kind == 'new_source':
        (target.parent / 'new_logic.py').write_text('unexpected = True', encoding='utf-8')
    elif kind == 'deleted_source':
        target.unlink()
    elif kind == 'wrapper':
        (root / 'scripts/run-tested-vkp.ps1').write_text('changed', encoding='utf-8')
    elif kind == 'config':
        with (root / 'pyproject.toml').open('a', encoding='utf-8') as stream:
            stream.write('\n# changed configuration')
    elif kind == 'config_file':
        (root / 'config').mkdir()
        (root / 'config/route.json').write_text('{}', encoding='utf-8')
    elif kind == 'contract_lock':
        (root / 'portable-contract.lock.json').write_text('{}', encoding='utf-8')
    else:
        target.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source manifest'):
        entry.validate_binding(root)


def test_deleting_and_rehashing_stored_file_membership_does_not_bypass_scan(tmp_path, fixed_dependencies):
    root = source(tmp_path)
    result = bind(root)
    path = Path(result['binding_path'])
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['source']['files'] = payload['source']['files'][1:]
    payload['source']['sha256'] = canonical_json_sha256({key: payload['source'][key] for key in ('policy', 'root_presence', 'files')})
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='source manifest'):
        entry.validate_binding(root)


def test_source_scanning_never_reads_env_credentials_models_or_runtime_outputs(tmp_path, monkeypatch):
    root = source(tmp_path)
    forbidden = [root / 'src/.env', root / 'scripts/.env.example', root / 'src/credentials/token.json',
                 root / 'src/models/model.safetensors', root / 'scripts/outputs/raw.json',
                 root / '.local/secret.json', root / 'media/video.mp4']
    for path in forbidden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic read trap')
    real = entry.sha256_file
    def guarded(path):
        assert path not in forbidden
        return real(path)
    monkeypatch.setattr(entry, 'sha256_file', guarded)
    with pytest.raises(ValueError, match='runtime/private directory'):
        entry.source_snapshot(root)
    # All private trees were refused by name without any content hash/read.
    for folder in ('src/credentials', 'src/models', 'scripts/outputs'):
        shutil.rmtree(root / folder)
    snapshot = entry.source_snapshot(root)
    assert all(row['path'] not in {p.relative_to(root).as_posix() for p in forbidden} for row in snapshot['files'])


@pytest.mark.parametrize('folder', ['models', 'outputs', 'credentials', 'secrets', 'media', 'node_modules', '.venv'])
def test_new_executable_under_excluded_source_trees_is_rejected_without_reading_it(tmp_path, monkeypatch, fixed_dependencies, folder):
    root = source(tmp_path)
    bind(root)
    hidden = root / 'src' / folder / 'extra.py'
    hidden.parent.mkdir()
    hidden.write_text('raise AssertionError("must not be imported or hashed")', encoding='utf-8')
    real = entry.sha256_file
    monkeypatch.setattr(entry, 'sha256_file', lambda path: (_ for _ in ()).throw(AssertionError('private file read'))
                        if path == hidden else real(path))
    with pytest.raises(ValueError, match='runtime/private directory'):
        entry.validate_binding(root)


def test_expected_bytecode_cache_is_ignored_but_hidden_source_under_it_is_rejected(tmp_path, fixed_dependencies):
    root = source(tmp_path)
    bind(root)
    cache = root / 'src/video_knowledge_pipeline/__pycache__'
    cache.mkdir()
    (cache / 'metadata.cpython-313.pyc').write_bytes(b'synthetic cache bytes')
    entry.validate_binding(root)
    (cache / 'extra.py').write_text('unexpected = True', encoding='utf-8')
    with pytest.raises(ValueError, match='excluded cache'):
        entry.validate_binding(root)


@pytest.mark.parametrize('poisoned', ['file_hash.py', 'canonical_json.py', '__init__.py'])
def test_direct_script_proof_rejects_helper_drift_before_importing_package_code(tmp_path, fixed_dependencies, poisoned):
    root = source(tmp_path)
    bind(root)
    path = root / 'src/video_knowledge_pipeline' / poisoned
    path.write_text('raise AssertionError("POISONED_PACKAGE_WAS_EXECUTED")', encoding='utf-8')
    result = subprocess.run([sys.executable, '-B', str(root / 'src/video_knowledge_pipeline/tested_entry.py'),
        '--version-proof'], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode != 0 and 'source manifest' in result.stderr
    assert 'POISONED_PACKAGE_WAS_EXECUTED' not in result.stderr


def test_binding_refuses_dependency_metadata_and_interpreter_drift(tmp_path, monkeypatch, fixed_dependencies):
    root = source(tmp_path)
    bind(root)
    changed = copy.deepcopy(fixed_dependencies)
    changed['distributions'][0]['version'] = '2'
    changed['sha256'] = canonical_json_sha256(changed['distributions'])
    monkeypatch.setattr(entry, 'dependency_snapshot', lambda: changed)
    with pytest.raises(ValueError, match='dependency metadata'):
        entry.validate_binding(root)
    monkeypatch.setattr(entry, 'dependency_snapshot', lambda: fixed_dependencies)
    identity = entry.python_identity()
    monkeypatch.setattr(entry, 'python_identity', lambda: {**identity, 'version': 'different'})
    with pytest.raises(ValueError, match='Python executable'):
        entry.validate_binding(root)


def test_creation_requires_explicit_interpreter_head_and_ignored_local_binding(tmp_path, fixed_dependencies):
    root = source(tmp_path)
    with pytest.raises(ValueError, match='explicitly selected Python'):
        entry.create_binding(root, python=root / 'missing-python', expected_head=HEAD)
    with pytest.raises(ValueError, match='expected-head'):
        entry.create_binding(root, python=sys.executable, expected_head=OTHER)
    with pytest.raises(ValueError, match='ignored .local'):
        entry.create_binding(root, python=sys.executable, expected_head=HEAD, binding=root / 'published-config.json')
    result = bind(root)
    before = Path(result['binding_path']).read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        bind(root)
    assert Path(result['binding_path']).read_bytes() == before
    (root / '.git/HEAD').write_text(OTHER, encoding='utf-8')
    with pytest.raises(ValueError, match='actual Git HEAD changed'):
        entry.validate_binding(root)


def test_git_head_supports_loose_packed_and_linked_worktree_refs(tmp_path):
    root = source(tmp_path)
    git = root / '.git'
    (git / 'HEAD').write_text('ref: refs/heads/master\n', encoding='utf-8')
    (git / 'refs/heads').mkdir(parents=True)
    (git / 'refs/heads/master').write_text(HEAD, encoding='utf-8')
    assert entry.read_git_head(root) == HEAD
    (git / 'refs/heads/master').unlink()
    (git / 'packed-refs').write_text(f'# packed\n{OTHER} refs/heads/master\n', encoding='utf-8')
    assert entry.read_git_head(root) == OTHER
    linked = tmp_path / 'linked'
    linked.mkdir()
    worktree_git = git / 'worktrees/linked'
    worktree_git.mkdir(parents=True)
    (worktree_git / 'HEAD').write_text('ref: refs/heads/master\n', encoding='utf-8')
    (worktree_git / 'commondir').write_text('../..', encoding='utf-8')
    (linked / '.git').write_text(f'gitdir: {worktree_git}\n', encoding='utf-8')
    assert entry.read_git_head(linked) == OTHER
    (worktree_git / 'HEAD').write_text(HEAD, encoding='utf-8')
    assert entry.read_git_head(linked) == HEAD
    (worktree_git / 'HEAD').write_text('ref: refs/../secret\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unsafe Git ref'):
        entry.read_git_head(linked)


def test_allowlist_preserves_declared_console_and_rejects_arbitrary_module(tmp_path, fixed_dependencies):
    root = source(tmp_path)
    bind(root)
    assert entry.validate_binding(root, entry_point='quick-health')['callable'] == 'video_knowledge_pipeline.quick_health:main'
    assert entry.validate_binding(root, entry_point='model_api_settings_ui')['callable'] == 'video_knowledge_pipeline.model_api_settings_http:main'
    for invalid in ('unrelated', 'os:system', 'video_knowledge_pipeline.cli:main'):
        with pytest.raises(ValueError, match='entry point'):
            entry.validate_binding(root, entry_point=invalid)


def test_dispatch_pins_child_pythonpath_preserves_other_values_and_native_argv(tmp_path, monkeypatch):
    root = source(tmp_path)
    source_path = str(root / 'src')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(['retained-path', source_path, 'retained-path']))
    original_argv = sys.argv
    captured = {}
    def native():
        captured.update(argv=list(sys.argv), pythonpath=os.environ['PYTHONPATH'])
        return 17
    monkeypatch.setattr(entry.importlib, 'import_module', lambda _: SimpleNamespace(
        __file__=str(root / 'src/video_knowledge_pipeline/cli.py'), main=native))
    result = entry.dispatch(root, {'entry_point': 'video-knowledge', 'callable': 'video_knowledge_pipeline.cli:main'},
                            ['export-creator-report', 'a b', '--no-write'])
    assert result == 17 and sys.argv is original_argv
    assert captured['argv'] == ['video-knowledge', 'export-creator-report', 'a b', '--no-write']
    assert captured['pythonpath'].split(os.pathsep) == [source_path, 'retained-path']


def test_proof_does_not_import_native_provider_or_dispatch(tmp_path, monkeypatch, fixed_dependencies, capsys):
    root = source(tmp_path)
    bind(root)
    monkeypatch.setattr(entry, 'repository_root', lambda: root)
    def forbidden(*args, **kwargs):
        raise AssertionError('proof may not import or dispatch a provider')
    monkeypatch.setattr(entry, 'dispatch', forbidden)
    assert entry.main(['--version-proof', '--entry-point', 'quick-health']) == 0
    assert json.loads(capsys.readouterr().out)['entry_point'] == 'video-knowledge-quick-health'


def test_powershell_local_binding_creation_proof_native_args_and_no_fallback(tmp_path):
    # Actual PowerShell/Python processes, synthetic code only; no browser/network/model.
    pwsh = shutil.which('pwsh') or 'C:/Program Files/PowerShell/7/pwsh.exe'
    if not Path(pwsh).is_file():
        pytest.skip('PowerShell is unavailable for this cross-platform smoke')
    root = source(tmp_path)
    wrapper = root / 'scripts/run-tested-vkp.ps1'
    def run(*args):
        return subprocess.run([pwsh, '-NoProfile', '-File', str(wrapper), *args],
            capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    missing = run('--version-proof')
    assert missing.returncode != 0 and 'no interpreter fallback' in missing.stderr
    created = run('-CreateBinding', '-Python', sys.executable, '-ExpectedHead', HEAD)
    assert created.returncode == 0, created.stderr
    proof = run('--version-proof')
    assert proof.returncode == 0, proof.stderr
    assert json.loads(proof.stdout)['git_head'] == HEAD
    native = run('export-creator-report', 'path with spaces', '--no-write')
    assert native.returncode == 0, native.stderr
    result = json.loads(native.stdout)
    assert result['native'] and result['args'] == ['export-creator-report', 'path with spaces', '--no-write']
    assert result['pythonpath'].split(os.pathsep)[0] == str(root / 'src')
    native_flags = run('some-native-command', '--python', 'native-value')
    assert native_flags.returncode == 0, native_flags.stderr
    assert json.loads(native_flags.stdout)['args'] == ['some-native-command', '--python', 'native-value']
    (root / 'src/video_knowledge_pipeline/cli.py').write_text('raise AssertionError("not executed")', encoding='utf-8')
    changed = run('--version-proof')
    assert changed.returncode != 0 and 'source manifest' in changed.stderr
