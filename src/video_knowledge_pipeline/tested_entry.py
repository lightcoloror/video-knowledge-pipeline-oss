"""A local binding over native VKP entrypoints, not another execution runtime.

Reuse the project's file hashes, canonical JSON and console declarations. The
ignored operator binding records actual Git HEAD, a fixed source scan and Python
dependency metadata. Creation/proof never import providers or start services.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tomllib
from typing import Any

# Formal PowerShell launch uses this file directly. The bootstrap must remain
# stdlib-only: importing even our normal hash helpers would execute unverified
# package code before drift is rejected. These two small functions preserve the
# existing file_hash/canonical_json contracts; native code is imported only by
# dispatch after full validation. Python -m startup is not this pre-import gate.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

SCHEMA = 'video_knowledge_pipeline.tested_entry.v1'
SOURCE_POLICY = 'src-scripts-config-static-and-root-contracts.v1'
SOURCE_ROOTS = ('src', 'scripts', 'config')
ROOT_CONFIGS = ('pyproject.toml', 'setup.cfg', 'setup.py', 'uv.lock', 'poetry.lock',
                'requirements.txt', 'requirements.lock', 'portable-contract.lock.json',
                'agent-tool-manifest.v1.json')
EXCLUDED_PARTS = frozenset({'.git', '.local', '.venv', '__pycache__', 'node_modules',
    'media', 'models', 'outputs', 'downloads', 'transcripts', 'credentials', 'secrets',
    '.cache', 'logs', 'tmp', '.tmp'})
EXCLUDED_SUFFIXES = frozenset({'.pyc', '.pyo', '.mp4', '.mkv', '.mov', '.webm', '.avi',
    '.wav', '.mp3', '.m4a', '.flac', '.onnx', '.safetensors', '.gguf', '.ckpt', '.pth',
    '.pt', '.pem', '.key', '.pfx', '.p12', '.token', '.secret', '.session'})
SOURCE_SUFFIXES = frozenset({'.py', '.ps1', '.cmd', '.bat', '.sh', '.js', '.cjs', '.mjs',
    '.ts', '.tsx', '.vue', '.css', '.html', '.json', '.toml', '.yaml', '.yml', '.cfg',
    '.ini', '.schema', '.sql', '.md', '.txt'})
STATIC_SUFFIXES = frozenset({'.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico', '.woff',
    '.woff2', '.ttf', '.wasm', '.map'})
EXECUTABLE_SUFFIXES = frozenset({'.py', '.pyc', '.pyo', '.js', '.cjs', '.mjs', '.ts',
    '.tsx', '.ps1', '.cmd', '.bat', '.sh'})
ALIASES = {'cli': 'video-knowledge', 'mcp': 'video-knowledge-mcp',
    'autopilot_mcp': 'video-knowledge-autopilot-mcp',
    'personal_vlog_mcp': 'video-knowledge-personal-vlog-mcp',
    'openclaw_http': 'video-knowledge-openclaw-http', 'review_http': 'video-knowledge-review',
    'quick-health': 'video-knowledge-quick-health'}
EXTRA_ENTRY_POINTS = {'model_api_settings_ui': 'video_knowledge_pipeline.model_api_settings_http:main'}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('local tested-version binding missing or too large')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('invalid local tested-version binding schema')
    return value


def read_git_head(root: str | Path) -> str:
    """Read detached, loose or packed refs, including linked-worktree gitdirs."""
    root = Path(root).resolve()
    marker = root / '.git'
    if marker.is_dir():
        directory = marker.resolve()
    elif marker.is_file():
        text = marker.read_text(encoding='utf-8-sig').strip()
        if not text.startswith('gitdir: '):
            raise ValueError('invalid .git directory marker')
        directory = (root / text[8:]).resolve()
    else:
        raise ValueError('selected source has no Git HEAD')
    head = (directory / 'HEAD').read_text(encoding='utf-8-sig').strip()
    if re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', head):
        return head.lower()
    if not head.startswith('ref: '):
        raise ValueError('invalid Git HEAD')
    ref = head[5:]
    parts = PurePosixPath(ref).parts
    if (not ref.startswith('refs/') or '\\' in ref or not parts
            or any(part in {'.', '..'} for part in parts) or ref != '/'.join(parts)):
        raise ValueError('unsafe Git ref')
    common_file = directory / 'commondir'
    common = (directory / common_file.read_text(encoding='utf-8-sig').strip()).resolve() if common_file.is_file() else directory
    for candidate in dict.fromkeys([directory, common]):
        loose = candidate / ref
        if loose.is_file():
            value = loose.read_text(encoding='utf-8-sig').strip()
            if re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', value):
                return value.lower()
            raise ValueError('invalid loose Git ref')
        packed = candidate / 'packed-refs'
        if packed.is_file():
            for line in packed.read_text(encoding='utf-8-sig').splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    if not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', fields[0]):
                        raise ValueError('invalid packed Git ref')
                    return fields[0].lower()
    raise ValueError('Git HEAD ref is unavailable')


def _excluded(name: str) -> bool:
    lower = name.casefold()
    return (lower in EXCLUDED_PARTS or lower.startswith('.env') or lower in {'.netrc', '.envrc'}
            or lower.startswith('cookies') or lower.startswith('id_rsa') or lower.startswith('id_ed25519')
            or lower.endswith('.credential.xml') or Path(lower).suffix in EXCLUDED_SUFFIXES)


def source_snapshot(root: str | Path) -> dict[str, Any]:
    """Derive membership from code policy; a binding cannot omit a new file."""
    root = Path(root).resolve()
    files = []
    present = {}
    for folder in SOURCE_ROOTS:
        base = root / folder
        present[folder] = base.is_dir()
        if not base.is_dir():
            if folder == 'config':
                continue
            raise ValueError(f'missing source directory: {folder}')
        for directory, dirs, names in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                if _excluded(name):
                    if name == '__pycache__':
                        # Normal bytecode caches are not source inputs. Inspect
                        # names only and reject executable source hidden there.
                        for cached in path.iterdir():
                            if cached.is_dir() or cached.suffix.casefold() in (EXECUTABLE_SUFFIXES - {'.pyc', '.pyo'}):
                                raise ValueError('source hidden under an excluded cache directory')
                        dirs.remove(name)
                    else:
                        # Runtime/private trees must sit outside managed source.
                        # Refuse without scanning weights, credentials or data.
                        raise ValueError('unsupported runtime/private directory under managed source')
                elif path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('source directory links are not supported by the binding')
            for name in names:
                if _excluded(name):
                    if Path(name).suffix.casefold() in EXECUTABLE_SUFFIXES:
                        raise ValueError('executable source uses an excluded private/cache filename')
                    continue
                path = Path(directory) / name
                suffix = path.suffix.casefold()
                static = any(part in {'static', 'assets', 'fixtures'} for part in path.relative_to(root).parts)
                if not (suffix in SOURCE_SUFFIXES or (static and suffix in STATIC_SUFFIXES)
                        or name in {'LICENSE', 'NOTICE', 'COPYING'}):
                    raise ValueError('unreviewed file type in managed source scope')
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('source file escapes the selected checkout')
                files.append({'path': path.relative_to(root).as_posix(), 'sha256': sha256_file(path)})
    for name in ROOT_CONFIGS:
        path = root / name
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('build configuration escapes the selected checkout')
            files.append({'path': name, 'sha256': sha256_file(path)})
    files.sort(key=lambda row: row['path'])
    policy = {'version': SOURCE_POLICY, 'source_roots': list(SOURCE_ROOTS),
        'root_configs': list(ROOT_CONFIGS), 'source_suffixes': sorted(SOURCE_SUFFIXES),
        'static_suffixes': sorted(STATIC_SUFFIXES), 'excluded_parts': sorted(EXCLUDED_PARTS),
        'excluded_suffixes': sorted(EXCLUDED_SUFFIXES), 'executable_suffixes': sorted(EXECUTABLE_SUFFIXES)}
    payload = {'policy': policy, 'root_presence': present, 'files': files}
    return {**payload, 'sha256': canonical_json_sha256(payload)}


def dependency_snapshot() -> dict[str, Any]:
    """Inspect installed distribution metadata, never import models/weights."""
    rows = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name')
        if not name:
            raise ValueError('installed distribution is missing its name')
        rows.append({'name': re.sub(r'[-_.]+', '-', name).lower(),
                     'version': str(distribution.version),
                     'location': str(Path(distribution.locate_file('')).resolve())})
    rows.sort(key=lambda row: (row['name'], row['version'], row['location']))
    return {'kind': 'installed_distribution_metadata_not_binary_or_model_attestation',
            'distributions': rows, 'sha256': canonical_json_sha256(rows)}


def python_identity() -> dict[str, str]:
    binary = Path(sys.executable).resolve()
    return {'path': str(binary), 'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable_sha256': sha256_file(binary)}


def entrypoints(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8-sig'))
    rows = dict(project.get('project', {}).get('scripts', {}))
    rows.update(EXTRA_ENTRY_POINTS)
    checked = {}
    for name, target in rows.items():
        if (isinstance(name, str) and isinstance(target, str)
                and re.fullmatch(r'video_knowledge_pipeline\.[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*', target)):
            checked[name] = target
    return checked


def _selected_entry(root: Path, name: str) -> tuple[str, str]:
    chosen = ALIASES.get(name, name)
    options = entrypoints(root)
    if chosen not in options:
        raise ValueError('entry point must be a declared VKP console or fixed local alias')
    return chosen, options[chosen]


def binding_path(root: Path, value: str | Path | None = None) -> Path:
    selected = value or os.environ.get('VKP_TESTED_VERSION_PATH')
    return Path(selected).expanduser().resolve() if selected else root / '.local/tested-version.json'


def create_binding(source_root: str | Path, *, python: str | Path, expected_head: str,
                   binding: str | Path | None = None) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve()
    target = binding_path(root, binding)
    if target.is_relative_to(root) and not target.is_relative_to(root / '.local'):
        raise ValueError('in-repository operator bindings must stay in ignored .local')
    if Path(python).expanduser().resolve() != Path(sys.executable).resolve():
        raise ValueError('create the binding using the explicitly selected Python')
    actual_head = read_git_head(root)
    if actual_head != expected_head.lower():
        raise ValueError('actual Git HEAD differs from expected-head')
    payload = {'schema': SCHEMA, 'source_root': str(root), 'git_head': actual_head,
               'python': python_identity(), 'source': source_snapshot(root),
               'dependencies': dependency_snapshot(),
               'operator_boundary': {'metadata_only': True, 'models_imported': False,
                    'network_calls_made': 0, 'quality_gate_bypassed': False}}
    if read_git_head(root) != actual_head:
        raise ValueError('Git HEAD changed while recording the binding')
    if target.exists():
        raise ValueError('operator binding already exists; preserve it and choose a new path')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'status': 'binding_created', 'binding_path': str(target), 'git_head': actual_head,
            'source_sha256': payload['source']['sha256'], 'dependency_metadata_sha256': payload['dependencies']['sha256'],
            'quality_gate_bypassed': False}


def validate_binding(root: str | Path, binding: str | Path | None = None,
                     *, entry_point: str = 'cli') -> dict[str, Any]:
    root = Path(root).resolve()
    path = binding_path(root, binding)
    payload = _json(path)
    if payload.get('source_root') != str(root):
        raise ValueError('binding belongs to another source checkout')
    actual_head = read_git_head(root)
    if payload.get('git_head') != actual_head:
        raise ValueError('actual Git HEAD changed from the bound commit')
    if payload.get('python') != python_identity():
        raise ValueError('Python executable, version or implementation drift')
    # Re-derive membership and hashes, rather than trusting the stored file list.
    source = source_snapshot(root)
    if payload.get('source') != source:
        raise ValueError('source manifest or source contents drift')
    dependencies = dependency_snapshot()
    if payload.get('dependencies') != dependencies:
        raise ValueError('dependency metadata drift')
    selected, target = _selected_entry(root, entry_point)
    if read_git_head(root) != actual_head:
        raise ValueError('Git HEAD changed during validation')
    return {'status': 'binding_verified', 'binding_path': str(path), 'source_root': str(root),
        'git_head': actual_head, 'source_sha256': source['sha256'], 'verified_source_files': len(source['files']),
        'python': payload['python'], 'dependency_metadata_sha256': dependencies['sha256'],
        'entry_point': selected, 'callable': target, 'quality_gate_bypassed': False,
        'content_quality': 'determined_by_native_workflow_not_this_software_binding'}


def _source_environment(root: Path) -> None:
    source = str(root / 'src')
    values = [source, *os.environ.get('PYTHONPATH', '').split(os.pathsep)]
    unique, seen = [], set()
    for value in values:
        if not value:
            continue
        identity = os.path.normcase(os.path.abspath(value))
        if identity not in seen:
            unique.append(value)
            seen.add(identity)
    os.environ['PYTHONPATH'] = os.pathsep.join(unique)
    sys.path.insert(0, source)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def dispatch(root: Path, proof: dict[str, Any], argv: list[str]) -> int:
    """Preserve the selected declared console function and its native arguments."""
    _source_environment(root)
    module_name, function_name = proof['callable'].split(':', 1)
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(root / 'src'):
        raise ValueError('native entrypoint imported from another checkout')
    original = sys.argv
    try:
        sys.argv = [proof['entry_point'], *argv]
        value = getattr(module, function_name)()
        return int(value or 0)
    finally:
        sys.argv = original


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own, native = (argv[:argv.index('--')], argv[argv.index('--')+1:]) if '--' in argv else (argv, [])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--binding')
    parser.add_argument('--entry-point', default='cli')
    parser.add_argument('--version-proof', action='store_true')
    parser.add_argument('--create-binding', action='store_true')
    parser.add_argument('--source-root')
    parser.add_argument('--python')
    parser.add_argument('--expected-head')
    args, trailing = parser.parse_known_args(own)
    if trailing:
        native = [*trailing, *native]
    if args.create_binding:
        if not args.source_root or not args.python or not args.expected_head or native or args.version_proof:
            raise ValueError('creation requires explicit source-root, python and expected-head only')
        result = create_binding(args.source_root, python=args.python, expected_head=args.expected_head, binding=args.binding)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.source_root or args.python or args.expected_head:
        raise ValueError('creation arguments are not runtime overrides')
    root = repository_root()
    proof = validate_binding(root, args.binding, entry_point=args.entry_point)
    if args.version_proof:
        if native:
            raise ValueError('version proof cannot dispatch a native command')
        print(json.dumps(proof, ensure_ascii=False))
        return 0
    return dispatch(root, proof, native)


if __name__ == '__main__':
    raise SystemExit(main())
