"""Synthetic exact-padding reuse: never approximate glyphs or skip detection."""
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from test_courseware_dedupe import image_bundle
from test_courseware_incremental_ocr import picture
from video_knowledge_pipeline import courseware_incremental_ocr as module
from video_knowledge_pipeline.courseware_material import read_scan


POLICY = 'uniform_padding_exact_v1'


def crop(padding=(2, 3, 2, 3)):
    # No font or OCR model: asymmetric strokes, internal whitespace and a dot.
    content = np.full((9, 21, 3), 255, dtype=np.uint8)
    content[0:9, 0:2] = 0
    content[0:2, :10] = 0
    content[4:6, 5:11] = 0
    content[7:9, 20:21] = 0
    top, right, bottom, left = padding
    return np.pad(content, ((top, bottom), (left, right), (0, 0)), constant_values=255)


def cache(root, recognize, policy=POLICY):
    runtime = {'model_actual': 'synthetic-v1', 'recognition_cache_policy': policy}
    return module.RegionRecognitionCache(root, runtime, recognize, SimpleNamespace)


def test_padding_only_change_reuses_with_exact_current_crop_proof(tmp_path):
    root = image_bundle(tmp_path, [picture(), picture()])
    candidates = read_scan(root)['candidates']
    calls = []

    def recognize(images):
        calls.extend(images)
        return SimpleNamespace(txts=('Condition: 10.5%',) * len(images), scores=[.99] * len(images))

    adapter = cache(root, recognize)
    adapter.begin_frame(candidates[0])
    original = crop()
    adapter([original])
    first = adapter.events[0]
    adapter.begin_frame(candidates[1])
    shifted = crop((4, 1, 1, 4))
    output = adapter([shifted])
    second = adapter.events[0]
    assert len(calls) == 1
    assert calls[0] is original  # No preprocessing of upstream recognizer input.
    assert output.imgs[0] is shifted
    assert second['crop_sha256'] != first['crop_sha256']
    assert second['cache_identity']['cache_key_sha256'] == first['cache_identity']['cache_key_sha256']
    assert second['cache_identity']['method'] == POLICY
    assert second['recognition_id'] == first['recognition_id']
    assert second['origin'] == 'cache_reuse'
    assert second['source'] == first['source']


@pytest.mark.parametrize('change', ['dot', 'internal_space', 'tint', 'border_noise', 'clipped'])
def test_any_glyph_background_or_crop_boundary_change_is_a_miss(tmp_path, change):
    root = image_bundle(tmp_path, [picture(), picture()])
    candidates = read_scan(root)['candidates']
    calls = []

    def recognize(images):
        calls.extend(images)
        return SimpleNamespace(txts=('Observed text',) * len(images), scores=[.99] * len(images))

    adapter = cache(root, recognize)
    adapter.begin_frame(candidates[0])
    adapter([crop()])
    image = crop((4, 1, 1, 4))
    if change == 'dot':
        image[11, 24] = 255  # A one-pixel change in the dot must never disappear.
    elif change == 'internal_space':
        image[7, 18] = 0
    elif change == 'tint':
        image[image == 255] = 254
    elif change == 'border_noise':
        image[0, 5] = 254
    elif change == 'clipped':
        image = image[4:, 4:]
    adapter.begin_frame(candidates[1])
    adapter([image])
    assert len(calls) == 2


def test_default_policy_preserves_exact_crop_behavior(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    calls = []
    def recognize(images):
        calls.extend(images)
        return SimpleNamespace(txts=('Observed text',) * len(images), scores=[.99] * len(images))
    adapter = cache(root, recognize, 'exact_current_crop')
    adapter.begin_frame(read_scan(root)['candidates'][0])
    adapter([crop(), crop((4, 1, 1, 4))])
    assert len(calls) == 2


def test_unknown_policy_is_rejected_before_recognition(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    with pytest.raises(ValueError, match='cache policy'):
        cache(root, lambda _: pytest.fail('recognizer executed'), 'fuzzy_ignore_small_text')


def test_blank_crops_do_not_canonicalize_different_extents():
    left = np.full((15, 30, 3), 255, dtype=np.uint8)
    right = np.full((17, 31, 3), 255, dtype=np.uint8)
    a = module.crop_cache_identity(left, runtime_sha256='a' * 64, method=POLICY)
    b = module.crop_cache_identity(right, runtime_sha256='a' * 64, method=POLICY)
    assert a['cache_key_sha256'] != b['cache_key_sha256']
    assert a['method'] == b['method'] == 'exact_current_crop'


def test_lru_touch_survives_checkpoint_and_avoids_hot_region_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MAX_CACHE_ROWS', 2)
    root = image_bundle(tmp_path, [picture()] * 4)
    candidates = read_scan(root)['candidates']
    calls = []
    def recognize(images):
        calls.extend(images)
        return SimpleNamespace(txts=('Observed text',) * len(images), scores=[.99] * len(images))
    images = [np.full((10, 20, 3), i, dtype=np.uint8) for i in (10, 20, 30)]
    adapter = cache(root, recognize, 'exact_current_crop')
    adapter.begin_frame(candidates[0])
    adapter(images[:2])
    adapter.begin_frame(candidates[1])
    adapter([images[0]])  # Touch hot A, no recognition call.
    adapter = cache(root, recognize, 'exact_current_crop')  # Simulate process restart.
    adapter.begin_frame(candidates[2])
    adapter([images[2]])  # Cold B must be evicted, not hot A.
    adapter.begin_frame(candidates[3])
    adapter([images[0]])
    assert len(calls) == 3


def test_padding_policy_is_bound_to_cache_and_run_identity(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    candidate = read_scan(root)['candidates'][0]
    calls = []
    def recognize(images):
        calls.extend(images)
        return SimpleNamespace(txts=('Observed text',) * len(images), scores=[.99] * len(images))
    for policy in ('exact_current_crop', POLICY):
        adapter = cache(root, recognize, policy)
        adapter.begin_frame(candidate)
        adapter([crop()])
    assert len(calls) == 2
    scan = read_scan(root)['identity']
    assert module.incremental_run_id(scan, [1], {'model_actual':'x'}) != module.incremental_run_id(
        scan, [1], {'model_actual':'x', 'recognition_cache_policy':POLICY})


def test_policy_rejects_invalid_cache_mode_before_opening_model_files(tmp_path):
    from video_knowledge_pipeline.courseware_incremental_runtime import model_policy
    from video_knowledge_pipeline.storage import write_json
    path = tmp_path / 'model-policy.json'
    for mode in ('approximate', [], None, True):
        write_json(path, {'schema':'video_knowledge_pipeline.incremental_ocr_models.v1',
                          'recognition_cache_policy':mode})
        with pytest.raises(ValueError, match='cache policy'):
            model_policy(path)


@pytest.mark.parametrize('field', ['cache_key_sha256', 'content_box', 'content_sha256', 'crop_sha256'])
def test_padding_identity_tampering_is_rejected(tmp_path, field):
    root = image_bundle(tmp_path, [picture()])
    adapter = cache(root, lambda images: SimpleNamespace(txts=('Text',), scores=[.99]))
    adapter.begin_frame(read_scan(root)['candidates'][0])
    adapter([crop()])
    events = deepcopy(adapter.events)
    events[0]['cache_identity'][field] = [0, 0, 1, 1] if field == 'content_box' else '0' * 64
    with pytest.raises(ValueError, match='crop identity'):
        module._region_projection(events, {'txts':['Text'], 'scores':[.99]},
                                  [{'text':'Text','score':.99}], cache_policy=POLICY)


@pytest.mark.parametrize('change', ['strip_proofs', 'recognition_id', 'source_frame', 'runtime_policy'])
def test_uniform_policy_cannot_downgrade_or_reassign_recognition_provenance(tmp_path, change):
    root = image_bundle(tmp_path, [picture()])
    adapter = cache(root, lambda images: SimpleNamespace(txts=('Text',), scores=[.99]))
    adapter.begin_frame(read_scan(root)['candidates'][0])
    adapter([crop()])
    events = deepcopy(adapter.events)
    if change == 'strip_proofs':
        del events[0]['cache_identity']
        del events[0]['source_crop_identity']
    elif change == 'recognition_id':
        events[0]['recognition_id'] = 'a' * 64
    elif change == 'source_frame':
        events[0]['source']['frame_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='identity'):
        module._region_projection(events, {'txts':['Text'], 'scores':[.99]},
            [{'text':'Text','score':.99}], cache_policy='exact_current_crop' if change == 'runtime_policy' else POLICY)


def test_padding_reuse_import_and_resume_preserve_target_geometry_and_sources(tmp_path, monkeypatch):
    from PIL import Image
    from video_knowledge_pipeline.storage import read_json
    root = image_bundle(tmp_path, [Image.fromarray(crop()), Image.fromarray(crop((4, 1, 1, 4)))])
    class Engine:
        return_word_box = False
        calls = 0
        def recognize_txt(self, images):
            self.calls += 1
            return SimpleNamespace(txts=('Payment 10.5%, subject to conditions.',), scores=[.99])
        def __call__(self, path):
            with Image.open(path) as image:
                data = np.array(image.convert('RGB'))
            return self.recognize_txt([data])
    engine = Engine()
    def normalize(raw):
        return [{'text':raw.txts[0], 'score':raw.scores[0], 'bbox':[2, 3, 21, 9]}]
    runtime = {'model_actual':'synthetic-v1', 'recognition_cache_policy':POLICY}
    original_import = module.import_courseware_page
    def interrupted_import(bundle, result):
        if result['page_id'] == 'page-1':
            raise RuntimeError('injected second page import interruption')
        return original_import(bundle, result)
    monkeypatch.setattr(module, 'import_courseware_page', interrupted_import)
    with pytest.raises(RuntimeError, match='interruption'):
        module.run_incremental_courseware_ocr(root, indexes=[1, 2], engine=engine,
            runtime_identity=runtime, output_factory=SimpleNamespace, normalize_blocks=normalize)
    first_bytes = (root / 'courseware/pages/page-0.json').read_bytes()
    assert not (root / 'courseware/pages/page-1.json').exists()
    monkeypatch.setattr(module, 'import_courseware_page', original_import)
    receipt = module.run_incremental_courseware_ocr(root, indexes=[1, 2], engine=engine,
        runtime_identity=runtime, output_factory=SimpleNamespace, normalize_blocks=normalize, resume=True)
    assert engine.calls == 1 and receipt['recognized_regions'] == 0
    assert (root / 'courseware/pages/page-0.json').read_bytes() == first_bytes
    final = read_json(root / 'courseware/pages/page-1.json')
    module.validate_incremental_page(root, final)
    raw = read_json(root / final['incremental_ocr']['raw_path'])
    assert raw['region_events'][0]['source']['page_id'] == 'page-0'
    assert raw['observations'][0]['recognition_id'] == raw['region_events'][0]['recognition_id']
    assert raw['observations'][0]['text'] == raw['region_events'][0]['text']
    assert 'cache_identity' not in raw['observations'][0]
    assert 'cache_identity' in raw['region_events'][0]
    assert final['blocks'][0]['structure']['bbox'] == [2, 3, 21, 9]


def test_optional_pinned_rapidocr_crop_source_smoke():
    """Execute upstream crop geometry, not an OCR model or service."""
    import ast
    import os
    from pathlib import Path
    source_root = os.environ.get('VKP_RAPIDOCR_SOURCE_SMOKE')
    if not source_root:
        pytest.skip('optional pinned local RapidOCR source path not supplied')
    import cv2
    source = Path(source_root) / 'python/rapidocr/utils/process_img.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'get_rotate_crop_image')
    namespace = {'np':np, 'cv2':cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    original = crop((5, 5, 5, 5))
    height, width = original.shape[:2]
    points = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    actual = namespace['get_rotate_crop_image'](original, points)
    assert actual.dtype == original.dtype and actual.shape == original.shape
    padded = np.pad(actual, ((2,1),(3,1),(0,0)), constant_values=255)
    a = module.crop_cache_identity(actual, runtime_sha256='a'*64, method=POLICY)
    b = module.crop_cache_identity(padded, runtime_sha256='a'*64, method=POLICY)
    assert a['cache_key_sha256'] == b['cache_key_sha256']
