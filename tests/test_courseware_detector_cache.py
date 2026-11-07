from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from test_courseware_dedupe import image_bundle
from video_knowledge_pipeline import courseware_detector_cache as module
from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.courseware_material import read_scan
from video_knowledge_pipeline.storage import read_json, write_json


def runtime(**changes):
    return {'model_actual': 'synthetic-detector', 'weights_sha256': 'a' * 64,
            'detector_cache_policy': module.EXACT_DETECTOR, **changes}


def fixture(tmp_path, count=3):
    root = image_bundle(tmp_path, [Image.new('RGB', (40, 30), 'white') for _ in range(count)])
    return root, read_scan(root)['candidates'], np.full((30, 40, 3), 255, dtype=np.uint8)


class Detector:
    def __init__(self, output=None):
        self.calls = 0
        self.marker = object()
        self.output = output

    def __call__(self, image):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        if self.output is not None:
            return self.output
        return SimpleNamespace(img=image, boxes=np.array([[[2, 3], [20, 3], [20, 12], [2, 12]]], dtype=np.float32),
                               scores=[.95], elapse=.01)


def cache(root, detector=None, identity=None):
    return module.DetectorResultCache(root, identity or runtime(), detector or Detector(), SimpleNamespace)


def test_exact_hit_after_restart_reconstructs_fresh_boxes_and_metrics(tmp_path):
    root, frames, image = fixture(tmp_path)
    detector = Detector()
    first = cache(root, detector)
    first.begin_frame(frames[0])
    result = first(image)
    result.boxes[:] = 0  # Upstream coordinates are mutated during final mapping.
    result.scores[0] = 0
    result.img[:] = 0
    current_image = np.full((30, 40, 3), 255, dtype=np.uint8)
    second = cache(root, detector)
    second.begin_frame(frames[1])
    hit = second(current_image)
    assert detector.calls == 1 and second.detector_calls == 0 and second.reused_inputs == 1
    assert hit.boxes[0, 0].tolist() == [2, 3] and hit.scores == [.95]
    assert hit.img is current_image
    assert second.events[0]['origin'] == 'cache_reuse'
    assert second.events[0]['source']['page_id'] == frames[0]['page_id']
    assert second.events[0]['target']['page_id'] == frames[1]['page_id']
    assert module.validate_detector_events(root, second.events, runtime()) == {
        'detector_calls': 0, 'reused_inputs': 1, 'detector_inputs': 1}
    hit.boxes[:] = 0
    assert second(current_image).boxes[0, 0].tolist() == [2, 3]


@pytest.mark.parametrize('change', ['pixel', 'shape', 'runtime'])
def test_any_pixel_shape_or_runtime_change_requires_detector(tmp_path, change):
    root, frames, image = fixture(tmp_path)
    detector = Detector()
    first = cache(root, detector)
    first.begin_frame(frames[0])
    first(image)
    changed = image.copy()
    identity = runtime()
    if change == 'pixel':
        changed[25, 35, 1] -= 1  # Outside every prior text box.
    elif change == 'shape':
        changed = changed.reshape(40, 30, 3)
    else:
        identity['weights_sha256'] = 'b' * 64
    second = cache(root, detector, identity)
    second.begin_frame(frames[1])
    second(changed)
    assert detector.calls == 2 and second.detector_calls == 1 and second.reused_inputs == 0


def test_disabled_hook_needs_neither_factory_nor_detector(tmp_path):
    with module.attach_detector_cache(object(), tmp_path, {}) as active:
        assert active is None
    with pytest.raises(ValueError, match='policy'):
        with module.attach_detector_cache(object(), tmp_path, {'detector_cache_policy': 'near'}):
            pass


def test_enabled_hook_forwards_attributes_and_restores_on_failure(tmp_path):
    root, frames, image = fixture(tmp_path)
    detector = Detector(RuntimeError('injected'))
    engine = SimpleNamespace(text_det=detector)
    with pytest.raises(RuntimeError, match='injected'):
        with module.attach_detector_cache(engine, root, runtime(), SimpleNamespace) as active:
            assert engine.text_det.marker is detector.marker
            active.begin_frame(frames[0])
            engine.text_det(image)
    assert engine.text_det is detector
    assert not (root / 'courseware/incremental-cache/detector/inputs.json').exists()


def test_empty_detection_is_reported_but_never_cached(tmp_path):
    root, frames, image = fixture(tmp_path)
    detector = Detector(SimpleNamespace(boxes=None, scores=None, elapse=.1))
    active = cache(root, detector)
    active.begin_frame(frames[0])
    assert active(image).boxes is None
    active.begin_frame(frames[1])
    assert active(image).boxes is None
    assert detector.calls == 2 and active.reused_inputs == 0
    assert module.validate_detector_events(root, active.events, runtime())['detector_calls'] == 1
    assert not (root / 'courseware/incremental-cache/detector/inputs.json').exists()


@pytest.mark.parametrize('kind', ['nan', 'outside', 'crossed', 'scores', 'count', 'too_many'])
def test_invalid_detector_geometry_or_scores_fail_before_cache(tmp_path, kind):
    root, frames, image = fixture(tmp_path)
    output = Detector()(image)
    if kind == 'nan':
        output.boxes[0, 0, 0] = float('nan')
    elif kind == 'outside':
        output.boxes[0, 0, 0] = -1
    elif kind == 'crossed':
        output.boxes[0] = [[2, 3], [20, 12], [20, 3], [2, 12]]
    elif kind == 'scores':
        output.scores = [float('inf')]
    elif kind == 'count':
        output.scores = []
    else:
        output.boxes = np.repeat(output.boxes, 513, axis=0)
        output.scores = [.95] * 513
    active = cache(root, Detector(output))
    active.begin_frame(frames[0])
    with pytest.raises(ValueError):
        active(image)
    assert not (root / 'courseware/incremental-cache/detector/inputs.json').exists()


@pytest.mark.parametrize('image', [np.zeros((30, 40, 3), dtype=np.float32),
                                  np.zeros((30, 40, 4), dtype=np.uint8),
                                  np.zeros((0, 40, 3), dtype=np.uint8)])
def test_invalid_input_rejected_without_detector(tmp_path, image):
    root, frames, _ = fixture(tmp_path)
    detector = Detector()
    active = cache(root, detector)
    active.begin_frame(frames[0])
    with pytest.raises(ValueError):
        active(image)
    assert detector.calls == 0


def test_source_drift_blocks_hits_and_event_validation(tmp_path):
    root, frames, image = fixture(tmp_path)
    active = cache(root)
    active.begin_frame(frames[0])
    active(image)
    events = deepcopy(active.events)
    (root / frames[0]['frame']).write_bytes(b'changed')
    active.begin_frame(frames[1])
    with pytest.raises(ValueError, match='hash drift'):
        active(image)
    with pytest.raises(ValueError, match='hash drift'):
        module.validate_detector_events(root, events, runtime())


@pytest.mark.parametrize('field', ['input', 'output', 'origin', 'target', 'source'])
def test_tampered_events_rejected_even_with_new_outer_hash(tmp_path, field):
    root, frames, image = fixture(tmp_path)
    active = cache(root)
    active.begin_frame(frames[0])
    active(image)
    event = deepcopy(active.events[0])
    if field == 'input':
        event['input_identity']['pixels_sha256'] = 'b' * 64
    elif field == 'output':
        event['scores'][0] = .1
    elif field == 'origin':
        event['origin'] = 'near_reuse'
    else:
        event[field]['page_id'] = 'page-nonexistent'
    event['event_sha256'] = canonical_json_sha256({k: v for k, v in event.items() if k != 'event_sha256'})
    with pytest.raises(ValueError):
        module.validate_detector_events(root, [event], runtime())


def test_load_budget_checked_before_json_parse(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    active = cache(root)
    path = root / 'courseware/incremental-cache/detector/inputs.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * 257)
    monkeypatch.setattr(module, 'MAX_CACHE_BYTES', 256)
    monkeypatch.setattr(active.cache.__class__, 'load_json', lambda *a, **kw: pytest.fail('unbounded parse'))
    with pytest.raises(ValueError, match='budget'):
        cache(root)


def test_hit_only_lru_order_persists_across_restart(tmp_path, monkeypatch):
    root, frames, image = fixture(tmp_path)
    monkeypatch.setattr(module, 'MAX_CACHE_ROWS', 2)
    detector = Detector()
    active = cache(root, detector)
    active.begin_frame(frames[0])
    active(image)
    other = image.copy()
    other[20, 20] = 10
    active(other)
    active(image)  # Refresh first input before restart.
    active = cache(root, detector)
    active.begin_frame(frames[1])
    third = image.copy()
    third[20, 20] = 11
    active(third)
    active(image)
    assert detector.calls == 3
    active(other)
    assert detector.calls == 4


def test_persisted_cache_tampering_is_rejected(tmp_path):
    root, frames, image = fixture(tmp_path)
    active = cache(root)
    active.begin_frame(frames[0])
    active(image)
    path = root / 'courseware/incremental-cache/detector/inputs.json'
    payload = read_json(path)
    payload['rows'][0]['scores'][0] = .1
    payload['sha256'] = canonical_json_sha256(payload['rows'])
    write_json(path, payload)
    with pytest.raises(ValueError, match='hash'):
        cache(root)


def test_historical_events_survive_adapter_change_but_new_cache_misses(tmp_path, monkeypatch):
    root, frames, image = fixture(tmp_path)
    detector = Detector()
    original_hash = module.sha256_file(module.Path(module.__file__))
    old_runtime = runtime(detector_adapter_sha256=original_hash)
    active = cache(root, detector, old_runtime)
    active.begin_frame(frames[0])
    active(image)
    events = deepcopy(active.events)
    actual_hash = module.sha256_file
    monkeypatch.setattr(module, 'sha256_file', lambda path: 'b' * 64
                        if module.Path(path) == module.Path(module.__file__) else actual_hash(path))
    assert module.validate_detector_events(root, events, old_runtime)['detector_calls'] == 1
    with pytest.raises(ValueError, match='adapter identity drift'):
        cache(root, detector, old_runtime)
    current = cache(root, detector, runtime(detector_adapter_sha256='b' * 64))
    current.begin_frame(frames[1])
    current(image)
    assert detector.calls == 2


def test_full_input_pixel_budget_is_checked_before_detection(tmp_path, monkeypatch):
    root, frames, image = fixture(tmp_path)
    active = cache(root)
    active.begin_frame(frames[0])
    monkeypatch.setattr(module, 'MAX_PIXELS', 1000)
    with pytest.raises(ValueError, match='pixel budget'):
        active(image)
    assert active.detector_calls == 0


def test_factory_failure_does_not_cache_successful_detector_output(tmp_path):
    root, frames, image = fixture(tmp_path)
    detector = Detector()

    def broken_factory(**kwargs):
        raise RuntimeError('injected factory failure')

    active = module.DetectorResultCache(root, runtime(), detector, broken_factory)
    active.begin_frame(frames[0])
    with pytest.raises(RuntimeError, match='factory failure'):
        active(image)
    assert detector.calls == 1 and not active.rows and not active.events
    assert not (root / 'courseware/incremental-cache/detector/inputs.json').exists()
    recovered = cache(root, detector)
    recovered.begin_frame(frames[0])
    recovered(image)
    assert detector.calls == 2
