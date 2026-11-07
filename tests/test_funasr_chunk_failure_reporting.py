from video_knowledge_pipeline.funasr_chunked_runner import _child_failure_detail


def test_child_failure_detail_prefers_structured_stdout_error() -> None:
    stdout = (
        'loading model\n'
        '{"ok": false, "status": "failed", '
        '"error": "speaker diarization length mismatch"}\n'
    )
    stderr = "100%|##########| 1/1\nWARNING:root:length mismatch"

    assert (
        _child_failure_detail(stdout, stderr)
        == "speaker diarization length mismatch"
    )


def test_child_failure_detail_falls_back_to_stderr() -> None:
    assert _child_failure_detail("not-json", "runtime failed") == "runtime failed"


def test_child_failure_detail_preserves_structured_traceback() -> None:
    stdout = (
        '{"ok": false, "error": "generate failed", '
        '"error_traceback": "Traceback line 1\\nTraceback line 2"}\n'
    )

    assert _child_failure_detail(stdout, "progress") == (
        "generate failed\nTraceback line 1\nTraceback line 2"
    )
