from pathlib import Path


def test_funasr_install_is_pinned_to_reviewed_speaker_release() -> None:
    """Keep new local ASR environments on the reviewed upstream contract.

    Intent: prevent a fresh setup from silently reinstalling an incompatible FunASR.
    Decision: assert the installer exposes and uses the reviewed version parameter.
    Reason: version 1.3.9 reproduced the SenseVoice/CAM++ timestamp-boundary failure.
    Evidence: FunASR 1.3.30 official timestamp/diarization tests passed 13/13 locally.
    Effective scope: installer contract only; this test never installs a package.
    """

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install-local-asr-env.ps1"
    ).read_text(encoding="utf-8-sig")

    assert '[string]$FunASRVersion = "1.3.30"' in script
    assert 'Invoke-PipInstall -Packages @("funasr==$FunASRVersion", "modelscope")' in script
    assert "funasr_required_version = $FunASRVersion" in script
