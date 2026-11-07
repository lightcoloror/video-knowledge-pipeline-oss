"""Native handoff regression for the VKP context-manifest ASR consumer.

Synthetic only. Fake media/audio bytes exist only to be hashed; nothing here
runs ASR, renders media, or claims workflow/business completion. The host runs
this file inside the VKP ``tests/`` directory with ``SELF_MEDIA_SYSTEM_ROOT``
(real public main flow) and ``TEST_TMP_ROOT`` (all temporary data) set.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_knowledge_pipeline.context_manifest_consumer_cli import decide_asr
from video_knowledge_pipeline.path_defaults import workspace_root


def _self_media_root() -> Path:
    configured = str(os.environ.get("SELF_MEDIA_SYSTEM_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (workspace_root() / "self-media-creation-system").resolve()


SELF_MEDIA_ROOT = _self_media_root()

_test_tmp_root = str(os.environ.get("TEST_TMP_ROOT") or "").strip()
if not _test_tmp_root:
    raise RuntimeError(
        "TEST_TMP_ROOT must be set so synthetic test data stays inside it"
    )
TEST_TMP_ROOT = Path(_test_tmp_root).expanduser().resolve()

FIXTURE_REQUEST = (
    SELF_MEDIA_ROOT
    / "fixtures"
    / "agent-entry"
    / "context-efficient-rework.request.json"
)


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "vkp_test_context_consumer_agent",
        SELF_MEDIA_ROOT / "scripts" / "self_media_agent.py",
    )
    if not spec or not spec.loader:
        raise RuntimeError(
            "self_media_agent.py is unavailable under SELF_MEDIA_SYSTEM_ROOT"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContextManifestConsumerContractTests(unittest.TestCase):
    """Drive the real native gates; never mock validate/prepare/purpose/hash."""

    def setUp(self) -> None:
        self.root = TEST_TMP_ROOT / "vkp-context-consumer" / (
            "context-consumer-" + uuid.uuid4().hex
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.root, True)

        self.agent = load_agent()
        self.shared = self.agent._load_context_efficiency()
        self.purpose_contract = importlib.import_module("purpose_continuity")

        request = self.agent.load_json(FIXTURE_REQUEST)
        self.capsule_path = self.root / "capsule.json"
        self.capsule_path.write_text(
            json.dumps(self.agent.resolve_request(request), ensure_ascii=False),
            encoding="utf-8",
        )
        self.capsule = self.agent.load_json(self.capsule_path)
        self.step_plan = self.capsule["step_index"][0]
        self.step_asr = self.capsule["step_index"][1]

        self.media = self.root / "source.mp4"
        self.media.write_bytes(b"synthetic-video")
        self.decision_sheet = self.root / "decision.json"
        self.decision_sheet.write_text('{"status":"approved"}', encoding="utf-8")
        self.rough_cut = self.root / "rough.json"
        self.rough_cut.write_text(
            '{"schema":"video_creation_pipeline.rough_cut_workspace.v2",'
            '"workspace_sha256":"synthetic"}',
            encoding="utf-8",
        )
        self.change_description = self.root / "change.json"
        self.change_description.write_text(
            '{"synthetic_only":true,"change":"initial review"}', encoding="utf-8"
        )
        self.purpose_source = self.root / "synthetic-purpose.json"
        self.purpose_source.write_text(
            json.dumps(
                {
                    "synthetic_only": True,
                    "purpose": "Decide synthetic ASR reuse only from a bound audio hash.",
                    "audience": "Synthetic offline regression testers.",
                }
            ),
            encoding="utf-8",
        )
        self.proxy_review = self.root / "proxy-review-plan.json"
        self.audio = self.root / "derived-audio.wav"
        self.audio.write_bytes(b"synthetic-audio-v1")
        self.sources = {
            "source_media": self.media,
            "decision_sheet": self.decision_sheet,
            "rough_cut_workspace": self.rough_cut,
            "change_description": self.change_description,
            "proxy-review-plan": self.proxy_review,
        }
        self.manifest = self.root / "video-review-input.json"
        self._write_review_manifest()
        self.predecessor_handoff, self.predecessor_receipt = self._build_predecessor()

    # -- native fixtures -------------------------------------------------
    def _write_review_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "self-media-video-review-input.v1",
                    "task_id": self.capsule["request_id"],
                    "mode": "initial_review",
                    "source_media": {
                        "path": self.media.name,
                        "expected_sha256": digest(self.media),
                        "duration_seconds": 30,
                    },
                    "decision_sheet": {
                        "path": self.decision_sheet.name,
                        "expected_sha256": digest(self.decision_sheet),
                    },
                    "rough_cut_workspace": {
                        "path": self.rough_cut.name,
                        "expected_sha256": digest(self.rough_cut),
                    },
                    "changed_windows": [],
                    "frozen_windows": [],
                    "acceptance_frame_timestamps": [3, 15, 27],
                    "change_kinds": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _purpose(self, step_id: str) -> dict:
        return {
            "version": 1,
            "content_id": "synthetic-context-consumer",
            "purpose": {
                "artifact_id": "synthetic-purpose",
                "quote": "Decide synthetic ASR reuse only from a bound audio hash.",
            },
            "audience": {
                "artifact_id": "synthetic-purpose",
                "quote": "Synthetic offline regression testers.",
            },
            "stage_duty": "Validate only synthetic ASR-decision metadata at " + step_id,
            "current_body_artifact_id": None,
            "feedback": [],
            "feedback_note": "No real author feedback, draft or approval is supplied.",
        }

    def _purpose_artifact(self) -> dict:
        return {
            "artifact_id": "synthetic-purpose",
            "role": "source_manifest",
            "source_role": "editorial-purpose",
            "path": self.purpose_source.name,
            "load_mode": "reference",
            "expected_sha256": digest(self.purpose_source),
        }

    def _input_artifact(self, index: int, role: str) -> dict:
        source = self.sources[role]
        return {
            "artifact_id": "input-" + str(index),
            # The recipe role stays in source_role. The schema artifact role is
            # the reference-proven generic source_manifest (the specific
            # alternative is proxy_review_plan); never copy source_role to role.
            "role": "source_manifest",
            "source_role": role,
            "path": source.name,
            "load_mode": "reference",
            "expected_sha256": digest(source),
        }

    def _derived_audio_artifact(self, audio: Path) -> dict:
        # Two roles with different jobs: schema role source_media lets the
        # consumer recognize a legal bound audio artifact, while the unique
        # recipe source_role derived-audio keeps source_role unique per artifact
        # (the native prepare gate rejects duplicate source_role values).
        return {
            "artifact_id": "derived-audio",
            "role": "source_media",
            "source_role": "derived-audio",
            "path": audio.name,
            "load_mode": "reference",
            "expected_sha256": digest(audio),
        }

    def _context_document(self, step: dict, extra_artifacts=()) -> dict:
        artifacts = [
            self._input_artifact(index, role)
            for index, role in enumerate(step["input_roles"])
        ]
        artifacts.extend(extra_artifacts)
        artifacts.append(self._purpose_artifact())
        return {
            "schema_version": "self-media-context-input.v1",
            "task_id": self.capsule["request_id"],
            "stage": step["context_stage"],
            "budget": {"max_inline_bytes": 1000},
            "artifacts": artifacts,
            "editorial_context": self._purpose(step["step_id"]),
        }

    def _purpose_review(
        self, step_id: str, context_path: Path, artifact_path: Path
    ) -> dict:
        return {
            "schema_version": self.purpose_contract.REVIEW_SCHEMA,
            "context_ref": {
                "path": context_path.name,
                "sha256": digest(context_path),
            },
            "artifact_ref": {
                "path": artifact_path.name,
                "sha256": digest(artifact_path),
            },
            "quote": artifact_path.read_text(encoding="utf-8"),
            "purpose_alignment": {
                "stage_duty": self._purpose(step_id)["stage_duty"],
                "contribution": (
                    "Bind the actual synthetic review plan to the next native "
                    "metadata step."
                ),
                "misunderstanding_risk": (
                    "This establishes no ASR, media, author or platform completion."
                ),
                "tradeoff": "Keep all execution and human approval pending.",
            },
            "feedback_resolutions": [],
            "disposition": "retained",
            "reason": (
                "Actual synthetic metadata output; no content quality or "
                "execution claim."
            ),
        }

    def _build_predecessor(self, tag: str = ""):
        """Build the real first step, its purpose review and a verified receipt."""
        step = self.step_plan
        prefix = tag + "-" if tag else ""
        context_path = self.root / (prefix + "plan-bounded-review-context.json")
        self.agent.write_or_print(self._context_document(step), context_path)
        handoff_path = self.root / (prefix + "plan-bounded-review-handoff.json")
        self.agent.prepare_step_handoff(
            self.capsule_path,
            step["step_id"],
            context_path,
            self.root / (prefix + "plan-bounded-review-context-receipt.json"),
            handoff_path,
        )
        plan_path = (
            self.proxy_review
            if not tag
            else self.root / (tag + "-proxy-review-plan.json")
        )
        self.shared.build_video_review_plan(self.manifest, plan_path)
        self.proxy_review = plan_path
        self.sources["proxy-review-plan"] = plan_path
        purpose_review_path = self.root / (
            prefix + "plan-bounded-review-purpose-review.json"
        )
        self.agent.write_or_print(
            self._purpose_review(step["step_id"], context_path, plan_path),
            purpose_review_path,
        )
        receipt_path = self.root / (prefix + "plan-bounded-review-output-receipt.json")
        receipt = {
            "schema_version": "self-media-step-output-receipt.v1",
            "status": "candidate_ready",
            "step_id": step["step_id"],
            "tool_id": step["tool_id"],
            "output_contract": step["output_contract"],
            "handoff": {"path": handoff_path.name, "sha256": digest(handoff_path)},
            "outputs": [
                {"path": plan_path.name, "sha256": digest(plan_path)},
                {"path": purpose_review_path.name, "sha256": digest(purpose_review_path)},
            ],
            "boundaries": self.capsule["boundaries"],
        }
        receipt["payload_sha256"] = self.agent.sha256_value(receipt)
        self.agent.write_or_print(receipt, receipt_path)
        verification = self.agent.verify_step_output(handoff_path, receipt_path)
        self.assertTrue(verification["ok"])
        self.assertFalse(verification["workflow_step_completed"])
        return handoff_path, receipt_path

    def _write_asr_context(self, audio: Path, tag: str) -> Path:
        step = self.step_asr
        context_path = self.root / (tag + "-decide-asr-context.json")
        self.agent.write_or_print(
            self._context_document(
                step, extra_artifacts=[self._derived_audio_artifact(audio)]
            ),
            context_path,
        )
        return context_path

    def _build_asr_handoff(self, audio: Path, previous_receipt: Path, tag: str):
        step = self.step_asr
        context_path = self._write_asr_context(audio, tag)
        handoff_path = self.root / (tag + "-decide-asr-handoff.json")
        self.agent.prepare_step_handoff(
            self.capsule_path,
            step["step_id"],
            context_path,
            self.root / (tag + "-decide-asr-context-receipt.json"),
            handoff_path,
            previous_receipt=previous_receipt,
        )
        return handoff_path, context_path

    # -- positives -------------------------------------------------------
    def test_predecessor_receipt_is_candidate_ready_not_completion(self) -> None:
        verification = self.agent.verify_step_output(
            self.predecessor_handoff, self.predecessor_receipt
        )
        self.assertTrue(verification["ok"])
        self.assertFalse(verification["workflow_step_completed"])
        receipt = self.agent.load_json(self.predecessor_receipt)
        self.assertEqual(receipt["status"], "candidate_ready")
        self.assertEqual(receipt["step_id"], "plan-bounded-review")

    def test_first_audio_hash_reruns_full_asr_and_same_hash_reuses(self) -> None:
        handoff, _ = self._build_asr_handoff(
            self.audio, self.predecessor_receipt, "initial"
        )
        first = self.root / "first-asr-decision.json"
        result = decide_asr(handoff, self.audio, None, first, [])
        self.assertEqual(result["decision"], "rerun_full_asr")
        self.assertTrue(first.is_file())

        second = self.root / "second-asr-decision.json"
        reused = decide_asr(handoff, self.audio, first, second, [])
        self.assertEqual(reused["decision"], "reuse_existing_asr")

    def test_changed_audio_and_window_rerun_changed_windows_with_context(self) -> None:
        handoff, _ = self._build_asr_handoff(
            self.audio, self.predecessor_receipt, "initial"
        )
        first = self.root / "first-asr-decision.json"
        decide_asr(handoff, self.audio, None, first, [])

        self.audio.write_bytes(b"synthetic-audio-v2")
        # Source changed: rebuild the valid native predecessor before consuming.
        _, rebuilt_receipt = self._build_predecessor("rebuilt")
        changed_handoff, _ = self._build_asr_handoff(
            self.audio, rebuilt_receipt, "changed"
        )
        third = self.root / "third-asr-decision.json"
        changed = decide_asr(
            changed_handoff, self.audio, first, third, ["12.0-18.0"]
        )
        self.assertEqual(changed["decision"], "rerun_changed_windows_with_context")
        self.assertEqual(changed["context_padding_seconds"], 2)

    # -- negatives -------------------------------------------------------
    def test_unbound_audio_is_rejected(self) -> None:
        handoff, _ = self._build_asr_handoff(
            self.audio, self.predecessor_receipt, "unbound"
        )
        other_audio = self.root / "other-spoken-track.wav"
        other_audio.write_bytes(b"unbound-audio")
        decision_path = self.root / "unbound-decision.json"
        with self.assertRaisesRegex(
            RuntimeError, "audio is not a bound context media artifact"
        ):
            decide_asr(handoff, other_audio, None, decision_path, [])
        self.assertFalse(decision_path.exists())

    def test_unchanged_audio_cannot_claim_changed_window(self) -> None:
        handoff, _ = self._build_asr_handoff(
            self.audio, self.predecessor_receipt, "unchanged-window"
        )
        first = self.root / "first-asr-decision.json"
        decide_asr(handoff, self.audio, None, first, [])
        decision_path = self.root / "unchanged-changed-window-decision.json"
        with self.assertRaisesRegex(
            RuntimeError, "unchanged audio cannot claim changed-window rerun"
        ):
            decide_asr(handoff, self.audio, first, decision_path, ["12.0-18.0"])
        self.assertFalse(decision_path.exists())

    def test_missing_previous_step_receipt_is_rejected(self) -> None:
        error_type = getattr(self.agent, "AgentEntryError", None)
        if error_type is None:
            # The loaded module may expose the class only through function globals.
            error_type = self.agent.prepare_step_handoff.__globals__.get(
                "AgentEntryError"
            )
        if error_type is None:
            self.fail(
                "native AgentEntryError is not reachable; refusing a broad catch"
            )
        context_path = self._write_asr_context(self.audio, "no-predecessor")
        handoff_path = self.root / "no-predecessor-decide-asr-handoff.json"
        with self.assertRaises(error_type) as caught:
            self.agent.prepare_step_handoff(
                self.capsule_path,
                self.step_asr["step_id"],
                context_path,
                self.root / "no-predecessor-decide-asr-context-receipt.json",
                handoff_path,
            )
        self.assertEqual(caught.exception.code, "previous_step_required")
        self.assertEqual(str(caught.exception), "plan-bounded-review")
        self.assertFalse(handoff_path.exists())

    def test_missing_purpose_context_is_rejected(self) -> None:
        step = self.step_asr
        document = self._context_document(
            step, extra_artifacts=[self._derived_audio_artifact(self.audio)]
        )
        document.pop("editorial_context")
        context_path = self.root / "missing-purpose-context.json"
        self.agent.write_or_print(document, context_path)
        handoff_path = self.root / "missing-purpose-decide-asr-handoff.json"
        # Only the purpose block is removed; all other conditions stay valid, so
        # the rejection must be the native context_missing purpose error, never
        # a generic context/manifest message that could false-pass.
        pattern = (
            r"context_missing"
            r"|Supply editorial_context in the existing context manifest"
        )
        try:
            self.agent.prepare_step_handoff(
                self.capsule_path,
                step["step_id"],
                context_path,
                self.root / "missing-purpose-decide-asr-context-receipt.json",
                handoff_path,
                previous_receipt=self.predecessor_receipt,
            )
        except Exception as exc:  # native gate must reject, never silently pass
            self.assertRegex(str(exc), pattern)
        else:
            # If preparation defers the purpose gate, the consumer must reject it.
            decision_path = self.root / "missing-purpose-decision.json"
            with self.assertRaisesRegex(Exception, pattern):
                decide_asr(handoff_path, self.audio, None, decision_path, [])
            self.assertFalse(decision_path.exists())

    def test_context_hash_drift_is_rejected(self) -> None:
        handoff, _ = self._build_asr_handoff(
            self.audio, self.predecessor_receipt, "drift"
        )
        self.proxy_review.write_text(
            '{"status":"tampered-after-handoff"}', encoding="utf-8"
        )
        decision_path = self.root / "drift-decision.json"
        with self.assertRaisesRegex(Exception, "context_receipt_drift"):
            decide_asr(handoff, self.audio, None, decision_path, [])
        self.assertFalse(decision_path.exists())


if __name__ == "__main__":
    unittest.main()
