from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .personal_vlog_runtime import (
    personal_vlog_doctor,
    personal_vlog_prepare,
    personal_vlog_resume,
    personal_vlog_run,
    personal_vlog_status,
)


def prepare_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="personal-vlog-prepare")
    parser.add_argument("event_root")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=2_000)
    _print(
        personal_vlog_prepare(
            **vars(parser.parse_args(argv)),
        )
    )


def run_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="personal-vlog-run")
    parser.add_argument("project_dir")
    parser.add_argument("--presets", default="all", choices=("all", "highlight", "standard", "long"))
    parser.add_argument("--stop-after", default="", choices=("", "inventory", "evidence", "event_plan", "recipes", "handoff_review"))
    _print(personal_vlog_run(**vars(parser.parse_args(argv))))


def resume_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="personal-vlog-resume")
    parser.add_argument("project_dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-after", default="", choices=("", "inventory", "evidence", "event_plan", "recipes", "handoff_review"))
    args = vars(parser.parse_args(argv))
    run_id = str(args.pop("run_id"))
    _print(personal_vlog_resume(run_id=run_id, **args))


def status_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="personal-vlog-status")
    parser.add_argument("project_dir")
    parser.add_argument("--run-id", default="")
    _print(personal_vlog_status(**vars(parser.parse_args(argv))))


def doctor_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="personal-vlog-doctor")
    parser.parse_args(argv)
    _print(personal_vlog_doctor())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="video-knowledge-personal-vlog")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("event_root")
    prepare.add_argument("--project-dir", required=True)
    prepare.add_argument("--max-depth", type=int, default=2)
    prepare.add_argument("--max-items", type=int, default=2_000)
    run = sub.add_parser("run")
    run.add_argument("project_dir")
    run.add_argument("--presets", default="all", choices=("all", "highlight", "standard", "long"))
    run.add_argument("--stop-after", default="", choices=("", "inventory", "evidence", "event_plan", "recipes", "handoff_review"))
    resume = sub.add_parser("resume")
    resume.add_argument("project_dir")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--stop-after", default="", choices=("", "inventory", "evidence", "event_plan", "recipes", "handoff_review"))
    status = sub.add_parser("status")
    status.add_argument("project_dir")
    status.add_argument("--run-id", default="")
    sub.add_parser("doctor")
    args = vars(parser.parse_args(argv))
    command = str(args.pop("command"))
    if command == "prepare":
        result = personal_vlog_prepare(**args)
    elif command == "run":
        result = personal_vlog_run(**args)
    elif command == "resume":
        result = personal_vlog_resume(run_id=str(args.pop("run_id")), **args)
    elif command == "status":
        result = personal_vlog_status(**args)
    else:
        result = personal_vlog_doctor()
    _print(result)


def _print(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
