from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .storage import read_json, write_json
from .time_utils import utc_now_iso_seconds

SCHEMA = "video_knowledge_pipeline.sqlite_vec_human_gold_evaluation.v1"
REVIEW_SCHEMA = "vkp.sqlite_vec_human_gold_review.v1"


def _stable_ids(rows: list[dict[str, Any]], limit: int) -> list[str]:
    return [
        str(row.get("stable_id") or "")
        for row in rows[:limit]
        if str(row.get("stable_id") or "").strip()
    ]


def _recall(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def evaluate_retrieval_goldset(
    review_path: str | Path,
    source_report_path: str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    review_file = Path(review_path).expanduser().resolve()
    report_file = Path(source_report_path).expanduser().resolve()
    review = read_json(review_file)
    report = read_json(report_file)
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"review schema must be {REVIEW_SCHEMA}")
    if not isinstance(report, dict) or report.get("schema") != "vkp.sqlite_vec_benchmark.v1":
        raise ValueError("source report schema must be vkp.sqlite_vec_benchmark.v1")

    report_queries = {
        str(row.get("query_id") or ""): row
        for row in report.get("queries", [])
        if isinstance(row, dict)
    }
    reviewed_rows: list[dict[str, Any]] = []
    pending: list[str] = []
    for row in review.get("queries", []):
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id") or "").strip()
        if not query_id or query_id not in report_queries:
            raise ValueError(f"review query is absent from source report: {query_id or '<empty>'}")
        review_status = str(row.get("review_status") or "").strip()
        if review_status != "human_confirmed":
            pending.append(query_id)
            continue
        relevant = {
            str(value).strip()
            for value in row.get("relevant_stable_ids", [])
            if str(value).strip()
        }
        if not relevant:
            raise ValueError(f"human-confirmed query has no relevant_stable_ids: {query_id}")
        source = report_queries[query_id]
        top_10 = [
            item
            for item in source.get("top_10", [])
            if isinstance(item, dict)
        ]
        retrieved_5 = _stable_ids(top_10, 5)
        retrieved_10 = _stable_ids(top_10, 10)
        reviewed_rows.append(
            {
                "query_id": query_id,
                "bundle_id": str(row.get("bundle_id") or source.get("bundle_id") or ""),
                "relevant_stable_ids": sorted(relevant),
                "recall_at_5": _recall(relevant, retrieved_5),
                "recall_at_10": _recall(relevant, retrieved_10),
                "review_notes": str(row.get("review_notes") or ""),
            }
        )

    if require_complete and pending:
        raise ValueError(f"human review is incomplete: {len(pending)} pending queries")
    count = len(reviewed_rows)
    identity = {
        "schema": SCHEMA,
        "source_report_sha256": canonical_json_sha256(report),
        "review_sha256": canonical_json_sha256(review),
        "queries": reviewed_rows,
    }
    status = "completed" if count and not pending else "partial" if count else "blocked_human_review"
    return {
        **identity,
        "generated_at": utc_now_iso_seconds(),
        "status": status,
        "evidence_status": (
            "human_gold"
            if status == "completed"
            else "partial_human_review"
            if status == "partial"
            else "blocked_human_review"
        ),
        "aggregate": {
            "reviewed_query_count": count,
            "pending_query_count": len(pending),
            "recall_at_5": (
                sum(row["recall_at_5"] for row in reviewed_rows) / count
                if count
                else None
            ),
            "recall_at_10": (
                sum(row["recall_at_10"] for row in reviewed_rows) / count
                if count
                else None
            ),
        },
        "pending_query_ids": pending,
        "evaluation_sha256": canonical_json_sha256(identity),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate sqlite-vec Recall@5/10 from explicit human relevance labels."
    )
    parser.add_argument("review_path")
    parser.add_argument("source_report_path")
    parser.add_argument("--output")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_retrieval_goldset(
        args.review_path,
        args.source_report_path,
        require_complete=args.require_complete,
    )
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked_human_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
