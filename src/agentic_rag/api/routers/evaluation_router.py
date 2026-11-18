"""Read-only access to past evaluation runs stored in the Postgres ledger.

For the full statistical analysis (nDCG/Recall against qrels, bootstrap CIs),
see `agentic_rag.reporting.ReportGenerator` (rendered via `agentic-rag report
<run-id>` into `report/results.md`); this router exposes a lighter-weight
live summary useful for spot-checking a run without regenerating the report.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.api.dependencies import get_session
from agentic_rag.storage.postgres.models import EvaluationRunRow, PipelineRunRecordRow

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/runs")
def list_runs(session: Session = Depends(get_session)) -> list[dict]:
    """List every evaluation run recorded in the ledger, most recent first."""
    rows = session.execute(
        select(EvaluationRunRow).order_by(EvaluationRunRow.started_at.desc())
    ).scalars().all()
    return [
        {
            "run_id": row.run_id,
            "started_at": row.started_at.isoformat(),
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }
        for row in rows
    ]


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str, session: Session = Depends(get_session)) -> dict:
    """Aggregate mean answer score and retrieval latency per (dataset, strategy)."""
    rows = session.execute(
        select(PipelineRunRecordRow).where(PipelineRunRecordRow.run_id == run_id)
    ).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No records found for run '{run_id}'")

    rows_by_dataset_strategy: dict[tuple[str, str], list[PipelineRunRecordRow]] = {}
    for row in rows:
        rows_by_dataset_strategy.setdefault((row.dataset, row.strategy), []).append(row)

    summary = [
        {
            "dataset": dataset,
            "strategy": strategy,
            "num_queries": len(group_rows),
            "mean_answer_score": _mean(
                r.metrics["answer_score"] for r in group_rows if "answer_score" in r.metrics
            ),
            "mean_retrieval_latency_seconds": _mean(
                r.retrieval_latency_seconds for r in group_rows
            ),
        }
        for (dataset, strategy), group_rows in rows_by_dataset_strategy.items()
    ]
    return {"run_id": run_id, "summary": summary}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
