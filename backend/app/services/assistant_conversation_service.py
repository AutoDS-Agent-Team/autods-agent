"""Persist bounded Ask AutoDS context without treating prior answers as truth."""
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.assistant_turn import AssistantTurn


_FOLLOW_UP = re.compile(r"^\s*(?:what about|how about|and\b|now\b|only\b|what if|for those|among those|why did you|why was that)", re.I)


def recent_turns(database: Session, user_id: str, conversation_id: str, limit: int) -> list[AssistantTurn]:
    rows = database.scalars(
        select(AssistantTurn)
        .where(AssistantTurn.user_id == user_id, AssistantTurn.conversation_id == conversation_id)
        .order_by(AssistantTurn.created_at.desc())
        .limit(max(1, min(limit, 20)))
    ).all()
    return list(reversed(rows))


def resolve_follow_up(question: str, turns: list[AssistantTurn]) -> tuple[str, bool]:
    """Restore intent only; callers must recalculate all dataset numbers."""
    if not turns:
        return question, False
    previous = turns[-1]
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    prior_columns = {re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip() for column in previous.referenced_columns or []}
    explicit_new_intent = re.search(r"\b(?:compare|comparing|versus|vs|between|highest|lowest|top|bottom)\b", normalized)
    short_contextual = len(normalized.split()) <= 6 and not explicit_new_intent and not any(column and column in normalized for column in prior_columns)
    if not _FOLLOW_UP.search(question) and not short_contextual:
        return question, False
    return f"{previous.resolved_question.rstrip(' ?')}. Additional follow-up constraint: {question}", True


def remembered_context(turns: list[AssistantTurn]) -> tuple[str | None, str | None]:
    for turn in reversed(turns):
        if turn.dataset_id or turn.experiment_id:
            return turn.dataset_id, turn.experiment_id
    return None, None


def save_turn(
    database: Session,
    *,
    user_id: str,
    conversation_id: str,
    dataset_id: str | None,
    experiment_id: str | None,
    question: str,
    resolved_question: str,
    answer: str,
    evidence_type: str,
    referenced_columns: list[str] | None = None,
    analytical_topic: str | None = None,
    source_types: list[str] | None = None,
) -> AssistantTurn:
    turn = AssistantTurn(
        user_id=user_id,
        conversation_id=conversation_id,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        question=question,
        resolved_question=resolved_question,
        answer=answer[:12000],
        evidence_type=evidence_type,
        referenced_columns=list(dict.fromkeys(referenced_columns or []))[:20],
        analytical_topic=(analytical_topic or "")[:255] or None,
        source_types=list(dict.fromkeys(source_types or []))[:10],
    )
    database.add(turn)
    database.commit()
    return turn


def conversation_context(turns: list[AssistantTurn], max_chars: int) -> str:
    """Provide recent topics—not trusted numeric evidence—to an intent model."""
    lines = ["RECENT CONVERSATION CONTEXT (intent only; recalculate all numbers):"]
    for turn in turns[-8:]:
        columns = ", ".join(turn.referenced_columns or []) or "none"
        lines.append(f"User topic: {turn.question[:300]} | evidence={turn.evidence_type} | columns={columns}")
    return "\n".join(lines)[:max_chars]
