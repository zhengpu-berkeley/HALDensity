"""Shared CVXPY solver selection and fallback utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import cvxpy as cp


DEFAULT_ACCEPTABLE_STATUSES: frozenset[str] = frozenset({"optimal", "optimal_inaccurate"})


@dataclass(frozen=True)
class CVXPYSolveResult:
    """Result metadata for a successful CVXPY solve attempt."""

    solver_used: str
    status: str
    objective_value: Optional[float]
    attempted_solvers: tuple[str, ...]


def _normalize_solver_name(solver_name: str) -> str:
    normalized = str(solver_name).strip().upper()
    if normalized == "":
        raise ValueError("Solver name must be non-empty")
    return normalized


def build_solver_sequence(
    primary_solver: str,
    *,
    use_secondary_solver: bool,
    solver_waterfall: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Build an ordered, de-duplicated solver candidate sequence."""
    seen: set[str] = set()
    candidates: list[str] = []

    def _append(solver_name: str) -> None:
        normalized = _normalize_solver_name(solver_name)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    _append(primary_solver)
    if use_secondary_solver and solver_waterfall is not None:
        for solver_name in solver_waterfall:
            _append(solver_name)

    return tuple(candidates)


def solve_cvxpy_problem(
    problem: cp.Problem,
    primary_solver: str,
    *,
    use_secondary_solver: bool,
    solver_waterfall: Sequence[str] | None = None,
    build_solve_kwargs: Callable[[str], dict[str, Any]] | None = None,
    before_attempt: Callable[[str], None] | None = None,
    acceptable_statuses: Iterable[str] = DEFAULT_ACCEPTABLE_STATUSES,
) -> CVXPYSolveResult:
    """Solve a CVXPY problem with a primary solver and optional fallback sequence."""
    candidates = build_solver_sequence(
        primary_solver=primary_solver,
        use_secondary_solver=use_secondary_solver,
        solver_waterfall=solver_waterfall,
    )
    if len(candidates) == 0:
        raise ValueError("At least one solver candidate is required")

    accepted = {str(status) for status in acceptable_statuses}
    attempted: list[str] = []
    failure_details: list[str] = []
    last_exception: Exception | None = None

    for solver_name in candidates:
        attempted.append(solver_name)

        if before_attempt is not None:
            before_attempt(solver_name)

        kwargs = {} if build_solve_kwargs is None else dict(build_solve_kwargs(solver_name))
        kwargs["solver"] = solver_name

        try:
            problem.solve(**kwargs)
        except Exception as exc:  # pragma: no cover - solver availability varies by environment
            last_exception = exc
            failure_details.append(f"{solver_name}: {exc}")
            continue

        status = str(problem.status)
        if status in accepted and problem.value is not None:
            return CVXPYSolveResult(
                solver_used=solver_name,
                status=status,
                objective_value=float(problem.value),
                attempted_solvers=tuple(attempted),
            )

        failure_details.append(f"{solver_name}: status={status}, value={problem.value}")

    if len(candidates) == 1 and last_exception is not None:
        raise RuntimeError(f"CVXPY optimization failed: {last_exception}") from last_exception

    attempt_msg = ", ".join(candidates)
    detail_msg = "; ".join(failure_details) if failure_details else "no solver details captured"
    raise RuntimeError(
        f"CVXPY optimization failed with solvers [{attempt_msg}]; details: {detail_msg}"
    ) from last_exception
