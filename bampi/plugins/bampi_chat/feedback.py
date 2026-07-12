"""User-facing failure feedback for group replies.

把底层异常与 assistant 的终止信息归类为具体、可执行的群内提示，
避免所有失败都笼统地呈现为“本次回复失败，请重试”。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

THRESHOLD_COMPACTION_NOTICE = "🧹 上下文长度接近上限，正在自动压缩前文，完成后继续。"

_DETAIL_PREVIEW_LIMIT = 80


class FailureKind(StrEnum):
    COMPACTION_FAILED = "compaction_failed"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    CONTEXT_OVERFLOW = "context_overflow"
    BILLING = "billing"
    AUTH = "auth"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureAssessment:
    kind: FailureKind
    max_turns: int | None = None
    detail: str = ""


_MAX_TURNS_PATTERN = re.compile(r"exceeded max_turns=(\d+)")

# 顺序即优先级：先匹配语义更具体的类别，再落到宽泛的 HTTP/网络类别。
# 例如 OpenAI 的 insufficient_quota 也带 429，必须先按额度归类。
_RULES: tuple[tuple[FailureKind, tuple[str, ...]], ...] = (
    (
        FailureKind.COMPACTION_FAILED,
        ("summarization failed",),
    ),
    (
        FailureKind.CONTEXT_OVERFLOW,
        (
            "prompt is too long",
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "input is too long",
            "token count exceeds",
        ),
    ),
    (
        FailureKind.BILLING,
        (
            "credit balance",
            "insufficient_quota",
            "billing",
            "purchase credits",
        ),
    ),
    (
        FailureKind.AUTH,
        (
            "error code: 401",
            "error code: 403",
            "authentication_error",
            "permission_error",
            "invalid x-api-key",
            "api key not valid",
            "unauthorized",
            "permission denied",
        ),
    ),
    (
        FailureKind.RATE_LIMITED,
        (
            "error code: 429",
            "rate_limit_error",
            "rate limit",
            "resource_exhausted",
            "too many requests",
        ),
    ),
    (
        FailureKind.UPSTREAM_UNAVAILABLE,
        (
            "error code: 529",
            "overloaded",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "server_error",
            "api_error",
        ),
    ),
    (
        FailureKind.NETWORK,
        (
            "connection error",
            "connection refused",
            "connection reset",
            "timed out",
            "timeout",
            "ssl",
            "name resolution",
            "network is unreachable",
            "remoteprotocolerror",
        ),
    ),
)

_FAILURE_SUMMARIES: dict[FailureKind, str] = {
    FailureKind.COMPACTION_FAILED: "自动压缩上下文出错",
    FailureKind.TOOL_BUDGET_EXHAUSTED: "工具调用次数达到上限",
    FailureKind.CONTEXT_OVERFLOW: "上下文超出模型长度限制",
    FailureKind.BILLING: "模型账户额度不足",
    FailureKind.AUTH: "模型接口鉴权失败",
    FailureKind.RATE_LIMITED: "模型接口触发限流",
    FailureKind.UPSTREAM_UNAVAILABLE: "模型服务暂时不可用",
    FailureKind.NETWORK: "网络异常或超时",
    FailureKind.UNKNOWN: "未知错误",
}


def assess_failure(error_text: str | None) -> FailureAssessment:
    normalized = " ".join((error_text or "").split())
    if not normalized:
        return FailureAssessment(kind=FailureKind.UNKNOWN)
    lowered = normalized.lower()

    match = _MAX_TURNS_PATTERN.search(lowered)
    if match is not None:
        return FailureAssessment(
            kind=FailureKind.TOOL_BUDGET_EXHAUSTED,
            max_turns=int(match.group(1)),
            detail=normalized,
        )

    for kind, needles in _RULES:
        if any(needle in lowered for needle in needles):
            return FailureAssessment(kind=kind, detail=normalized)
    return FailureAssessment(kind=FailureKind.UNKNOWN, detail=normalized)


def summarize_failure(assessment: FailureAssessment) -> str:
    if assessment.kind is FailureKind.UNKNOWN and assessment.detail:
        return _preview_detail(assessment.detail)
    return _FAILURE_SUMMARIES[assessment.kind]


def build_reply_failure_message(assessment: FailureAssessment) -> str:
    kind = assessment.kind
    if kind is FailureKind.TOOL_BUDGET_EXHAUSTED:
        limit_text = f"（{assessment.max_turns} 次）" if assessment.max_turns else ""
        return f"⚠️ 工具调用次数达到上限{limit_text}，如需继续请发送“继续”。"
    if kind is FailureKind.COMPACTION_FAILED:
        return (
            "⚠️ 自动压缩上下文时出错。"
            "请再试一次；若反复失败，可使用 /clear 清空上下文。"
        )
    if kind is FailureKind.CONTEXT_OVERFLOW:
        return (
            "⚠️ 上下文超出了模型长度限制。"
            "可以用 /compact 压缩上下文，或用 /clear 清空后再试。"
        )
    if kind is FailureKind.BILLING:
        return "⚠️ 模型账户额度不足，这次请求没有成功，请联系管理员处理。"
    if kind is FailureKind.AUTH:
        return "⚠️ 模型接口鉴权失败，请联系管理员检查 API Key 配置。"
    if kind is FailureKind.RATE_LIMITED:
        return "⚠️ 模型接口触发限流，这次请求没有成功，请过一两分钟再试。"
    if kind is FailureKind.UPSTREAM_UNAVAILABLE:
        return "⚠️ 模型服务暂时过载或故障，这次请求没有成功，请稍后再试。"
    if kind is FailureKind.NETWORK:
        return "⚠️ 连接模型服务失败（网络异常或超时），请稍后再试。"
    if assessment.detail:
        return (
            f"⚠️ 本次回复失败：{_preview_detail(assessment.detail)}。"
            "可以重试一次，反复失败请联系管理员。"
        )
    return "⚠️ 本次回复失败，请重试；反复失败请联系管理员。"


def build_background_failure_message(assessment: FailureAssessment) -> str:
    return (
        f"⚠️ 后台任务已结束，但后续回复失败（{summarize_failure(assessment)}）。"
        "可以发送新消息继续。"
    )


def _preview_detail(detail: str) -> str:
    if len(detail) <= _DETAIL_PREVIEW_LIMIT:
        return detail
    return detail[: _DETAIL_PREVIEW_LIMIT - 1] + "…"
