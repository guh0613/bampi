import pytest

from bampi.plugins.bampi_chat.feedback import (
    FailureKind,
    assess_failure,
    build_background_failure_message,
    build_reply_failure_message,
    summarize_failure,
)


@pytest.mark.parametrize(
    ("error_text", "expected_kind"),
    [
        ("Agent loop exceeded max_turns=50", FailureKind.TOOL_BUDGET_EXHAUSTED),
        ("Summarization failed: Connection error.", FailureKind.COMPACTION_FAILED),
        ("Turn prefix summarization failed: Unknown error", FailureKind.COMPACTION_FAILED),
        (
            "Error code: 400 - {'error': {'message': 'prompt is too long: 210000 tokens > 200000 maximum'}}",
            FailureKind.CONTEXT_OVERFLOW,
        ),
        (
            "Error code: 400 - {'error': {'code': 'context_length_exceeded'}}",
            FailureKind.CONTEXT_OVERFLOW,
        ),
        (
            "Error code: 400 - {'error': {'message': 'Your credit balance is too low'}}",
            FailureKind.BILLING,
        ),
        (
            "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
            FailureKind.BILLING,
        ),
        (
            "Error code: 401 - {'error': {'type': 'authentication_error'}}",
            FailureKind.AUTH,
        ),
        ("API key not valid. Please pass a valid API key.", FailureKind.AUTH),
        (
            "Error code: 429 - {'error': {'type': 'rate_limit_error'}}",
            FailureKind.RATE_LIMITED,
        ),
        ("429 RESOURCE_EXHAUSTED", FailureKind.RATE_LIMITED),
        (
            "Error code: 529 - {'error': {'type': 'overloaded_error'}}",
            FailureKind.UPSTREAM_UNAVAILABLE,
        ),
        (
            "Error code: 500 - {'error': {'type': 'api_error'}}",
            FailureKind.UPSTREAM_UNAVAILABLE,
        ),
        ("503 Service Unavailable", FailureKind.UPSTREAM_UNAVAILABLE),
        ("Connection error.", FailureKind.NETWORK),
        ("Request timed out.", FailureKind.NETWORK),
        ("something totally unexpected", FailureKind.UNKNOWN),
        ("", FailureKind.UNKNOWN),
        (None, FailureKind.UNKNOWN),
    ],
)
def test_assess_failure_classification(error_text, expected_kind):
    assert assess_failure(error_text).kind is expected_kind


def test_assess_failure_extracts_max_turns():
    assessment = assess_failure("Agent loop exceeded max_turns=50")

    assert assessment.kind is FailureKind.TOOL_BUDGET_EXHAUSTED
    assert assessment.max_turns == 50


def test_reply_failure_message_mentions_tool_budget():
    assessment = assess_failure("Agent loop exceeded max_turns=50")
    message = build_reply_failure_message(assessment)

    assert "工具调用次数达到上限" in message
    assert "50 次" in message
    assert "继续" in message


def test_reply_failure_message_for_rate_limit_suggests_waiting():
    assessment = assess_failure("Error code: 429 - {'error': {'type': 'rate_limit_error'}}")
    message = build_reply_failure_message(assessment)

    assert "限流" in message
    assert "再试" in message


def test_reply_failure_message_unknown_includes_truncated_detail():
    detail = "x" * 200
    message = build_reply_failure_message(assess_failure(detail))

    assert "本次回复失败" in message
    assert "x" * 79 + "…" in message
    assert "x" * 100 not in message


def test_reply_failure_message_unknown_without_detail_stays_generic():
    message = build_reply_failure_message(assess_failure(None))

    assert message == "⚠️ 本次回复失败，请重试；反复失败请联系管理员。"


def test_background_failure_message_includes_reason():
    assessment = assess_failure("Error code: 529 - {'error': {'type': 'overloaded_error'}}")
    message = build_background_failure_message(assessment)

    assert "后台任务已结束" in message
    assert "模型服务暂时不可用" in message
    assert "发送新消息" in message


def test_summarize_failure_uses_detail_for_unknown():
    assessment = assess_failure("weird failure nobody expected")

    assert summarize_failure(assessment) == "weird failure nobody expected"


def test_assess_failure_normalizes_whitespace_in_detail():
    assessment = assess_failure("line one\n   line two")

    assert assessment.detail == "line one line two"
