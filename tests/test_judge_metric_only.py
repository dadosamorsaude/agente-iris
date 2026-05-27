import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import evaluator
from app.agent import iris_orchestrator as orchestrator


class DummyHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message):
        self.messages.append(message)

    def add_ai_message(self, message):
        self.messages.append(message)


@pytest.mark.asyncio
async def test_run_iris_agent_delivers_before_judge(monkeypatch):
    created_tasks = []
    post_results = []

    async def fake_post_execution(*args):
        post_results.append(args[4])

    async def fake_synthesize(*args, **kwargs):
        return {
            "final_answer": "resposta original",
            "analysis_type": "relatorio",
            "periodo": {"inicio": None, "fim_exclusivo": None},
            "unit_of_analysis": "id_atendimento",
            "error": False,
            "errorType": None,
            "errorMessage": None,
            "rag_used": True,
            "sql_used": True,
            "user_asked_for_sql": False,
        }

    def capture_task(coro):
        created_tasks.append(coro)
        return coro

    monkeypatch.setattr(orchestrator, "detect_intent", lambda _: {"is_simple": False, "output_mode": "summary"})
    monkeypatch.setattr(orchestrator, "load_curated_lessons", lambda: "{}")
    monkeypatch.setattr(orchestrator, "get_session_history", lambda _: DummyHistory())
    monkeypatch.setattr(orchestrator, "_run_clinical_rag_in_thread", lambda _: ("rag context", [{"source": "rag"}]))
    monkeypatch.setattr(
        orchestrator,
        "_run_sql_analyst_in_thread",
        lambda *args: (
            {
                "execution_status": "success",
                "summary": {"total_registros": 1},
                "rows": [],
                "row_count": 1,
            },
            [{"sql": "select 1", "results": [{"total_registros": 1}]}],
        ),
    )
    monkeypatch.setattr(orchestrator, "_iris_synthesize", fake_synthesize)
    monkeypatch.setattr(orchestrator, "_post_execution", fake_post_execution)
    monkeypatch.setattr(orchestrator.asyncio, "create_task", capture_task)

    chunks = []
    async for chunk in orchestrator.run_iris_agent("user-1", "gere um relatorio", stream=False):
        chunks.append(chunk)

    assert "".join(chunks) == "resposta original"
    assert len(created_tasks) == 1

    await created_tasks[0]
    assert post_results[0]["final_answer"] == "resposta original"
    assert post_results[0]["final_delivery_policy"] == "delivered_without_blocking"
    assert "judge_output" not in post_results[0]


@pytest.mark.asyncio
async def test_post_execution_attaches_judge_metrics_for_learning(monkeypatch):
    result = {
        "final_answer": "resposta entregue",
        "error": False,
        "rag_used": True,
        "sql_used": True,
    }
    seen_by_learning = {}
    seen_by_log = {}

    async def fake_evaluate_response(**kwargs):
        return {
            "judge_passed": False,
            "overall_score": 0.2,
            "should_block_callback": True,
            "block_reason": "hallucinated_data",
            "issues": [{"tipo": "hallucinated_data"}],
            "metric_only": True,
            "metric_available": True,
        }

    def fake_generate_lessons(execution_data):
        seen_by_learning.update(execution_data)
        return []

    async def fake_save_execution_log(**kwargs):
        seen_by_log.update(kwargs["result"])

    monkeypatch.setattr(orchestrator, "evaluate_response", fake_evaluate_response)
    monkeypatch.setattr(orchestrator, "generate_lessons_from_execution", fake_generate_lessons)
    monkeypatch.setattr(orchestrator, "save_execution_log", fake_save_execution_log)

    await orchestrator._post_execution(
        job_id="job-1",
        session_id="session-1",
        conversation_id="conversation-1",
        original_input="pergunta",
        result=result,
        raw_athena=[{"sql": "select 1", "results": []}],
        raw_rag=[{"source": "rag"}],
    )

    assert result["judge_score"] == 0.2
    assert result["judge_passed"] is False
    assert result["final_delivery_policy"] == "delivered_without_blocking"
    assert seen_by_learning["judge_output"]["metric_only"] is True
    assert seen_by_log["judge_output"]["block_reason"] == "hallucinated_data"


@pytest.mark.asyncio
async def test_evaluator_parse_error_is_unavailable_metric(monkeypatch):
    class BadJudge:
        async def ainvoke(self, messages):
            class Response:
                content = "isso nao e json"

            return Response()

    monkeypatch.setattr(evaluator, "_get_evaluator_llm", lambda: BadJudge())

    metric = await evaluator.evaluate_response(
        user_question="pergunta",
        agent_response="resposta",
        raw_athena_data=[{"sql": "select 1", "results": []}],
        rag_context=[{"source": "rag", "query": "q", "chunks": ["chunk"]}],
    )

    assert metric["judge_passed"] is None
    assert metric["should_block_callback"] is False
    assert metric["metric_available"] is False
    assert metric["block_reason"] == "judge_unavailable"
