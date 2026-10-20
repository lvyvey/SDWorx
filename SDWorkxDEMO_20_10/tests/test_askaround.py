import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import pytest  # noqa: E402

from askaround import create_demo  # noqa: E402

PAYSLIPS = "How long must we keep payslips after an employee leaves?"
BONUS = "Is the end-of-year bonus subject to social security contributions?"
FIRST_HIRE = "What are the employer social security contributions for a first hire?"
DISMISSAL = "Can we dismiss an employee who is on sick leave?"
PRICING = "What price do we charge a retail customer with 500 employees for payroll outsourcing?"


@pytest.fixture
def engine():
    return create_demo()


def test_graph_is_a_weighted_networkx_graph_with_four_node_types(engine):
    G = engine.graph.G
    assert isinstance(G, nx.Graph) and not G.is_directed()
    assert {t for _, t in G.nodes(data="type")} == {"employee", "document", "customer", "expertise"}
    assert all(0 < d["weight"] <= 1 for _, _, d in G.edges(data=True))


def test_answered_and_verified_question_is_high(engine):
    r = engine.ask(PAYSLIPS)
    assert r["level"] == "high"
    assert r["answer"]["verified_on"] == "2026-09-15"
    assert any("<mark>" in p["html"] for p in r["sources"][0]["passages"])


def test_answer_with_changed_source_and_one_colleague_is_medium(engine):
    r = engine.ask(BONUS)
    assert r["level"] == "medium"
    assert any("Source changed" in reason for reason in r["reasons"])
    assert r["message"]["text"].startswith("Hi ")


def test_unanswered_question_with_good_document_is_medium_with_draft(engine):
    r = engine.ask(FIRST_HIRE)
    assert r["level"] == "medium"
    assert r["answer"]["draft_from_document"]
    assert r["experts"][0]["employee_id"] == "emp-chloe"


@pytest.mark.parametrize("question, expected_expert", [(DISMISSAL, "emp-lotte"), (PRICING, "emp-ines")])
def test_high_risk_without_verified_answer_routes_to_people(engine, question, expected_expert):
    r = engine.ask(question)
    assert r["level"] == "low"
    assert r["risk_topics"]
    assert r["experts"][0]["employee_id"] == expected_expert


def test_unknown_topic_is_low_without_experts(engine):
    r = engine.ask("Where is the coffee machine?")
    assert r["level"] == "low" and r["experts"] == []


def test_expert_verification_upgrades_medium_to_high(engine):
    first = engine.ask(BONUS)
    engine.verify(BONUS, first["answer"]["text"], first["experts"][0]["employee_id"],
                  [s["id"] for s in first["sources"]], correct=True, answer_id=first["answer"]["id"])
    assert engine.ask(BONUS)["level"] == "high"


def test_saved_expert_answer_is_reused_next_time(engine):
    assert engine.ask(DISMISSAL)["not_answered_before"]
    engine.save_answer(DISMISSAL, "Yes, as long as the illness is not the reason.", "emp-lotte", ["doc-dismissal"])
    r = engine.ask("Is it allowed to dismiss someone during sick leave?")
    assert not r["not_answered_before"]
    assert r["level"] in {"medium", "high"}


def test_feedback_nudges_expert_scores(engine):
    before = engine.ask(FIRST_HIRE)["experts"][0]
    for _ in range(3):
        engine.rate_answer("ans-2", helpful=False)     # Bram and the payroll guide get thumbs down
    assert engine.expertise_feedback[("emp-bram", "payroll")] < 1.0
    assert engine.ask(FIRST_HIRE)["experts"][0]["employee_id"] == before["employee_id"]


def test_pagerank_restart_probability_is_15_percent():
    from askaround.engine import PAGERANK_ALPHA
    assert PAGERANK_ALPHA == pytest.approx(0.85)


def test_flask_api_round_trip():
    import app as server

    client = server.app.test_client()
    assert client.get("/").status_code == 200
    r = client.post("/api/ask", json={"question": PAYSLIPS}).get_json()
    assert r["level"] == "high"
    assert client.post("/api/rate", json={"answer_id": r["answer"]["id"], "helpful": True}).status_code == 200
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/reset").get_json() == {"ok": True}
