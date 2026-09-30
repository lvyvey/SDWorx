from __future__ import annotations

import os

from flask import Flask, render_template, request

from sdworx_graph import ExpertiseGraph

app = Flask(__name__)


def build_demo_graph() -> ExpertiseGraph:
    graph = ExpertiseGraph()

    graph.add_person("emp-001", "Alice", team="Data Protection")
    graph.add_person("emp-002", "Bob", team="Payroll")
    graph.add_person("emp-003", "Charlie", team="Labor Law")
    graph.add_person("emp-004", "Diana", team="Customer Compliance")

    graph.add_document("doc-001", "GDPR retention policy", topics={"gdpr", "data_protection", "retention"})
    graph.add_document("doc-002", "Cross-border payroll tax guide", topics={"payroll", "tax", "compliance"})
    graph.add_document("doc-003", "Works council obligations", topics={"works_council", "labor_law", "employment"})
    graph.add_document("doc-004", "Customer confidentiality framework", topics={"customer", "confidentiality", "data_security"})

    graph.link_person_to_document("emp-001", "doc-001", expertise=0.90, interactions=28, authored=True)
    graph.link_person_to_document("emp-002", "doc-002", expertise=0.82, interactions=44, authored=True)
    graph.link_person_to_document("emp-003", "doc-003", expertise=0.88, interactions=19, authored=False)
    graph.link_person_to_document("emp-004", "doc-004", expertise=0.75, interactions=15, authored=True)
    graph.link_person_to_document("emp-001", "doc-004", expertise=0.62, interactions=12, authored=False)
    graph.link_person_to_document("emp-004", "doc-001", expertise=0.58, interactions=7, authored=False)

    return graph


graph = build_demo_graph()


@app.route("/", methods=["GET", "POST"])
def index():
    question = "Which GDPR retention rules apply to employee data for EU countries?"
    topics = "gdpr, data_protection, retention"
    experts = []
    update_message = None
    update_score = None

    if request.method == "POST":
        if "question" in request.form:
            question = request.form.get("question", "")
            topics_input = request.form.get("topics", "")
            topics = topics_input.strip()
            topic_list = [value.strip() for value in topics.split(",") if value.strip()]
            experts = graph.find_experts(question=question, question_topics=topic_list, limit=5)

        if "update_person" in request.form:
            person_id = request.form.get("person_id", "")
            document_id = request.form.get("document_id", "")
            answer_quality = float(request.form.get("answer_quality", "0.0"))
            update_score = graph.update_score_after_answer(person_id, document_id, answer_quality)
            update_message = f"Updated {person_id} for {document_id}: score is now {update_score:.4f}"

    return render_template(
        "index.html",
        question=question,
        topics=topics,
        experts=experts,
        people=graph.people,
        documents=graph.documents,
        update_message=update_message,
        update_score=update_score,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
