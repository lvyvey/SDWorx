from sdworx_graph import ExpertiseGraph


def test_find_experts_prefers_relevant_and_authoring_signal():
    graph = ExpertiseGraph()
    graph.add_person("p1", "Alice")
    graph.add_person("p2", "Bob")
    graph.add_document("d1", "GDPR retention policy", topics={"gdpr", "data_protection"})
    graph.add_document("d2", "Payroll tax guidance", topics={"payroll", "tax"})

    graph.link_person_to_document("p1", "d1", expertise=0.9, interactions=25, authored=True)
    graph.link_person_to_document("p2", "d1", expertise=0.6, interactions=8, authored=False)

    experts = graph.find_experts(question="Which GDPR retention rules apply to employee data?", question_topics={"gdpr", "data_protection"}, limit=3)

    assert experts[0]["person_id"] == "p1"
    assert experts[0]["score"] >= experts[1]["score"]


def test_answer_quality_updates_document_score():
    graph = ExpertiseGraph()
    graph.add_person("p1", "Alice")
    graph.add_document("d1", "Work council law", topics={"works_council", "labor_law"})
    graph.link_person_to_document("p1", "d1", expertise=0.5, interactions=3)

    graph.update_score_after_answer("p1", "d1", answer_quality=0.9)

    person = graph.people["p1"]
    assert person.document_scores["d1"] > 0.5
    assert person.document_scores["d1"] > 0.0

    experts = graph.find_experts(question="What should we know about work council obligations?", question_topics={"works_council", "labor_law"}, limit=5)
    assert experts[0]["person_id"] == "p1"
