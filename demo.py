from sdworx_graph import ExpertiseGraph


def main() -> None:
    graph = ExpertiseGraph()

    graph.add_person("emp-001", "Alice", team="Data Protection")
    graph.add_person("emp-002", "Bob", team="Payroll")
    graph.add_person("emp-003", "Charlie", team="Labor Law")

    graph.add_document("doc-001", "GDPR retention policy", topics={"gdpr", "data_protection", "retention"})
    graph.add_document("doc-002", "Cross-border payroll tax guide", topics={"payroll", "tax", "compliance"})
    graph.add_document("doc-003", "Works council obligations", topics={"works_council", "labor_law", "employment"})

    graph.link_person_to_document("emp-001", "doc-001", expertise=0.90, interactions=28, authored=True)
    graph.link_person_to_document("emp-002", "doc-002", expertise=0.82, interactions=44, authored=True)
    graph.link_person_to_document("emp-003", "doc-003", expertise=0.88, interactions=19, authored=False)

    experts = graph.find_experts(
        question="Which GDPR retention rules apply to employee data for EU countries?",
        question_topics={"gdpr", "data_protection", "retention"},
        limit=5,
    )
    print("Top experts for GDPR question:")
    for item in experts:
        print(f"- {item['person_name']} ({item['person_id']}): score={item['score']}")

    graph.update_score_after_answer("emp-001", "doc-001", answer_quality=0.95)
    print("\nAlice's updated score on the GDPR document:")
    print(graph.people["emp-001"].document_scores["doc-001"])


if __name__ == "__main__":
    main()
