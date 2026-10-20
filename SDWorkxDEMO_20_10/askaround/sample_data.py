"""Fictional demo data. Replace with real sources (HR / Entra ID, SharePoint, Teams, CRM)."""

from datetime import date
from typing import List, Tuple

from .graph import KnowledgeGraph
from .models import (
    EXPERTISE_AREAS,
    Customer,
    CustomerInteraction,
    Document,
    DocumentActivity,
    Employee,
    EmployeeContact,
    Expertise,
    StoredAnswer,
)

DEMO_TODAY = date(2026, 10, 20)

d = date


def build_sample() -> Tuple[KnowledgeGraph, List[StoredAnswer]]:
    g = KnowledgeGraph()

    for area in EXPERTISE_AREAS:
        g.add_expertise(Expertise(area, area.replace("_", " ").title()))

    # ---------------------------------------------------------------- employees
    for e in [
        Employee("emp-lotte", "Lotte Peeters", "Senior Legal Advisor", "Master in Law", 12, "BE", "Antwerp", "Legal",
                 {"legal", "data_handling"}),
        Employee("emp-bram", "Bram Janssens", "Payroll Consultant", "Bachelor in Accountancy", 6, "BE", "Antwerp",
                 "Payroll", {"payroll", "tax"}),
        Employee("emp-chloe", "Chloé Dubois", "Social Security Expert", "Master in Economics", 15, "BE", "Brussels",
                 "Social Security", {"social_security", "payroll"}),
        Employee("emp-daan", "Daan de Vries", "Tax Specialist", "Master in Fiscal Law", 8, "NL", "Utrecht", "Payroll",
                 {"tax"}),
        Employee("emp-eva", "Eva Maes", "Data Protection Officer", "Master in IT Law", 4, "BE", "Antwerp", "Legal",
                 {"data_handling", "legal"}),
        Employee("emp-felix", "Felix Wagner", "Account Manager", "Bachelor in Business", 3, "DE", "Munich", "Sales",
                 set()),
        Employee("emp-ines", "Ines Claes", "Commercial Director", "MBA", 18, "BE", "Brussels", "Sales", set()),
    ]:
        g.add_employee(e)

    # ---------------------------------------------------------------- documents
    for doc in [
        Document(
            "doc-gdpr", "GDPR retention policy for employee data",
            "Payslips and payroll records must be kept for five years after the end of the employment. "
            "Employment contracts and work regulations are kept for five years after the employee leaves. "
            "Medical records from the occupational health service are kept for ten years. "
            "After the retention period, personal data must be deleted or anonymised under GDPR.",
            "EU", d(2022, 3, 1), d(2026, 6, 12), {"gdpr", "retention", "data_protection"},
            {"retail", "manufacturing", "healthcare"}, "pdf", {"data_handling", "legal"}),
        Document(
            "doc-dpa", "Customer data processing agreement template",
            "This template describes how SD Worx processes personal data of the customer's employees. "
            "It lists subprocessors, security measures and data breach notification within 48 hours. "
            "The customer remains data controller under GDPR, SD Worx acts as processor.",
            "EU", d(2023, 1, 15), d(2025, 11, 2), {"gdpr", "data_protection", "contracts"},
            {"retail", "manufacturing", "healthcare"}, "docx", {"data_handling", "legal"}),
        Document(
            "doc-be-payroll", "Belgian payroll calculation guide 2026",
            "The gross to net calculation starts from gross salary minus employee social security contributions of 13.07 percent. "
            "Withholding tax is then calculated on the taxable amount using the 2026 scales. "
            "The end-of-year bonus is subject to employee social security contributions and to withholding tax at the exceptional rate. "
            "Holiday pay differs for blue collar and white collar workers.",
            "BE", d(2026, 1, 5), d(2026, 9, 1), {"payroll", "tax", "social_security"},
            {"retail", "manufacturing"}, "pdf", {"payroll", "tax"}),
        Document(
            "doc-onss", "Employer social security contributions (RSZ/ONSS)",
            "Employers pay basic social security contributions of about 25 percent on gross salary. "
            "A first hire reduction lowers employer contributions for the first employee. "
            "Contributions are declared each quarter to RSZ/ONSS through the DmfA declaration.",
            "BE", d(2024, 4, 10), d(2026, 2, 20), {"social_security", "contributions"},
            {"retail", "manufacturing", "healthcare"}, "pdf", {"social_security"}),
        Document(
            "doc-crossborder", "Cross-border workers: tax and social security",
            "Cross-border workers usually pay social security in the country where they work. "
            "An A1 certificate proves which country's social security applies when working in several countries. "
            "Tax residence and the double tax treaty decide where withholding tax is due.",
            "EU", d(2023, 9, 1), d(2026, 5, 30), {"tax", "social_security", "cross_border"},
            {"manufacturing", "logistics"}, "pdf", {"tax", "social_security", "payroll"}),
        Document(
            "doc-nl-30", "Dutch 30% ruling for expats",
            "The 30% ruling lets employers pay 30 percent of the salary of eligible expats tax free. "
            "The expat must be recruited from abroad and earn above the salary norm. "
            "From 2027 the tax-free part is reduced to 27 percent.",
            "NL", d(2024, 2, 1), d(2026, 1, 10), {"tax", "expats"}, {"logistics", "tech"}, "docx", {"tax", "payroll"}),
        Document(
            "doc-dismissal", "Dismissal and notice periods in Belgium",
            "Notice periods depend on seniority and are the same for blue collar and white collar workers. "
            "Dismissal during sick leave is allowed, but the reason must not be the illness itself. "
            "Dismissal of protected employees, such as pregnant workers, requires special justification.",
            "BE", d(2024, 6, 1), d(2026, 3, 3), {"labour_law", "dismissal"},
            {"retail", "manufacturing", "healthcare", "logistics"}, "pdf", {"legal"}),
        Document(
            "doc-pricing", "Service catalogue and price list 2026",
            "Payroll outsourcing is priced per payslip with volume tiers. "
            "Discounts above 500 employees need approval from the commercial director. "
            "Implementation fees depend on the number of countries.",
            "EU", d(2026, 1, 1), d(2026, 1, 1), {"pricing", "sales"},
            {"retail", "manufacturing", "healthcare", "logistics", "tech"}, "xlsx", set()),
    ]:
        g.add_document(doc)

    # ---------------------------------------------------------------- customers
    for c in [
        Customer("cust-fresh", "FreshMart", "BE", "retail", 2500, "mixed"),
        Customer("cust-steel", "SteelWorks", "BE", "manufacturing", 900, "blue"),
        Customer("cust-care", "CareHospital", "BE", "healthcare", 1800, "white"),
        Customer("cust-ship", "ShipFast", "NL", "logistics", 600, "blue"),
        Customer("cust-motor", "MotorTeile", "DE", "manufacturing", 1200, "blue"),
    ]:
        g.add_customer(c)

    # ---------------------------------------------------------------- employee - document activity
    for a in [
        DocumentActivity("emp-lotte", "doc-gdpr", "author", [d(2025, 2, 1), d(2026, 6, 12)], 14, 40, 9, 1),
        DocumentActivity("emp-eva", "doc-gdpr", "contributor", [d(2026, 6, 10)], 3, 25),
        DocumentActivity("emp-eva", "doc-dpa", "author", [d(2025, 11, 2)], 6, 12, 2),
        DocumentActivity("emp-lotte", "doc-dpa", used=8),
        DocumentActivity("emp-bram", "doc-be-payroll", "author", [d(2026, 1, 5), d(2026, 4, 3), d(2026, 9, 1)], 20, 60, 12, 2),
        DocumentActivity("emp-chloe", "doc-be-payroll", "contributor", [d(2026, 3, 15)], used=15),
        DocumentActivity("emp-chloe", "doc-onss", "author", [d(2024, 4, 10), d(2026, 2, 20)], 10, 35, 7),
        DocumentActivity("emp-chloe", "doc-crossborder", "contributor", [d(2025, 8, 1)], used=10),
        DocumentActivity("emp-daan", "doc-crossborder", "author", [d(2023, 9, 1), d(2026, 5, 30)], 8, 22, 4, 1),
        DocumentActivity("emp-daan", "doc-nl-30", "author", [d(2026, 1, 10)], 5, 18),
        DocumentActivity("emp-bram", "doc-crossborder", used=6),
        DocumentActivity("emp-lotte", "doc-dismissal", "author", [d(2024, 6, 1), d(2026, 3, 3)], 9, 30, 5),
        DocumentActivity("emp-eva", "doc-dismissal", used=4),
        DocumentActivity("emp-ines", "doc-pricing", "author", [d(2026, 1, 1)], 12, 25),
        DocumentActivity("emp-felix", "doc-pricing", "contributor", [d(2026, 1, 1)], 15, 40),
    ]:
        g.add_activity(a)

    # ---------------------------------------------------------------- employee - employee contact
    for c in [
        EmployeeContact("emp-lotte", "emp-eva", 120, 20, 2),
        EmployeeContact("emp-bram", "emp-chloe", 60, 8),
        EmployeeContact("emp-bram", "emp-daan", 25, 4, 1),
        EmployeeContact("emp-chloe", "emp-daan", 10, 2),
        EmployeeContact("emp-felix", "emp-bram", 15, 3),
        EmployeeContact("emp-felix", "emp-ines", 80, 15, 3),
        EmployeeContact("emp-felix", "emp-lotte", 5, 1),
    ]:
        g.add_contact(c)

    # ---------------------------------------------------------------- employee - customer interaction
    for employee_id, customer_id, dates in [
        ("emp-bram", "cust-fresh", [d(2026, 9, 1), d(2026, 9, 20), d(2026, 10, 5), d(2026, 10, 15)]),
        ("emp-bram", "cust-steel", [d(2026, 6, 1), d(2026, 8, 12)]),
        ("emp-chloe", "cust-care", [d(2026, 7, 7), d(2026, 9, 9), d(2026, 10, 1)]),
        ("emp-daan", "cust-ship", [d(2026, 5, 5), d(2026, 10, 10)]),
        ("emp-felix", "cust-motor", [d(2026, 8, 8), d(2026, 9, 2), d(2026, 9, 25), d(2026, 10, 14)]),
        ("emp-ines", "cust-fresh", [d(2026, 4, 4), d(2026, 9, 30)]),
        ("emp-lotte", "cust-care", [d(2025, 11, 11)]),
    ]:
        g.add_customer_interaction(CustomerInteraction(employee_id, customer_id, dates))

    # ---------------------------------------------------------------- answer memory
    answers = [
        # Verified recently, confirmed by two colleagues, lots of thumbs up -> HIGH
        StoredAnswer(
            "ans-1", "How long do we have to keep payslips of former employees?",
            "Payslips and payroll records must be kept for five years after the employment ends. "
            "After that, delete or anonymise them under GDPR.",
            answered_by="emp-lotte", source_documents=["doc-gdpr"], confirmed_by={"emp-eva"},
            verified_on=d(2026, 9, 15), helpful=14, not_helpful=1),
        # Only one colleague, and the payroll guide changed after the last verification -> MEDIUM
        StoredAnswer(
            "ans-2", "Is the end-of-year bonus subject to social security contributions?",
            "Yes. The end-of-year bonus is subject to employee social security contributions "
            "and to withholding tax at the exceptional rate.",
            answered_by="emp-bram", source_documents=["doc-be-payroll"],
            verified_on=d(2025, 12, 1), helpful=6, not_helpful=1),
        # Fresh but only one colleague -> MEDIUM
        StoredAnswer(
            "ans-3", "Can an expat get the Dutch 30% ruling?",
            "Yes, if the expat was recruited from abroad and earns above the salary norm. "
            "Note that the tax-free part drops to 27 percent from 2027.",
            answered_by="emp-daan", source_documents=["doc-nl-30"],
            verified_on=d(2026, 8, 1), helpful=5),
    ]

    return g.build(today=DEMO_TODAY), answers
