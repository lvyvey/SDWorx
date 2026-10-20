"""The AskAround knowledge graph, stored as a weighted undirected networkx graph.

Node attributes: `type` (employee / document / customer / expertise), `label`, `obj` (the dataclass).
Edge attributes: `kind` (e.g. "employee-document") and `weight` in [0, 1].
"""

from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import networkx as nx

from . import weights as w
from .models import (
    Customer,
    CustomerInteraction,
    Document,
    DocumentActivity,
    Employee,
    EmployeeContact,
    Expertise,
)
from .text import TfidfIndex

# Weaker links are left out so the graph stays readable.
MIN_DOCUMENT_SIMILARITY = 0.10
MIN_CUSTOMER_SIMILARITY = 0.40


class KnowledgeGraph:
    def __init__(self) -> None:
        self.employees: Dict[str, Employee] = {}
        self.documents: Dict[str, Document] = {}
        self.customers: Dict[str, Customer] = {}
        self.expertise: Dict[str, Expertise] = {}

        self.activities: Dict[Tuple[str, str], DocumentActivity] = {}
        self.contacts: Dict[frozenset, EmployeeContact] = {}
        self.customer_interactions: Dict[Tuple[str, str], CustomerInteraction] = {}

        self.G = nx.Graph()
        self.today = date.today()

    # ------------------------------------------------------------------ nodes and raw signals

    def add_employee(self, e: Employee) -> None:
        self.employees[e.id] = e

    def add_document(self, d: Document) -> None:
        self.documents[d.id] = d

    def add_customer(self, c: Customer) -> None:
        self.customers[c.id] = c

    def add_expertise(self, x: Expertise) -> None:
        self.expertise[x.id] = x

    def add_activity(self, a: DocumentActivity) -> None:
        self._require(a.employee_id, self.employees)
        self._require(a.document_id, self.documents)
        self.activities[(a.employee_id, a.document_id)] = a

    def add_contact(self, c: EmployeeContact) -> None:
        self._require(c.employee_a, self.employees)
        self._require(c.employee_b, self.employees)
        self.contacts[frozenset((c.employee_a, c.employee_b))] = c

    def add_customer_interaction(self, i: CustomerInteraction) -> None:
        self._require(i.employee_id, self.employees)
        self._require(i.customer_id, self.customers)
        self.customer_interactions[(i.employee_id, i.customer_id)] = i

    @staticmethod
    def _require(node_id: str, collection: Dict[str, Any]) -> None:
        if node_id not in collection:
            raise KeyError(f"Unknown node '{node_id}'")

    # ------------------------------------------------------------------ building the networkx graph

    def build(self, today: Optional[date] = None) -> "KnowledgeGraph":
        """(Re)compute all nodes and edge weights from the raw signals."""
        self.today = today or date.today()
        G = self.G = nx.Graph()

        for node_type, collection, label in (
            ("employee", self.employees, lambda n: n.name),
            ("document", self.documents, lambda n: n.title),
            ("customer", self.customers, lambda n: n.company),
            ("expertise", self.expertise, lambda n: n.name),
        ):
            for item in collection.values():
                G.add_node(item.id, type=node_type, label=label(item), obj=item)

        for activity in self.activities.values():
            self._set_employee_document_edge(activity)

        self.text_index = TfidfIndex({d.id: f"{d.title} {d.text}" for d in self.documents.values()})
        for a, b in combinations(self.documents.values(), 2):
            weight = w.document_document_weight(self.text_index.similarity(a.id, b.id), a, b)
            if weight >= MIN_DOCUMENT_SIMILARITY:
                self._link(a.id, b.id, "document-document", weight)

        for a, b in combinations(self.employees.values(), 2):
            weight = w.employee_employee_weight(a, b, self.contacts.get(frozenset((a.id, b.id))))
            if weight > 0:
                self._link(a.id, b.id, "employee-employee", weight)

        for i in self.customer_interactions.values():
            weight = w.employee_customer_weight(i.interaction_dates, self.today)
            if weight > 0:
                self._link(i.employee_id, i.customer_id, "employee-customer", weight)

        for a, b in combinations(self.customers.values(), 2):
            weight = w.customer_customer_weight(a, b)
            if weight >= MIN_CUSTOMER_SIMILARITY:
                self._link(a.id, b.id, "customer-customer", weight)

        for c in self.customers.values():
            for d in self.documents.values():
                weight = w.customer_document_weight(c, d)
                if weight > 0:
                    self._link(c.id, d.id, "customer-document", weight)

        for kind, collection in (("employee-expertise", self.employees), ("document-expertise", self.documents)):
            for item in collection.values():
                for area in item.expertise:
                    if area in self.expertise:
                        self._link(item.id, area, kind, 1.0)
        return self

    def _set_employee_document_edge(self, activity: DocumentActivity) -> None:
        weight = w.employee_document_weight(activity, self.today)
        if weight > 0:
            self._link(activity.employee_id, activity.document_id, "employee-document", weight)
        elif self.G.has_edge(activity.employee_id, activity.document_id):
            self.G.remove_edge(activity.employee_id, activity.document_id)

    def _link(self, a: str, b: str, kind: str, weight: float) -> None:
        self.G.add_edge(a, b, kind=kind, weight=weight)

    # ------------------------------------------------------------------ helpers

    def weight(self, a: str, b: str) -> float:
        return self.G.edges[a, b]["weight"] if self.G.has_edge(a, b) else 0.0

    def nodes_of_type(self, node_type: str):
        return [n for n, t in self.G.nodes(data="type") if t == node_type]

    def label(self, node_id: str) -> str:
        return self.G.nodes[node_id]["label"]

    def record_document_feedback(self, employee_id: str, document_id: str, helpful: bool) -> None:
        """Thumbs up/down on an answer based on this person's link to this document."""
        key = (employee_id, document_id)
        if key not in self.activities:
            self.add_activity(DocumentActivity(employee_id, document_id))
        if helpful:
            self.activities[key].thumbs_up += 1
        else:
            self.activities[key].thumbs_down += 1
        self._set_employee_document_edge(self.activities[key])

    def to_json(self) -> Dict[str, Any]:
        nodes = [{"id": n, "type": d["type"], "label": d["label"]} for n, d in self.G.nodes(data=True)]
        edges = [{"source": a, "target": b, "kind": d["kind"], "weight": d["weight"]}
                 for a, b, d in self.G.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
