from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class DocumentLink:
    expertise: float = 0.0
    interaction_count: int = 0
    authored: bool = False
    response_quality_sum: float = 0.0
    response_count: int = 0

    @property
    def quality_average(self) -> float:
        if self.response_count == 0:
            return 0.0
        return self.response_quality_sum / self.response_count


@dataclass
class Person:
    id: str
    name: str
    team: str = ""
    document_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class Document:
    id: str
    title: str
    topics: Set[str] = field(default_factory=set)
    created_by: Optional[str] = None


class ExpertiseGraph:
    def __init__(self) -> None:
        self.people: Dict[str, Person] = {}
        self.documents: Dict[str, Document] = {}
        self.person_document_links: Dict[str, Dict[str, DocumentLink]] = {}

    def add_person(self, person_id: str, name: str, team: str = "") -> Person:
        person = Person(id=person_id, name=name, team=team)
        self.people[person_id] = person
        self.person_document_links.setdefault(person_id, {})
        return person

    def add_document(self, document_id: str, title: str, topics: Optional[Iterable[str]] = None) -> Document:
        document = Document(id=document_id, title=title, topics=set(topics or set()))
        self.documents[document_id] = document
        return document

    def link_person_to_document(
        self,
        person_id: str,
        document_id: str,
        expertise: float = 0.0,
        interactions: int = 0,
        authored: bool = False,
    ) -> DocumentLink:
        if person_id not in self.people:
            raise KeyError(f"Unknown person '{person_id}'")
        if document_id not in self.documents:
            raise KeyError(f"Unknown document '{document_id}'")

        link = self.person_document_links.setdefault(person_id, {}).setdefault(document_id, DocumentLink())
        link.expertise = max(link.expertise, max(0.0, min(1.0, float(expertise))))
        link.interaction_count += max(0, int(interactions))
        link.authored = link.authored or authored
        if authored:
            self.documents[document_id].created_by = person_id

        self.people[person_id].document_scores[document_id] = self._compute_document_score(person_id, document_id)
        return link

    def update_score_after_answer(self, person_id: str, document_id: str, answer_quality: float) -> float:
        if person_id not in self.people:
            raise KeyError(f"Unknown person '{person_id}'")
        if document_id not in self.documents:
            raise KeyError(f"Unknown document '{document_id}'")

        quality = max(0.0, min(1.0, float(answer_quality)))
        link = self.person_document_links.setdefault(person_id, {}).setdefault(document_id, DocumentLink())
        link.response_quality_sum += quality
        link.response_count += 1
        link.expertise = min(1.0, link.expertise + (quality * 0.15))

        self.people[person_id].document_scores[document_id] = self._compute_document_score(person_id, document_id)
        return self.people[person_id].document_scores[document_id]

    def _compute_document_score(self, person_id: str, document_id: str) -> float:
        document = self.documents[document_id]
        link = self.person_document_links.get(person_id, {}).get(document_id, DocumentLink())

        interaction_signal = min(1.0, math.log(link.interaction_count + 1) / math.log(50 + 1))
        authored_signal = 0.25 if link.authored or document.created_by == person_id else 0.0
        quality_signal = link.quality_average

        score = (
            (link.expertise * 0.45)
            + (interaction_signal * 0.20)
            + (authored_signal * 0.15)
            + (quality_signal * 0.20)
        )
        return round(min(1.0, score), 6)

    def _document_relevance_score(self, document: Document, question: str, question_topics: Optional[Set[str]] = None) -> float:
        question_text = (question or "").lower()
        topic_hits = 0
        if question_topics:
            normalized_topics = {item.lower().strip() for item in question_topics if item and item.strip()}
            topic_hits = len(normalized_topics & {topic.lower().strip() for topic in document.topics})

        if not question_text and not question_topics:
            return 0.0

        title_score = 0.0
        for term in (question_topics or set()):
            normalized_term = term.lower().strip()
            if normalized_term and normalized_term in {item.lower() for item in document.topics}:
                title_score += 1.0
            if normalized_term and normalized_term in document.title.lower():
                title_score += 0.5

        if question_text:
            question_terms = {
                term for term in question_text.replace("?", "").replace("-", " ").split()
                if len(term) > 3
            }
            overlap = len(question_terms & {term.lower() for term in document.title.lower().split()})
            title_score += overlap * 0.2

        return min(1.0, (topic_hits * 0.6) + (title_score * 0.4))

    def find_experts(
        self,
        question: str,
        question_topics: Optional[Iterable[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        topics = set(question_topics or [])
        scores: Dict[str, float] = {}
        top_documents_by_person: Dict[str, List[str]] = {}

        for document_id, document in self.documents.items():
            relevance = self._document_relevance_score(document, question, topics)
            if relevance <= 0:
                continue

            for person_id, links in self.person_document_links.items():
                if document_id not in links:
                    continue

                link = links[document_id]
                document_score = self._compute_document_score(person_id, document_id)
                authored_boost = 0.25 if link.authored or document.created_by == person_id else 0.0
                candidate_score = (
                    (relevance * 1.4)
                    + (document_score * 1.1)
                    + (link.expertise * 0.8)
                    + (authored_boost * 0.6)
                    + (min(1.0, link.interaction_count / 25) * 0.5)
                )

                scores[person_id] = max(scores.get(person_id, 0.0), candidate_score)
                top_documents_by_person.setdefault(person_id, []).append(document_id)

        ranked = []
        for person_id, person in self.people.items():
            if person_id not in scores:
                continue
            ranked.append(
                {
                    "person_id": person_id,
                    "person_name": person.name,
                    "score": round(scores[person_id], 4),
                    "documents": top_documents_by_person.get(person_id, []),
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]
