"""AskAround: turns a question into a high / medium / low confidence response.

1. The question is embedded (TF-IDF; swap for text embeddings later) and compared with every document.
2. Personalized PageRank starts from the closest documents (weighted by similarity) and the matching
   expertise nodes, follows the weighted edges and jumps back to the start 15 % of the time (alpha=0.85).
3. That gives a confidence score for every person and document. Together with the answer memory
   (earlier answers, their feedback and verification date) this picks one of three levels:

   HIGH    reuse an earlier answer, with sources, highlighted passages and the last verification date
   MEDIUM  give the answer / document plus the matching expert and a prepared "please verify" message
   LOW     no answer: route to the best matching experts (always for high-risk topics without a verified answer)
"""

from __future__ import annotations

import html
import itertools
import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .graph import KnowledgeGraph
from .models import StoredAnswer
from .text import TfidfIndex, cosine, tokenize

# --------------------------------------------------------------------------- tuning knobs

PAGERANK_ALPHA = 0.85            # 15 % chance per step to jump back to the question's starting points
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40
MIN_ANSWER_MATCH = 0.35          # question similarity needed before an earlier answer is considered
FULL_ANSWER_MATCH = 0.60         # similarity that counts as "the same question"
FULL_RETRIEVAL = 0.30            # document similarity that counts as a strong match
MAX_VERIFICATION_AGE_DAYS = 365
FEEDBACK_STEP = 0.05             # how much one thumbs up/down nudges a person's score in an expertise area

# How easily the walk follows each kind of edge (multiplied with the edge weight). Knowledge links
# carry trust best; customer links mostly add context, so they get less of the flow.
WALK_FACTOR = {
    "employee-document": 1.0,
    "document-document": 0.6,
    "employee-expertise": 0.6,
    "document-expertise": 0.6,
    "employee-employee": 0.3,
    "employee-customer": 0.2,
    "customer-document": 0.1,
    "customer-customer": 0.1,
}

EXPERTISE_KEYWORDS = {
    "payroll": {"payroll", "payslip", "salary", "wage", "bonus", "gross", "net", "holiday", "pay"},
    "legal": {"legal", "law", "contract", "dismissal", "dismiss", "notice", "clause", "labour", "court", "fire"},
    "social_security": {"social", "security", "rsz", "onss", "contribution", "a1", "pension"},
    "tax": {"tax", "withholding", "fiscal", "ruling", "vat", "deduction", "expat"},
    "data_handling": {"gdpr", "data", "privacy", "retention", "personal", "keep", "delete", "processing"},
}

# Topics where a wrong AI answer is costly: route to a person unless a verified answer exists.
RISK_KEYWORDS = {
    "legal": {"dismissal", "dismiss", "fire", "lawsuit", "court", "liability", "legal", "contract"},
    "compliance": {"compliance", "compliant", "audit", "fine", "penalty", "breach", "sanction"},
    "pricing": {"price", "pricing", "cost", "fee", "tariff", "quote", "discount", "charge"},
}


def _keywords_in(tokens: Set[str], keywords: Dict[str, Set[str]]) -> List[str]:
    return [name for name, words in keywords.items() if tokens & {w for kw in words for w in tokenize(kw)}]


class AskAround:
    def __init__(self, graph: KnowledgeGraph, answers: Iterable[StoredAnswer] = (), today: Optional[date] = None):
        self.graph = graph
        self.today = today or graph.today
        self.answers: Dict[str, StoredAnswer] = {a.id: a for a in answers}
        # Per (employee, expertise area) multiplier, nudged by feedback. 1.0 = neutral.
        self.expertise_feedback: Dict[Tuple[str, str], float] = {}
        self._reindex()

    def _reindex(self) -> None:
        texts = {d.id: f"{d.title} {d.text}" for d in self.graph.documents.values()}
        texts.update({a.id: a.question for a in self.answers.values()})
        self.index = TfidfIndex(texts)

    # ------------------------------------------------------------------ the main entry point

    def ask(self, question: str) -> Dict[str, Any]:
        q_vec = self.index.vector(question)
        q_tokens = set(tokenize(question))
        areas = _keywords_in(q_tokens, EXPERTISE_KEYWORDS)
        risks = _keywords_in(q_tokens, RISK_KEYWORDS)

        doc_sims = {d: cosine(q_vec, self.index.vectors[d]) for d in self.graph.documents}
        max_sim = max(doc_sims.values(), default=0.0)
        retrieval = min(1.0, max_sim / FULL_RETRIEVAL)
        if areas:
            retrieval = max(retrieval, 0.4)      # an expertise match alone still says something

        start = self._start_points(doc_sims, areas)
        pagerank = self._personalized_pagerank(start)
        people = self._score_people(pagerank, start, retrieval, areas)
        documents = self._score_documents(pagerank, doc_sims, retrieval)

        answer, match = self._best_answer(q_vec)
        evaluation = self._evaluate_answer(answer, match, people) if answer else None

        level, confidence, reasons = self._decide(evaluation, documents, people, risks)

        result: Dict[str, Any] = {
            "question": question,
            "level": level,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "expertise_areas": [self.graph.label(a) for a in areas if a in self.graph.G],
            "risk_topics": risks,
            "not_answered_before": answer is None,
            "propagation": self._top_nodes(pagerank),
        }

        if level == "high":
            result["answer"] = self._answer_view(answer, evaluation)
            result["sources"] = [self._source_view(d, q_vec, documents) for d in answer.source_documents]
        elif level == "medium":
            top_doc = answer.source_documents[0] if answer and answer.source_documents else documents[0]["id"]
            source = self._source_view(top_doc, q_vec, documents)
            expert = self._best_expert_for(top_doc, people)
            if answer:
                result["answer"] = self._answer_view(answer, evaluation)
            else:
                draft = " ".join(p["text"] for p in source["passages"][:1])
                result["answer"] = {"text": draft, "draft_from_document": True}
            result["sources"] = [source]
            result["experts"] = [expert]
            result["message"] = self._verify_message(expert, question, result["answer"]["text"], source)
        else:
            experts = [p for p in people if p["confidence"] > 0][:3]
            result["experts"] = experts
            # Documents a captured expert answer would be linked to.
            result["related_documents"] = [d["id"] for d in documents if d["confidence"] > 0][:2]
            for expert in experts:
                expert["message"] = self._question_message(expert, question)
        return result

    # ------------------------------------------------------------------ propagation

    def _start_points(self, doc_sims: Dict[str, float], areas: List[str]) -> Dict[str, float]:
        # Start from the documents closest to the question (at least 40 % of the best match).
        max_sim = max(doc_sims.values(), default=0.0)
        start = {d: s for d, s in doc_sims.items() if s >= max(0.05, 0.4 * max_sim)}
        start_mass = max(start.values(), default=0.3)
        known_areas = [a for a in areas if a in self.graph.G]
        for area in known_areas:
            # Matching expertise nodes together get a quarter of the best document's weight.
            start[area] = start.get(area, 0) + 0.25 * start_mass / len(known_areas)
        return start

    def _personalized_pagerank(self, start: Dict[str, float]) -> Dict[str, float]:
        if not start:
            return {}
        G = self.graph.G
        for _, _, data in G.edges(data=True):
            data["walk_weight"] = data["weight"] * WALK_FACTOR.get(data["kind"], 1.0)
        return nx.pagerank(G, alpha=PAGERANK_ALPHA, personalization=start, weight="walk_weight")

    def _feedback_multiplier(self, employee_id: str, areas: List[str]) -> float:
        relevant = areas or sorted(self.graph.employees[employee_id].expertise)
        if not relevant:
            return 1.0
        return sum(self.expertise_feedback.get((employee_id, a), 1.0) for a in relevant) / len(relevant)

    def _score_people(self, pagerank, start, retrieval, areas) -> List[Dict[str, Any]]:
        """Half from the PageRank walk (indirect links), half from direct links to the starting points.

        PageRank alone favours well-connected people; the direct part keeps the author of the
        best matching document on top.
        """
        employees = self.graph.nodes_of_type("employee")
        direct = {e: sum(s * self.graph.weight(e, n) for n, s in start.items()) for e in employees}
        top_pr = max((pagerank.get(e, 0.0) for e in employees), default=0.0)
        top_direct = max(direct.values(), default=0.0)
        scored = []
        for e in employees:
            relative_pr = pagerank.get(e, 0.0) / top_pr if top_pr else 0.0
            relative_direct = direct[e] / top_direct if top_direct else 0.0
            relative = 0.5 * relative_pr + 0.5 * relative_direct
            confidence = min(1.0, retrieval * relative * self._feedback_multiplier(e, areas))
            person = self.graph.employees[e]
            scored.append({
                "employee_id": e,
                "name": person.name,
                "position": person.position,
                "office": person.office,
                "confidence": round(confidence, 2),
                "pagerank": round(pagerank.get(e, 0.0), 4),
                "why": self._why(e, areas),
            })
        scored.sort(key=lambda p: p["confidence"], reverse=True)
        return scored

    def _score_documents(self, pagerank, doc_sims, retrieval) -> List[Dict[str, Any]]:
        docs = self.graph.nodes_of_type("document")
        top_pr = max((pagerank.get(d, 0.0) for d in docs), default=0.0)
        top_sim = max(doc_sims.values(), default=0.0)
        scored = []
        for d in docs:
            relative_pr = pagerank.get(d, 0.0) / top_pr if top_pr else 0.0
            relative_sim = doc_sims[d] / top_sim if top_sim else 0.0
            scored.append({"id": d, "confidence": round(retrieval * (0.5 * relative_sim + 0.5 * relative_pr), 2)})
        scored.sort(key=lambda x: x["confidence"], reverse=True)
        return scored

    def _why(self, employee_id: str, areas: List[str]) -> List[str]:
        reasons = []
        person = self.graph.employees[employee_id]
        for area in areas:
            if area in person.expertise:
                reasons.append(f"{self.graph.label(area)} expert")
        for (e, d), activity in self.graph.activities.items():
            if e == employee_id and activity.role:
                verb = "Wrote" if activity.role == "author" else "Contributed to"
                reasons.append(f"{verb} “{self.graph.documents[d].title}”")
        if person.years_experience:
            reasons.append(f"{person.years_experience} years of experience")
        return reasons[:3]

    def _top_nodes(self, pagerank: Dict[str, float], n: int = 8) -> List[Dict[str, Any]]:
        top = sorted(pagerank.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"id": k, "label": self.graph.label(k), "type": self.graph.G.nodes[k]["type"], "score": round(v, 4)}
                for k, v in top]

    # ------------------------------------------------------------------ answer memory

    def _best_answer(self, q_vec) -> Tuple[Optional[StoredAnswer], float]:
        best, best_match = None, 0.0
        for a in self.answers.values():
            match = cosine(q_vec, self.index.vectors[a.id])
            if match > best_match:
                best, best_match = a, match
        return (best, best_match) if best_match >= MIN_ANSWER_MATCH else (None, 0.0)

    def _evaluate_answer(self, answer: StoredAnswer, match: float, people: List[Dict[str, Any]]) -> Dict[str, Any]:
        helpful = (answer.helpful + 1) / (answer.helpful + answer.not_helpful + 2)
        age = (self.today - answer.verified_on).days if answer.verified_on else None
        freshness = 0.5 ** (age / 365) if age is not None else 0.3
        stale_sources = [
            d for d in answer.source_documents
            if answer.verified_on and (self.graph.documents[d].last_update_date or date.min) > answer.verified_on
        ]
        colleagues = len(answer.confirmed_by | {answer.answered_by})
        corroboration = 1.0 if colleagues >= 2 else 0.6
        author_conf = next((p["confidence"] for p in people if p["employee_id"] == answer.answered_by), 0.0)
        authority = 0.5 + 0.5 * author_conf

        trust = 0.35 * helpful + 0.25 * freshness + 0.20 * corroboration + 0.20 * authority
        if stale_sources:
            trust *= 0.7
        confidence = min(1.0, match / FULL_ANSWER_MATCH) * trust
        return {
            "confidence": confidence,
            "match": match,
            "verified": age is not None and age <= MAX_VERIFICATION_AGE_DAYS and not stale_sources,
            "verification_age_days": age,
            "stale_sources": stale_sources,
            "colleagues": colleagues,
        }

    def _decide(self, evaluation, documents, people, risks) -> Tuple[str, float, List[str]]:
        reasons: List[str] = []
        if evaluation:
            if evaluation["stale_sources"]:
                titles = ", ".join(self.graph.documents[d].title for d in evaluation["stale_sources"])
                reasons.append(f"Source changed since the last verification: {titles}")
            elif evaluation["verification_age_days"] is None:
                reasons.append("This answer was never verified by an expert")
            elif evaluation["verification_age_days"] > MAX_VERIFICATION_AGE_DAYS:
                reasons.append("Last expert verification is more than a year old")
            if evaluation["colleagues"] < 2:
                reasons.append("Only one colleague gave this answer")
        else:
            reasons.append("Not answered before")
        if risks:
            reasons.append(f"High-risk topic ({', '.join(risks)})")

        answer_conf = evaluation["confidence"] if evaluation else 0.0
        verified = bool(evaluation and evaluation["verified"])

        if evaluation and answer_conf >= HIGH_THRESHOLD and verified and evaluation["colleagues"] >= 2:
            return "high", answer_conf, ["Answered before, verified and confirmed by several colleagues"]
        if risks and not verified:
            reasons.append("Routed to an expert because there is no verified answer")
            return "low", people[0]["confidence"] if people else 0.0, reasons
        if evaluation and answer_conf >= MEDIUM_THRESHOLD:
            return "medium", answer_conf, reasons
        if not evaluation and documents and documents[0]["confidence"] >= 0.45 and not risks:
            reasons.append("A matching document exists, but nobody answered this exact question yet")
            # A draft taken straight from a document is never more than "medium" sure.
            return "medium", min(documents[0]["confidence"], HIGH_THRESHOLD) * 0.85, reasons
        return "low", people[0]["confidence"] if people else 0.0, reasons

    # ------------------------------------------------------------------ views

    def _answer_view(self, answer: StoredAnswer, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        employees = self.graph.employees
        return {
            "id": answer.id,
            "text": answer.answer,
            "original_question": answer.question,
            "answered_by": employees[answer.answered_by].name,
            "confirmed_by": sorted(employees[e].name for e in answer.confirmed_by if e != answer.answered_by),
            "verified_on": answer.verified_on.isoformat() if answer.verified_on else None,
            "helpful": answer.helpful,
            "not_helpful": answer.not_helpful,
            "confidence": round(evaluation["confidence"], 2),
        }

    def _source_view(self, doc_id: str, q_vec, documents) -> Dict[str, Any]:
        doc = self.graph.documents[doc_id]
        confidence = next((d["confidence"] for d in documents if d["id"] == doc_id), 0.0)
        return {
            "id": doc_id,
            "title": doc.title,
            "last_update": doc.last_update_date.isoformat() if doc.last_update_date else None,
            "confidence": confidence,
            "passages": self._passages(doc.text, q_vec),
        }

    def _passages(self, text: str, q_vec, limit: int = 2) -> List[Dict[str, str]]:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        scored = sorted(((cosine(q_vec, self.index.vector(s)), s) for s in sentences), reverse=True)
        query_terms = set(q_vec)
        passages = []
        for score, sentence in scored[:limit]:
            if score <= 0:
                break
            highlighted = re.sub(
                r"[A-Za-zÀ-ÿ0-9]+",
                lambda m: f"<mark>{m.group(0)}</mark>" if set(tokenize(m.group(0))) & query_terms else m.group(0),
                html.escape(sentence),
            )
            passages.append({"text": sentence, "html": highlighted})
        return passages

    def _best_expert_for(self, doc_id: str, people: List[Dict[str, Any]]) -> Dict[str, Any]:
        linked = {n for n in self.graph.G.neighbors(doc_id) if self.graph.G.nodes[n]["type"] == "employee"}
        return next((p for p in people if p["employee_id"] in linked), people[0])

    def _verify_message(self, expert, question, answer_text, source) -> Dict[str, str]:
        first_name = expert["name"].split()[0]
        text = (
            f"Hi {first_name}, I have a question: “{question}”\n\n"
            f"AskAround suggests this answer, based on “{source['title']}” "
            f"(last updated {source['last_update']}):\n“{answer_text}”\n\n"
            f"Could you quickly check whether the answer and the document are still correct? Thanks!"
        )
        return {"to": expert["name"], "employee_id": expert["employee_id"], "text": text}

    def _question_message(self, expert, question) -> str:
        first_name = expert["name"].split()[0]
        return (f"Hi {first_name}, AskAround suggested you as an expert for this question: "
                f"“{question}”. Do you have a moment to help me? Thanks!")

    # ------------------------------------------------------------------ feedback loop

    def _nudge(self, employee_id: str, areas: Iterable[str], helpful: bool) -> None:
        for area in areas:
            key = (employee_id, area)
            current = self.expertise_feedback.get(key, 1.0)
            step = FEEDBACK_STEP if helpful else -FEEDBACK_STEP
            self.expertise_feedback[key] = round(min(1.5, max(0.5, current + step)), 4)

    def _areas_for(self, employee_id: str, document_ids: Iterable[str]) -> Set[str]:
        areas = set()
        for d in document_ids:
            areas |= self.graph.documents[d].expertise
        return areas & self.graph.employees[employee_id].expertise or areas

    def rate_answer(self, answer_id: str, helpful: bool) -> StoredAnswer:
        """The asker gives a thumbs up / down on a reused answer."""
        answer = self.answers[answer_id]
        if helpful:
            answer.helpful += 1
        else:
            answer.not_helpful += 1
        for employee_id in answer.confirmed_by | {answer.answered_by}:
            for d in answer.source_documents:
                self.graph.record_document_feedback(employee_id, d, helpful)
            self._nudge(employee_id, self._areas_for(employee_id, answer.source_documents), helpful)
        return answer

    def verify(self, question: str, answer_text: str, employee_id: str, document_ids: List[str],
               correct: bool, answer_id: Optional[str] = None) -> Optional[StoredAnswer]:
        """An expert answers the 'please verify' message: correct -> stored as a verified answer."""
        self._nudge(employee_id, self._areas_for(employee_id, document_ids), correct)
        existing = self.answers.get(answer_id) if answer_id else None
        if not correct:
            if existing:
                existing.verified_on = None
                existing.not_helpful += 1
            return existing
        if existing:
            existing.verified_on = self.today
            existing.confirmed_by.add(employee_id)
            return existing
        return self.save_answer(question, answer_text, employee_id, document_ids)

    def save_answer(self, question: str, answer_text: str, employee_id: str, document_ids: List[str]) -> StoredAnswer:
        """Store an expert's reply so the next colleague with a similar question gets it directly."""
        new_id = f"ans-{next(i for i in itertools.count(len(self.answers) + 1) if f'ans-{i}' not in self.answers)}"
        answer = StoredAnswer(new_id, question, answer_text, answered_by=employee_id,
                              source_documents=list(document_ids), confirmed_by={employee_id},
                              verified_on=self.today)
        self.answers[new_id] = answer
        self._reindex()
        return answer
