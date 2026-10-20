"""Edge-weight formulas. Every weight is in [0, 1].

The coefficients below are first guesses; tune them in one place here.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Iterable, Set

from .models import Customer, Document, DocumentActivity, Employee, EmployeeContact

EDIT_HALF_LIFE_DAYS = 365        # an edit from a year ago counts half
INTERACTION_HALF_LIFE_DAYS = 180


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def log_saturation(count: float, saturate_at: float) -> float:
    """0 for 0, 1 at `saturate_at`, logarithmic in between (first contacts matter most)."""
    if count <= 0:
        return 0.0
    return clamp(math.log1p(count) / math.log1p(saturate_at))


def recency_weighted_count(dates: Iterable[date], today: date, half_life_days: float) -> float:
    return sum(0.5 ** (max(0, (today - d).days) / half_life_days) for d in dates)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- employee - document

AUTHORSHIP_SCORE = {"author": 1.0, "contributor": 0.5}   # self-written (++) vs added (+)
FEEDBACK_IMPACT = 0.2                                    # thumbs up/down move the weight by at most ±20 %


def employee_document_weight(activity: DocumentActivity, today: date) -> float:
    authorship = AUTHORSHIP_SCORE.get(activity.role, 0.0)
    edits = log_saturation(recency_weighted_count(activity.edit_dates, today, EDIT_HALF_LIFE_DAYS), 10)
    usage = log_saturation(activity.forwarded * 2 + activity.used, 50)   # forwarding counts double

    base = 0.45 * authorship + 0.30 * edits + 0.25 * usage

    # Bayesian average of thumbs up/down: 0.5 with no feedback, so no feedback = no effect.
    feedback = (activity.thumbs_up + 1) / (activity.thumbs_up + activity.thumbs_down + 2)
    return round(clamp(base * (1 + FEEDBACK_IMPACT * (feedback - 0.5) * 2)), 4)


# --------------------------------------------------------------------------- document - document


def document_document_weight(text_similarity: float, doc_a: Document, doc_b: Document) -> float:
    """Text overlap (TF-IDF / embedding) + shared topics; file type only as a small bonus."""
    topics = jaccard(doc_a.topics, doc_b.topics)
    same_type = 1.0 if doc_a.file_type and doc_a.file_type == doc_b.file_type else 0.0
    return round(clamp(0.60 * text_similarity + 0.35 * topics + 0.05 * same_type), 4)


# --------------------------------------------------------------------------- employee - employee


def employee_employee_weight(a: Employee, b: Employee, contact: EmployeeContact | None) -> float:
    same_office = 1.0 if a.office and a.office == b.office else 0.0
    same_department = 1.0 if a.department and a.department == b.department else 0.0
    contact_score = 0.0
    if contact:
        contact_score = log_saturation(contact.messages + 3 * contact.meetings + 3 * contact.events, 100)
    return round(clamp(0.20 * same_office + 0.30 * same_department + 0.50 * contact_score), 4)


# --------------------------------------------------------------------------- employee - customer


def employee_customer_weight(interaction_dates: Iterable[date], today: date) -> float:
    return round(log_saturation(recency_weighted_count(interaction_dates, today, INTERACTION_HALF_LIFE_DAYS), 20), 4)


# --------------------------------------------------------------------------- customer - customer


def customer_customer_weight(a: Customer, b: Customer) -> float:
    same_sector = 1.0 if a.sector and a.sector == b.sector else 0.0
    same_country = 1.0 if a.country and a.country == b.country else 0.0
    same_collar = 1.0 if a.collar and a.collar == b.collar else 0.0
    size = 0.0
    if a.num_employees > 0 and b.num_employees > 0:
        # 1 for equal size, 0 when one is 100x bigger.
        size = clamp(1 - abs(math.log10(a.num_employees / b.num_employees)) / 2)
    return round(clamp(0.35 * same_sector + 0.20 * same_country + 0.25 * size + 0.20 * same_collar), 4)


# --------------------------------------------------------------------------- customer - document


def customer_document_weight(customer: Customer, document: Document) -> float:
    """Only connected when the document covers the customer's sector; same country strengthens it."""
    if customer.sector not in document.sectors:
        return 0.0
    country_match = 1.0 if not document.country or document.country in (customer.country, "EU") else 0.0
    return round(0.7 + 0.3 * country_match, 4)
