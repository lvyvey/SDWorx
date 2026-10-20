"""Node types, raw interaction records and stored answers for AskAround."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Set

# Expertise domains from the concept: payroll, legal, social security, tax and data handling.
EXPERTISE_AREAS = ["payroll", "legal", "social_security", "tax", "data_handling"]


# --------------------------------------------------------------------------- nodes


@dataclass
class Employee:
    id: str
    name: str
    position: str = ""
    education: str = ""
    years_experience: int = 0
    country: str = ""
    office: str = ""
    department: str = ""
    expertise: Set[str] = field(default_factory=set)


@dataclass
class Document:
    id: str
    title: str
    text: str = ""
    country: str = ""
    publication_date: Optional[date] = None
    last_update_date: Optional[date] = None
    topics: Set[str] = field(default_factory=set)
    sectors: Set[str] = field(default_factory=set)
    file_type: str = ""
    expertise: Set[str] = field(default_factory=set)


@dataclass
class Customer:
    id: str
    company: str
    country: str = ""
    sector: str = ""
    num_employees: int = 0
    collar: str = ""                   # "blue", "white" or "mixed"


@dataclass
class Expertise:
    id: str
    name: str


# --------------------------------------------------------------------------- raw signals (edge weights are computed from these)


@dataclass
class DocumentActivity:
    employee_id: str
    document_id: str
    role: str = ""                                        # "author" (++), "contributor" (+) or ""
    edit_dates: List[date] = field(default_factory=list)
    forwarded: int = 0
    used: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0


@dataclass
class EmployeeContact:
    employee_a: str
    employee_b: str
    messages: int = 0
    meetings: int = 0
    events: int = 0


@dataclass
class CustomerInteraction:
    employee_id: str
    customer_id: str
    interaction_dates: List[date] = field(default_factory=list)


# --------------------------------------------------------------------------- answer memory


@dataclass
class StoredAnswer:
    """An earlier answer that can be reused for similar questions."""

    id: str
    question: str
    answer: str
    answered_by: str                                      # employee id
    source_documents: List[str] = field(default_factory=list)
    confirmed_by: Set[str] = field(default_factory=set)   # colleagues who gave or confirmed the same answer
    verified_on: Optional[date] = None                    # last expert verification
    helpful: int = 0
    not_helpful: int = 0
