"""AskAround: trusted answers from the people who know."""

from .engine import AskAround
from .graph import KnowledgeGraph
from .sample_data import DEMO_TODAY, build_sample


def create_demo() -> AskAround:
    graph, answers = build_sample()
    return AskAround(graph, answers, today=DEMO_TODAY)


__all__ = ["AskAround", "KnowledgeGraph", "create_demo"]
