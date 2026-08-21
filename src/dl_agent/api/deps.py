from dl_agent.knowledge.service import KnowledgeService, get_service

__all__ = ["get_knowledge"]


def get_knowledge() -> KnowledgeService:
    return get_service()
