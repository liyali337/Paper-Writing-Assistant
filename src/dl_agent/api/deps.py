from dl_agent.knowledge.service import KnowledgeService, get_service
from dl_agent.translate.service import TranslateService, get_translate_service

__all__ = ["get_knowledge", "get_translate"]


def get_knowledge() -> KnowledgeService:
    return get_service()


def get_translate() -> TranslateService:
    return get_translate_service()
