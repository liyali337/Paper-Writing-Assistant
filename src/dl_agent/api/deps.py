from fastapi import Depends

from dl_agent.config import get_settings
from dl_agent.knowledge.service import KnowledgeService, get_service
from dl_agent.translate.service import TranslateService, get_translate_service
from dl_agent.understand.service import UnderstandService

__all__ = ["get_knowledge", "get_translate", "get_understand"]


def get_knowledge() -> KnowledgeService:
    return get_service()


def get_translate() -> TranslateService:
    return get_translate_service()


def get_understand(knowledge: KnowledgeService = Depends(get_knowledge)) -> UnderstandService:
    return UnderstandService(knowledge, get_settings())
