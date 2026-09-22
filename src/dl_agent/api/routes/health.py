from fastapi import APIRouter

from dl_agent import __version__

router = APIRouter(tags=["health"])


@router.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "dl-agent",
        "health": "/health",
        "docs": "/docs",
        "ui": "http://localhost:5173",
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dl-agent", "version": __version__}
