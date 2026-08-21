from fastapi import APIRouter

from dl_agent import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dl-agent", "version": __version__}
