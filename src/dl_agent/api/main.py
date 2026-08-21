import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dl_agent.api.routes.health import router as health_router
from dl_agent.api.routes.papers import router as papers_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="论文理解",
    version="0.1.0",
    description="M1：上传 PDF，按标题分解章节并抽出插图。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(papers_router)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "dl_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
    )
