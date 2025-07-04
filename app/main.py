from fastapi import FastAPI, Request
from app.core.config import settings
from app.api.v1.router import api_router
import time
import logging

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME)

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"请求 {request.url.path} 耗时: {process_time:.4f} 秒")
        return response

    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    return app

app = create_app()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Service project"} 