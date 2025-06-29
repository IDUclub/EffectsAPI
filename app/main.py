from contextlib import asynccontextmanager

from app.effects_api import effects_controller
# from app.effects_api.schemas.task_schema import TaskSchema
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse


@asynccontextmanager
async def lifespan(router : FastAPI):
    yield

app = FastAPI(
    title="Effects API",
    description="API for calculating effects of territory transformation with BlocksNet library",
    lifespan=lifespan,
)

# disable cors
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='http://.*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=100)
app.include_router(effects_controller.development_router)

@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse('/docs')
