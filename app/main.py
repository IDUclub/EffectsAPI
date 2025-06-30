from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse

from app.effects_api.effects_controller import development_router
from app.system_router.system_controller import system_router

#TODO add app version
app = FastAPI(
    title="Effects API",
    description="API for calculating effects of territory transformation with BlocksNet library",
)

origins = ["*"]

# disable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=100)


@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse("/docs")


app.include_router(system_router)
app.include_router(development_router)
