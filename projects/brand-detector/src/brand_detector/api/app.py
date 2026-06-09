"""FastAPI application scaffold for brand-detector."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="brand-detector", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
