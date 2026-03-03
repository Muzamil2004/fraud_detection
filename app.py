"""FastAPI app for fraud detection inference."""
# pyright: reportMissingImports=false
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:
    if __name__ == "__main__":
        raise SystemExit(
            "Missing API dependencies in current interpreter.\n"
            "Use project env:\n"
            "  .\\venv\\Scripts\\uvicorn.exe app:app --reload\n"
            "or install packages in this interpreter:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    raise

from src.api.routes import build_router

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts fraud probability and decision with tuned threshold.",
    version="1.1.0",
    openapi_url="/api/openapi.json",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(build_router(model_path=MODEL_PATH, frontend_dir=FRONTEND_DIR))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
