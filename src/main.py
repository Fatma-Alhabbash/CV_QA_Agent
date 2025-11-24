# backend/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.routes.routes import router, get_service
from src.services import CVService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="CV QA Agent")

    # Allow Lovable (or any origin during dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # include router
    app.include_router(router)

    # --- Resolve the static folder and CV file in a robust, repo-relative way ---
    # project_root: one level above backend/ (where this file lives)
    project_root = Path(__file__).resolve().parents[1]
    log.info("project_root resolved to: %s", project_root)

    # candidate static directories (common layouts)
    static_dirs = [
        project_root / "src" / "static",  # if you keep src/static
        project_root / "static",          # or top-level static/
        project_root / "backend" / "static", # sometimes backend/static
    ]

    # pick the first existing static dir
    static_dir = next((d for d in static_dirs if d.exists() and d.is_dir()), None)
    if static_dir is None:
        # helpful log and fallback to project_root/static (so you can still see error)
        log.error("No static directory found. Checked: %s", static_dirs)
        # keep trying to create it so StaticFiles won't crash
        static_dir = project_root / "src" / "static"

    log.info("Using static directory: %s (exists=%s)", static_dir, static_dir.exists())

    # mount static so files are served under /static
    try:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    except Exception:
        log.exception("Failed to mount static directory: %s", static_dir)

    # CV filename - change this if your file has a different name
    cv_filename = "Fatma Alzahraa Alhabbash - CV.pdf"

    # try explicit filename first, otherwise pick the first pdf in static_dir
    cv_path_candidates = [
        static_dir / cv_filename,
        *(static_dir.glob("*.pdf")),   # yields Path objects if any pdfs exist
        *(static_dir.glob("*.PDF")),
    ]

    cv_path = None
    for p in cv_path_candidates:
        if p and p.exists():
            cv_path = p
            break

    if cv_path is None:
        log.error(
            "CV file not found in static dir. Looked for: %s\nStatic dir listing: %s",
            cv_filename,
            list(static_dir.glob("*")) if static_dir.exists() else "static dir does not exist"
        )
        # raise an informative error so startup fails and you see logs in Render
        raise FileNotFoundError(
            f"CV file not found. Tried: {cv_filename} and any *.pdf inside {static_dir}"
        )

    log.info("Using CV file at: %s", cv_path)

    # attach service to app.state so it's accessible globally
    app.state.cv_service = CVService(cv_path=str(cv_path))

    # override the dependency function from routes.get_service
    app.dependency_overrides[get_service] = lambda: app.state.cv_service

    @app.on_event("startup")
    def startup_event():
        log.info("Starting up - initializing CVService (loading CV, building index)...")
        try:
            app.state.cv_service.initialize()
            log.info("CVService initialized successfully.")
        except Exception as e:
            log.exception("Failed to initialize CVService: %s", e)
            raise

    @app.on_event("shutdown")
    def shutdown_event():
        log.info("Shutting down - cleaning up resources if necessary.")

    return app

app = create_app()
