# backend/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router, get_service
from services import CVService

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

    # attach service to app.state so it's accessible globally, storing this CV service globally on the app so you can call it from anywhere in the app
    app.state.cv_service = CVService(cv_path="C:\\Users\\hp\\Desktop\\All\\AI Agentic\\lectures\\code\\Fatma Alzahraa Alhabbash - CV.pdf")

    # override the dependency function from routes.get_service
    # so that routes can use Depends(get_service)
    app.dependency_overrides[get_service] = lambda: app.state.cv_service

    @app.on_event("startup")
    def startup_event():
        log.info("Starting up - initializing CVService (loading CV, building index)...")
        try:
            app.state.cv_service.initialize()
            log.info("CVService initialized successfully.")
        except Exception as e:
            log.exception("Failed to initialize CVService: %s", e)
            # re-raise if you want the app to fail to start
            raise

    @app.on_event("shutdown")
    def shutdown_event():
        log.info("Shutting down - cleaning up resources if necessary.")
        # If you had db connections or other resources, close them here.

    return app

app = create_app()

# For uvicorn: uvicorn backend.main:app --reload --port 8000
