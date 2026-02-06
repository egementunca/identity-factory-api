"""
FastAPI server setup for Identity Circuit Factory API.
"""

import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..factory_manager import FactoryConfig, IdentityFactory
from .endpoints import router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global factory instance
_factory: Optional[IdentityFactory] = None

# In-memory log buffer (shared)
LOG_BUFFER = deque(maxlen=200)


class BufferLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        LOG_BUFFER.append(msg)


# Attach buffer handler to root logger
buffer_handler = BufferLogHandler()
buffer_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(buffer_handler)


def get_log_buffer():
    return list(LOG_BUFFER)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _factory

    # Startup
    logger.info("Starting Identity Circuit Factory API...")

    try:
        # Initialize factory
        config = FactoryConfig()
        _factory = IdentityFactory(config)
        logger.info("Factory initialized successfully")

        # Add factory to app state
        app.state.factory = _factory

    except Exception as e:
        logger.error(f"Failed to initialize factory: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Identity Circuit Factory API...")

    if _factory:
        # Cleanup factory resources
        logger.info("Factory cleanup completed")


def create_app(
    title: str = "Identity Circuit Factory API",
    description: str = "RESTful API for generating, unrolling, and managing identity circuits",
    version: str = "1.0.0",
    debug: bool = False,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        description: API description
        version: API version
        debug: Enable debug mode
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI application
    """

    # Default CORS origins
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Vue dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "*",  # Allow all origins in development
        ]

    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        import time

        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    # Add error handling middleware
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        import uuid

        request_id = str(uuid.uuid4())

        logger.error(f"Request {request_id} failed: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if debug else "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            },
        )

    # Include API routes
    app.include_router(router, prefix="/api/v1", tags=["identity-factory"])

    # Include generator routes
    from .generator_endpoints import router as generator_router

    app.include_router(generator_router, prefix="/api/v1", tags=["generators"])

    # Include Go database routes
    from .go_database_endpoints import router as go_database_router

    app.include_router(go_database_router, prefix="/api/v1", tags=["go-database"])

    # Include SAT database routes
    from .sat_database_endpoints import router as sat_database_router

    app.include_router(sat_database_router, prefix="/api/v1", tags=["sat-database"])

    # Include advanced endpoints (compression, identities, synthesis)
    from .advanced_endpoints import router as advanced_router

    app.include_router(advanced_router, prefix="/api/v1", tags=["advanced"])

    # Include automation endpoints (scheduler for batch operations)
    from .automation_endpoints import router as automation_router

    app.include_router(
        automation_router, prefix="/api/v1/automation", tags=["automation"]
    )

    # Include irreducible circuit endpoints (forward + inverse synthesis)
    from .irreducible_endpoints import router as irreducible_router

    app.include_router(
        irreducible_router, prefix="/api/v1/irreducible", tags=["irreducible"]
    )

    # Include ECA57 LMDB endpoints (fresh enumeration database)
    from .eca57_lmdb_endpoints import router as eca57_lmdb_router

    app.include_router(eca57_lmdb_router, prefix="/api/v1", tags=["eca57-lmdb"])

    # Include identity browser endpoints (saved identity circuits from local_mixing)
    from .identity_browser_endpoints import router as identity_browser_router

    app.include_router(
        identity_browser_router, prefix="/api/v1/local-mixing/identities", tags=["identity-browser"]
    )

    # Include Local Mixing endpoints (canonicalization, heatmap, inflation preview)
    from .local_mixing_endpoints import router as local_mixing_router

    app.include_router(
        local_mixing_router, prefix="/api/v1/local-mixing", tags=["local-mixing"]
    )

    # Include Cluster Database endpoints (SAT-generated ECA57/MCT circuits from cluster)
    from .cluster_endpoints import router as cluster_router

    app.include_router(cluster_router, prefix="/api/v1", tags=["cluster-database"])
    
    # Include Permutation Database endpoints (Rust LMDB Permutation Tables)
    from .perm_db_endpoints import router as perm_db_router

    app.include_router(perm_db_router, prefix="/api/v1", tags=["perm-database"])

    # Include Wire Shuffler database endpoints
    from .wire_shuffler_endpoints import router as wire_shuffler_router

    app.include_router(wire_shuffler_router, prefix="/api/v1", tags=["wire-shuffler"])

    # Include Waksman network endpoints (O(n log n) permutation circuits)
    from .waksman_endpoints import router as waksman_router

    app.include_router(waksman_router, prefix="/api/v1", tags=["waksman"])

    # Include Experiment endpoints (run local_mixing experiments with progress streaming)
    from .experiment_endpoints import router as experiment_router

    app.include_router(
        experiment_router, prefix="/api/v1/experiments", tags=["experiments"]
    )

    # Include Skeleton Database endpoints (skeleton chain identity circuits)
    from .skeleton_db_endpoints import router as skeleton_db_router

    app.include_router(skeleton_db_router, prefix="/api/v1", tags=["skeleton-database"])

    # Include Unified Search endpoints (search across all databases)
    from .unified_search_endpoints import router as unified_search_router

    app.include_router(unified_search_router, prefix="/api/v1", tags=["unified-search"])

    # Mount static files directory for frontend assets (if exists)
    import os

    static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Serve frontend.html at the root (if exists)
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend.html")
    if os.path.exists(frontend_path):

        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(frontend_path)

    else:

        @app.get("/", include_in_schema=False)
        async def root():
            return {"message": "Identity Circuit Factory API", "docs": "/docs"}

    # API info endpoint
    @app.get("/api/info")
    async def api_info():
        """API information endpoint."""
        return {
            "name": title,
            "version": version,
            "description": description,
            "endpoints": {
                "health": "/api/v1/health",
                "stats": "/api/v1/stats",
                "generate": "/api/v1/generate",
                "unroll": "/api/v1/unroll",
                "simplify": "/api/v1/simplify",
                "circuits": "/api/v1/circuits",
                "dim-groups": "/api/v1/dim-groups",
                "export": "/api/v1/export",
                "import": "/api/v1/import",
                "recommendations": "/api/v1/recommendations",
            },
        }

    # Logs endpoint (moved from endpoints.py to avoid circular import)
    @app.get("/logs")
    async def get_logs():
        """Get the latest server logs (live)."""
        logs = list(LOG_BUFFER)
        return {"logs": logs}

    logger.info(f"FastAPI application created: {title} v{version}")

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    debug: bool = False,
    workers: int = 1,
):
    """
    Run the FastAPI server.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload
        debug: Enable debug mode
        workers: Number of worker processes
    """

    app = create_app(debug=debug)

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Auto-reload: {reload}")
    logger.info(f"Workers: {workers}")

    uvicorn.run(
        "identity_factory.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        log_level="info" if not debug else "debug",
        workers=workers if not reload else 1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identity Circuit Factory API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        debug=args.debug,
        workers=args.workers,
    )
