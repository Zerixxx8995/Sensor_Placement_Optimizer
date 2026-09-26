"""
main.py
-------
FastAPI application factory.

Responsibilities:
  - Create the FastAPI app instance.
  - Register middleware (CORS for now; error handler and logger added in step 6).
  - Mount all routers under /api/v1.
  - Expose the /api/v1/health endpoint.

Nothing else lives here — no business logic, no algorithm code.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.error_handler import add_error_handlers
from app.middleware.request_logger import RequestLoggerMiddleware
from app.routers.optimize import router as optimize_router
from app.routers.compare import router as compare_router
from app.routers.fault import router as fault_router
from app.routers.history import router as history_router

app = FastAPI(
    title="PSO Sensor Placement Optimizer",
    description=(
        "GPU-accelerated Particle Swarm Optimization for Wireless Sensor "
        "Network deployment."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tightened in production via .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# RequestLoggerMiddleware registered last → outermost layer (wraps CORS too)
app.add_middleware(RequestLoggerMiddleware)

# Register global exception handlers
add_error_handlers(app)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(optimize_router, prefix="/api/v1", tags=["Optimize"])
app.include_router(compare_router, prefix="/api/v1", tags=["Compare"])
app.include_router(fault_router, prefix="/api/v1", tags=["Fault"])
app.include_router(history_router, prefix="/api/v1", tags=["History"])

# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", tags=["Health"])
def health():
    """Liveness probe — must respond < 100 ms."""
    return {"status": "ok"}
