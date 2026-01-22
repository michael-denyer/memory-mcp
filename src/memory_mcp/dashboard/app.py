"""FastAPI application for the Memory MCP dashboard."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from memory_mcp.config import get_settings
from memory_mcp.storage import MemoryType, Storage

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Global storage instance
storage: Storage | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage storage lifecycle."""
    global storage
    settings = get_settings()
    storage = Storage(settings)
    yield
    if storage:
        storage.close()


app = FastAPI(
    title="Memory MCP Dashboard",
    description="Web dashboard for Memory MCP",
    lifespan=lifespan,
)


def get_storage() -> Storage:
    """Get the storage instance."""
    if storage is None:
        raise RuntimeError("Storage not initialized")
    return storage


def format_bytes(size: int | float) -> str:
    """Format bytes to human readable string."""
    size_f = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} TB"


# Type badge colors
TYPE_COLORS = {
    "project": ("blue", "bg-blue-500/20 text-blue-400 border-blue-500/30"),
    "pattern": ("purple", "bg-purple-500/20 text-purple-400 border-purple-500/30"),
    "reference": ("green", "bg-green-500/20 text-green-400 border-green-500/30"),
    "conversation": ("yellow", "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"),
    "episodic": ("gray", "bg-gray-500/20 text-gray-400 border-gray-500/30"),
}


def get_type_badge_class(memory_type: str) -> str:
    """Get Tailwind classes for a memory type badge."""
    return TYPE_COLORS.get(memory_type, TYPE_COLORS["project"])[1]


# Add template globals
templates.env.globals["get_type_badge_class"] = get_type_badge_class
templates.env.globals["format_bytes"] = format_bytes


# ============================================================================
# HTML Pages
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Overview page with stats cards."""
    s = get_storage()
    stats = s.get_stats()
    hot_stats = s.get_hot_cache_stats()

    # Calculate DB size
    db_path = get_settings().db_path
    db_size = db_path.stat().st_size if db_path.exists() else 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "hot_stats": hot_stats,
            "db_size": db_size,
            "active_page": "overview",
        },
    )


@app.get("/hot-cache", response_class=HTMLResponse)
async def hot_cache_page(request: Request) -> HTMLResponse:
    """Hot cache management page."""
    s = get_storage()
    hot_memories = s.get_hot_memories()
    hot_stats = s.get_hot_cache_stats()

    return templates.TemplateResponse(
        "hot_cache.html",
        {
            "request": request,
            "memories": hot_memories,
            "hot_stats": hot_stats,
            "active_page": "hot_cache",
        },
    )


def _parse_memory_type(type_str: str | None) -> MemoryType | None:
    """Parse a string to MemoryType enum or None."""
    if type_str is None:
        return None
    try:
        return MemoryType(type_str)
    except ValueError:
        return None


@app.get("/memories", response_class=HTMLResponse)
async def memories_page(
    request: Request,
    type_filter: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> HTMLResponse:
    """Memory browser page."""
    s = get_storage()
    offset = (page - 1) * limit
    mem_type = _parse_memory_type(type_filter)
    memories = s.list_memories(limit=limit, offset=offset, memory_type=mem_type)
    stats = s.get_stats()
    total = stats.get("total_memories", 0)
    total_pages = (total + limit - 1) // limit

    return templates.TemplateResponse(
        "memories.html",
        {
            "request": request,
            "memories": memories,
            "type_filter": type_filter,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "active_page": "memories",
        },
    )


# ============================================================================
# HTMX API Endpoints
# ============================================================================


@app.get("/api/stats", response_class=HTMLResponse)
async def api_stats(request: Request) -> HTMLResponse:
    """Return stats cards partial for HTMX polling."""
    s = get_storage()
    stats = s.get_stats()
    hot_stats = s.get_hot_cache_stats()
    db_path = get_settings().db_path
    db_size = db_path.stat().st_size if db_path.exists() else 0

    return templates.TemplateResponse(
        "partials/stats_cards.html",
        {
            "request": request,
            "stats": stats,
            "hot_stats": hot_stats,
            "db_size": db_size,
        },
    )


@app.post("/api/hot-cache/{memory_id}/demote", response_class=HTMLResponse)
async def api_demote(memory_id: int, request: Request) -> HTMLResponse:
    """Demote a memory from hot cache."""
    s = get_storage()
    s.demote_from_hot(memory_id)
    # Return empty response - HTMX will remove the row
    return HTMLResponse(content="")


@app.post("/api/hot-cache/{memory_id}/pin", response_class=HTMLResponse)
async def api_pin(memory_id: int, request: Request) -> HTMLResponse:
    """Pin a memory in hot cache."""
    s = get_storage()
    s.pin_memory(memory_id)
    memory = s.get_memory(memory_id)
    return templates.TemplateResponse(
        "partials/hot_item.html",
        {"request": request, "memory": memory, "is_pinned": True},
    )


@app.post("/api/hot-cache/{memory_id}/unpin", response_class=HTMLResponse)
async def api_unpin(memory_id: int, request: Request) -> HTMLResponse:
    """Unpin a memory in hot cache."""
    s = get_storage()
    s.unpin_memory(memory_id)
    memory = s.get_memory(memory_id)
    return templates.TemplateResponse(
        "partials/hot_item.html",
        {"request": request, "memory": memory, "is_pinned": False},
    )


@app.get("/api/memories/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    query: str = "",
    type_filter: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> HTMLResponse:
    """Search memories and return table partial."""
    s = get_storage()
    offset = (page - 1) * limit
    mem_type = _parse_memory_type(type_filter)

    if query.strip():
        # Semantic search
        mem_types = [mem_type] if mem_type else None
        results = s.recall(query, limit=limit, memory_types=mem_types)
        memories = results.memories
        total = len(memories)
    else:
        # List with filter
        memories = s.list_memories(limit=limit, offset=offset, memory_type=mem_type)
        stats = s.get_stats()
        total = stats.get("total_memories", 0)

    total_pages = max(1, (total + limit - 1) // limit)

    return templates.TemplateResponse(
        "partials/memory_table.html",
        {
            "request": request,
            "memories": memories,
            "query": query,
            "type_filter": type_filter,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
        },
    )


@app.post("/api/memories/{memory_id}/promote", response_class=HTMLResponse)
async def api_promote(memory_id: int, request: Request) -> HTMLResponse:
    """Promote a memory to hot cache."""
    s = get_storage()
    s.promote_to_hot(memory_id)
    memory = s.get_memory(memory_id)
    return templates.TemplateResponse(
        "partials/memory_row.html",
        {"request": request, "memory": memory, "is_hot": True},
    )


@app.delete("/api/memories/{memory_id}", response_class=HTMLResponse)
async def api_delete(memory_id: int, request: Request) -> HTMLResponse:
    """Delete a memory."""
    s = get_storage()
    s.delete_memory(memory_id)
    return HTMLResponse(content="")


@app.get("/api/hot-cache", response_class=HTMLResponse)
async def api_hot_cache_list(request: Request) -> HTMLResponse:
    """Return hot cache list partial for HTMX polling."""
    s = get_storage()
    hot_memories = s.get_hot_memories()

    # Check which are pinned
    hot_stats = s.get_hot_cache_stats()
    pinned_ids = set(hot_stats.get("pinned_ids", []))

    return templates.TemplateResponse(
        "partials/hot_list.html",
        {
            "request": request,
            "memories": hot_memories,
            "pinned_ids": pinned_ids,
        },
    )
