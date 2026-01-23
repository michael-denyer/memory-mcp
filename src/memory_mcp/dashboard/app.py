"""FastAPI application for the Memory MCP dashboard."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import fastmcp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from memory_mcp import __version__
from memory_mcp.config import get_settings
from memory_mcp.storage import MemorySource, MemoryType, PatternStatus, Storage

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


def get_type_value(memory_type: MemoryType | str) -> str:
    """Extract string value from MemoryType enum or pass through string."""
    if hasattr(memory_type, "value"):
        return memory_type.value
    return str(memory_type)


def get_type_badge_class(memory_type: MemoryType | str) -> str:
    """Get Tailwind classes for a memory type badge."""
    type_str = get_type_value(memory_type)
    return TYPE_COLORS.get(type_str, TYPE_COLORS["project"])[1]


# Add template globals and filters
templates.env.globals["get_type_badge_class"] = get_type_badge_class
templates.env.globals["format_bytes"] = format_bytes
templates.env.globals["version"] = __version__
templates.env.globals["mcp_version"] = fastmcp.__version__
templates.env.filters["type_value"] = get_type_value


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
    settings = get_settings()
    promoted_memories = s.get_promoted_memories()
    promoted_stats = s.get_promoted_stats()
    hot_cache = s.get_hot_cache()

    return templates.TemplateResponse(
        "hot_cache.html",
        {
            "request": request,
            "promoted_memories": promoted_memories,
            "promoted_stats": promoted_stats,
            "hot_cache": hot_cache,
            "hot_cache_enabled": settings.hot_cache_enabled,
            "promoted_resource_enabled": settings.promoted_resource_enabled,
            "active_page": "hot_cache",
        },
    )


def _parse_memory_type(type_str: str | None) -> MemoryType | None:
    """Parse a string to MemoryType enum or None."""
    if not type_str:
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

    return templates.TemplateResponse(
        "partials/hot_list.html",
        {
            "request": request,
            "memories": hot_memories,
        },
    )


# ============================================================================
# Mining Page and API
# ============================================================================


def _get_mining_stats(s: Storage) -> dict:
    """Get mining statistics."""
    # Count output logs
    output_count = s._conn.execute("SELECT COUNT(*) FROM output_log").fetchone()[0]
    # Count mined patterns by status
    pattern_stats = s._conn.execute(
        """
        SELECT status, COUNT(*) as count
        FROM mined_patterns
        GROUP BY status
        """
    ).fetchall()
    stats = {row["status"]: row["count"] for row in pattern_stats}
    return {
        "output_count": output_count,
        "total_patterns": sum(stats.values()),
        "pending_count": stats.get("pending", 0),
        "promoted_count": stats.get("promoted", 0),
        "rejected_count": stats.get("rejected", 0),
    }


@app.get("/mining", response_class=HTMLResponse)
async def mining_page(request: Request) -> HTMLResponse:
    """Pattern mining review page."""
    s = get_storage()
    mining_stats = _get_mining_stats(s)
    patterns = s.get_promotion_candidates(threshold=1, status=PatternStatus.PENDING)

    return templates.TemplateResponse(
        "mining.html",
        {
            "request": request,
            "mining_stats": mining_stats,
            "patterns": patterns[:50],  # Limit display
            "active_page": "mining",
        },
    )


@app.post("/api/mining/run", response_class=HTMLResponse)
async def api_run_mining(request: Request) -> HTMLResponse:
    """Run pattern mining and return results."""
    from memory_mcp.mining import run_mining

    s = get_storage()
    result = run_mining(s, hours=24)

    return HTMLResponse(
        content=f"""
        <div class="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-green-400">
            <p class="font-medium">Mining completed</p>
            <p class="text-sm mt-1">
                Processed {result["outputs_processed"]} outputs,
                found {result["patterns_found"]} patterns,
                created {result["new_memories"]} new memories
            </p>
        </div>
        """
    )


def _pattern_type_to_memory_type(pattern_type: str) -> MemoryType:
    """Map pattern type to memory type."""
    if pattern_type == "fact":
        return MemoryType.PROJECT
    if pattern_type == "command":
        return MemoryType.REFERENCE
    return MemoryType.PATTERN


@app.post("/api/mining/{pattern_id}/approve", response_class=HTMLResponse)
async def api_approve_pattern(pattern_id: int, request: Request) -> HTMLResponse:
    """Approve a mined pattern and promote to memory."""
    s = get_storage()
    pattern = s.get_mined_pattern(pattern_id)
    if not pattern:
        return HTMLResponse(content="")

    # Create a session for dashboard approvals
    session_id = str(uuid.uuid4())
    s.create_or_get_session(session_id, topic="Dashboard approval")

    mem_type = _pattern_type_to_memory_type(pattern.pattern_type)
    memory_id, _ = s.store_memory(
        content=pattern.pattern,
        memory_type=mem_type,
        source=MemorySource.MINED,
        tags=["approved"],
        session_id=session_id,
    )
    s.promote_to_hot(memory_id)
    s.delete_mined_pattern(pattern_id)

    return HTMLResponse(content="")


@app.post("/api/mining/{pattern_id}/reject", response_class=HTMLResponse)
async def api_reject_pattern(pattern_id: int, request: Request) -> HTMLResponse:
    """Reject a mined pattern."""
    s = get_storage()
    s.update_pattern_status(pattern_id, PatternStatus.REJECTED)
    return HTMLResponse(content="")


# ============================================================================
# Injection History Page and API
# ============================================================================


def _get_injection_stats(s: Storage) -> dict:
    """Get injection statistics."""
    conn = s._conn
    today = conn.execute(
        "SELECT COUNT(*) FROM injection_log WHERE injected_at >= date('now')"
    ).fetchone()[0]
    week = conn.execute(
        "SELECT COUNT(*) FROM injection_log WHERE injected_at >= date('now', '-7 days')"
    ).fetchone()[0]
    hot_cache = conn.execute(
        "SELECT COUNT(*) FROM injection_log WHERE resource = 'hot-cache'"
    ).fetchone()[0]
    working_set = conn.execute(
        "SELECT COUNT(*) FROM injection_log WHERE resource = 'working-set'"
    ).fetchone()[0]
    return {
        "today": today,
        "week": week,
        "hot_cache": hot_cache,
        "working_set": working_set,
    }


def _get_injections(
    s: Storage,
    days: int = 7,
    resource: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get recent injections with memory content."""
    conn = s._conn
    params: list = [f"-{days} days"]
    resource_filter = ""
    if resource:
        resource_filter = "AND il.resource = ?"
        params.append(resource)
    params.extend([limit, offset])

    rows = conn.execute(
        f"""
        SELECT il.id, il.memory_id, il.resource, il.injected_at, il.session_id,
               m.content
        FROM injection_log il
        LEFT JOIN memories m ON m.id = il.memory_id
        WHERE il.injected_at >= datetime('now', ?)
        {resource_filter}
        ORDER BY il.injected_at DESC
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


@app.get("/injections", response_class=HTMLResponse)
async def injections_page(
    request: Request,
    resource_filter: str | None = None,
    days: int = 7,
) -> HTMLResponse:
    """Injection history page."""
    s = get_storage()
    injection_stats = _get_injection_stats(s)
    injections = _get_injections(s, days=days, resource=resource_filter)

    return templates.TemplateResponse(
        "injections.html",
        {
            "request": request,
            "injection_stats": injection_stats,
            "injections": injections,
            "resource_filter": resource_filter,
            "days": days,
            "page": 1,
            "total_pages": 1,
            "active_page": "injections",
        },
    )


@app.get("/api/injections", response_class=HTMLResponse)
async def api_injections(
    request: Request,
    resource_filter: str | None = None,
    days: int = 7,
    page: int = 1,
    limit: int = 50,
) -> HTMLResponse:
    """Return injections table partial."""
    s = get_storage()
    offset = (page - 1) * limit
    injections = _get_injections(s, days=days, resource=resource_filter, limit=limit, offset=offset)

    # Rough count for pagination
    total = len(injections) if len(injections) < limit else limit * 2
    total_pages = max(1, (total + limit - 1) // limit)

    return templates.TemplateResponse(
        "partials/injection_table.html",
        {
            "request": request,
            "injections": injections,
            "page": page,
            "total_pages": total_pages,
        },
    )


# ============================================================================
# Sessions Page and API
# ============================================================================


def _get_sessions(s: Storage, limit: int = 50) -> list[dict]:
    """Get recent sessions with memory counts."""
    with s._connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id as session_id,
                s.project_path,
                s.topic,
                s.started_at as created_at,
                s.last_activity_at as ended_at,
                COUNT(m.id) as memory_count,
                SUM(CASE WHEN m.is_hot = 1 THEN 1 ELSE 0 END) as hot_count
            FROM sessions s
            LEFT JOIN memories m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def _get_session(s: Storage, session_id: str) -> dict | None:
    """Get a single session by ID."""
    with s._connection() as conn:
        row = conn.execute(
            """
            SELECT id as session_id, project_path, topic,
                   started_at as created_at, last_activity_at as ended_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


def _get_session_memories(s: Storage, session_id: str) -> list[dict]:
    """Get all memories for a session."""
    with s._connection() as conn:
        rows = conn.execute(
            """
            SELECT m.*
            FROM memories m
            WHERE m.session_id = ?
            ORDER BY m.created_at DESC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request) -> HTMLResponse:
    """Sessions list page."""
    s = get_storage()
    sessions = _get_sessions(s)

    return templates.TemplateResponse(
        "sessions.html",
        {
            "request": request,
            "sessions": sessions,
            "active_page": "sessions",
        },
    )


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail_page(session_id: str, request: Request) -> HTMLResponse:
    """Session detail page."""
    s = get_storage()
    session = _get_session(s, session_id)
    if not session:
        return HTMLResponse(content="Session not found", status_code=404)

    memories = _get_session_memories(s, session_id)

    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "session": session,
            "memories": memories,
            "active_page": "sessions",
        },
    )


# ============================================================================
# Stats History API (for charts)
# ============================================================================


@app.get("/api/stats/history")
async def api_stats_history(days: int = 30):
    """Get memory counts by day for time-series charts."""
    s = get_storage()
    with s._connection() as conn:
        rows = conn.execute(
            """
            SELECT
                date(created_at) as day,
                COUNT(*) as count,
                SUM(CASE WHEN is_hot = 1 THEN 1 ELSE 0 END) as hot_count
            FROM memories
            WHERE created_at >= date('now', ?)
            GROUP BY date(created_at)
            ORDER BY day
            """,
            (f"-{days} days",),
        ).fetchall()
        return {"days": [dict(row) for row in rows]}


# ============================================================================
# Knowledge Graph Page and API
# ============================================================================


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request) -> HTMLResponse:
    """Knowledge graph visualization page."""
    s = get_storage()
    stats = s.get_relationship_stats()

    return templates.TemplateResponse(
        "graph.html",
        {
            "request": request,
            "stats": stats,
            "active_page": "graph",
        },
    )


@app.get("/api/graph")
async def api_graph_data(limit: int = 100):
    """Get knowledge graph data for visualization."""
    s = get_storage()
    with s._connection() as conn:
        nodes_rows = conn.execute(
            """
            SELECT DISTINCT m.id, m.content, m.memory_type
            FROM memories m
            WHERE m.id IN (
                SELECT from_memory_id FROM memory_relationships
                UNION
                SELECT to_memory_id FROM memory_relationships
            )
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        edges_rows = conn.execute(
            """
            SELECT from_memory_id, to_memory_id, relation_type
            FROM memory_relationships
            LIMIT ?
            """,
            (limit * 2,),
        ).fetchall()

    nodes = [
        {
            "id": row["id"],
            "label": row["content"][:40] + "..." if len(row["content"]) > 40 else row["content"],
            "type": row["memory_type"],
        }
        for row in nodes_rows
    ]

    edges = [
        {
            "from": row["from_memory_id"],
            "to": row["to_memory_id"],
            "type": row["relation_type"],
        }
        for row in edges_rows
    ]

    return {"nodes": nodes, "edges": edges}
