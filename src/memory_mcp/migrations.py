"""Database schema and migrations for Memory MCP.

This module contains the database schema definition and all migration functions.
Migrations are versioned and run incrementally when upgrading databases.
"""

import sqlite3

from memory_mcp.config import Settings
from memory_mcp.logging import get_logger

log = get_logger("migrations")

# Current schema version - increment when making breaking changes
SCHEMA_VERSION = 17

SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- All memories (hot + cold)
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    memory_type TEXT NOT NULL,
    source TEXT NOT NULL,
    is_hot INTEGER DEFAULT 0,
    is_pinned INTEGER DEFAULT 0,
    promotion_source TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Tags
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

-- Output log (7-day rolling)
CREATE TABLE IF NOT EXISTS output_log (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Mined patterns (candidates for promotion)
CREATE TABLE IF NOT EXISTS mined_patterns (
    id INTEGER PRIMARY KEY,
    pattern TEXT NOT NULL,
    pattern_hash TEXT UNIQUE,
    pattern_type TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Trust history (audit trail for trust changes)
CREATE TABLE IF NOT EXISTS trust_history (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    old_trust REAL NOT NULL,
    new_trust REAL NOT NULL,
    delta REAL NOT NULL,
    similarity REAL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Memory relationships (knowledge graph edges)
CREATE TABLE IF NOT EXISTS memory_relationships (
    id INTEGER PRIMARY KEY,
    from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_memory_id, to_memory_id, relation_type)
);

-- Sessions (conversation provenance tracking)
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,  -- UUID or transcript path hash
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TEXT DEFAULT CURRENT_TIMESTAMP,
    topic TEXT,  -- Auto-detected or user-provided topic
    project_path TEXT,  -- Working directory path
    memory_count INTEGER DEFAULT 0,
    log_count INTEGER DEFAULT 0
);

-- Key-value metadata (for embedding model tracking, etc.)
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Audit log for destructive operations
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    operation TEXT NOT NULL,
    target_type TEXT,
    target_id INTEGER,
    details TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_hot ON memories(is_hot);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_output_log_timestamp ON output_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_mined_patterns_hash ON mined_patterns(pattern_hash);
CREATE INDEX IF NOT EXISTS idx_trust_history_memory ON trust_history(memory_id);
CREATE INDEX IF NOT EXISTS idx_trust_history_reason ON trust_history(reason);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON memory_relationships(from_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON memory_relationships(to_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relation_type);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);
"""


def get_vector_schema(dim: int) -> str:
    """Generate vector schema with correct dimension."""
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    embedding FLOAT[{dim}]
);
"""


class SchemaVersionError(Exception):
    """Raised when database schema version is incompatible."""


class EmbeddingDimensionError(Exception):
    """Raised when embedding dimension doesn't match stored vectors."""


# ========== Migration Helper Functions ==========


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Get set of column names for a table."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> bool:
    """Add column if it doesn't exist. Returns True if added."""
    columns = get_table_columns(conn, table)
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        log.debug("Added {} column to {} table", column, table)
        return True
    return False


# ========== Individual Migrations ==========


def migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add is_pinned and promotion_source columns for hot cache LRU."""
    add_column_if_missing(conn, "memories", "is_pinned", "INTEGER DEFAULT 0")
    add_column_if_missing(conn, "memories", "promotion_source", "TEXT")


def migrate_v2_to_v3(conn: sqlite3.Connection, settings: Settings) -> None:
    """Add trust_score and provenance columns."""
    add_column_if_missing(conn, "memories", "trust_score", "REAL DEFAULT 1.0")
    add_column_if_missing(conn, "memories", "source_log_id", "INTEGER")
    add_column_if_missing(conn, "memories", "extracted_at", "TEXT")

    # Set initial trust scores based on source type
    conn.execute(
        """
        UPDATE memories
        SET trust_score = CASE
            WHEN source = 'manual' THEN ?
            WHEN source = 'mined' THEN ?
            ELSE 1.0
        END
        WHERE trust_score IS NULL OR trust_score = 1.0
        """,
        (settings.trust_score_manual, settings.trust_score_mined),
    )


def migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Add trust_history table for audit trail."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trust_history (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
            reason TEXT NOT NULL,
            old_trust REAL NOT NULL,
            new_trust REAL NOT NULL,
            delta REAL NOT NULL,
            similarity REAL,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trust_history_memory ON trust_history(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trust_history_reason ON trust_history(reason)")
    log.info("Created trust_history table for audit trail")


def migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Add memory_relationships table for knowledge graph."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_relationships (
            id INTEGER PRIMARY KEY,
            from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(from_memory_id, to_memory_id, relation_type)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_from ON memory_relationships(from_memory_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_to ON memory_relationships(to_memory_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relation_type)"
    )
    log.info("Created memory_relationships table for knowledge graph")


def migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Add sessions table and session_id columns for conversation provenance."""
    # Create sessions table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TEXT DEFAULT CURRENT_TIMESTAMP,
            topic TEXT,
            project_path TEXT,
            memory_count INTEGER DEFAULT 0,
            log_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity_at)")

    # Add session_id to memories table
    add_column_if_missing(conn, "memories", "session_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")

    # Add session_id to output_log table
    add_column_if_missing(conn, "output_log", "session_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_output_log_session ON output_log(session_id)")

    log.info("Created sessions table and session tracking columns")


def migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Add access_sequences table for predictive hot cache warming."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS access_sequences (
            from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            count INTEGER DEFAULT 1,
            last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (from_memory_id, to_memory_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sequences_from ON access_sequences(from_memory_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sequences_last ON access_sequences(last_seen)")
    log.info("Created access_sequences table for predictive cache")


def migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    """Enhance mined_patterns table for mining quality improvements."""
    # Add status column for approval workflow
    add_column_if_missing(conn, "mined_patterns", "status", "TEXT DEFAULT 'pending'")
    # Add source_log_id for provenance tracking
    add_column_if_missing(
        conn, "mined_patterns", "source_log_id", "INTEGER REFERENCES output_log(id)"
    )
    # Add confidence score from extraction
    add_column_if_missing(conn, "mined_patterns", "confidence", "REAL DEFAULT 0.5")
    # Add computed promotion score
    add_column_if_missing(conn, "mined_patterns", "score", "REAL DEFAULT 0.0")
    # Index for filtering by status
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mined_patterns_status ON mined_patterns(status)")
    log.info("Enhanced mined_patterns table with status, provenance, and scoring")


def migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    """Add audit_log table for destructive operation tracking."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            operation TEXT NOT NULL,
            target_type TEXT,
            target_id INTEGER,
            details TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation)")
    log.info("Created audit_log table for destructive operation tracking")


def migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    """Add importance scoring and retrieval tracking (research-inspired features).

    - importance_score: MemGPT-inspired admission-time content scoring
    - retrieval_events: RAG-inspired tracking of recall usage
    """
    # Add importance_score column to memories
    add_column_if_missing(conn, "memories", "importance_score", "REAL DEFAULT 0.5")

    # Create retrieval tracking table (RAG-inspired)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_events (
            id INTEGER PRIMARY KEY,
            query_hash TEXT NOT NULL,
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            similarity REAL NOT NULL,
            was_used INTEGER DEFAULT 0,
            feedback TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_query ON retrieval_events(query_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_memory ON retrieval_events(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_created ON retrieval_events(created_at)")
    log.info("Added importance_score column and retrieval_events table")


def migrate_v10_to_v11(conn: sqlite3.Connection) -> None:
    """Add project awareness - project_id column and index."""
    # Add project_id column to memories (NULL = global/no project)
    add_column_if_missing(conn, "memories", "project_id", "TEXT")

    # Create index for project-based queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id)")

    # Create projects table for tracking project metadata
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT,
            path TEXT,
            last_accessed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_last_accessed ON projects(last_accessed_at)"
    )

    log.info("Added project awareness (project_id column and projects table)")


def migrate_v11_to_v12(conn: sqlite3.Connection) -> None:
    """Add FTS5 full-text search for hybrid semantic+keyword search.

    Creates an FTS5 virtual table for keyword-based matching, improving recall
    for queries using indirect phrasings or named entities (e.g., "FastAPI"
    won't semantically match "framework" but will keyword-match).
    """
    # Create FTS5 virtual table for memory content
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            content,
            content='memories',
            content_rowid='id',
            tokenize='porter unicode61 remove_diacritics 2'
        )
        """
    )

    # Populate FTS table with existing memories
    conn.execute(
        """
        INSERT OR IGNORE INTO memory_fts(rowid, content)
        SELECT id, content FROM memories
        """
    )

    # Create triggers to keep FTS in sync with memories table
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
        END
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE OF content ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
        END
        """
    )

    log.info("Created memory_fts table and triggers for hybrid keyword search")


def migrate_v12_to_v13(conn: sqlite3.Connection) -> None:
    """Add project_id to output_log for project-scoped mining.

    Without this, mining would process logs from all projects regardless
    of which project is currently active, potentially leaking patterns
    across project boundaries.
    """
    # Add project_id column to output_log
    add_column_if_missing(conn, "output_log", "project_id", "TEXT")

    # Create index for project-based filtering
    conn.execute("CREATE INDEX IF NOT EXISTS idx_output_log_project ON output_log(project_id)")

    log.info("Added project_id to output_log for project-scoped mining")


def migrate_v13_to_v14(conn: sqlite3.Connection) -> None:
    """Add category column for memory subcategorization.

    Allows subcategories within memory types, e.g.:
    - project + category="decision" for design decisions
    - project + category="architecture" for architecture notes
    - pattern + category="import" for import patterns
    """
    add_column_if_missing(conn, "memories", "category", "TEXT")

    # Create index for category-based filtering
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")

    log.info("Added category column for memory subcategorization")


def migrate_v14_to_v15(conn: sqlite3.Connection) -> None:
    """Add injection_log table for tracking hot cache injections.

    Tracks which memories were injected via hot-cache/working-set resources
    to enable feedback loop analysis (injection → used correlation).
    Retention: 7 days, cleaned during maintenance.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS injection_log (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            resource TEXT NOT NULL,
            injected_at TEXT DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            project_id TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_injection_memory ON injection_log(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_injection_time ON injection_log(injected_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_injection_resource ON injection_log(resource)")

    log.info("Created injection_log table for tracking hot cache injections")


def migrate_v15_to_v16(conn: sqlite3.Connection) -> None:
    """Add helpfulness tracking columns for feedback loop.

    New columns on memories table:
    - retrieved_count: Times memory was returned in recall results
    - used_count: Times memory was marked as actually helpful
    - last_used_at: When memory was last marked as used
    - utility_score: Precomputed Bayesian helpfulness score

    These enable:
    - Tracking injection → used correlation
    - Bayesian helpfulness for promotion criteria
    - Helpfulness-weighted recall ranking
    """
    add_column_if_missing(conn, "memories", "retrieved_count", "INTEGER DEFAULT 0")
    add_column_if_missing(conn, "memories", "used_count", "INTEGER DEFAULT 0")
    add_column_if_missing(conn, "memories", "last_used_at", "TEXT")
    add_column_if_missing(conn, "memories", "utility_score", "REAL DEFAULT 0.25")

    # Index for ordering by last_used_at (session recency)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at)")

    log.info("Added helpfulness tracking columns (v16)")


def migrate_v16_to_v17(conn: sqlite3.Connection) -> None:
    """Add memory_id to mined_patterns for exact-match promotion.

    When patterns are approved and stored as memories, link the memory_id
    back to the pattern so promotion can use exact match instead of
    semantic search (which can miss short patterns or match wrong memories).
    """
    add_column_if_missing(
        conn, "mined_patterns", "memory_id", "INTEGER REFERENCES memories(id) ON DELETE SET NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mined_patterns_memory ON mined_patterns(memory_id)"
    )
    log.info("Added memory_id column to mined_patterns for exact-match promotion (v17)")


# ========== Migration Runner ==========


def run_migrations(conn: sqlite3.Connection, from_version: int, settings: Settings) -> None:
    """Run schema migrations from from_version to SCHEMA_VERSION."""
    if from_version < 2:
        migrate_v1_to_v2(conn)
    if from_version < 3:
        migrate_v2_to_v3(conn, settings)
    if from_version < 4:
        migrate_v3_to_v4(conn)
    if from_version < 5:
        migrate_v4_to_v5(conn)
    if from_version < 6:
        migrate_v5_to_v6(conn)
    if from_version < 7:
        migrate_v6_to_v7(conn)
    if from_version < 8:
        migrate_v7_to_v8(conn)
    if from_version < 9:
        migrate_v8_to_v9(conn)
    if from_version < 10:
        migrate_v9_to_v10(conn)
    if from_version < 11:
        migrate_v10_to_v11(conn)
    if from_version < 12:
        migrate_v11_to_v12(conn)
    if from_version < 13:
        migrate_v12_to_v13(conn)
    if from_version < 14:
        migrate_v13_to_v14(conn)
    if from_version < 15:
        migrate_v14_to_v15(conn)
    if from_version < 16:
        migrate_v15_to_v16(conn)
    if from_version < 17:
        migrate_v16_to_v17(conn)


def check_schema_version(conn: sqlite3.Connection) -> None:
    """Check schema version compatibility.

    Raises:
        SchemaVersionError: If database schema is newer than supported version.
    """
    # Check if schema_version table exists
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()

    if not table_exists:
        # New database or pre-versioning database
        # Check if other tables exist (pre-versioning database)
        memories_exist = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if memories_exist:
            log.info("Upgrading pre-versioning database to version {}", SCHEMA_VERSION)
        return

    # Get current version
    current_version = conn.execute(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()

    if current_version and current_version[0] > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Database schema version {current_version[0]} is newer than "
            f"supported version {SCHEMA_VERSION}. Please upgrade memory-mcp."
        )
