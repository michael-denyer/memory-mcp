"""Core Storage class combining all mixins."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from memory_mcp.config import Settings, ensure_data_dir, get_settings
from memory_mcp.embeddings import EmbeddingEngine
from memory_mcp.logging import get_logger
from memory_mcp.migrations import (
    SCHEMA,
    SCHEMA_VERSION,
    EmbeddingDimensionError,
    SchemaVersionError,
    check_schema_version,
    get_vector_schema,
    run_migrations,
)
from memory_mcp.models import HotCacheMetrics

# Import all mixins
from memory_mcp.storage.audit import AuditMixin
from memory_mcp.storage.bootstrap import BootstrapMixin
from memory_mcp.storage.consolidation import ConsolidationMixin
from memory_mcp.storage.contradictions import ContradictionsMixin
from memory_mcp.storage.hot_cache import HotCacheMixin
from memory_mcp.storage.injection_tracking import InjectionTrackingMixin
from memory_mcp.storage.maintenance import MaintenanceMixin
from memory_mcp.storage.memory_crud import MemoryCrudMixin, ValidationError
from memory_mcp.storage.mining_store import MiningStoreMixin
from memory_mcp.storage.output_logging import OutputLoggingMixin
from memory_mcp.storage.predictions import PredictionsMixin
from memory_mcp.storage.relationships import RelationshipsMixin
from memory_mcp.storage.retrieval import RetrievalMixin
from memory_mcp.storage.search import SearchMixin
from memory_mcp.storage.sessions import SessionsMixin
from memory_mcp.storage.trust import TrustMixin

log = get_logger("storage")


class Storage(
    AuditMixin,
    TrustMixin,
    MiningStoreMixin,
    MaintenanceMixin,
    RetrievalMixin,
    RelationshipsMixin,
    ContradictionsMixin,
    SessionsMixin,
    HotCacheMixin,
    ConsolidationMixin,
    MemoryCrudMixin,
    SearchMixin,
    PredictionsMixin,
    BootstrapMixin,
    OutputLoggingMixin,
    InjectionTrackingMixin,
):
    """SQLite storage manager with thread-safe connection handling.

    Combines functionality from all mixins:
    - AuditMixin: Audit logging for destructive operations
    - TrustMixin: Trust score management and history
    - MiningStoreMixin: Mined pattern storage
    - MaintenanceMixin: Database maintenance operations
    - RetrievalMixin: RAG-inspired retrieval tracking
    - RelationshipsMixin: Knowledge graph relationships
    - ContradictionsMixin: Contradiction detection
    - SessionsMixin: Session management and episodic memory
    - HotCacheMixin: Hot cache operations
    - ConsolidationMixin: Memory consolidation
    - MemoryCrudMixin: Memory CRUD operations
    - SearchMixin: Vector search and recall
    - PredictionsMixin: Predictive cache warming
    - BootstrapMixin: Bootstrap from files
    - OutputLoggingMixin: Output logging for mining
    - InjectionTrackingMixin: Track hot cache/working set injections
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        # Use settings-aware embedding engine, not global singleton
        self._embedding_engine = EmbeddingEngine(self.settings)
        self._hot_cache_metrics: HotCacheMetrics | None = None  # Lazy-loaded from DB
        log.info("Storage initialized with db_path={}", self.settings.db_path)

    @property
    def db_path(self) -> Path:
        return self.settings.db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection. Must be called with lock held."""
        if self._conn is None:
            ensure_data_dir(self.settings)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,  # Wait up to 30s for locks
            )
            self._conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            self._conn.execute("PRAGMA foreign_keys=ON")  # Enable cascade deletes

            # Load sqlite-vec extension
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            # Initialize schema
            self._init_schema()
        return self._conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for read operations with thread safety."""
        with self._lock:
            yield self._get_connection()

    def _init_schema(self) -> None:
        """Initialize database schema with version tracking."""
        conn = self._conn
        if conn is None:
            return

        # Check existing schema version before applying migrations
        check_schema_version(conn)

        # Apply base schema first (uses IF NOT EXISTS, safe to re-run)
        conn.executescript(SCHEMA)
        conn.execute(get_vector_schema(self.settings.embedding_dim))

        # Get current version for migrations (now table exists)
        existing_version = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        current_version = existing_version[0] if existing_version else 0

        # Run migrations
        run_migrations(conn, current_version, self.settings)

        # Record new schema version
        if current_version < SCHEMA_VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            log.info("Database migrated from v{} to v{}", current_version, SCHEMA_VERSION)

        conn.commit()

        # Validate embedding dimension (fail fast, not just warn)
        self._validate_vector_dimension(conn)
        log.debug("Database schema initialized (version={})", SCHEMA_VERSION)

    def _validate_vector_dimension(self, conn: sqlite3.Connection) -> None:
        """Check that existing vector table matches configured dimension. Fails fast on mismatch."""
        result = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
        ).fetchone()
        if result and result[0]:
            schema_sql = result[0]
            expected_dim = self.settings.embedding_dim

            # Check if dimension in schema matches
            if f"FLOAT[{expected_dim}]" not in schema_sql.upper():
                # Check if there are any existing vectors
                count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]

                if count > 0:
                    raise EmbeddingDimensionError(
                        f"Embedding dimension mismatch: database has {count} vectors with "
                        f"different dimension than configured ({expected_dim}).\n\n"
                        f"To fix this, use one of these options:\n"
                        f"  1. CLI: memory-mcp-cli db-rebuild-vectors\n"
                        f"  2. MCP tool: db_rebuild_vectors()\n"
                        f"  3. Set MEMORY_MCP_EMBEDDING_DIM to match existing vectors\n"
                        f"  4. Delete the database: {self.db_path}"
                    )
                else:
                    log.warning(
                        "Vector table dimension mismatch but no data. Consider recreating: {}",
                        self.db_path,
                    )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for transactions with thread safety."""
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_schema_version(self) -> int:
        """Get current database schema version."""
        with self._connection() as conn:
            result = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return result[0] if result else 0


# Re-export for backwards compatibility
__all__ = [
    "Storage",
    "ValidationError",
    "SchemaVersionError",
    "EmbeddingDimensionError",
]
