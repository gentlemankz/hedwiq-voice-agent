"""
Persistent Document Store for Luframe Agent

Provides persistent storage for documents with room scoping and TTL.
Supports both SQLite (for development/single-instance) and Redis
(for production/multi-instance) backends.

Key Features:
- Room-scoped document storage
- TTL-based automatic cleanup
- Max limits enforcement
- Embedding storage with numpy serialization

Usage:
    # SQLite (default)
    store = PersistentDocumentStore(backend="sqlite")

    # Redis
    store = PersistentDocumentStore(backend="redis", redis_url="redis://localhost:6379")

    # Add document
    doc_id = store.add_document(
        room_id="room-123",
        filename="report.pdf",
        title="Q4 Report",
        summary="Quarterly financial report...",
        page_count=10,
        segments=[...],
        embeddings=np.array([...]),
        pdf_data=pdf_bytes
    )
"""

import json
import sqlite3
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
import threading

import numpy as np

from schemas.documents import (
    MAX_DOCUMENTS_PER_ROOM,
    MAX_SEGMENTS_PER_DOCUMENT,
    DOCUMENT_TTL_HOURS,
)

if TYPE_CHECKING:
    from hybrid_retriever import RoomRetrieverManager

logger = logging.getLogger("luframe-document-store")


def _validate_doc_id(doc_id: str) -> str:
    """
    Validate/sanitize a document ID to prevent path traversal or overwrite.

    Allows alphanumerics, dash, underscore, and dot. Rejects slashes and '..'.
    """
    import re

    if not doc_id:
        raise ValueError("Document ID cannot be empty")

    if "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise ValueError("Invalid document ID")

    if not re.match(r"^[A-Za-z0-9._-]+$", doc_id):
        raise ValueError("Document ID contains invalid characters")

    return doc_id


@dataclass
class StoredDocument:
    """Internal representation of a stored document."""
    id: str
    room_id: str
    filename: str
    title: str
    summary: str
    page_count: int
    segments: List[dict]
    embeddings: List[List[float]]  # Stored as lists for JSON serialization
    created_at: int
    uploaded_by: str
    pdf_path: Optional[str] = None  # Path to stored PDF file


class PersistentDocumentStore:
    """
    Persistent document storage with room scoping and TTL.

    Key improvements over in-memory storage:
    - Survives agent restarts
    - Room-scoped isolation
    - TTL for automatic cleanup
    - Max limits enforced
    - PDF file storage

    Supports SQLite (default) and Redis backends.
    """

    def __init__(
        self,
        backend: str = "sqlite",
        redis_url: Optional[str] = None,
        db_path: Optional[str] = None,
        storage_dir: Optional[str] = None
    ):
        """
        Initialize document store.

        Args:
            backend: "sqlite" or "redis"
            redis_url: Redis connection URL (for redis backend)
            db_path: SQLite database path (for sqlite backend)
            storage_dir: Directory for PDF file storage
        """
        self.backend = backend
        self._lock = threading.Lock()

        # Set up storage directory for PDFs
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path(__file__).parent / "document_storage"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if backend == "redis":
            self._init_redis(redis_url)
        else:
            self.db_path = db_path or str(Path(__file__).parent / "documents.db")
            self._init_sqlite()

    def _init_redis(self, redis_url: Optional[str]):
        """Initialize Redis backend."""
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis is required for redis backend. "
                "Install it with: pip install redis"
            )

        self.redis = redis.from_url(redis_url or "redis://localhost:6379")
        logger.info("Initialized Redis document store")

    def _init_sqlite(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_room ON documents(room_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at)")
        conn.commit()
        conn.close()
        logger.info(f"Initialized SQLite document store at {self.db_path}")

    def generate_document_id(self, room_id: str) -> str:
        """
        Generate a unique document ID for a room.

        This should be called BEFORE creating segments so the segments
        have the correct document_id from the start.

        Args:
            room_id: LiveKit room ID

        Returns:
            Document ID string

        Raises:
            ValueError: If room limit exceeded
        """
        with self._lock:
            existing = self.get_documents_for_room(room_id)
            if len(existing) >= MAX_DOCUMENTS_PER_ROOM:
                raise ValueError(f"Max {MAX_DOCUMENTS_PER_ROOM} documents per room")
            return f"doc-{int(time.time())}-{len(existing)}"

    def add_document(
        self,
        room_id: str,
        filename: str,
        title: str,
        summary: str,
        page_count: int,
        segments: List[dict],
        embeddings: np.ndarray,
        uploaded_by: str,
        pdf_data: Optional[bytes] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a processed document to the store.

        Args:
            room_id: LiveKit room ID
            filename: Original filename
            title: Document title
            summary: Document summary
            page_count: Number of pages
            segments: List of segment dictionaries
            embeddings: numpy array of embeddings
            uploaded_by: User ID who uploaded
            pdf_data: Optional PDF file content
            doc_id: Optional pre-generated document ID (from generate_document_id)

        Returns:
            Document ID

        Raises:
            ValueError: If room limit exceeded
        """
        with self._lock:
            # Check room limits
            existing = self.get_documents_for_room(room_id)
            if len(existing) >= MAX_DOCUMENTS_PER_ROOM:
                raise ValueError(f"Max {MAX_DOCUMENTS_PER_ROOM} documents per room")

            # Limit segments
            segments = segments[:MAX_SEGMENTS_PER_DOCUMENT]

            # Use provided doc_id or generate new one
            if doc_id is None:
                doc_id = f"doc-{int(time.time())}-{len(existing)}"

            # Sanitize/validate doc_id to prevent path traversal
            doc_id = _validate_doc_id(doc_id)

            # Store PDF file if provided
            pdf_path = None
            if pdf_data:
                pdf_path = str(self.storage_dir / f"{doc_id}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(pdf_data)

            doc = StoredDocument(
                id=doc_id,
                room_id=room_id,
                filename=filename,
                title=title,
                summary=summary,
                page_count=page_count,
                segments=segments,
                embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                created_at=int(time.time() * 1000),
                uploaded_by=uploaded_by,
                pdf_path=pdf_path
            )

            if self.backend == "redis":
                self._add_redis(doc)
            else:
                self._add_sqlite(doc)

            logger.info(f"Added document {doc_id} to room {room_id}")
            return doc_id

    def _add_redis(self, doc: StoredDocument):
        """Add document to Redis."""
        key = f"luframe:doc:{doc.room_id}:{doc.id}"
        ttl_seconds = DOCUMENT_TTL_HOURS * 3600
        self.redis.setex(
            key,
            ttl_seconds,
            json.dumps(asdict(doc))
        )

    def _add_sqlite(self, doc: StoredDocument):
        """Add document to SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO documents (id, room_id, data, created_at) VALUES (?, ?, ?, ?)",
            (doc.id, doc.room_id, json.dumps(asdict(doc)), doc.created_at)
        )
        conn.commit()
        conn.close()

    def get_documents_for_room(self, room_id: str) -> List[StoredDocument]:
        """
        Get all documents for a room.

        Args:
            room_id: LiveKit room ID

        Returns:
            List of StoredDocument objects
        """
        if self.backend == "redis":
            return self._get_docs_redis(room_id)
        else:
            return self._get_docs_sqlite(room_id)

    def _get_docs_redis(self, room_id: str) -> List[StoredDocument]:
        """Get documents from Redis."""
        pattern = f"luframe:doc:{room_id}:*"
        keys = self.redis.keys(pattern)
        docs = []
        for key in keys:
            data = self.redis.get(key)
            if data:
                doc_dict = json.loads(data)
                docs.append(StoredDocument(**doc_dict))
        return docs

    def _get_docs_sqlite(self, room_id: str) -> List[StoredDocument]:
        """Get documents from SQLite."""
        # Clean up expired documents first
        self.cleanup_expired()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT data FROM documents WHERE room_id = ?",
            (room_id,)
        )
        docs = []
        for row in cursor.fetchall():
            doc_dict = json.loads(row[0])
            docs.append(StoredDocument(**doc_dict))
        conn.close()
        return docs

    def get_document(self, room_id: str, doc_id: str) -> Optional[StoredDocument]:
        """
        Get a specific document.

        Args:
            room_id: LiveKit room ID
            doc_id: Document ID

        Returns:
            StoredDocument or None if not found
        """
        if self.backend == "redis":
            return self._get_doc_redis(room_id, doc_id)
        else:
            return self._get_doc_sqlite(room_id, doc_id)

    def _get_doc_redis(self, room_id: str, doc_id: str) -> Optional[StoredDocument]:
        """Get document from Redis."""
        key = f"luframe:doc:{room_id}:{doc_id}"
        data = self.redis.get(key)
        if data:
            return StoredDocument(**json.loads(data))
        return None

    def _get_doc_sqlite(self, room_id: str, doc_id: str) -> Optional[StoredDocument]:
        """Get document from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT data FROM documents WHERE id = ? AND room_id = ?",
            (doc_id, room_id)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return StoredDocument(**json.loads(row[0]))
        return None

    def get_all_segments_for_room(self, room_id: str) -> List[dict]:
        """
        Get all segments for all documents in a room.

        Args:
            room_id: LiveKit room ID

        Returns:
            List of segment dictionaries
        """
        docs = self.get_documents_for_room(room_id)
        all_segments = []
        for doc in docs:
            all_segments.extend(doc.segments)
        return all_segments

    def get_embeddings_for_room(self, room_id: str) -> np.ndarray:
        """
        Get all embeddings for a room as numpy array.

        Args:
            room_id: LiveKit room ID

        Returns:
            numpy array of shape (num_segments, embedding_dim)
        """
        docs = self.get_documents_for_room(room_id)
        all_embeddings = []
        for doc in docs:
            all_embeddings.extend(doc.embeddings)
        return np.array(all_embeddings) if all_embeddings else np.array([])

    def get_pdf_data(self, room_id: str, doc_id: str) -> Optional[bytes]:
        """
        Get PDF file content for a document.

        Args:
            room_id: LiveKit room ID
            doc_id: Document ID

        Returns:
            PDF bytes or None if not found
        """
        doc = self.get_document(room_id, doc_id)
        if doc and doc.pdf_path and os.path.exists(doc.pdf_path):
            with open(doc.pdf_path, "rb") as f:
                return f.read()
        return None

    def remove_document(self, room_id: str, doc_id: str):
        """
        Remove a document.

        Args:
            room_id: LiveKit room ID
            doc_id: Document ID
        """
        with self._lock:
            # Get document to find PDF path
            doc = self.get_document(room_id, doc_id)

            if self.backend == "redis":
                key = f"luframe:doc:{room_id}:{doc_id}"
                self.redis.delete(key)
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "DELETE FROM documents WHERE id = ? AND room_id = ?",
                    (doc_id, room_id)
                )
                conn.commit()
                conn.close()

            # Remove PDF file
            if doc and doc.pdf_path and os.path.exists(doc.pdf_path):
                try:
                    os.remove(doc.pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to remove PDF file: {e}")

            logger.info(f"Removed document {doc_id} from room {room_id}")

    def clear_room(self, room_id: str):
        """
        Clear all documents for a room.

        Args:
            room_id: LiveKit room ID
        """
        with self._lock:
            # Get documents to find PDF paths
            docs = self.get_documents_for_room(room_id)

            if self.backend == "redis":
                pattern = f"luframe:doc:{room_id}:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute("DELETE FROM documents WHERE room_id = ?", (room_id,))
                conn.commit()
                conn.close()

            # Remove PDF files
            for doc in docs:
                if doc.pdf_path and os.path.exists(doc.pdf_path):
                    try:
                        os.remove(doc.pdf_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove PDF file: {e}")

            logger.info(f"Cleared all documents for room {room_id}")

    def cleanup_expired(self):
        """
        Remove expired documents (for SQLite backend).

        Redis handles TTL automatically.
        """
        if self.backend == "redis":
            return  # Redis handles TTL

        cutoff = int(time.time() * 1000) - (DOCUMENT_TTL_HOURS * 3600 * 1000)

        conn = sqlite3.connect(self.db_path)

        # Get expired documents to find PDF paths
        cursor = conn.execute(
            "SELECT data FROM documents WHERE created_at < ?",
            (cutoff,)
        )
        for row in cursor.fetchall():
            try:
                doc_dict = json.loads(row[0])
                pdf_path = doc_dict.get("pdf_path")
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to clean up PDF: {e}")

        # Delete expired documents
        conn.execute("DELETE FROM documents WHERE created_at < ?", (cutoff,))
        conn.commit()
        conn.close()

    def get_document_count(self, room_id: str) -> int:
        """
        Get number of documents in a room.

        Args:
            room_id: LiveKit room ID

        Returns:
            Number of documents
        """
        return len(self.get_documents_for_room(room_id))

    def document_exists(self, room_id: str, doc_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            room_id: LiveKit room ID
            doc_id: Document ID

        Returns:
            True if document exists
        """
        return self.get_document(room_id, doc_id) is not None
