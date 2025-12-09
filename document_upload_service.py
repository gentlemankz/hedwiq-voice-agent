"""
Document upload service extracted from persistent_store to reduce file size.
Handles PDF parsing, embeddings, summaries, storage, and index rebuild notifications.
"""

import asyncio
import logging
from typing import Optional

from schemas.documents import MAX_DOCUMENTS_PER_ROOM
from persistent_store import PersistentDocumentStore

logger = logging.getLogger("hedwiq-document-store")


class DocumentUploadService:
    """High-level service for document upload and processing."""

    def __init__(
        self,
        store: Optional[PersistentDocumentStore] = None,
        embedding_model: str = "text-embedding-3-large",
        summary_model: str = "gpt-4o-mini",
        retriever_manager: Optional["RoomRetrieverManager"] = None,
    ):
        from document_processor import PDFProcessor, EmbeddingGenerator, DocumentSummarizer

        self.store = store or PersistentDocumentStore()
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.summarizer = DocumentSummarizer(model_name=summary_model)
        self._retriever_manager = retriever_manager

    def set_retriever_manager(self, manager: "RoomRetrieverManager"):
        self._retriever_manager = manager

    def _notify_index_rebuild(self, room_id: str):
        if self._retriever_manager:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Not in an event loop; rebuild synchronously
                    self._retriever_manager.rebuild_room_index(room_id)
                    logger.info(f"Triggered retrieval index rebuild for room {room_id}")
                else:
                    loop.create_task(asyncio.to_thread(self._retriever_manager.rebuild_room_index, room_id))
                    logger.info(f"Scheduled retrieval index rebuild for room {room_id}")
            except Exception as e:
                logger.error(f"Failed to rebuild retrieval index for room {room_id}: {e}")

    async def upload_document(
        self,
        room_id: str,
        filename: str,
        pdf_data: bytes,
        uploaded_by: str,
        doc_id: Optional[str] = None,
    ) -> dict:
        if doc_id is None:
            try:
                doc_id = self.store.generate_document_id(room_id)
            except ValueError as e:
                raise e
        else:
            existing_count = len(self.store.get_documents_for_room(room_id))
            if existing_count >= MAX_DOCUMENTS_PER_ROOM:
                raise ValueError(f"Max {MAX_DOCUMENTS_PER_ROOM} documents per room")
            logger.info(f"Using frontend-provided document ID: {doc_id}")

        try:
            pages = await asyncio.to_thread(
                self.pdf_processor.parse_pdf_from_bytes,
                pdf_data,
                filename,
            )
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")

        if not pages:
            raise ValueError("PDF contains no readable content")

        title = self.pdf_processor.extract_title(pages)
        segments = self.pdf_processor.segment_document(pages, doc_id)
        if not segments:
            raise ValueError("Could not extract any segments from PDF")

        segment_dicts = [s.to_dict() for s in segments]

        try:
            embeddings = await asyncio.to_thread(
                self.embedding_generator.generate_embeddings,
                segments,
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}")

        full_text = " ".join(page.text for page in pages)
        try:
            summary = await self.summarizer.generate_summary(title, full_text)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            summary = full_text[:500] + "..."

        try:
            stored_doc_id = await asyncio.to_thread(
                self.store.add_document,
                room_id,
                filename,
                title,
                summary,
                len(pages),
                segment_dicts,
                embeddings,
                uploaded_by,
                pdf_data,
                doc_id,
            )
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise ValueError(f"Failed to store document: {e}")

        self._notify_index_rebuild(room_id)

        return {
            "documentId": stored_doc_id,
            "title": title,
            "pageCount": len(pages),
            "segmentCount": len(segments),
            "status": "ready",
        }

    def upload_document_sync(
        self,
        room_id: str,
        filename: str,
        pdf_data: bytes,
        uploaded_by: str,
        doc_id: Optional[str] = None,
    ) -> dict:
        if doc_id is None:
            try:
                doc_id = self.store.generate_document_id(room_id)
            except ValueError as e:
                raise e
        else:
            existing_count = len(self.store.get_documents_for_room(room_id))
            if existing_count >= MAX_DOCUMENTS_PER_ROOM:
                raise ValueError(f"Max {MAX_DOCUMENTS_PER_ROOM} documents per room")
            logger.info(f"Using frontend-provided document ID: {doc_id}")

        try:
            pages = self.pdf_processor.parse_pdf_from_bytes(pdf_data, filename)
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")

        if not pages:
            raise ValueError("PDF contains no readable content")

        title = self.pdf_processor.extract_title(pages)
        segments = self.pdf_processor.segment_document(pages, doc_id)
        if not segments:
            raise ValueError("Could not extract any segments from PDF")

        segment_dicts = [s.to_dict() for s in segments]

        try:
            embeddings = self.embedding_generator.generate_embeddings(segments)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}")

        full_text = " ".join(page.text for page in pages)
        try:
            summary = self.summarizer.generate_summary_sync(title, full_text)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            summary = full_text[:500] + "..."

        try:
            stored_doc_id = self.store.add_document(
                room_id=room_id,
                filename=filename,
                title=title,
                summary=summary,
                page_count=len(pages),
                segments=segment_dicts,
                embeddings=embeddings,
                uploaded_by=uploaded_by,
                pdf_data=pdf_data,
                doc_id=doc_id,
            )
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise ValueError(f"Failed to store document: {e}")

        self._notify_index_rebuild(room_id)

        return {
            "documentId": stored_doc_id,
            "title": title,
            "pageCount": len(pages),
            "segmentCount": len(segments),
            "status": "ready",
        }
