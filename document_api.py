"""
Document Upload API for Hedwiq Agent

A FastAPI server that handles document upload and processing.
Run alongside the main LiveKit agent to provide HTTP endpoints
for document operations.

Usage:
    # Run with uvicorn
    uvicorn document_api:app --host 0.0.0.0 --port 8000

    # Or run directly
    python document_api.py

Environment Variables:
    - INTERNAL_SERVICE_TOKEN: Token for authenticating frontend requests
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key for embeddings
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
    - LIVEKIT_API_KEY: LiveKit API key for room access validation
    - LIVEKIT_API_SECRET: LiveKit API secret for room access validation
    - LIVEKIT_URL: LiveKit server URL
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from persistent_store import PersistentDocumentStore, DocumentUploadService
from hybrid_retriever import RoomRetrieverManager
from schemas.documents import MAX_DOCUMENTS_PER_ROOM
from supabase_client import (
    download_document_from_supabase_sync,
    check_supabase_configured,
    STORAGE_BUCKET,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hedwiq-document-api")

# Initialize FastAPI app
app = FastAPI(
    title="Hedwiq Document API",
    description="Document upload and processing API for Hedwiq meetings",
    version="1.0.0"
)

# CORS configuration - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Initialize services
document_store = PersistentDocumentStore(backend="sqlite")
retriever_manager = RoomRetrieverManager.get_instance(document_store)
upload_service = DocumentUploadService(
    store=document_store,
    retriever_manager=retriever_manager  # Wire up retriever for index rebuilding
)

logger.info("Document API services initialized with retriever manager")

# Security constants
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Room access validation settings
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
ENFORCE_ROOM_ACCESS = os.getenv("ENFORCE_ROOM_ACCESS", "false").lower() == "true"


def verify_internal_token(x_internal_token: Optional[str] = Header(None)) -> bool:
    """Verify the internal service token."""
    if not INTERNAL_SERVICE_TOKEN:
        # If no token configured, allow requests (development mode)
        logger.warning("INTERNAL_SERVICE_TOKEN not set - running in development mode")
        return True

    if x_internal_token != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    return True


async def verify_room_access(user_id: str, room_id: str) -> bool:
    """
    Verify that a user has access to upload documents to a room.

    In production, this should check:
    1. User is a current participant in the room (via LiveKit API)
    2. Or user has admin/owner role for the room
    3. Or room allows document uploads from this user

    Args:
        user_id: The user ID attempting the upload
        room_id: The room ID to upload to

    Returns:
        True if access is allowed

    Raises:
        HTTPException: If access is denied
    """
    # If enforcement is disabled, allow all (development mode)
    if not ENFORCE_ROOM_ACCESS:
        logger.debug(f"Room access check skipped (ENFORCE_ROOM_ACCESS=false): user={user_id}, room={room_id}")
        return True

    # Check if LiveKit credentials are configured
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        logger.warning(
            "LiveKit credentials not configured - room access validation disabled. "
            "Set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and ENFORCE_ROOM_ACCESS=true "
            "to enable room membership validation."
        )
        return True

    try:
        # Use LiveKit Server SDK to check room participants
        from livekit.api import LiveKitAPI

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        # List participants in the room
        participants = await lk_api.room.list_participants(room_id)

        # Check if user is a participant (identity prefix matches user_id)
        # Note: identities are typically formatted as "userId-randomSuffix"
        user_in_room = any(
            p.identity.startswith(f"{user_id}-") or p.identity == user_id
            for p in participants
        )

        if not user_in_room:
            logger.warning(f"Room access denied: user {user_id} not in room {room_id}")
            raise HTTPException(
                status_code=403,
                detail="You must be a participant in the room to upload documents"
            )

        logger.info(f"Room access verified: user {user_id} in room {room_id}")
        return True

    except HTTPException:
        raise
    except ImportError:
        logger.warning(
            "livekit-api not installed - room access validation disabled. "
            "Install with: pip install livekit-api"
        )
        return True
    except Exception as e:
        # Log error but don't block upload if validation fails
        # This is a security trade-off - in production you may want to deny on error
        logger.error(f"Room access validation error: {e}")
        if ENFORCE_ROOM_ACCESS:
            raise HTTPException(
                status_code=500,
                detail="Failed to validate room access"
            )
        return True


class UploadResponse(BaseModel):
    """Response model for document upload."""
    documentId: str
    title: str
    pageCount: int
    segmentCount: Optional[int] = None
    status: str = "ready"


class ProcessDocumentRequest(BaseModel):
    """Request model for processing a document from Supabase Storage."""
    documentId: str
    roomId: str
    storagePath: str
    uploadedBy: Optional[str] = None


class DocumentInfo(BaseModel):
    """Response model for document info."""
    id: str
    roomId: str
    filename: str
    title: str
    summary: str
    pageCount: int
    status: str
    uploadedAt: int
    uploadedBy: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing failed"},
    }
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    roomId: str = Form(..., description="LiveKit room ID"),
    uploadedBy: str = Form(..., description="User ID who uploaded"),
    x_internal_token: Optional[str] = Header(None),
):
    """
    Upload and process a PDF document.

    The document will be:
    1. Parsed to extract text and coordinates
    2. Split into segments for retrieval
    3. Embedded using Azure OpenAI
    4. Stored with the room for reference detection

    Security checks:
    - Internal service token validation
    - Room membership validation (when ENFORCE_ROOM_ACCESS=true)
    """
    # Verify internal token
    verify_internal_token(x_internal_token)

    # Verify user has access to upload to this room
    await verify_room_access(uploadedBy, roomId)

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    try:
        pdf_data = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Validate file size
    if len(pdf_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_FILE_SIZE / 1024 / 1024}MB)"
        )

    # Validate PDF content (check magic bytes)
    if not pdf_data[:5] == b"%PDF-":
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    # Check room document limit
    existing_count = document_store.get_document_count(roomId)
    if existing_count >= MAX_DOCUMENTS_PER_ROOM:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_DOCUMENTS_PER_ROOM} documents per room"
        )

    # Process document
    try:
        result = upload_service.upload_document_sync(
            room_id=roomId,
            filename=file.filename or "document.pdf",
            pdf_data=pdf_data,
            uploaded_by=uploadedBy
        )
        return UploadResponse(**result)

    except ValueError as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")


@app.post(
    "/documents/process",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        500: {"model": ErrorResponse, "description": "Processing failed"},
        503: {"model": ErrorResponse, "description": "Supabase not configured"},
    }
)
async def process_document_from_supabase(
    request: ProcessDocumentRequest,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Process a document that was uploaded to Supabase Storage.

    This endpoint is called by the frontend after uploading a PDF to Supabase.
    It downloads the PDF from Supabase Storage and processes it for:
    1. Text extraction with coordinates
    2. Segmentation for retrieval
    3. Embedding generation
    4. Storage in the agent's document store

    This enables the document reference feature to work with documents
    uploaded via the frontend's pre-join upload flow.
    """
    # Verify internal token
    verify_internal_token(x_internal_token)

    # Check if Supabase is configured
    if not check_supabase_configured():
        logger.error("Supabase is not configured - cannot download document")
        raise HTTPException(
            status_code=503,
            detail="Supabase storage is not configured on the agent. "
                   "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables."
        )

    # Validate room ID format
    if not request.roomId or len(request.roomId) > 100:
        raise HTTPException(status_code=400, detail="Invalid room ID")

    # Check room document limit
    existing_count = document_store.get_document_count(request.roomId)
    if existing_count >= MAX_DOCUMENTS_PER_ROOM:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_DOCUMENTS_PER_ROOM} documents per room"
        )

    # Download PDF from Supabase
    logger.info(f"Downloading document from Supabase: {request.storagePath}")
    try:
        pdf_data = download_document_from_supabase_sync(request.storagePath)
    except ValueError as e:
        logger.error(f"Supabase configuration error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to download from Supabase: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download document from storage: {e}"
        )

    if not pdf_data:
        logger.error(f"Document not found in Supabase: {request.storagePath}")
        raise HTTPException(
            status_code=404,
            detail="Document not found in storage"
        )

    # Validate PDF content (check magic bytes)
    if not pdf_data[:5] == b"%PDF-":
        raise HTTPException(status_code=400, detail="Invalid PDF file in storage")

    # Extract filename from storage path
    # Format: "meeting-documents/{roomId}/{documentId}.pdf" or "{roomId}/{documentId}.pdf"
    path_parts = request.storagePath.split("/")
    filename = path_parts[-1] if path_parts else f"{request.documentId}.pdf"

    # Process document - use frontend's documentId to maintain consistency
    logger.info(f"Processing document {request.documentId} for room {request.roomId}")
    try:
        result = upload_service.upload_document_sync(
            room_id=request.roomId,
            filename=filename,
            pdf_data=pdf_data,
            uploaded_by=request.uploadedBy or "unknown",
            doc_id=request.documentId  # Use frontend's document ID for consistency
        )

        logger.info(
            f"Document processed successfully: {result['documentId']} "
            f"({result['pageCount']} pages, {result.get('segmentCount', 0)} segments)"
        )

        return UploadResponse(**result)

    except ValueError as e:
        logger.error(f"Document processing validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")


@app.get(
    "/documents/{document_id}/pdf",
    responses={
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    }
)
async def get_document_pdf(
    document_id: str,
    roomId: Optional[str] = None,
    x_internal_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    """
    Retrieve PDF file for a document.

    Requires room ID for access control.
    """
    # Verify internal token
    verify_internal_token(x_internal_token)

    if not roomId:
        raise HTTPException(status_code=400, detail="roomId is required")

    # Get PDF data
    pdf_data = document_store.get_pdf_data(roomId, document_id)

    if not pdf_data:
        raise HTTPException(status_code=404, detail="Document not found")

    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{document_id}.pdf"',
            "Cache-Control": "private, max-age=3600",
        }
    )


@app.get(
    "/documents/{document_id}",
    response_model=DocumentInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    }
)
async def get_document_info(
    document_id: str,
    roomId: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Get document metadata.
    """
    verify_internal_token(x_internal_token)

    doc = document_store.get_document(roomId, document_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentInfo(
        id=doc.id,
        roomId=doc.room_id,
        filename=doc.filename,
        title=doc.title,
        summary=doc.summary,
        pageCount=doc.page_count,
        status="ready",
        uploadedAt=doc.created_at,
        uploadedBy=doc.uploaded_by
    )


@app.get("/documents/room/{room_id}")
async def list_room_documents(
    room_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    List all documents for a room.
    """
    verify_internal_token(x_internal_token)

    docs = document_store.get_documents_for_room(room_id)

    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "pageCount": doc.page_count,
                "uploadedAt": doc.created_at,
            }
            for doc in docs
        ],
        "count": len(docs),
        "maxAllowed": MAX_DOCUMENTS_PER_ROOM
    }


@app.delete(
    "/documents/{document_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    }
)
async def delete_document(
    document_id: str,
    roomId: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete a document.

    Also triggers retrieval index rebuild for the room.
    """
    verify_internal_token(x_internal_token)

    if not document_store.document_exists(roomId, document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    document_store.remove_document(roomId, document_id)

    # Rebuild retrieval index after deletion
    retriever_manager.rebuild_room_index(roomId)

    return {"status": "deleted", "documentId": document_id}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "hedwiq-document-api"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("DOCUMENT_API_PORT", "8000"))
    host = os.getenv("DOCUMENT_API_HOST", "0.0.0.0")

    logger.info(f"Starting Hedwiq Document API on {host}:{port}")

    uvicorn.run(
        "document_api:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
