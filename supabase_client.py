"""
Supabase Client for Hedwiq Agent

Provides functionality to download documents from Supabase Storage.
This enables the agent to process documents that were uploaded via
the frontend to Supabase Storage.

Usage:
    from supabase_client import download_document_from_supabase

    pdf_data = await download_document_from_supabase(
        storage_path="meeting-documents/room-123/doc-abc.pdf"
    )
"""

import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger("hedwiq-supabase")

# Storage bucket name (must match frontend STORAGE_BUCKETS.DOCUMENTS)
STORAGE_BUCKET = "meeting-documents"


def get_supabase_client():
    """
    Create a Supabase client for storage operations.

    Uses SUPABASE_SERVICE_ROLE_KEY for admin access to download files.

    Returns:
        Supabase client instance

    Raises:
        ValueError: If required environment variables are not set
        ImportError: If supabase package is not installed
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "Supabase SDK is required for document downloads. "
            "Install it with: pip install supabase"
        )

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url:
        raise ValueError(
            "SUPABASE_URL environment variable is required. "
            "Set it to your Supabase project URL (e.g., https://xxx.supabase.co)"
        )

    if not supabase_key:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY environment variable is required. "
            "Get it from your Supabase project settings."
        )

    return create_client(supabase_url, supabase_key)


def parse_storage_path(storage_path: str) -> Tuple[str, str]:
    """
    Parse a storage path to extract bucket and file path.

    Args:
        storage_path: Full storage path (e.g., "meeting-documents/room-123/doc.pdf")
                     or just the file path (e.g., "room-123/doc.pdf")

    Returns:
        Tuple of (bucket_name, file_path)
    """
    # Check if path includes bucket name
    if storage_path.startswith(f"{STORAGE_BUCKET}/"):
        # Remove bucket prefix
        file_path = storage_path[len(f"{STORAGE_BUCKET}/"):]
        return STORAGE_BUCKET, file_path
    else:
        # Assume it's just the file path
        return STORAGE_BUCKET, storage_path


async def download_document_from_supabase(
    storage_path: str,
    bucket: Optional[str] = None
) -> Optional[bytes]:
    """
    Download a document from Supabase Storage.

    Args:
        storage_path: The storage path (can include bucket name or just file path)
                     Format: "{roomId}/{documentId}.pdf" or
                            "meeting-documents/{roomId}/{documentId}.pdf"
        bucket: Optional bucket name override

    Returns:
        PDF file content as bytes, or None if download fails

    Raises:
        ValueError: If Supabase is not configured
    """
    import asyncio

    try:
        supabase = get_supabase_client()
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise

    # Parse storage path
    bucket_name, file_path = parse_storage_path(storage_path)
    if bucket:
        bucket_name = bucket

    logger.info(f"Downloading document from Supabase: bucket={bucket_name}, path={file_path}")

    try:
        # Download file (sync operation, run in thread)
        def _download():
            response = supabase.storage.from_(bucket_name).download(file_path)
            return response

        pdf_data = await asyncio.to_thread(_download)

        if pdf_data:
            logger.info(f"Successfully downloaded document: {len(pdf_data)} bytes")
            return pdf_data
        else:
            logger.error(f"Downloaded empty file from {file_path}")
            return None

    except Exception as e:
        logger.error(f"Failed to download document from Supabase: {e}")
        return None


def download_document_from_supabase_sync(
    storage_path: str,
    bucket: Optional[str] = None
) -> Optional[bytes]:
    """
    Synchronous version of download_document_from_supabase.

    Args:
        storage_path: The storage path
        bucket: Optional bucket name override

    Returns:
        PDF file content as bytes, or None if download fails
    """
    try:
        supabase = get_supabase_client()
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise

    # Parse storage path
    bucket_name, file_path = parse_storage_path(storage_path)
    if bucket:
        bucket_name = bucket

    logger.info(f"Downloading document from Supabase: bucket={bucket_name}, path={file_path}")

    try:
        response = supabase.storage.from_(bucket_name).download(file_path)

        if response:
            logger.info(f"Successfully downloaded document: {len(response)} bytes")
            return response
        else:
            logger.error(f"Downloaded empty file from {file_path}")
            return None

    except Exception as e:
        logger.error(f"Failed to download document from Supabase: {e}")
        return None


def check_supabase_configured() -> bool:
    """
    Check if Supabase is properly configured.

    Returns:
        True if all required environment variables are set
    """
    return bool(
        os.getenv("SUPABASE_URL") and
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
