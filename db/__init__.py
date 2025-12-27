"""
Database utilities for Luframe Agent

Provides database access for agenda tracking and other features
that need to interact with the frontend's PostgreSQL database.
"""

from .agenda import AgendaDB

__all__ = ["AgendaDB"]
