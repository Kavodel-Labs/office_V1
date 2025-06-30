"""
Core memory system for Project Aethelred.

This module implements the multi-tiered memory architecture:
- Hot Memory (Redis): Fast access, short-term storage
- Warm Memory (PostgreSQL): Structured data, medium-term storage  
- Cold Memory (Neo4j): Graph relationships, long-term storage
- Archive Memory (Filesystem/S3): Long-term archival
"""

from .tier_manager import MemoryTierManager, MemoryTier
from .coherence import MemoryCoherenceManager
from .interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory

__all__ = [
    'MemoryTierManager',
    'MemoryTier', 
    'MemoryCoherenceManager',
    'HotMemory',
    'WarmMemory', 
    'ColdMemory',
    'ArchiveMemory'
]