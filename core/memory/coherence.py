"""
Memory coherence management for Project Aethelred.

Ensures consistency across memory tiers through write-through policies,
eventual consistency guarantees, and conflict resolution.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum

from .tier_manager import MemoryTierManager

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency levels for memory operations."""
    EVENTUAL = "eventual"      # Best effort, async propagation
    STRONG = "strong"         # Synchronous write to multiple tiers
    IMMEDIATE = "immediate"   # Synchronous write to all tiers


class MemoryCoherenceManager:
    """
    Manages consistency across memory tiers.
    
    Implements write-through policies, eventual consistency,
    and conflict resolution strategies.
    """
    
    def __init__(self, tier_manager: MemoryTierManager):
        self.tier_manager = tier_manager
        self.write_through_events: Set[str] = set()
        self.consistency_config: Dict[str, Any] = {}
        self.pending_writes: Dict[str, List[Dict[str, Any]]] = {}
        self._background_tasks: Set[asyncio.Task] = set()
        
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure coherence policies."""
        self.write_through_events = set(config.get('write_through', {}).get('critical_events', []))
        self.consistency_config = config.get('consistency_windows', {})
        
        logger.info(f"Configured coherence with {len(self.write_through_events)} critical events")
        
    async def write(self, key: str, value: Any, event_type: Optional[str] = None,
                   consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
                   **kwargs) -> Dict[str, bool]:
        """
        Write data with coherence guarantees.
        
        Args:
            key: The key to write
            value: The value to write
            event_type: Type of event triggering the write
            consistency: Required consistency level
            **kwargs: Additional parameters for write operation
            
        Returns:
            Dictionary of write results per tier
        """
        target_tiers = self._determine_target_tiers(event_type, consistency)
        
        # Add timestamp and metadata
        enriched_value = self._enrich_value(value, event_type)
        
        if consistency == ConsistencyLevel.IMMEDIATE:
            # Write to all tiers synchronously
            return await self._write_immediate(key, enriched_value, target_tiers, **kwargs)
        elif consistency == ConsistencyLevel.STRONG:
            # Write to primary tiers synchronously
            return await self._write_strong(key, enriched_value, target_tiers, **kwargs)
        else:
            # Write to fastest tier, then propagate asynchronously
            return await self._write_eventual(key, enriched_value, target_tiers, **kwargs)
            
    def _determine_target_tiers(self, event_type: Optional[str], 
                               consistency: ConsistencyLevel) -> List[str]:
        """Determine which tiers to write to based on event type and consistency."""
        tier_order = self.tier_manager.tier_order
        
        if not tier_order:
            return []
            
        if consistency == ConsistencyLevel.IMMEDIATE:
            return tier_order  # All tiers
        elif consistency == ConsistencyLevel.STRONG or \
             (event_type and event_type in self.write_through_events):
            # Hot and warm memory for critical events
            return tier_order[:2] if len(tier_order) >= 2 else tier_order
        else:
            # Just hot memory for eventual consistency
            return [tier_order[0]]
            
    def _enrich_value(self, value: Any, event_type: Optional[str]) -> Dict[str, Any]:
        """Enrich value with metadata."""
        if isinstance(value, dict):
            enriched = value.copy()
        else:
            enriched = {'data': value}
            
        enriched['_metadata'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'coherence_version': 1
        }
        
        return enriched
        
    async def _write_immediate(self, key: str, value: Any, target_tiers: List[str],
                              **kwargs) -> Dict[str, bool]:
        """Write to all tiers synchronously."""
        logger.debug(f"Immediate write for key '{key}' to tiers: {target_tiers}")
        return await self.tier_manager.write(key, value, target_tiers, **kwargs)
        
    async def _write_strong(self, key: str, value: Any, target_tiers: List[str],
                           **kwargs) -> Dict[str, bool]:
        """Write to primary tiers synchronously."""
        logger.debug(f"Strong consistency write for key '{key}' to tiers: {target_tiers}")
        return await self.tier_manager.write(key, value, target_tiers, **kwargs)
        
    async def _write_eventual(self, key: str, value: Any, target_tiers: List[str],
                             **kwargs) -> Dict[str, bool]:
        """Write to fastest tier, then schedule async propagation."""
        primary_tier = target_tiers[0] if target_tiers else None
        if not primary_tier:
            return {}
            
        # Write to primary tier first
        primary_result = await self.tier_manager.write(key, value, [primary_tier], **kwargs)
        
        # Schedule async propagation to other tiers
        remaining_tiers = target_tiers[1:]
        if remaining_tiers:
            self._schedule_propagation(key, value, remaining_tiers, **kwargs)
            
        logger.debug(f"Eventual consistency write for key '{key}' to primary tier: {primary_tier}")
        return primary_result
        
    def _schedule_propagation(self, key: str, value: Any, target_tiers: List[str],
                             **kwargs) -> None:
        """Schedule asynchronous propagation to remaining tiers."""
        task = asyncio.create_task(
            self._propagate_write(key, value, target_tiers, **kwargs)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
    async def _propagate_write(self, key: str, value: Any, target_tiers: List[str],
                              **kwargs) -> None:
        """Propagate write to remaining tiers asynchronously."""
        try:
            # Add small delay for eventual consistency
            eventual_delay = self.consistency_config.get('eventual', 5)  # 5 seconds default
            await asyncio.sleep(eventual_delay)
            
            await self.tier_manager.write(key, value, target_tiers, **kwargs)
            logger.debug(f"Propagated write for key '{key}' to tiers: {target_tiers}")
            
        except Exception as e:
            logger.error(f"Failed to propagate write for key '{key}': {e}")
            # Could implement retry logic here
            
    async def read_consistent(self, key: str, consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
                             prefer_tier: Optional[str] = None) -> Optional[Any]:
        """
        Read data with consistency guarantees.
        
        Args:
            key: The key to read
            consistency: Required consistency level
            prefer_tier: Preferred tier to read from
            
        Returns:
            The value if found, None otherwise
        """
        if consistency == ConsistencyLevel.IMMEDIATE:
            # Read from all tiers and resolve conflicts
            return await self._read_immediate(key)
        elif consistency == ConsistencyLevel.STRONG:
            # Read from primary tiers with conflict resolution
            return await self._read_strong(key, prefer_tier)
        else:
            # Read from fastest available tier
            return await self.tier_manager.read(key, prefer_tier)
            
    async def _read_immediate(self, key: str) -> Optional[Any]:
        """Read from all tiers and resolve conflicts."""
        all_tiers = list(self.tier_manager.tiers.keys())
        results = []
        
        for tier_name in all_tiers:
            try:
                value = await self.tier_manager.read(key, prefer_tier=tier_name)
                if value is not None:
                    results.append((tier_name, value))
            except Exception as e:
                logger.warning(f"Read failed from {tier_name} for key '{key}': {e}")
                
        if not results:
            return None
            
        # Resolve conflicts by preferring latest timestamp
        return self._resolve_conflicts(results)
        
    async def _read_strong(self, key: str, prefer_tier: Optional[str]) -> Optional[Any]:
        """Read with strong consistency."""
        primary_tiers = self.tier_manager.tier_order[:2]  # Hot and warm
        
        if prefer_tier and prefer_tier in primary_tiers:
            return await self.tier_manager.read(key, prefer_tier)
            
        # Try primary tiers in order
        for tier_name in primary_tiers:
            try:
                value = await self.tier_manager.read(key, prefer_tier=tier_name)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(f"Strong read failed from {tier_name} for key '{key}': {e}")
                
        return None
        
    def _resolve_conflicts(self, results: List[tuple]) -> Any:
        """Resolve conflicts between different versions of data."""
        if len(results) == 1:
            return results[0][1]
            
        # Sort by timestamp if available
        def get_timestamp(result):
            tier_name, value = result
            if isinstance(value, dict) and '_metadata' in value:
                return value['_metadata'].get('timestamp', '')
            return ''
            
        sorted_results = sorted(results, key=get_timestamp, reverse=True)
        latest_result = sorted_results[0][1]
        
        # Log conflict resolution
        if len(set(str(r[1]) for r in results)) > 1:
            logger.warning(f"Resolved conflict between {len(results)} versions, "
                          f"using latest from tier: {sorted_results[0][0]}")
            
        return latest_result
        
    async def ensure_consistency(self, key: str) -> Dict[str, Any]:
        """
        Ensure consistency for a specific key across all tiers.
        
        Reads from all tiers, resolves conflicts, and propagates canonical version.
        """
        all_tiers = list(self.tier_manager.tiers.keys())
        results = []
        
        # Read from all tiers
        for tier_name in all_tiers:
            try:
                value = await self.tier_manager.read(key, prefer_tier=tier_name)
                if value is not None:
                    results.append((tier_name, value))
            except Exception as e:
                logger.warning(f"Consistency check failed for {tier_name}, key '{key}': {e}")
                
        if not results:
            return {'status': 'not_found', 'key': key}
            
        # Resolve conflicts
        canonical_value = self._resolve_conflicts(results)
        
        # Check if all tiers have the canonical version
        inconsistent_tiers = []
        for tier_name, value in results:
            if str(value) != str(canonical_value):
                inconsistent_tiers.append(tier_name)
                
        # Propagate canonical version to inconsistent tiers
        if inconsistent_tiers:
            await self.tier_manager.write(key, canonical_value, inconsistent_tiers)
            logger.info(f"Synchronized key '{key}' to {len(inconsistent_tiers)} tiers")
            
        return {
            'status': 'synchronized' if inconsistent_tiers else 'consistent',
            'key': key,
            'tiers_synchronized': inconsistent_tiers,
            'canonical_version': canonical_value
        }
        
    async def cleanup_background_tasks(self) -> None:
        """Clean up background propagation tasks."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            logger.debug("Cleaned up background coherence tasks")