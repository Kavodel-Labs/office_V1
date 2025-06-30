"""
Memory tier management for Project Aethelred.

Implements the abstract base class for memory tiers and the manager
that coordinates access across all tiers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryTier(ABC):
    """Abstract base class for memory tiers."""
    
    def __init__(self, name: str, tier_level: int):
        self.name = name
        self.tier_level = tier_level
        self.is_connected = False
        
    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the memory tier."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the memory tier."""
        pass
    
    @abstractmethod
    async def read(self, key: str) -> Optional[Any]:
        """Read data from the tier."""
        pass
    
    @abstractmethod
    async def write(self, key: str, value: Any, **kwargs) -> None:
        """Write data to the tier."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from the tier."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in the tier."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the tier."""
        pass


class MemoryTierManager:
    """
    Manages access across multiple memory tiers.
    
    Provides unified interface for reading/writing across hot, warm, 
    cold, and archive memory tiers with automatic failover and caching.
    """
    
    def __init__(self):
        self.tiers: Dict[str, MemoryTier] = {}
        self.tier_order: List[str] = []  # Ordered by access speed
        
    def register_tier(self, tier: MemoryTier) -> None:
        """Register a memory tier with the manager."""
        self.tiers[tier.name] = tier
        self.tier_order = sorted(self.tiers.keys(), 
                                key=lambda x: self.tiers[x].tier_level)
        logger.info(f"Registered memory tier: {tier.name} (level {tier.tier_level})")
        
    async def connect_all(self) -> None:
        """Connect to all registered memory tiers."""
        connection_tasks = [tier.connect() for tier in self.tiers.values()]
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        for tier_name, result in zip(self.tiers.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {tier_name}: {result}")
            else:
                self.tiers[tier_name].is_connected = True
                logger.info(f"Connected to memory tier: {tier_name}")
                
    async def disconnect_all(self) -> None:
        """Disconnect from all memory tiers."""
        disconnect_tasks = [tier.disconnect() for tier in self.tiers.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        for tier in self.tiers.values():
            tier.is_connected = False
            
        logger.info("Disconnected from all memory tiers")
        
    async def read(self, key: str, prefer_tier: Optional[str] = None) -> Optional[Any]:
        """
        Read data from memory tiers.
        
        Searches tiers in order of access speed unless a specific tier is preferred.
        Returns the first successful result.
        """
        search_order = self.tier_order
        if prefer_tier and prefer_tier in self.tiers:
            search_order = [prefer_tier] + [t for t in self.tier_order if t != prefer_tier]
            
        for tier_name in search_order:
            tier = self.tiers[tier_name]
            if not tier.is_connected:
                continue
                
            try:
                result = await tier.read(key)
                if result is not None:
                    logger.debug(f"Found key '{key}' in tier: {tier_name}")
                    return result
            except Exception as e:
                logger.warning(f"Read failed from {tier_name} for key '{key}': {e}")
                continue
                
        logger.debug(f"Key '{key}' not found in any tier")
        return None
        
    async def write(self, key: str, value: Any, target_tiers: Optional[List[str]] = None,
                   **kwargs) -> Dict[str, bool]:
        """
        Write data to memory tiers.
        
        Writes to specified tiers or just the fastest tier by default.
        Returns success status for each tier.
        """
        if target_tiers is None:
            target_tiers = [self.tier_order[0]] if self.tier_order else []
            
        results = {}
        write_tasks = []
        
        for tier_name in target_tiers:
            if tier_name not in self.tiers:
                results[tier_name] = False
                continue
                
            tier = self.tiers[tier_name]
            if not tier.is_connected:
                results[tier_name] = False
                continue
                
            write_tasks.append(self._write_to_tier(tier_name, key, value, **kwargs))
            
        if write_tasks:
            write_results = await asyncio.gather(*write_tasks, return_exceptions=True)
            for tier_name, result in zip(target_tiers, write_results):
                results[tier_name] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    logger.error(f"Write failed to {tier_name} for key '{key}': {result}")
                    
        return results
        
    async def _write_to_tier(self, tier_name: str, key: str, value: Any, **kwargs) -> None:
        """Helper method to write to a specific tier."""
        tier = self.tiers[tier_name]
        await tier.write(key, value, **kwargs)
        logger.debug(f"Wrote key '{key}' to tier: {tier_name}")
        
    async def delete(self, key: str, target_tiers: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Delete data from memory tiers.
        
        Deletes from specified tiers or all tiers by default.
        Returns success status for each tier.
        """
        if target_tiers is None:
            target_tiers = list(self.tiers.keys())
            
        results = {}
        delete_tasks = []
        
        for tier_name in target_tiers:
            if tier_name not in self.tiers:
                results[tier_name] = False
                continue
                
            tier = self.tiers[tier_name]
            if not tier.is_connected:
                results[tier_name] = False
                continue
                
            delete_tasks.append(self._delete_from_tier(tier_name, key))
            
        if delete_tasks:
            delete_results = await asyncio.gather(*delete_tasks, return_exceptions=True)
            for tier_name, result in zip(target_tiers, delete_results):
                results[tier_name] = not isinstance(result, Exception) and result
                if isinstance(result, Exception):
                    logger.error(f"Delete failed from {tier_name} for key '{key}': {result}")
                    
        return results
        
    async def _delete_from_tier(self, tier_name: str, key: str) -> bool:
        """Helper method to delete from a specific tier."""
        tier = self.tiers[tier_name]
        result = await tier.delete(key)
        if result:
            logger.debug(f"Deleted key '{key}' from tier: {tier_name}")
        return result
        
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all memory tiers."""
        health_results = {}
        
        for tier_name, tier in self.tiers.items():
            try:
                if tier.is_connected:
                    health_results[tier_name] = await tier.health_check()
                else:
                    health_results[tier_name] = {
                        'status': 'disconnected',
                        'error': 'Tier not connected'
                    }
            except Exception as e:
                health_results[tier_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return health_results
        
    def get_tier_status(self) -> Dict[str, Any]:
        """Get status of all memory tiers."""
        return {
            'connected_tiers': [name for name, tier in self.tiers.items() 
                               if tier.is_connected],
            'total_tiers': len(self.tiers),
            'tier_order': self.tier_order
        }