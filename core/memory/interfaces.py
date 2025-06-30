"""
Memory tier implementations for Project Aethelred.

Concrete implementations of the memory tiers:
- HotMemory: Redis-based fast cache
- WarmMemory: PostgreSQL-based structured storage
- ColdMemory: Neo4j-based graph storage
- ArchiveMemory: Filesystem/S3-based archival storage
"""

import os
import json
import pickle
import asyncio
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path

try:
    import redis.asyncio as aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        AIOREDIS_AVAILABLE = True
    except ImportError:
        AIOREDIS_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from .tier_manager import MemoryTier

logger = logging.getLogger(__name__)


class HotMemory(MemoryTier):
    """Redis-based hot memory tier for fast access."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("hot", 0)
        self.config = config
        self.redis: Optional[Any] = None
        self.default_ttl = config.get('ttl', 3600)
        
    async def connect(self) -> None:
        """Connect to Redis."""
        if not AIOREDIS_AVAILABLE:
            raise ImportError("aioredis not available. Install with: pip install aioredis")
            
        try:
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6379)
            db = self.config.get('db', 0)
            
            # Use redis.asyncio module
            import redis.asyncio as aioredis
            self.redis = aioredis.Redis(
                host=host,
                port=port,
                db=db,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.is_connected = False
            
    async def read(self, key: str) -> Optional[Any]:
        """Read from Redis."""
        if not self.redis:
            return None
            
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
                
            # Try to deserialize JSON, fall back to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis read error for key '{key}': {e}")
            return None
            
    async def write(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> None:
        """Write to Redis."""
        if not self.redis:
            raise ConnectionError("Redis not connected")
            
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
                
            # Set with TTL
            ttl_value = ttl or self.default_ttl
            await self.redis.setex(key, ttl_value, serialized_value)
            
        except Exception as e:
            logger.error(f"Redis write error for key '{key}': {e}")
            raise
            
    async def delete(self, key: str) -> bool:
        """Delete from Redis."""
        if not self.redis:
            return False
            
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key '{key}': {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.redis:
            return False
            
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key '{key}': {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        if not self.redis:
            return {'status': 'disconnected'}
            
        try:
            latency = await self.redis.ping()
            info = await self.redis.info()
            
            return {
                'status': 'healthy',
                'latency_ms': latency,
                'memory_used': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed')
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class WarmMemory(MemoryTier):
    """PostgreSQL-based warm memory tier for structured data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("warm", 1)
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            dsn = f"postgresql://{self.config['user']}:{self.config['password']}@" \
                  f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
                  
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=5,
                max_size=self.config.get('pool_size', 20)
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
            self.is_connected = True
            logger.info("Connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            
    async def read(self, key: str) -> Optional[Any]:
        """Read from PostgreSQL."""
        if not self.pool:
            return None
            
        try:
            async with self.pool.acquire() as conn:
                # Check memory_snapshots table first
                result = await conn.fetchrow(
                    "SELECT data FROM aethelred.memory_snapshots WHERE id = $1::uuid",
                    key
                )
                
                if result:
                    return result['data']
                    
                # Check other tables based on key pattern
                if key.startswith('task:'):
                    task_id = key.replace('task:', '')
                    result = await conn.fetchrow(
                        "SELECT * FROM aethelred.tasks WHERE id = $1::uuid",
                        task_id
                    )
                    if result:
                        return dict(result)
                        
                elif key.startswith('agent:'):
                    agent_id = key.replace('agent:', '')
                    result = await conn.fetchrow(
                        "SELECT * FROM aethelred.agents WHERE agent_id = $1",
                        agent_id
                    )
                    if result:
                        return dict(result)
                        
                return None
                
        except Exception as e:
            logger.error(f"PostgreSQL read error for key '{key}': {e}")
            return None
            
    async def write(self, key: str, value: Any, **kwargs) -> None:
        """Write to PostgreSQL."""
        if not self.pool:
            raise ConnectionError("PostgreSQL not connected")
            
        try:
            async with self.pool.acquire() as conn:
                # Store in memory_snapshots table
                await conn.execute("""
                    INSERT INTO aethelred.memory_snapshots (id, snapshot_type, tier, data)
                    VALUES ($1::uuid, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        created_at = CURRENT_TIMESTAMP
                """, key, 'general', 'warm', json.dumps(value) if not isinstance(value, str) else value)
                
        except Exception as e:
            logger.error(f"PostgreSQL write error for key '{key}': {e}")
            raise
            
    async def delete(self, key: str) -> bool:
        """Delete from PostgreSQL."""
        if not self.pool:
            return False
            
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM aethelred.memory_snapshots WHERE id = $1::uuid",
                    key
                )
                return "DELETE 1" in result
                
        except Exception as e:
            logger.error(f"PostgreSQL delete error for key '{key}': {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in PostgreSQL."""
        if not self.pool:
            return False
            
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM aethelred.memory_snapshots WHERE id = $1::uuid)",
                    key
                )
                return result or False
                
        except Exception as e:
            logger.error(f"PostgreSQL exists error for key '{key}': {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform PostgreSQL health check."""
        if not self.pool:
            return {'status': 'disconnected'}
            
        try:
            async with self.pool.acquire() as conn:
                # Check connection
                await conn.execute("SELECT 1")
                
                # Get database stats
                stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM aethelred.memory_snapshots) as snapshot_count,
                        (SELECT count(*) FROM aethelred.tasks) as task_count,
                        (SELECT count(*) FROM aethelred.agents) as agent_count
                """)
                
                return {
                    'status': 'healthy',
                    'database_size_bytes': stats['db_size'],
                    'snapshot_count': stats['snapshot_count'],
                    'task_count': stats['task_count'],
                    'agent_count': stats['agent_count']
                }
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class ColdMemory(MemoryTier):
    """Neo4j-based cold memory tier for graph relationships."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("cold", 2)
        self.config = config
        self.driver = None
        
    async def connect(self) -> None:
        """Connect to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['user'], self.config['password'])
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            self.is_connected = True
            logger.info("Connected to Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self.driver:
            await self.driver.close()
            self.is_connected = False
            
    async def read(self, key: str) -> Optional[Any]:
        """Read from Neo4j."""
        if not self.driver:
            return None
            
        try:
            async with self.driver.session() as session:
                # Try to find node by various properties
                result = await session.run(
                    "MATCH (n) WHERE n.id = $key OR n.agent_id = $key OR n.task_id = $key "
                    "RETURN n LIMIT 1",
                    key=key
                )
                
                record = await result.single()
                if record:
                    node = record['n']
                    return dict(node)
                    
                return None
                
        except Exception as e:
            logger.error(f"Neo4j read error for key '{key}': {e}")
            return None
            
    async def write(self, key: str, value: Any, **kwargs) -> None:
        """Write to Neo4j."""
        if not self.driver:
            raise ConnectionError("Neo4j not connected")
            
        try:
            async with self.driver.session() as session:
                # Create or update a Memory node
                await session.run("""
                    MERGE (m:Memory {id: $key})
                    SET m.data = $data,
                        m.tier = 'cold',
                        m.updated_at = datetime()
                """, key=key, data=json.dumps(value) if not isinstance(value, str) else value)
                
        except Exception as e:
            logger.error(f"Neo4j write error for key '{key}': {e}")
            raise
            
    async def delete(self, key: str) -> bool:
        """Delete from Neo4j."""
        if not self.driver:
            return False
            
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (m:Memory {id: $key}) DELETE m RETURN count(m) as deleted",
                    key=key
                )
                
                record = await result.single()
                return record and record['deleted'] > 0
                
        except Exception as e:
            logger.error(f"Neo4j delete error for key '{key}': {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in Neo4j."""
        if not self.driver:
            return False
            
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (m:Memory {id: $key}) RETURN count(m) > 0 as exists",
                    key=key
                )
                
                record = await result.single()
                return record and record['exists']
                
        except Exception as e:
            logger.error(f"Neo4j exists error for key '{key}': {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform Neo4j health check."""
        if not self.driver:
            return {'status': 'disconnected'}
            
        try:
            async with self.driver.session() as session:
                # Check connection and get stats
                result = await session.run("""
                    CALL db.info() YIELD name, value
                    WHERE name IN ['StoreSize', 'TransactionCommitted']
                    RETURN collect({name: name, value: value}) as info
                """)
                
                record = await result.single()
                info = {item['name']: item['value'] for item in record['info']}
                
                # Get node counts
                node_result = await session.run("""
                    MATCH (n) 
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY count DESC
                    LIMIT 5
                """)
                
                node_counts = {}
                async for record in node_result:
                    if record['label']:
                        node_counts[record['label']] = record['count']
                
                return {
                    'status': 'healthy',
                    'store_size': info.get('StoreSize'),
                    'transactions_committed': info.get('TransactionCommitted'),
                    'node_counts': node_counts
                }
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class ArchiveMemory(MemoryTier):
    """Filesystem-based archive memory tier for long-term storage."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("archive", 3)
        self.config = config
        self.base_path = Path(config.get('filesystem', {}).get('path', './aethelred_archive'))
        
    async def connect(self) -> None:
        """Initialize archive directory."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for subdir in ['data', 'snapshots', 'logs']:
                (self.base_path / subdir).mkdir(exist_ok=True)
                
            self.is_connected = True
            logger.info(f"Archive directory initialized at {self.base_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize archive: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Nothing to disconnect for filesystem."""
        self.is_connected = False
        
    async def read(self, key: str) -> Optional[Any]:
        """Read from filesystem."""
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None
                
            with open(file_path, 'rb') as f:
                # Try pickle first, then JSON, then plain text
                try:
                    return pickle.load(f)
                except (pickle.PickleError, EOFError):
                    f.seek(0)
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        return f.read().decode('utf-8')
                        
        except Exception as e:
            logger.error(f"Archive read error for key '{key}': {e}")
            return None
            
    async def write(self, key: str, value: Any, **kwargs) -> None:
        """Write to filesystem."""
        try:
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                # Try to pickle for efficiency, fall back to JSON
                try:
                    pickle.dump(value, f)
                except (pickle.PickleError, TypeError):
                    f.seek(0)
                    f.truncate()
                    if isinstance(value, (dict, list)):
                        json.dump(value, f)
                    else:
                        f.write(str(value).encode('utf-8'))
                        
        except Exception as e:
            logger.error(f"Archive write error for key '{key}': {e}")
            raise
            
    async def delete(self, key: str) -> bool:
        """Delete from filesystem."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Archive delete error for key '{key}': {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in filesystem."""
        try:
            return self._get_file_path(key).exists()
        except Exception as e:
            logger.error(f"Archive exists error for key '{key}': {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform filesystem health check."""
        try:
            # Check disk space
            stat = os.statvfs(self.base_path)
            free_space = stat.f_bavail * stat.f_frsize
            total_space = stat.f_blocks * stat.f_frsize
            
            # Count files
            file_count = sum(1 for _ in self.base_path.rglob('*') if _.is_file())
            
            return {
                'status': 'healthy',
                'free_space_bytes': free_space,
                'total_space_bytes': total_space,
                'file_count': file_count,
                'base_path': str(self.base_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
            
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Create a safe filename from the key
        safe_key = key.replace(':', '_').replace('/', '_')
        
        # Distribute files into subdirectories based on key hash
        hash_dir = str(hash(key) % 100).zfill(2)
        
        return self.base_path / 'data' / hash_dir / f"{safe_key}.dat"