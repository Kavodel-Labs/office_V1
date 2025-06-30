"""
Base agent implementation for Project Aethelred.

Provides the foundational Agent class that all specific agents inherit from,
along with core functionality for task processing, health monitoring, and
communication.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentCapability(Enum):
    """Standard agent capabilities."""
    # System capabilities
    SYSTEM_CONFIG_READ = "system.config.read"
    SYSTEM_CONFIG_WRITE = "system.config.write"
    SYSTEM_VISION_READ = "system.vision.read"
    SYSTEM_VISION_WRITE = "system.vision.write"
    
    # Agent management
    AGENTS_STRATEGY_DEFINE = "agents.strategy.define"
    AGENTS_EVOLUTION_MANAGE = "agents.evolution.*"
    AGENTS_OBSERVE = "agents.observe"
    
    # Task management
    TASKS_ROUTING = "tasks.routing"
    TASKS_ASSIGN = "tasks.assign"
    TASKS_EXECUTE = "tasks.execute"
    
    # Code development
    CODE_BACKEND_DEVELOP = "code.backend.*"
    CODE_FRONTEND_DEVELOP = "code.frontend.*"
    CODE_REVIEW = "code.review"
    CODE_COMMIT = "git.commit"
    
    # Testing
    TESTS_WRITE = "tests.write"
    TESTS_EXECUTE = "tests.execute"
    
    # Monitoring and metrics
    METRICS_WRITE = "metrics.write"
    SCORES_CALCULATE = "scores.calculate"
    
    # Rules and governance
    RULES_GOVERNANCE = "rules.governance.*"
    RULES_ENFORCE = "rules.enforce"
    
    # Communications
    COMMUNICATIONS_EXTERNAL = "communications.external"
    TASKS_RECEIVE = "tasks.receive"
    NOTIFICATIONS_SEND = "notifications.send"
    
    # Agora Consensus
    AGORA_CONSENSUS = "agora.consensus"
    AGORA_INVOKE = "agora.invoke"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Agent(ABC):
    """
    Base agent class for Project Aethelred.
    
    All agents inherit from this class and implement the abstract methods
    for task processing and health monitoring.
    """
    
    def __init__(
        self,
        agent_id: str,
        version: int,
        tier: str,
        role: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict[str, Any]] = None,
        memory_manager: Optional[Any] = None
    ):
        self.agent_id = agent_id
        self.version = version
        self.tier = tier
        self.role = role
        self.capabilities = set(capabilities)
        self.config = config or {}
        self.memory_manager = memory_manager
        
        # Agora integration (if available)
        self.agora_orchestrator = None
        
        # Runtime state
        self.status = AgentStatus.INITIALIZING
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[str] = []
        self.error_count = 0
        self.last_health_check = datetime.utcnow()
        
        # Performance metrics
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_response_time_ms': 0,
            'total_response_time_ms': 0,
            'uptime_seconds': 0
        }
        
        # Lifecycle
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"Agent {self.agent_id} v{self.version} initialized")
        
    async def initialize(self) -> None:
        """Initialize the agent and start background tasks."""
        try:
            self.status = AgentStatus.INITIALIZING
            
            # Initialize Agora if agent has consensus capabilities
            if (AgentCapability.AGORA_CONSENSUS in self.capabilities or 
                AgentCapability.AGORA_INVOKE in self.capabilities):
                await self._initialize_agora()
            
            # Initialize agent-specific resources
            await self.on_initialize()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.status = AgentStatus.IDLE
            self.started_at = datetime.utcnow()
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            logger.error(f"Agent {self.agent_id} initialization failed: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        logger.info(f"Shutting down agent {self.agent_id}")
        
        self.status = AgentStatus.SHUTDOWN
        self._shutdown_event.set()
        
        # Wait for current tasks to complete (with timeout)
        if self.current_tasks:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    self._wait_for_current_tasks(),
                    timeout=30.0  # 30 second grace period
                )
            except asyncio.TimeoutError:
                logger.warning(f"Agent {self.agent_id} tasks did not complete within timeout")
                
        # Clean up background tasks
        if self._background_tasks:
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
        # Agent-specific cleanup
        await self.on_shutdown()
        
        logger.info(f"Agent {self.agent_id} shutdown complete")
        
    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task and return the result.
        
        Args:
            task: Task definition containing type, data, and metadata
            
        Returns:
            TaskResult with success status and result data
        """
        task_id = task.get('id', str(uuid.uuid4()))
        start_time = datetime.utcnow()
        
        # Check if agent can handle this task
        if not await self.can_handle_task(task):
            return TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=f"Agent {self.agent_id} cannot handle task type: {task.get('type')}"
            )
            
        # Check agent status
        if self.status not in [AgentStatus.ACTIVE, AgentStatus.IDLE]:
            return TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=f"Agent {self.agent_id} is not available (status: {self.status.value})"
            )
            
        # Add to current tasks
        self.current_tasks[task_id] = {
            'task': task,
            'started_at': start_time
        }
        self.status = AgentStatus.BUSY
        
        try:
            logger.debug(f"Agent {self.agent_id} processing task {task_id}")
            
            # Execute the task
            result = await self.execute_task(task)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update performance metrics
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_response_time_ms'] += duration_ms
            self.performance_metrics['average_response_time_ms'] = (
                self.performance_metrics['total_response_time_ms'] // 
                self.performance_metrics['tasks_completed']
            )
            
            # Create successful result
            task_result = TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                duration_ms=duration_ms,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_version': self.version,
                    'completed_at': end_time.isoformat()
                }
            )
            
            logger.debug(f"Agent {self.agent_id} completed task {task_id} in {duration_ms}ms")
            
        except Exception as e:
            # Calculate duration for failed task
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update error metrics
            self.performance_metrics['tasks_failed'] += 1
            self.error_count += 1
            
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration_ms,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_version': self.version,
                    'failed_at': end_time.isoformat()
                }
            )
            
            logger.error(f"Agent {self.agent_id} failed task {task_id}: {e}")
            
        finally:
            # Remove from current tasks
            self.current_tasks.pop(task_id, None)
            self.completed_tasks.append(task_id)
            
            # Update status
            self.status = AgentStatus.IDLE if not self.current_tasks else AgentStatus.BUSY
            
        return task_result
        
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """
        Check if the agent can handle a specific task.
        
        Args:
            task: Task definition
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        task_type = task.get('type')
        if not task_type:
            return False
            
        required_capabilities = task.get('required_capabilities', [])
        
        # Check if agent has required capabilities
        for capability in required_capabilities:
            capability_enum = None
            try:
                capability_enum = AgentCapability(capability)
            except ValueError:
                # Check for wildcard capabilities
                for agent_cap in self.capabilities:
                    if agent_cap.value.endswith('*') and capability.startswith(agent_cap.value[:-1]):
                        capability_enum = agent_cap
                        break
                        
            if capability_enum not in self.capabilities:
                return False
                
        # Agent-specific validation
        return await self.validate_task(task)
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status and metrics
        """
        self.last_health_check = datetime.utcnow()
        
        # Calculate uptime
        if self.started_at:
            uptime = self.last_health_check - self.started_at
            self.performance_metrics['uptime_seconds'] = int(uptime.total_seconds())
            
        health_data = {
            'agent_id': self.agent_id,
            'version': self.version,
            'status': self.status.value,
            'uptime_seconds': self.performance_metrics['uptime_seconds'],
            'current_tasks': len(self.current_tasks),
            'completed_tasks': len(self.completed_tasks),
            'error_count': self.error_count,
            'performance_metrics': self.performance_metrics.copy(),
            'last_health_check': self.last_health_check.isoformat(),
            'capabilities': [cap.value for cap in self.capabilities]
        }
        
        # Add agent-specific health data
        try:
            agent_health = await self.check_agent_health()
            health_data.update(agent_health)
        except Exception as e:
            health_data['health_check_error'] = str(e)
            
        return health_data
        
    def get_info(self) -> Dict[str, Any]:
        """Get basic agent information."""
        return {
            'agent_id': self.agent_id,
            'version': self.version,
            'tier': self.tier,
            'role': self.role,
            'status': self.status.value,
            'capabilities': [cap.value for cap in self.capabilities],
            'current_tasks': len(self.current_tasks),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None
        }
        
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a specific task. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate if the agent can execute the task. Must be implemented by subclasses."""
        pass
        
    # Optional lifecycle hooks
    
    async def on_initialize(self) -> None:
        """Called during agent initialization. Override for custom initialization."""
        pass
        
    async def on_shutdown(self) -> None:
        """Called during agent shutdown. Override for custom cleanup."""
        pass
        
    async def check_agent_health(self) -> Dict[str, Any]:
        """Agent-specific health checks. Override for custom health checks."""
        return {'status': 'healthy'}
        
    # Private methods
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)
        
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Perform periodic health checks
                if datetime.utcnow() - self.last_health_check > timedelta(minutes=5):
                    await self.health_check()
                    
                # Clean up old completed tasks (keep last 1000)
                if len(self.completed_tasks) > 1000:
                    self.completed_tasks = self.completed_tasks[-1000:]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error for agent {self.agent_id}: {e}")
                
    async def _wait_for_current_tasks(self) -> None:
        """Wait for all current tasks to complete."""
        while self.current_tasks:
            await asyncio.sleep(0.1)
    
    async def _initialize_agora(self) -> None:
        """Initialize Agora consensus integration."""
        try:
            from ...core.agora import AgoraOrchestrator
            
            # Get Agora configuration from agent config
            agora_config = self.config.get('agora', {})
            
            # Default configuration for Agora
            if not agora_config:
                agora_config = {
                    'llm_config': {
                        'openai_api_key': self.config.get('openai_api_key'),
                        'anthropic_api_key': self.config.get('anthropic_api_key'),
                        'google_api_key': self.config.get('google_api_key'),
                        'perplexity_api_key': self.config.get('perplexity_api_key')
                    },
                    'budget_limits': {
                        'daily_limit': 5.0,
                        'session_limit': 1.0,
                        'loop_limit': 0.25
                    },
                    'quality_threshold': 0.8
                }
            
            self.agora_orchestrator = AgoraOrchestrator(agora_config, self.memory_manager)
            logger.info(f"Agora consensus engine initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Agora for agent {self.agent_id}: {e}")
            # Don't fail agent initialization if Agora fails
    
    async def invoke_agora_consensus(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Agora consensus for complex decision making.
        
        Args:
            task: Task definition for consensus
            
        Returns:
            Consensus result with reasoning trace
        """
        if not self.agora_orchestrator:
            raise RuntimeError(f"Agora not available for agent {self.agent_id}")
        
        # Add agent context to task
        task_with_context = {
            **task,
            'invoking_agent': self.agent_id,
            'agent_tier': self.tier,
            'agent_role': self.role,
            'context': {
                **task.get('context', {}),
                'agent_capabilities': [cap.value for cap in self.capabilities]
            }
        }
        
        logger.info(f"Agent {self.agent_id} invoking Agora consensus for task: {task.get('type', 'unknown')}")
        
        try:
            result = await self.agora_orchestrator.invoke(task_with_context, self.agent_id)
            logger.info(f"Agora consensus completed for agent {self.agent_id} with confidence {result.get('confidence', 0):.2f}")
            return result
        except Exception as e:
            logger.error(f"Agora consensus failed for agent {self.agent_id}: {e}")
            raise