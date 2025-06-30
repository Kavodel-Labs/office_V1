"""
Main entry point for Project Aethelred.

Initializes and orchestrates the complete Aethelred system including:
- Memory tier management
- Agent system initialization
- Task management integration
- System monitoring and health checks
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from datetime import datetime
from dotenv import load_dotenv

# Core system imports
from core.memory.tier_manager import MemoryTierManager
from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
from core.memory.coherence import MemoryCoherenceManager
from core.task_management.task_master_integration import TaskMasterIntegration
from core.task_management.task_router import TaskRouter

import uvicorn
from web_server import app, set_secretary_agent

# Agent imports
from agents.apex.chief_of_staff.agent import ChiefOfStaff
from agents.service.auditor.agent import Auditor
from agents.service.secretary.agent import Secretary
from agents.doer.developer.agent import Developer

# Agora Consensus Engine
from core.agora import AgoraOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/aethelred.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class AethelredSystem:
    """
    Main Aethelred system orchestrator.
    
    Manages the complete lifecycle of the Aethelred autonomous AI system.
    """
    
    def __init__(self, config_path: str = "config/aethelred-config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        # Core components
        self.memory_manager: Optional[MemoryTierManager] = None
        self.coherence_manager: Optional[MemoryCoherenceManager] = None
        self.task_master: Optional[TaskMasterIntegration] = None
        self.task_router: Optional[TaskRouter] = None
        
        # Core agents
        self.chief_of_staff: Optional[ChiefOfStaff] = None
        self.auditor: Optional[Auditor] = None
        self.secretary: Optional[Secretary] = None
        self.developer: Optional[Developer] = None
        
        # Agora Consensus Engine
        self.agora_orchestrator: Optional[AgoraOrchestrator] = None
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize the complete Aethelred system."""
        logger.info("🚀 Initializing Project Aethelred...")
        
        try:
            # Load configuration
            await self._load_configuration()
            
            # Initialize memory system
            await self._initialize_memory_system()
            
            # Initialize task management
            await self._initialize_task_management()
            
            # Initialize core agents
            await self._initialize_core_agents()
            
            # Initialize Agora Consensus Engine
            await self._initialize_agora()
            
            # Perform system health check
            await self._perform_startup_health_check()
            
            self.start_time = datetime.utcnow()
            self.is_running = True
            
            logger.info("✅ Project Aethelred initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Aethelred initialization failed: {e}")
            raise
            
    async def run(self) -> None:
        """Run the Aethelred system."""
        if not self.is_running:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        logger.info("🎯 Starting Project Aethelred autonomous operation...")
        
        try:
            # Start the web server
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
            server = uvicorn.Server(config)
            web_server_task = asyncio.create_task(server.serve())
            
            # Start background monitoring
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start core agent tasks
            agent_tasks = []
            if self.chief_of_staff:
                # Chief of Staff doesn't need a continuous loop - it responds to tasks
                pass
            if self.auditor:
                # Auditor runs continuous monitoring
                pass
                
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel background tasks
            web_server_task.cancel()
            monitoring_task.cancel()
            for task in agent_tasks:
                task.cancel()
                
            await asyncio.gather(web_server_task, monitoring_task, *agent_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error during system operation: {e}")
            raise
        finally:
            await self.shutdown()
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the Aethelred system."""
        logger.info("🔄 Shutting down Project Aethelred...")
        
        self.is_running = False
        
        try:
            # Shutdown agents
            if self.chief_of_staff:
                await self.chief_of_staff.shutdown()
            if self.auditor:
                await self.auditor.shutdown()
            if self.secretary:
                await self.secretary.shutdown()
            if self.developer:
                await self.developer.shutdown()
                
            # Cleanup coherence manager
            if self.coherence_manager:
                await self.coherence_manager.cleanup_background_tasks()
                
            # Disconnect memory tiers
            if self.memory_manager:
                await self.memory_manager.disconnect_all()
                
            logger.info("✅ Project Aethelred shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    # Initialization methods
    
    async def _load_configuration(self) -> None:
        """Load system configuration."""
        logger.info("📋 Loading configuration...")
        
        # Load environment variables from .env file
        load_dotenv()
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_str = f.read()
        
        # Expand environment variables
        config_str = os.path.expandvars(config_str)
        
        self.config = yaml.safe_load(config_str)
            
        logger.info(f"Configuration loaded from {self.config_path}")
        
    async def _initialize_memory_system(self) -> None:
        """Initialize the memory tier system."""
        logger.info("🧠 Initializing memory system...")
        
        # Create memory tier manager
        self.memory_manager = MemoryTierManager()
        
        # Initialize memory tiers based on configuration
        memory_config = self.config.get('cognitive', {}).get('memory_tiers', [])
        
        for tier_config in memory_config:
            tier_name = tier_config['name']
            tier_backend = tier_config['backend']
            tier_settings = tier_config.get('config', {})
            
            logger.info(f"Initializing {tier_name} memory tier ({tier_backend})...")
            
            if tier_backend == 'redis':
                tier = HotMemory(tier_settings)
            elif tier_backend == 'postgresql':
                tier = WarmMemory(tier_settings)
            elif tier_backend == 'neo4j':
                tier = ColdMemory(tier_settings)
            elif tier_backend == 'filesystem':
                tier = ArchiveMemory(tier_settings)
            else:
                logger.warning(f"Unknown memory backend: {tier_backend}")
                continue
                
            self.memory_manager.register_tier(tier)
            
        # Connect to all memory tiers
        await self.memory_manager.connect_all()
        
        # Initialize coherence manager
        self.coherence_manager = MemoryCoherenceManager(self.memory_manager)
        coherence_config = self.config.get('cognitive', {}).get('coherence', {})
        self.coherence_manager.configure(coherence_config)
        
        logger.info("Memory system initialization complete")
        
    async def _initialize_task_management(self) -> None:
        """Initialize task management system."""
        logger.info("📋 Initializing task management...")
        
        # Initialize task-master-ai integration
        task_config = self.config.get('communication', {}).get('task_master', {})
        task_config['project_root'] = Path.cwd()
        
        self.task_master = TaskMasterIntegration(self.memory_manager, task_config)
        await self.task_master.initialize()
        
        # Initialize task router
        self.task_router = TaskRouter(self.memory_manager, self.task_master)
        
        logger.info("Task management initialization complete")
        
    async def _initialize_core_agents(self) -> None:
        """Initialize core system agents."""
        logger.info("🤖 Initializing core agents...")
        
        # Initialize Chief of Staff
        chief_config = self._get_agent_config("A_ChiefOfStaff_v1")
        self.chief_of_staff = ChiefOfStaff(
            self.memory_manager,
            self.task_router,
            self.task_master,
            chief_config
        )
        await self.chief_of_staff.initialize()
        
        # Initialize Auditor
        auditor_config = self._get_agent_config("S_Auditor_v1")
        self.auditor = Auditor(self.memory_manager, auditor_config)
        await self.auditor.initialize()
        
        # Initialize Secretary (Communication Manager)
        secretary_config = self._get_agent_config("S_Secretary_v1")
        secretary_config.update({
            'mcp': {
                'slack_enabled': True,
                'cursor_integration': True
            }
        })
        self.secretary = Secretary(self.memory_manager, secretary_config)
        await self.secretary.initialize()
        
        # Initialize Developer (MCP Development Specialist)
        developer_config = self._get_agent_config("D_Developer_v1")
        self.developer = Developer(self.memory_manager, developer_config)
        await self.developer.initialize()
        
        # Pass the secretary agent to the web server
        set_secretary_agent(self.secretary)
        
        logger.info("Core agents initialization complete")
        
    async def _initialize_agora(self) -> None:
        """Initialize Agora Consensus Engine."""
        logger.info("🏛️ Initializing Agora Consensus Engine...")
        
        try:
            # Get Agora configuration
            agora_config = self.config.get('agora', {})
            
            # Default configuration if not in config
            if not agora_config:
                agora_config = {
                    'llm_config': {
                        'openai_api_key': self.config.get('openai_api_key'),
                        'anthropic_api_key': self.config.get('anthropic_api_key'),
                        'google_api_key': self.config.get('google_api_key'),
                        'perplexity_api_key': self.config.get('perplexity_api_key')
                    },
                    'budget_limits': {
                        'daily_limit': 10.0,
                        'session_limit': 2.0,
                        'loop_limit': 0.5
                    },
                    'quality_threshold': 0.8
                }
            
            # Initialize Agora Orchestrator
            self.agora_orchestrator = AgoraOrchestrator(agora_config, self.memory_manager)
            
            logger.info("Agora Consensus Engine initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agora: {e}")
            # Don't fail system startup if Agora fails
        
    def _get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        agent_roster = self.config.get('agents', {}).get('roster', [])
        for agent_config in agent_roster:
            if agent_config.get('id') == agent_id:
                return agent_config
        return {}
        
    async def _perform_startup_health_check(self) -> None:
        """Perform comprehensive startup health check."""
        logger.info("🏥 Performing startup health check...")
        
        # Check memory system
        memory_health = await self.memory_manager.health_check()
        healthy_tiers = sum(1 for status in memory_health.values() 
                           if status.get('status') == 'healthy')
        total_tiers = len(memory_health)
        
        logger.info(f"Memory system: {healthy_tiers}/{total_tiers} tiers healthy")
        
        # Check agents
        if self.chief_of_staff:
            chief_health = await self.chief_of_staff.health_check()
            logger.info(f"Chief of Staff: {chief_health['status']}")
            
        if self.auditor:
            auditor_health = await self.auditor.health_check()
            logger.info(f"Auditor: {auditor_health['status']}")
            
        if self.secretary:
            secretary_health = await self.secretary.health_check()
            logger.info(f"Secretary: {secretary_health['status']}")
            
        if self.developer:
            developer_health = await self.developer.health_check()
            logger.info(f"Developer: {developer_health['status']}")
            
        # Check Agora
        if self.agora_orchestrator:
            logger.info("Agora Consensus Engine: initialized")
            
        # Check task master
        task_stats = self.task_master.get_statistics()
        logger.info(f"Task Master: {task_stats['integration_status']}")
        
        logger.info("Startup health check complete")
        
    async def _monitoring_loop(self) -> None:
        """Background system monitoring loop."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Perform periodic health checks
                if self.auditor:
                    # Request system health check via auditor
                    health_task = {
                        'id': f"health_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        'type': 'system_health_check',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    try:
                        await self.auditor.process_task(health_task)
                    except Exception as e:
                        logger.error(f"Health check failed: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'system_name': 'aethelred',
            'version': '6.0.0',
            'edition': 'developer',
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (
                (datetime.utcnow() - self.start_time).total_seconds() 
                if self.start_time else 0
            ),
            'memory_manager_status': self.memory_manager is not None,
            'task_master_status': self.task_master is not None,
            'chief_of_staff_status': self.chief_of_staff is not None,
            'auditor_status': self.auditor is not None,
            'secretary_status': self.secretary is not None,
            'developer_status': self.developer is not None,
            'agora_status': self.agora_orchestrator is not None
        }


async def main():
    """Main entry point."""
    # Setup signal handlers for graceful shutdown
    system = AethelredSystem()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        system.shutdown_event.set()
        
    # Register signal handlers
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize and run system
        await system.initialize()
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        if system.is_running:
            await system.shutdown()


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("aethelred_archive").mkdir(exist_ok=True)
    
    # Run the system
    asyncio.run(main())