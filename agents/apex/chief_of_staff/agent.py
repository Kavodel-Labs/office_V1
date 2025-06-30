"""
Chief of Staff Agent for Project Aethelred.

The Chief of Staff is an Apex tier agent responsible for:
- System governance and configuration management
- Task routing and agent coordination
- Rule enforcement and compliance
- Strategic decision making
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from agents.base.agent import Agent, AgentCapability, AgentStatus, TaskResult
from core.memory.tier_manager import MemoryTierManager
from core.task_management.task_router import TaskRouter
from core.task_management.task_master_integration import TaskMasterIntegration

logger = logging.getLogger(__name__)


class ChiefOfStaff(Agent):
    """
    Chief of Staff - System Governor and Strategic Coordinator.
    
    Responsibilities:
    - Route tasks to appropriate agents
    - Enforce governance rules
    - Manage system configuration
    - Coordinate agent evolution
    - Handle strategic decisions
    """
    
    def __init__(self, memory_manager: MemoryTierManager,
                 task_router: TaskRouter,
                 task_master: TaskMasterIntegration,
                 config: Optional[Dict[str, Any]] = None):
        
        super().__init__(
            agent_id="A_ChiefOfStaff",
            version=1,
            tier="apex", 
            role="System Governor",
            capabilities=[
                AgentCapability.SYSTEM_CONFIG_READ,
                AgentCapability.SYSTEM_CONFIG_WRITE,
                AgentCapability.AGENTS_EVOLUTION_MANAGE,
                AgentCapability.RULES_GOVERNANCE,
                AgentCapability.TASKS_ROUTING,
                AgentCapability.TASKS_ASSIGN
            ],
            config=config or {}
        )
        
        self.memory_manager = memory_manager
        self.task_router = task_router
        self.task_master = task_master
        
        # Chief of Staff specific state
        self.governance_rules: Dict[str, Any] = {}
        self.system_config: Dict[str, Any] = {}
        self.agent_roster: Dict[str, Dict[str, Any]] = {}
        self.routing_statistics: Dict[str, Any] = {}
        
        # Decision making metrics
        self.decisions_made = 0
        self.successful_routings = 0
        self.failed_routings = 0
        
    async def on_initialize(self) -> None:
        """Initialize Chief of Staff specific resources."""
        logger.info("Initializing Chief of Staff agent...")
        
        # Load system configuration
        await self._load_system_config()
        
        # Load governance rules
        await self._load_governance_rules()
        
        # Register with task router
        await self.task_router.register_agent(self.get_info())
        
        # Load agent roster
        await self._load_agent_roster()
        
        logger.info("Chief of Staff initialization complete")
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task assigned to the Chief of Staff.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        logger.debug(f"CHIEF OF STAFF: Received task: {task.get('type')}")
        task_type = task.get('type')
        
        if task_type == 'route_task':
            return await self._handle_task_routing(task)
        elif task_type == 'system_config':
            return await self._handle_system_config(task)
        elif task_type == 'governance_check':
            return await self._handle_governance_check(task)
        elif task_type == 'agent_management':
            return await self._handle_agent_management(task)
        elif task_type == 'strategic_decision':
            return await self._handle_strategic_decision(task)
        elif task_type == 'system_status':
            return await self._handle_system_status(task)
        else:
            logger.error(f"CHIEF OF STAFF: Unknown task type: {task_type}")
            raise ValueError(f"Unknown task type: {task_type}")
            
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the Chief of Staff can execute the task.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task can be executed
        """
        valid_task_types = {
            'route_task',
            'system_config', 
            'governance_check',
            'agent_management',
            'strategic_decision',
            'system_status'
        }
        
        task_type = task.get('type')
        return task_type in valid_task_types
        
    async def check_agent_health(self) -> Dict[str, Any]:
        """Chief of Staff specific health checks."""
        health_data = {
            'decisions_made': self.decisions_made,
            'successful_routings': self.successful_routings,
            'failed_routings': self.failed_routings,
            'routing_success_rate': (
                self.successful_routings / max(1, self.successful_routings + self.failed_routings)
            ),
            'governance_rules_count': len(self.governance_rules),
            'registered_agents': len(self.agent_roster),
            'memory_tiers_status': 'healthy'  # Will be updated with actual check
        }
        
        # Check memory system health
        try:
            memory_health = await self.memory_manager.health_check()
            healthy_tiers = sum(1 for tier_health in memory_health.values() 
                              if tier_health.get('status') == 'healthy')
            total_tiers = len(memory_health)
            
            health_data['memory_tiers_status'] = f"{healthy_tiers}/{total_tiers} healthy"
            health_data['memory_details'] = memory_health
            
        except Exception as e:
            health_data['memory_tiers_status'] = f"error: {e}"
            
        return health_data
        
    # Task handlers
    
    async def _handle_task_routing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task routing requests."""
        target_task = task.get('target_task')
        if not target_task:
            raise ValueError("No target_task provided for routing")
            
        routing_strategy = task.get('routing_strategy')
        
        logger.info(f"Routing task {target_task.get('id')} with strategy {routing_strategy}")
        
        try:
            # Route the task
            selected_agent = await self.task_router.route_task(target_task, routing_strategy)
            
            if selected_agent:
                self.successful_routings += 1
                self.decisions_made += 1
                
                # Update routing statistics
                await self._update_routing_statistics()
                
                return {
                    'action': 'task_routed',
                    'target_task_id': target_task.get('id'),
                    'assigned_to': selected_agent,
                    'routing_strategy': routing_strategy,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                self.failed_routings += 1
                return {
                    'action': 'routing_failed',
                    'target_task_id': target_task.get('id'),
                    'reason': 'No suitable agent found',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.failed_routings += 1
            logger.error(f"Task routing failed: {e}")
            raise
            
    async def _handle_system_config(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system configuration tasks."""
        action = task.get('action')
        config_key = task.get('config_key')
        config_value = task.get('config_value')
        
        if action == 'get':
            value = self.system_config.get(config_key)
            return {
                'action': 'config_retrieved',
                'key': config_key,
                'value': value
            }
        elif action == 'set':
            if not config_key:
                raise ValueError("config_key required for set action")
                
            self.system_config[config_key] = config_value
            
            # Persist to memory
            await self.memory_manager.write(
                "system_config",
                self.system_config,
                target_tiers=['hot', 'warm']
            )
            
            self.decisions_made += 1
            
            return {
                'action': 'config_updated',
                'key': config_key,
                'value': config_value,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            raise ValueError(f"Unknown config action: {action}")
            
    async def _handle_governance_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle governance rule checking."""
        rule_type = task.get('rule_type')
        context = task.get('context', {})
        
        violations = []
        applicable_rules = []
        
        for rule_id, rule in self.governance_rules.items():
            if rule.get('type') == rule_type or rule_type is None:
                applicable_rules.append(rule_id)
                
                # Basic rule checking (would be more sophisticated in practice)
                if await self._check_rule_violation(rule, context):
                    violations.append({
                        'rule_id': rule_id,
                        'rule_name': rule.get('name'),
                        'violation_type': 'policy_violation',
                        'context': context
                    })
                    
        return {
            'action': 'governance_checked',
            'rule_type': rule_type,
            'applicable_rules': applicable_rules,
            'violations': violations,
            'compliant': len(violations) == 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_agent_management(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent management tasks."""
        action = task.get('action')
        agent_id = task.get('agent_id')
        
        if action == 'register':
            agent_info = task.get('agent_info')
            if not agent_info:
                raise ValueError("agent_info required for register action")
                
            self.agent_roster[agent_id] = agent_info
            await self.task_router.register_agent(agent_info)
            
            # Persist to memory
            await self.memory_manager.write(
                f"agent_roster:{agent_id}",
                agent_info,
                target_tiers=['hot', 'warm']
            )
            
            return {
                'action': 'agent_registered',
                'agent_id': agent_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        elif action == 'unregister':
            self.agent_roster.pop(agent_id, None)
            await self.task_router.unregister_agent(agent_id)
            
            # Remove from memory
            await self.memory_manager.delete(f"agent_roster:{agent_id}")
            
            return {
                'action': 'agent_unregistered',
                'agent_id': agent_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        elif action == 'list':
            return {
                'action': 'agents_listed',
                'agents': list(self.agent_roster.keys()),
                'count': len(self.agent_roster)
            }
            
        else:
            raise ValueError(f"Unknown agent management action: {action}")
            
    async def _handle_strategic_decision(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategic decision making."""
        decision_type = task.get('decision_type')
        context = task.get('context', {})
        
        self.decisions_made += 1
        
        if decision_type == 'resource_allocation':
            return await self._make_resource_allocation_decision(context)
        elif decision_type == 'priority_adjustment':
            return await self._make_priority_adjustment_decision(context)
        elif decision_type == 'system_scaling':
            return await self._make_scaling_decision(context)
        else:
            return {
                'action': 'decision_deferred',
                'decision_type': decision_type,
                'reason': 'Decision type not yet implemented',
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _handle_system_status(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system status requests."""
        # Get routing statistics
        routing_stats = await self.task_router.get_routing_statistics()
        
        # Get task master statistics
        task_stats = self.task_master.get_statistics()
        
        # Get memory system status
        memory_status = await self.memory_manager.health_check()
        
        return {
            'action': 'system_status',
            'chief_of_staff': {
                'decisions_made': self.decisions_made,
                'routing_success_rate': (
                    self.successful_routings / max(1, self.successful_routings + self.failed_routings)
                ),
                'governance_rules': len(self.governance_rules),
                'registered_agents': len(self.agent_roster)
            },
            'routing_statistics': routing_stats,
            'task_statistics': task_stats,
            'memory_status': memory_status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    # Helper methods
    
    async def _load_system_config(self) -> None:
        """Load system configuration from memory."""
        try:
            config = await self.memory_manager.read("system_config")
            if config:
                self.system_config = config
            else:
                # Initialize with defaults
                self.system_config = {
                    'max_tasks_per_agent': 5,
                    'task_timeout_seconds': 3600,
                    'evolution_enabled': False,
                    'log_level': 'INFO'
                }
                await self.memory_manager.write(
                    "system_config",
                    self.system_config,
                    target_tiers=['hot', 'warm']
                )
        except Exception as e:
            logger.error(f"Failed to load system config: {e}")
            self.system_config = {}
            
    async def _load_governance_rules(self) -> None:
        """Load governance rules from memory."""
        try:
            rules = await self.memory_manager.read("governance_rules")
            if rules:
                self.governance_rules = rules
            else:
                # Initialize with basic rules
                self.governance_rules = {
                    'code_quality': {
                        'type': 'code_standards',
                        'name': 'Code Quality Standards',
                        'enabled': True,
                        'rules': ['pep8_compliance', 'test_coverage_80']
                    },
                    'task_assignment': {
                        'type': 'task_routing',
                        'name': 'Task Assignment Rules',
                        'enabled': True,
                        'rules': ['capability_match', 'workload_balance']
                    }
                }
                await self.memory_manager.write(
                    "governance_rules",
                    self.governance_rules,
                    target_tiers=['hot', 'warm']
                )
        except Exception as e:
            logger.error(f"Failed to load governance rules: {e}")
            self.governance_rules = {}
            
    async def _load_agent_roster(self) -> None:
        """Load agent roster from memory."""
        try:
            # This would typically load from database
            # For now, initialize empty and agents will register themselves
            self.agent_roster = {}
        except Exception as e:
            logger.error(f"Failed to load agent roster: {e}")
            self.agent_roster = {}
            
    async def _check_rule_violation(self, rule: Dict[str, Any], 
                                   context: Dict[str, Any]) -> bool:
        """Check if a context violates a governance rule."""
        # Simplified rule checking - would be more sophisticated in practice
        rule_type = rule.get('type')
        
        if rule_type == 'code_standards':
            # Check code quality metrics
            code_quality = context.get('code_quality', {})
            if code_quality.get('pep8_violations', 0) > 0:
                return True
            if code_quality.get('test_coverage', 100) < 80:
                return True
                
        elif rule_type == 'task_routing':
            # Check task assignment rules
            if context.get('assigned_agent') and not context.get('capability_verified'):
                return True
                
        return False
        
    async def _update_routing_statistics(self) -> None:
        """Update routing statistics."""
        self.routing_statistics = await self.task_router.get_routing_statistics()
        
        # Store in memory
        await self.memory_manager.write(
            "routing_statistics",
            self.routing_statistics,
            target_tiers=['hot']
        )
        
    async def _make_resource_allocation_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make resource allocation decisions."""
        # Simplified decision making
        return {
            'action': 'resource_allocation_decided',
            'decision': 'maintain_current_allocation',
            'reasoning': 'System load within acceptable parameters',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _make_priority_adjustment_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make priority adjustment decisions."""
        return {
            'action': 'priority_adjustment_decided',
            'decision': 'no_adjustment_needed',
            'reasoning': 'Task priorities aligned with system goals',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _make_scaling_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make system scaling decisions."""
        return {
            'action': 'scaling_decision_made',
            'decision': 'maintain_current_scale',
            'reasoning': 'Resource utilization within target range',
            'timestamp': datetime.utcnow().isoformat()
        }