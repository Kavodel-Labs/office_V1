"""
Task routing system for Project Aethelred.

Routes tasks to appropriate agents based on capabilities, workload,
and performance metrics.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..memory.tier_manager import MemoryTierManager
from .task_master_integration import TaskMasterIntegration

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Task routing strategies."""
    CAPABILITY_FIRST = "capability_first"  # Route based on capabilities first
    LOAD_BALANCED = "load_balanced"       # Balance load across agents
    PERFORMANCE_BASED = "performance_based"  # Route to best performing agent
    ROUND_ROBIN = "round_robin"           # Simple round-robin assignment


class TaskRouter:
    """
    Intelligent task routing system.
    
    Routes tasks to the most appropriate agents based on various factors
    including capabilities, current workload, and historical performance.
    """
    
    def __init__(self, memory_manager: MemoryTierManager, 
                 task_master: TaskMasterIntegration):
        self.memory_manager = memory_manager
        self.task_master = task_master
        
        # Routing configuration
        self.default_strategy = RoutingStrategy.CAPABILITY_FIRST
        self.max_tasks_per_agent = 5
        self.performance_weight = 0.4
        self.load_weight = 0.3
        self.capability_weight = 0.3
        
        # Agent registry and metrics
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        
        # Round-robin state
        self._round_robin_index = 0
        
    async def register_agent(self, agent_info: Dict[str, Any]) -> None:
        """
        Register an agent with the router.
        
        Args:
            agent_info: Agent information including capabilities
        """
        agent_id = agent_info['agent_id']
        self.agents[agent_id] = agent_info.copy()
        
        # Initialize performance metrics if not exists
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'success_rate': 1.0,
                'avg_completion_time': 1000.0,  # ms
                'total_tasks': 0,
                'failed_tasks': 0,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        # Store in memory
        await self.memory_manager.write(
            f"agent_registry:{agent_id}",
            agent_info,
            target_tiers=['hot', 'warm']
        )
        
        logger.info(f"Registered agent {agent_id} with router")
        
    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the router.
        
        Args:
            agent_id: Agent identifier
        """
        self.agents.pop(agent_id, None)
        
        # Remove from memory
        await self.memory_manager.delete(f"agent_registry:{agent_id}")
        
        logger.info(f"Unregistered agent {agent_id} from router")
        
    async def route_task(self, task: Dict[str, Any], 
                        strategy: Optional[RoutingStrategy] = None) -> Optional[str]:
        """
        Route a task to the best available agent.
        
        Args:
            task: Task to route
            strategy: Routing strategy to use
            
        Returns:
            Agent ID if successful, None if no suitable agent found
        """
        logger.debug(f"TASKROUTER: Routing task: {task.get('id', 'N/A')}")
        routing_strategy = strategy or self.default_strategy
        
        # Get eligible agents
        eligible_agents = await self._get_eligible_agents(task)
        if not eligible_agents:
            logger.warning(f"TASKROUTER: No eligible agents found for task {task.get('id')}")
            return None
            
        logger.debug(f"TASKROUTER: Eligible agents for task {task.get('id', 'N/A')}: {eligible_agents}")
            
        # Apply routing strategy
        selected_agent = await self._apply_routing_strategy(
            task, eligible_agents, routing_strategy
        )
        
        if selected_agent:
            logger.debug(f"TASKROUTER: Selected agent {selected_agent} for task {task.get('id', 'N/A')}")
            # Record routing decision
            await self._record_routing_decision(task, selected_agent, routing_strategy)
            
            # Assign task via task-master-ai
            assignment_result = await self.task_master.assign_task_to_agent(
                task['id'], selected_agent
            )
            
            if assignment_result['success']:
                logger.info(f"TASKROUTER: Routed task {task.get('id')} to agent {selected_agent}")
                return selected_agent
            else:
                logger.error(f"TASKROUTER: Failed to assign task {task.get('id')} to agent {selected_agent}")
                return None
        else:
            logger.warning(f"TASKROUTER: No suitable agent selected for task {task.get('id')}")
            return None
            
    async def _get_eligible_agents(self, task: Dict[str, Any]) -> List[str]:
        """
        Get list of agents eligible to handle the task.
        
        Args:
            task: Task to check eligibility for
            
        Returns:
            List of eligible agent IDs
        """
        eligible_agents = []
        required_capabilities = task.get('required_capabilities', [])
        
        for agent_id, agent_info in self.agents.items():
            # Check if agent is active
            if not agent_info.get('is_active', True):
                continue
                
            # Check capabilities
            agent_capabilities = set(agent_info.get('capabilities', []))
            
            # Check if agent has all required capabilities
            has_capabilities = True
            for required_cap in required_capabilities:
                if not self._agent_has_capability(agent_capabilities, required_cap):
                    has_capabilities = False
                    break
                    
            if has_capabilities:
                # Check current workload
                current_tasks = await self._get_agent_current_tasks(agent_id)
                if len(current_tasks) < self.max_tasks_per_agent:
                    eligible_agents.append(agent_id)
                    
        return eligible_agents
        
    def _agent_has_capability(self, agent_capabilities: set, required_capability: str) -> bool:
        """
        Check if agent has a specific capability.
        
        Args:
            agent_capabilities: Set of agent capabilities
            required_capability: Required capability
            
        Returns:
            True if agent has the capability
        """
        # Direct match
        if required_capability in agent_capabilities:
            return True
            
        # Wildcard match (e.g., "code.backend.*" matches "code.backend.develop")
        for agent_cap in agent_capabilities:
            if agent_cap.endswith('*'):
                if required_capability.startswith(agent_cap[:-1]):
                    return True
                    
        return False
        
    async def _get_agent_current_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get current tasks assigned to an agent."""
        try:
            return await self.task_master.get_agent_tasks(agent_id)
        except Exception as e:
            logger.error(f"Failed to get tasks for agent {agent_id}: {e}")
            return []
            
    async def _apply_routing_strategy(self, task: Dict[str, Any], 
                                     eligible_agents: List[str],
                                     strategy: RoutingStrategy) -> Optional[str]:
        """
        Apply the specified routing strategy to select an agent.
        
        Args:
            task: Task to route
            eligible_agents: List of eligible agent IDs
            strategy: Routing strategy to apply
            
        Returns:
            Selected agent ID or None
        """
        if not eligible_agents:
            return None
            
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(eligible_agents)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._route_load_balanced(eligible_agents)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return await self._route_performance_based(eligible_agents)
        else:  # CAPABILITY_FIRST
            return await self._route_capability_first(task, eligible_agents)
            
    def _route_round_robin(self, eligible_agents: List[str]) -> str:
        """Simple round-robin routing."""
        selected_agent = eligible_agents[self._round_robin_index % len(eligible_agents)]
        self._round_robin_index += 1
        return selected_agent
        
    async def _route_load_balanced(self, eligible_agents: List[str]) -> str:
        """Route to agent with lowest current workload."""
        agent_loads = {}
        
        for agent_id in eligible_agents:
            current_tasks = await self._get_agent_current_tasks(agent_id)
            agent_loads[agent_id] = len(current_tasks)
            
        # Select agent with minimum load
        return min(agent_loads.keys(), key=lambda x: agent_loads[x])
        
    async def _route_performance_based(self, eligible_agents: List[str]) -> str:
        """Route to best performing agent."""
        if len(eligible_agents) == 1:
            return eligible_agents[0]
            
        # Calculate performance scores
        agent_scores = {}
        for agent_id in eligible_agents:
            performance = self.agent_performance.get(agent_id, {})
            
            # Calculate composite score
            success_rate = performance.get('success_rate', 1.0)
            avg_time = performance.get('avg_completion_time', 1000.0)
            
            # Normalize and combine metrics (higher is better)
            time_score = max(0, 1.0 - (avg_time / 10000.0))  # Normalize to 0-1
            composite_score = (success_rate * 0.7) + (time_score * 0.3)
            
            agent_scores[agent_id] = composite_score
            
        # Select agent with highest score
        return max(agent_scores.keys(), key=lambda x: agent_scores[x])
        
    async def _route_capability_first(self, task: Dict[str, Any], 
                                     eligible_agents: List[str]) -> str:
        """Route based on best capability match."""
        if len(eligible_agents) == 1:
            return eligible_agents[0]
            
        # Score agents based on capability match
        required_capabilities = task.get('required_capabilities', [])
        agent_scores = {}
        
        for agent_id in eligible_agents:
            agent_info = self.agents[agent_id]
            agent_capabilities = set(agent_info.get('capabilities', []))
            
            # Calculate capability match score
            match_score = 0
            for req_cap in required_capabilities:
                if self._agent_has_capability(agent_capabilities, req_cap):
                    match_score += 1
                    
            # Bonus for exact matches vs wildcard matches
            exact_matches = len(set(required_capabilities) & agent_capabilities)
            match_score += exact_matches * 0.5
            
            agent_scores[agent_id] = match_score
            
        # Select agent with highest capability match
        return max(agent_scores.keys(), key=lambda x: agent_scores[x])
        
    async def _record_routing_decision(self, task: Dict[str, Any], 
                                      selected_agent: str,
                                      strategy: RoutingStrategy) -> None:
        """Record routing decision for analysis."""
        routing_record = {
            'task_id': task.get('id'),
            'task_type': task.get('type'),
            'selected_agent': selected_agent,
            'strategy': strategy.value,
            'timestamp': datetime.utcnow().isoformat(),
            'eligible_agents': len([a for a in self.agents.keys() 
                                  if self.agents[a].get('is_active', True)])
        }
        
        self.routing_history.append(routing_record)
        
        # Keep only last 1000 routing decisions
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
            
        # Store in memory
        await self.memory_manager.write(
            f"routing_decision:{task.get('id')}",
            routing_record,
            target_tiers=['warm']
        )
        
    async def update_agent_performance(self, agent_id: str, 
                                      task_result: Dict[str, Any]) -> None:
        """
        Update agent performance metrics based on task result.
        
        Args:
            agent_id: Agent identifier
            task_result: Task execution result
        """
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'success_rate': 1.0,
                'avg_completion_time': 1000.0,
                'total_tasks': 0,
                'failed_tasks': 0,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        metrics = self.agent_performance[agent_id]
        
        # Update task counts
        metrics['total_tasks'] += 1
        if not task_result.get('success', True):
            metrics['failed_tasks'] += 1
            
        # Update success rate
        metrics['success_rate'] = 1.0 - (metrics['failed_tasks'] / metrics['total_tasks'])
        
        # Update average completion time
        if 'duration_ms' in task_result:
            duration = task_result['duration_ms']
            current_avg = metrics['avg_completion_time']
            total_tasks = metrics['total_tasks']
            
            # Weighted average
            metrics['avg_completion_time'] = (
                (current_avg * (total_tasks - 1) + duration) / total_tasks
            )
            
        metrics['last_updated'] = datetime.utcnow().isoformat()
        
        # Store updated metrics
        await self.memory_manager.write(
            f"agent_performance:{agent_id}",
            metrics,
            target_tiers=['hot', 'warm']
        )
        
        logger.debug(f"Updated performance metrics for agent {agent_id}")
        
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and analytics."""
        total_routes = len(self.routing_history)
        if total_routes == 0:
            return {'total_routes': 0}
            
        # Strategy usage
        strategy_counts = {}
        for record in self.routing_history:
            strategy = record['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
        # Agent utilization
        agent_utilization = {}
        for record in self.routing_history:
            agent = record['selected_agent']
            agent_utilization[agent] = agent_utilization.get(agent, 0) + 1
            
        # Recent routing rate
        recent_routes = [
            r for r in self.routing_history
            if datetime.fromisoformat(r['timestamp']) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return {
            'total_routes': total_routes,
            'strategy_distribution': strategy_counts,
            'agent_utilization': agent_utilization,
            'routes_last_hour': len(recent_routes),
            'active_agents': len([a for a in self.agents.values() 
                                if a.get('is_active', True)]),
            'total_registered_agents': len(self.agents)
        }
        
    async def get_agent_recommendations(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get agent recommendations for a task with scoring.
        
        Args:
            task: Task to get recommendations for
            
        Returns:
            List of agent recommendations with scores
        """
        eligible_agents = await self._get_eligible_agents(task)
        recommendations = []
        
        for agent_id in eligible_agents:
            agent_info = self.agents[agent_id]
            performance = self.agent_performance.get(agent_id, {})
            current_tasks = await self._get_agent_current_tasks(agent_id)
            
            # Calculate composite score
            capability_score = self._calculate_capability_score(task, agent_info)
            performance_score = performance.get('success_rate', 1.0)
            load_score = max(0, 1.0 - (len(current_tasks) / self.max_tasks_per_agent))
            
            composite_score = (
                capability_score * self.capability_weight +
                performance_score * self.performance_weight +
                load_score * self.load_weight
            )
            
            recommendations.append({
                'agent_id': agent_id,
                'agent_info': agent_info,
                'composite_score': composite_score,
                'capability_score': capability_score,
                'performance_score': performance_score,
                'load_score': load_score,
                'current_tasks': len(current_tasks)
            })
            
        # Sort by composite score (descending)
        recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return recommendations
        
    def _calculate_capability_score(self, task: Dict[str, Any], 
                                   agent_info: Dict[str, Any]) -> float:
        """Calculate how well an agent's capabilities match a task."""
        required_capabilities = task.get('required_capabilities', [])
        if not required_capabilities:
            return 1.0
            
        agent_capabilities = set(agent_info.get('capabilities', []))
        matches = 0
        
        for req_cap in required_capabilities:
            if self._agent_has_capability(agent_capabilities, req_cap):
                matches += 1
                
        return matches / len(required_capabilities)