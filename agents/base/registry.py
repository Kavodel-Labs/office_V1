"""
Agent registry for Project Aethelred.

Manages agent registration, discovery, and lifecycle.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .agent import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for all agents in the Aethelred system.
    
    Manages agent lifecycle, discovery, and communication routing.
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def register_agent(self, agent: Agent) -> None:
        """Register an agent with the registry."""
        agent_id = agent.agent_id
        
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, updating...")
            
        self.agents[agent_id] = agent
        self.agent_metadata[agent_id] = {
            'registered_at': datetime.utcnow(),
            'last_seen': datetime.utcnow(),
            'status': agent.status,
            'capabilities': [cap.value for cap in agent.capabilities]
        }
        
        logger.info(f"Registered agent: {agent_id}")
        
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_metadata[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False
        
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
        
    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.agents.keys())
        
    def get_agents_by_capability(self, capability: str) -> List[Agent]:
        """Get all agents with a specific capability."""
        matching_agents = []
        
        for agent in self.agents.values():
            if any(cap.value == capability or 
                   (cap.value.endswith('*') and capability.startswith(cap.value[:-1]))
                   for cap in agent.capabilities):
                matching_agents.append(agent)
                
        return matching_agents
        
    def get_agents_by_tier(self, tier: str) -> List[Agent]:
        """Get all agents in a specific tier."""
        return [agent for agent in self.agents.values() if agent.tier == tier]
        
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all registered agents."""
        health_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                health = await agent.health_check()
                health_results[agent_id] = health
                
                # Update last seen
                self.agent_metadata[agent_id]['last_seen'] = datetime.utcnow()
                self.agent_metadata[agent_id]['status'] = agent.status
                
            except Exception as e:
                health_results[agent_id] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return health_results
        
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status."""
        total_agents = len(self.agents)
        agents_by_tier = {}
        agents_by_status = {}
        
        for agent in self.agents.values():
            # Count by tier
            tier = agent.tier
            agents_by_tier[tier] = agents_by_tier.get(tier, 0) + 1
            
            # Count by status
            status = agent.status.value
            agents_by_status[status] = agents_by_status.get(status, 0) + 1
            
        return {
            'total_agents': total_agents,
            'agents_by_tier': agents_by_tier,
            'agents_by_status': agents_by_status,
            'registry_metadata': self.agent_metadata
        }