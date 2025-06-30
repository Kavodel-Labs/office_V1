"""
Agent system for Project Aethelred.

Multi-tiered agent architecture:
- Apex: Strategic decision makers (Grandmaster, Chief of Staff)
- Brigade: Tactical coordinators (Project Manager, Protocol Enforcer)
- Doer: Task executors (Backend Dev, Frontend Dev, etc.)
- Service: System observers (Auditor, Reflector)
"""

from .base.agent import Agent, AgentCapability, AgentStatus
from .base.persona import AgentPersona
from .base.registry import AgentRegistry

__all__ = [
    'Agent',
    'AgentCapability', 
    'AgentStatus',
    'AgentPersona',
    'AgentRegistry'
]