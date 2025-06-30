"""
Agora Consensus Engine for AETHELRED
Multi-LLM consensus framework for superior AI reasoning
"""

from .orchestrator import AgoraOrchestrator, AgoraSession, SpecialistResponse
from .merge_engine import MergeEngine
from .critic_engine import CriticEngine
from .cost_controller import CostController
from .role_assignment import RoleAssignment

__all__ = [
    'AgoraOrchestrator',
    'AgoraSession', 
    'SpecialistResponse',
    'MergeEngine',
    'CriticEngine',
    'CostController',
    'RoleAssignment'
]