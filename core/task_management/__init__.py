"""
Task management integration for Project Aethelred.

Integrates task-master-ai with the Aethelred agent system for 
advanced task orchestration and management.
"""

from .task_master_integration import TaskMasterIntegration
from .task_router import TaskRouter
from .workflow_engine import WorkflowEngine

__all__ = [
    'TaskMasterIntegration',
    'TaskRouter', 
    'WorkflowEngine'
]