"""
Task Master AI integration for Project Aethelred.

Provides integration with the task-master-ai npm package for 
advanced task orchestration and management.
"""

import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..memory.tier_manager import MemoryTierManager

logger = logging.getLogger(__name__)


class TaskMasterIntegration:
    """
    Integration with task-master-ai for enhanced task management.
    
    Provides a bridge between Aethelred's agent system and task-master-ai's
    sophisticated task orchestration capabilities.
    """
    
    def __init__(self, memory_manager: MemoryTierManager, config: Dict[str, Any]):
        self.memory_manager = memory_manager
        self.config = config
        self.task_master_path = config.get('task_master_path', 'task-master-ai')
        self.project_root = Path(config.get('project_root', '.'))
        self.is_initialized = False
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
    async def initialize(self) -> None:
        """Initialize task-master-ai integration."""
        try:
            # Check if task-master-ai is available
            result = await self._run_command(['--version'])
            if result['success']:
                logger.info(f"Task Master AI available: {result['output'].strip()}")
            else:
                raise RuntimeError("task-master-ai not found or not working")
                
            # Initialize task-master-ai project if needed
            taskmaster_config = self.project_root / '.taskmaster'
            if not taskmaster_config.exists():
                await self._initialize_taskmaster_project()
                
            self.is_initialized = True
            logger.info("Task Master AI integration initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Task Master AI: {e}")
            raise
            
    async def _initialize_taskmaster_project(self) -> None:
        """Initialize task-master-ai project configuration."""
        logger.info("Initializing Task Master AI project...")
        
        # Run task-master-ai init
        result = await self._run_command(['init'], cwd=self.project_root)
        if not result['success']:
            raise RuntimeError(f"Failed to initialize Task Master AI: {result['error']}")
            
        # Customize configuration for Aethelred
        config_file = self.project_root / '.taskmaster' / 'config.json'
        if config_file.exists():
            await self._customize_taskmaster_config(config_file)
            
    async def _customize_taskmaster_config(self, config_file: Path) -> None:
        """Customize task-master-ai configuration for Aethelred."""
        try:
            # Read existing config
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Customize for Aethelred
            config.update({
                'project_name': 'aethelred',
                'agent_integration': True,
                'memory_integration': True,
                'max_concurrent_tasks': self.config.get('max_concurrent_tasks', 50),
                'task_timeout': self.config.get('task_timeout', 3600),  # 1 hour
                'checkpoint_interval': self.config.get('checkpoint_interval', 300),  # 5 minutes
                'aethelred_config': {
                    'memory_tiers': ['hot', 'warm', 'cold'],
                    'agent_tiers': ['apex', 'brigade', 'doer', 'service'],
                    'enable_evolution': False  # Start disabled
                }
            })
            
            # Write updated config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info("Customized Task Master AI configuration for Aethelred")
            
        except Exception as e:
            logger.error(f"Failed to customize Task Master AI config: {e}")
            
    async def create_task(self, title: str, description: str, 
                         task_type: str = 'general',
                         priority: int = 5,
                         required_capabilities: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new task using task-master-ai.
        
        Args:
            title: Task title
            description: Task description
            task_type: Type of task
            priority: Task priority (1-10)
            required_capabilities: Required agent capabilities
            metadata: Additional task metadata
            
        Returns:
            Task creation result
        """
        if not self.is_initialized:
            raise RuntimeError("Task Master AI not initialized")
            
        task_data = {
            'title': title,
            'description': description,
            'type': task_type,
            'priority': priority,
            'required_capabilities': required_capabilities or [],
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat(),
            'created_by': 'aethelred_system'
        }
        
        try:
            # Create task via task-master-ai
            cmd_args = [
                'create',
                '--title', title,
                '--description', description,
                '--priority', str(priority),
                '--format', 'json'
            ]
            
            if task_type:
                cmd_args.extend(['--type', task_type])
                
            result = await self._run_command(cmd_args)
            
            if result['success']:
                # Parse task-master-ai response
                try:
                    task_response = json.loads(result['output'])
                    task_id = task_response.get('id')
                    
                    if task_id:
                        # Store in our tracking system
                        self.active_tasks[task_id] = {
                            'id': task_id,
                            'data': task_data,
                            'status': 'created',
                            'created_at': datetime.utcnow(),
                            'taskmaster_data': task_response
                        }
                        
                        # Store in memory
                        await self._store_task_in_memory(task_id, task_data)
                        
                        logger.info(f"Created task {task_id}: {title}")
                        
                        return {
                            'success': True,
                            'task_id': task_id,
                            'task_data': task_data,
                            'taskmaster_response': task_response
                        }
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse task-master-ai response: {e}")
                    
            return {
                'success': False,
                'error': f"Task creation failed: {result.get('error', 'Unknown error')}"
            }
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks from task-master-ai.
        
        Args:
            status: Filter by task status
            
        Returns:
            List of tasks
        """
        if not self.is_initialized:
            return []
            
        try:
            cmd_args = ['list', '--format', 'json']
            if status:
                cmd_args.extend(['--status', status])
                
            result = await self._run_command(cmd_args)
            
            if result['success']:
                try:
                    tasks = json.loads(result['output'])
                    return tasks if isinstance(tasks, list) else []
                except json.JSONDecodeError:
                    logger.error("Failed to parse task list response")
                    
            return []
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []
            
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        if not self.is_initialized:
            return None
            
        try:
            result = await self._run_command(['status', task_id, '--format', 'json'])
            
            if result['success']:
                try:
                    status_data = json.loads(result['output'])
                    
                    # Update our tracking
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]['last_status_check'] = datetime.utcnow()
                        self.active_tasks[task_id]['taskmaster_status'] = status_data
                        
                    return status_data
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse status response for task {task_id}")
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
            
    async def complete_task(self, task_id: str, result: Any, 
                           success: bool = True) -> Dict[str, Any]:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            result: Task execution result
            success: Whether task completed successfully
            
        Returns:
            Completion result
        """
        if not self.is_initialized:
            return {'success': False, 'error': 'Not initialized'}
            
        try:
            # Update task-master-ai
            status = 'completed' if success else 'failed'
            cmd_args = ['update', task_id, '--status', status, '--format', 'json']
            
            tm_result = await self._run_command(cmd_args)
            
            # Update our tracking
            if task_id in self.active_tasks:
                task_data = self.active_tasks.pop(task_id)
                task_data.update({
                    'status': status,
                    'completed_at': datetime.utcnow(),
                    'result': result,
                    'success': success
                })
                self.completed_tasks.append(task_data)
                
                # Update memory
                await self._update_task_in_memory(task_id, task_data)
                
            logger.info(f"Task {task_id} marked as {status}")
            
            return {
                'success': tm_result['success'],
                'task_id': task_id,
                'status': status,
                'taskmaster_response': tm_result.get('output')
            }
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def assign_task_to_agent(self, task_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Assign a task to a specific agent.
        
        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            
        Returns:
            Assignment result
        """
        try:
            # Update task-master-ai
            result = await self._run_command([
                'assign', task_id, '--assignee', agent_id, '--format', 'json'
            ])
            
            # Update our tracking
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['assigned_to'] = agent_id
                self.active_tasks[task_id]['assigned_at'] = datetime.utcnow()
                
                # Update memory
                await self._update_task_in_memory(task_id, self.active_tasks[task_id])
                
            logger.info(f"Assigned task {task_id} to agent {agent_id}")
            
            return {
                'success': result['success'],
                'task_id': task_id,
                'agent_id': agent_id,
                'taskmaster_response': result.get('output')
            }
            
        except Exception as e:
            logger.error(f"Failed to assign task {task_id} to agent {agent_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_agent_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all tasks assigned to a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of assigned tasks
        """
        try:
            all_tasks = await self.list_tasks()
            agent_tasks = [
                task for task in all_tasks 
                if task.get('assignee') == agent_id
            ]
            return agent_tasks
            
        except Exception as e:
            logger.error(f"Failed to get tasks for agent {agent_id}: {e}")
            return []
            
    async def _store_task_in_memory(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Store task data in memory system."""
        try:
            await self.memory_manager.write(
                f"task:{task_id}",
                task_data,
                target_tiers=['hot', 'warm']
            )
        except Exception as e:
            logger.error(f"Failed to store task {task_id} in memory: {e}")
            
    async def _update_task_in_memory(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Update task data in memory system."""
        try:
            await self.memory_manager.write(
                f"task:{task_id}",
                task_data,
                target_tiers=['hot', 'warm']
            )
        except Exception as e:
            logger.error(f"Failed to update task {task_id} in memory: {e}")
            
    async def _run_command(self, args: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run task-master-ai command.
        
        Args:
            args: Command arguments
            cwd: Working directory
            
        Returns:
            Command execution result
        """
        try:
            cmd = [self.task_master_path] + args
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd or self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode('utf-8') if stdout else '',
                'error': stderr.decode('utf-8') if stderr else '',
                'return_code': process.returncode
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'return_code': -1
            }
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get task management statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_tasks': len(self.active_tasks) + len(self.completed_tasks),
            'integration_status': 'initialized' if self.is_initialized else 'not_initialized'
        }