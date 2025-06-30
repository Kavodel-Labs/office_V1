"""
Terminal Passthrough Protocol for AETHELRED
Secure command execution bridge between Slack and local machines.
Implementation from aethelred-slack-interface specification.
"""

import asyncio
import logging
import json
import time
import uuid
import re
import subprocess
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CommandStatus(Enum):
    """Command execution status."""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    REJECTED = "rejected"

@dataclass
class CommandRequest:
    """Terminal command request."""
    id: str
    command: str
    user_id: str
    user_name: str
    channel: str
    timestamp: float
    status: CommandStatus
    approval_required: bool = True
    timeout_seconds: int = 30
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class CommandResult:
    """Terminal command execution result."""
    command_id: str
    status: CommandStatus
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SecurityValidator:
    """Security validation for terminal commands."""
    
    def __init__(self, config: Dict[str, Any]):
        self.whitelist_patterns = config.get("whitelist_patterns", [])
        self.blacklist_patterns = config.get("blacklist_patterns", [
            r".*rm\s+-rf.*",
            r".*sudo.*",
            r".*passwd.*",
            r".*shutdown.*",
            r".*reboot.*",
            r".*mkfs.*",
            r".*dd\s+if=.*",
            r".*>.*\/dev\/.*"
        ])
        self.max_command_length = config.get("max_command_length", 1000)
        self.allowed_binaries = set(config.get("allowed_binaries", [
            "git", "npm", "node", "python", "python3", "pip", "pip3", 
            "docker", "kubectl", "make", "cargo", "go", "ls", "cat", 
            "grep", "find", "ps", "top", "df", "du", "curl", "wget"
        ]))
        
    def validate_command(self, command: str, user_id: str) -> tuple[bool, str]:
        """Validate command for security compliance."""
        
        # Basic length check
        if len(command) > self.max_command_length:
            return False, f"Command too long (max {self.max_command_length} chars)"
        
        # Check blacklist patterns
        for pattern in self.blacklist_patterns:
            if re.match(pattern, command, re.IGNORECASE):
                return False, f"Command matches blacklisted pattern: {pattern}"
        
        # Check whitelist patterns (if configured)
        if self.whitelist_patterns:
            whitelist_match = False
            for pattern in self.whitelist_patterns:
                if re.match(pattern, command, re.IGNORECASE):
                    whitelist_match = True
                    break
            if not whitelist_match:
                return False, "Command not in whitelist"
        
        # Check allowed binaries
        if self.allowed_binaries:
            command_parts = command.strip().split()
            if command_parts:
                binary = command_parts[0].split('/')[-1]  # Handle full paths
                if binary not in self.allowed_binaries:
                    return False, f"Binary '{binary}' not in allowed list"
        
        # Additional security checks
        if self._contains_injection_attempts(command):
            return False, "Potential command injection detected"
        
        return True, "Command validated"
    
    def _contains_injection_attempts(self, command: str) -> bool:
        """Detect potential command injection attempts."""
        injection_patterns = [
            r".*;.*",     # Command chaining
            r".*\|\|.*",  # OR operators
            r".*&&.*",    # AND operators  
            r".*`.*`.*",  # Command substitution
            r".*\$\(.*\).*",  # Command substitution
            r".*>.*",     # Output redirection to sensitive locations
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, command):
                return True
        return False

class TerminalPassthroughManager:
    """Manager for terminal command passthrough system."""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.security = SecurityValidator(config.get("security", {}))
        
        # Redis streams for command coordination
        self.command_stream = "commands:terminal"
        self.response_stream = "responses:terminal" 
        self.approval_stream = "approvals:terminal"
        
        # Tracking
        self.pending_commands: Dict[str, CommandRequest] = {}
        self.active_commands: Dict[str, CommandRequest] = {}
        self.command_history: List[CommandResult] = []
        
        # Settings
        self.global_approval_required = config.get("require_approval", True)
        self.default_timeout = config.get("default_timeout", 30)
        self.max_concurrent_commands = config.get("max_concurrent", 5)
        
        # User permissions
        self.admin_users = set(config.get("admin_users", []))
        self.allowed_users = set(config.get("allowed_users", []))
        
    async def queue_command(self, command: str, user_id: str, user_name: str, 
                          channel: str, **kwargs) -> CommandRequest:
        """Queue a terminal command for execution."""
        
        # Validate user permissions
        if self.allowed_users and user_id not in self.allowed_users:
            raise PermissionError(f"User {user_id} not authorized for terminal access")
        
        # Validate command security
        is_valid, validation_message = self.security.validate_command(command, user_id)
        if not is_valid:
            raise ValueError(f"Command validation failed: {validation_message}")
        
        # Check concurrent command limit
        if len(self.active_commands) >= self.max_concurrent_commands:
            raise RuntimeError(f"Maximum concurrent commands ({self.max_concurrent_commands}) reached")
        
        # Create command request
        command_id = str(uuid.uuid4())
        
        # Admin users can bypass approval for simple commands
        needs_approval = self.global_approval_required
        if user_id in self.admin_users and self._is_simple_command(command):
            needs_approval = False
        
        request = CommandRequest(
            id=command_id,
            command=command,
            user_id=user_id,
            user_name=user_name,
            channel=channel,
            timestamp=time.time(),
            status=CommandStatus.PENDING,
            approval_required=needs_approval,
            timeout_seconds=kwargs.get("timeout", self.default_timeout),
            working_directory=kwargs.get("cwd"),
            environment=kwargs.get("env")
        )
        
        # Store in tracking
        if needs_approval:
            self.pending_commands[command_id] = request
        else:
            self.active_commands[command_id] = request
            request.status = CommandStatus.APPROVED
        
        # Publish to Redis stream for daemon pickup
        await self._publish_command_request(request)
        
        logger.info(f"Queued command {command_id}: {command} (approval: {needs_approval})")
        
        return request
    
    async def approve_command(self, command_id: str, approver_user_id: str) -> bool:
        """Approve a pending command."""
        
        if command_id not in self.pending_commands:
            logger.warning(f"Command {command_id} not found in pending")
            return False
        
        request = self.pending_commands[command_id]
        
        # Move to active commands
        del self.pending_commands[command_id]
        self.active_commands[command_id] = request
        
        request.status = CommandStatus.APPROVED
        
        # Publish approval
        await self._publish_command_approval(command_id, approver_user_id)
        
        logger.info(f"Command {command_id} approved by {approver_user_id}")
        return True
    
    async def reject_command(self, command_id: str, rejector_user_id: str, reason: str = "") -> bool:
        """Reject a pending command."""
        
        if command_id not in self.pending_commands:
            return False
        
        request = self.pending_commands[command_id]
        del self.pending_commands[command_id]
        
        # Create result record
        result = CommandResult(
            command_id=command_id,
            status=CommandStatus.REJECTED,
            error_message=f"Rejected by {rejector_user_id}: {reason}"
        )
        
        self.command_history.append(result)
        
        # Publish rejection
        await self._publish_command_rejection(command_id, rejector_user_id, reason)
        
        logger.info(f"Command {command_id} rejected by {rejector_user_id}")
        return True
    
    async def get_command_status(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a command."""
        
        # Check pending commands
        if command_id in self.pending_commands:
            return self.pending_commands[command_id].to_dict()
        
        # Check active commands  
        if command_id in self.active_commands:
            return self.active_commands[command_id].to_dict()
        
        # Check command history
        for result in self.command_history:
            if result.command_id == command_id:
                return result.to_dict()
        
        return None
    
    async def process_command_result(self, result_data: Dict[str, Any]) -> None:
        """Process command execution result from daemon."""
        
        command_id = result_data.get("command_id")
        if not command_id:
            logger.warning("Received result without command_id")
            return
        
        # Remove from active commands
        if command_id in self.active_commands:
            request = self.active_commands[command_id]
            del self.active_commands[command_id]
        else:
            logger.warning(f"Result for unknown command: {command_id}")
            return
        
        # Create result record
        result = CommandResult(
            command_id=command_id,
            status=CommandStatus(result_data.get("status", "failed")),
            exit_code=result_data.get("exit_code"),
            stdout=result_data.get("stdout"),
            stderr=result_data.get("stderr"),
            execution_time=result_data.get("execution_time"),
            error_message=result_data.get("error_message")
        )
        
        self.command_history.append(result)
        
        # Publish result for notification
        await self._publish_command_result(result)
        
        logger.info(f"Processed result for command {command_id}: {result.status}")
    
    async def get_user_command_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get command history for a user."""
        user_commands = []
        
        # Get from history
        for result in reversed(self.command_history):
            # We need to cross-reference with original requests
            # This is simplified - in production you'd store user_id with results
            user_commands.append(result.to_dict())
            if len(user_commands) >= limit:
                break
        
        return user_commands
    
    async def cleanup_expired_commands(self) -> None:
        """Clean up expired pending commands."""
        current_time = time.time()
        expired_commands = []
        
        for command_id, request in self.pending_commands.items():
            if current_time - request.timestamp > request.timeout_seconds:
                expired_commands.append(command_id)
        
        for command_id in expired_commands:
            request = self.pending_commands[command_id]
            del self.pending_commands[command_id]
            
            result = CommandResult(
                command_id=command_id,
                status=CommandStatus.TIMEOUT,
                error_message="Command approval timed out"
            )
            
            self.command_history.append(result)
            
            logger.info(f"Command {command_id} expired")
    
    def _is_simple_command(self, command: str) -> bool:
        """Check if command is considered 'simple' and safe."""
        simple_commands = [
            "git status", "git log", "git diff", "git branch",
            "npm list", "npm version", "npm outdated",
            "docker ps", "docker images", "docker version",
            "ls", "pwd", "date", "whoami", "ps aux"
        ]
        
        command_lower = command.lower().strip()
        return any(command_lower.startswith(simple) for simple in simple_commands)
    
    async def _publish_command_request(self, request: CommandRequest) -> None:
        """Publish command request to Redis stream."""
        try:
            await self.redis.xadd(
                f"{self.command_stream}:{request.user_id}",
                request.to_dict()
            )
        except Exception as e:
            logger.error(f"Failed to publish command request: {e}")
    
    async def _publish_command_approval(self, command_id: str, approver_id: str) -> None:
        """Publish command approval."""
        try:
            await self.redis.xadd(
                self.approval_stream,
                {
                    "command_id": command_id,
                    "action": "approved", 
                    "approver_id": approver_id,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Failed to publish approval: {e}")
    
    async def _publish_command_rejection(self, command_id: str, rejector_id: str, reason: str) -> None:
        """Publish command rejection."""
        try:
            await self.redis.xadd(
                self.approval_stream,
                {
                    "command_id": command_id,
                    "action": "rejected",
                    "rejector_id": rejector_id,
                    "reason": reason,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Failed to publish rejection: {e}")
    
    async def _publish_command_result(self, result: CommandResult) -> None:
        """Publish command execution result."""
        try:
            await self.redis.xadd(
                self.response_stream,
                result.to_dict()
            )
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")

class TerminalDaemon:
    """Local daemon for secure command execution."""
    
    def __init__(self, user_id: str, redis_url: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.redis_url = redis_url
        self.config = config
        self.redis_client = None
        
        # Command stream for this user
        self.command_stream = f"commands:terminal:{user_id}"
        self.response_stream = "responses:terminal"
        self.approval_stream = "approvals:terminal"
        
        # Security
        self.security = SecurityValidator(config.get("security", {}))
        
        # State
        self.running = False
        self.active_commands: Dict[str, subprocess.Popen] = {}
        
    async def start(self) -> None:
        """Start the terminal daemon."""
        logger.info(f"Starting AETHELRED Terminal Daemon for user: {self.user_id}")
        
        # Connect to Redis
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
        
        # Create consumer group
        try:
            await self.redis_client.xgroup_create(
                self.command_stream,
                "daemon", 
                id="0"
            )
        except redis.ResponseError:
            pass  # Group already exists
        
        self.running = True
        
        # Start command processor
        await self._process_commands()
    
    async def stop(self) -> None:
        """Stop the daemon."""
        logger.info("Stopping AETHELRED Terminal Daemon")
        self.running = False
        
        # Terminate any active commands
        for command_id, process in self.active_commands.items():
            logger.info(f"Terminating active command: {command_id}")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def _process_commands(self) -> None:
        """Main command processing loop."""
        while self.running:
            try:
                # Read from command stream
                messages = await self.redis_client.xreadgroup(
                    "daemon",
                    self.user_id,
                    {self.command_stream: ">"},
                    count=1,
                    block=1000
                )
                
                for stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        await self._handle_command_message(msg_id, data)
                        
            except Exception as e:
                logger.error(f"Error processing commands: {e}")
                await asyncio.sleep(5)
    
    async def _handle_command_message(self, msg_id: str, data: Dict[str, Any]) -> None:
        """Handle individual command message."""
        command_id = data.get("id")
        command = data.get("command")
        status = data.get("status")
        
        logger.info(f"Received command {command_id}: {command} (status: {status})")
        
        # Only execute approved commands
        if status != CommandStatus.APPROVED.value:
            logger.info(f"Skipping command {command_id} - not approved")
            return
        
        # Validate command again locally
        is_valid, validation_message = self.security.validate_command(command, self.user_id)
        if not is_valid:
            await self._send_result(command_id, {
                "status": CommandStatus.FAILED.value,
                "error_message": f"Local validation failed: {validation_message}"
            })
            return
        
        # Request user confirmation if required
        if self.config.get("require_confirmation", True):
            if not self._request_confirmation(command, command_id):
                await self._send_result(command_id, {
                    "status": CommandStatus.CANCELLED.value,
                    "error_message": "User declined execution"
                })
                return
        
        # Execute command
        await self._execute_command(command_id, command, data)
    
    def _request_confirmation(self, command: str, command_id: str) -> bool:
        """Request user confirmation for command execution."""
        print(f"\n{'='*60}")
        print(f"AETHELRED Terminal Command Request")
        print(f"{'='*60}")
        print(f"Command ID: {command_id}")
        print(f"Command: {command}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        response = input("Execute this command? [y/N]: ").strip().lower()
        return response == 'y'
    
    async def _execute_command(self, command_id: str, command: str, data: Dict[str, Any]) -> None:
        """Execute the command safely."""
        start_time = time.time()
        
        try:
            # Prepare execution environment
            cwd = data.get("working_directory") or os.getcwd()
            env = os.environ.copy()
            if data.get("environment"):
                env.update(data["environment"])
            
            timeout = int(data.get("timeout_seconds", 30))
            
            # Execute command
            logger.info(f"Executing command {command_id}: {command}")
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )
            
            self.active_commands[command_id] = process
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                # Send successful result
                await self._send_result(command_id, {
                    "status": CommandStatus.COMPLETED.value,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "execution_time": execution_time
                })
                
                logger.info(f"Command {command_id} completed successfully")
                
            except asyncio.TimeoutError:
                process.kill()
                await self._send_result(command_id, {
                    "status": CommandStatus.TIMEOUT.value,
                    "error_message": f"Command timed out after {timeout} seconds"
                })
                
            finally:
                if command_id in self.active_commands:
                    del self.active_commands[command_id]
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            await self._send_result(command_id, {
                "status": CommandStatus.FAILED.value,
                "error_message": str(e)
            })
    
    async def _send_result(self, command_id: str, result_data: Dict[str, Any]) -> None:
        """Send command execution result."""
        result_data["command_id"] = command_id
        result_data["timestamp"] = time.time()
        result_data["daemon_user"] = self.user_id
        
        try:
            await self.redis_client.xadd(self.response_stream, result_data)
            logger.info(f"Sent result for command {command_id}")
        except Exception as e:
            logger.error(f"Failed to send result: {e}")

# Factory functions
def create_passthrough_manager(redis_client: redis.Redis, config: Dict[str, Any]) -> TerminalPassthroughManager:
    """Create terminal passthrough manager."""
    return TerminalPassthroughManager(redis_client, config)

def create_terminal_daemon(user_id: str, redis_url: str, config: Dict[str, Any]) -> TerminalDaemon:
    """Create terminal daemon for local execution."""
    return TerminalDaemon(user_id, redis_url, config)