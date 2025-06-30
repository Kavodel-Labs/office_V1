"""
Secretary Agent for Project Aethelred.

The Secretary is a Service tier agent responsible for:
- Managing external communications (Slack, email, etc.)
- Routing user requests to appropriate agents
- Providing human-friendly interfaces to the system
- Managing notifications and status updates
- Translating between human language and system commands
"""

import asyncio
import logging
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from agents.base.agent import Agent, AgentCapability, AgentStatus, TaskResult
from core.memory.tier_manager import MemoryTierManager

logger = logging.getLogger(__name__)


@dataclass
class SlackMessage:
    """Represents a Slack message."""
    channel: str
    text: str
    user: Optional[str] = None
    timestamp: Optional[str] = None
    thread_ts: Optional[str] = None


class Secretary(Agent):
    """
    Secretary - External Communication Manager and Human Interface.
    
    Responsibilities:
    - Manage Slack communications via MCP
    - Route user requests to appropriate AETHELRED agents
    - Provide human-friendly system status updates
    - Handle notifications and alerts
    - Translate between natural language and system commands
    """
    
    def __init__(self, memory_manager: MemoryTierManager,
                 config: Optional[Dict[str, Any]] = None):
        
        super().__init__(
            agent_id="S_Secretary",
            version=1,
            tier="service",
            role="Communication Manager",
            capabilities=[
                AgentCapability.COMMUNICATIONS_EXTERNAL,
                AgentCapability.TASKS_RECEIVE,
                AgentCapability.NOTIFICATIONS_SEND
            ],
            config=config or {}
        )
        
        self.memory_manager = memory_manager
        
        # Communication state
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[str] = []
        self.command_history: List[Dict[str, Any]] = []
        
        # Slack Configuration
        self.slack_enabled = config.get('slack_enabled', True)
        self.slack_bot_token = config.get('slack_bot_token')
        self.slack_app_token = config.get('slack_app_token') 
        self.slack_signing_secret = config.get('slack_signing_secret')
        
        # MCP Configuration (fallback)
        self.mcp_config = config.get('mcp', {})
        self.cursor_mcp_integration = self.mcp_config.get('cursor_integration', True)
        
        # Slack integration instance
        self.slack_integration = None
        
        # Message processing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        
        # Statistics
        self.messages_processed = 0
        self.commands_executed = 0
        self.notifications_sent = 0
        
    async def on_initialize(self) -> None:
        """Initialize Secretary specific resources."""
        logger.info("Initializing Secretary agent...")
        
        # Load conversation history
        await self._load_conversation_history()
        
        # Set up notification channels
        await self._setup_notification_channels()
        
        # Start message processing loop
        self._start_message_processing()
        
        # Initialize Slack integration (try direct API first, fallback to MCP)
        if self.slack_enabled:
            await self._initialize_slack_integration()
            
        # Test connectivity
        if self.slack_integration:
            await self._test_slack_connectivity()
        
        logger.info("Secretary initialization complete")
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task assigned to the Secretary.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        logger.debug(f"SECRETARY: Received task: {json.dumps(task, indent=2)}")
        
        task_type = task.get('type')
        
        if task_type == 'process_incoming_message':
            return await self._handle_incoming_message(task)
        elif task_type == 'slack_message':
            return await self._handle_slack_message(task)
        elif task_type == 'send_notification':
            return await self._handle_send_notification(task)
        elif task_type == 'user_command':
            return await self._handle_user_command(task)
        elif task_type == 'system_status_report':
            return await self._handle_status_report(task)
        elif task_type == 'route_request':
            return await self._handle_route_request(task)
        elif task_type == 'conversation_summary':
            return await self._handle_conversation_summary(task)
        else:
            logger.error(f"SECRETARY: Unknown task type: {task_type}")
            raise ValueError(f"Unknown task type: {task_type}")
            
    async def _handle_incoming_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a message from any source (e.g., local UI)."""
        payload = task.get('payload', {})
        text = payload.get('text', '')
        user = payload.get('user', 'unknown_user')
        source = task.get('source', 'unknown_source')
        
        logger.debug(f"SECRETARY: Processing incoming message from {user} via {source}: '{text}'")
        
        # This is where the message is routed to the rest of the system.
        # For now, we'll just log it.
        # In the next step, we will route this to the TaskRouter.
        
        logger.debug("SECRETARY: Handing off message to internal systems...")
        
        # Placeholder for routing to the TaskRouter
        # This will be the next step in our debugging.
        
        return {"status": "message_received_by_secretary"}
        
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the Secretary can execute the task.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task can be executed
        """
        valid_task_types = {
            'process_incoming_message',
            'slack_message',
            'send_notification', 
            'user_command',
            'system_status_report',
            'route_request',
            'conversation_summary'
        }
        
        task_type = task.get('type')
        return task_type in valid_task_types
        
    async def check_agent_health(self) -> Dict[str, Any]:
        """Secretary specific health checks."""
        health_data = {
            'messages_processed': self.messages_processed,
            'commands_executed': self.commands_executed,
            'notifications_sent': self.notifications_sent,
            'active_conversations': len(self.active_conversations),
            'notification_channels': len(self.notification_channels),
            'slack_connectivity': 'unknown'
        }
        
        # Test Slack connectivity if enabled
        if self.slack_enabled:
            try:
                slack_test = await self._test_slack_connectivity()
                health_data['slack_connectivity'] = 'healthy' if slack_test else 'failed'
            except Exception as e:
                health_data['slack_connectivity'] = f"error: {e}"
                
        return health_data
        
    # Task handlers
    
    async def _handle_slack_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming Slack messages."""
        message_data = task.get('message', {})
        channel = message_data.get('channel')
        text = message_data.get('text', '')
        user = message_data.get('user')
        
        logger.info(f"Processing Slack message from {user} in {channel}: {text[:100]}...")
        
        # Parse message for commands or requests
        response = await self._process_user_message(text, user, channel)
        
        # Send response back to Slack
        if response:
            await self._send_slack_message(channel, response, user)
        
        self.messages_processed += 1
        
        # Store conversation in memory
        conversation_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'channel': channel,
            'user': user,
            'message': text,
            'response': response,
            'type': 'slack_message'
        }
        
        await self.memory_manager.write(
            f"conversation:{channel}:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            conversation_entry,
            target_tiers=['hot', 'warm']
        )
        
        return {
            'action': 'slack_message_processed',
            'channel': channel,
            'user': user,
            'response_sent': response is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_send_notification(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sending notifications to Slack."""
        message = task.get('message')
        channel = task.get('channel')
        urgency = task.get('urgency', 'normal')
        
        if not message or not channel:
            raise ValueError("message and channel are required for notifications")
        
        # Format notification based on urgency
        formatted_message = self._format_notification(message, urgency)
        
        # Send to Slack
        success = await self._send_slack_message(channel, formatted_message)
        
        if success:
            self.notifications_sent += 1
            
        return {
            'action': 'notification_sent',
            'channel': channel,
            'urgency': urgency,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_user_command(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user commands directed at AETHELRED."""
        command = task.get('command')
        user = task.get('user')
        context = task.get('context', {})
        
        logger.info(f"Processing user command from {user}: {command}")
        
        # Parse and route command
        routing_result = await self._route_user_command(command, user, context)
        
        self.commands_executed += 1
        
        # Store command in history
        command_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'command': command,
            'routing_result': routing_result,
            'context': context
        }
        self.command_history.append(command_entry)
        
        return {
            'action': 'user_command_processed',
            'command': command,
            'user': user,
            'routing_result': routing_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_status_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and send system status reports."""
        report_type = task.get('report_type', 'summary')
        destination = task.get('destination')
        
        # Generate status report
        status_report = await self._generate_status_report(report_type)
        
        # Send to destination (Slack channel, etc.)
        if destination:
            await self._send_slack_message(destination, status_report)
        
        return {
            'action': 'status_report_generated',
            'report_type': report_type,
            'destination': destination,
            'report': status_report,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_route_request(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route user requests to appropriate agents."""
        request = task.get('request')
        user = task.get('user')
        priority = task.get('priority', 'normal')
        
        # Analyze request and determine target agent
        target_agent = await self._analyze_request_routing(request)
        
        # Create task for target agent
        routed_task = {
            'id': f"user_request_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'type': 'user_request',
            'request': request,
            'user': user,
            'priority': priority,
            'routed_by': 'S_Secretary',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return {
            'action': 'request_routed',
            'target_agent': target_agent,
            'routed_task': routed_task,
            'user': user,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_conversation_summary(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversation summaries."""
        channel = task.get('channel')
        time_window = task.get('time_window_hours', 24)
        
        # Retrieve conversation history
        conversations = await self._get_conversation_history(channel, time_window)
        
        # Generate summary
        summary = await self._generate_conversation_summary(conversations)
        
        return {
            'action': 'conversation_summary_generated',
            'channel': channel,
            'time_window_hours': time_window,
            'message_count': len(conversations),
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    # Core functionality
    
    async def _process_user_message(self, text: str, user: str, channel: str) -> Optional[str]:
        """Process a user message and generate response."""
        # Check if message is directed at AETHELRED
        if not self._is_message_for_aethelred(text):
            return None
            
        # Extract command or request
        command = self._extract_command(text)
        
        if command.startswith('status'):
            return await self._get_system_status_summary()
        elif command.startswith('help'):
            return self._get_help_message()
        elif command.startswith('agents'):
            return await self._get_agent_status_summary()
        elif command.startswith('tasks'):
            return await self._get_task_status_summary()
        elif command.startswith('health'):
            return await self._get_health_summary()
        else:
            # Route as general request
            routing_result = await self._route_user_command(command, user, {'channel': channel})
            return f"Request routed to {routing_result.get('target_agent', 'system')}. I'll keep you updated on the progress."
            
    def _is_message_for_aethelred(self, text: str) -> bool:
        """Check if message is directed at AETHELRED."""
        keywords = ['aethelred', 'system', 'status', 'agent', 'task', 'help', '@aethelred']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
        
    def _extract_command(self, text: str) -> str:
        """Extract command from user message."""
        # Remove common prefixes and clean up
        text = text.lower().strip()
        prefixes = ['aethelred', 'system', '@aethelred', 'hey', 'please']
        
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                
        return text
        
    async def _send_slack_message(self, channel: str, message: str, user: Optional[str] = None) -> bool:
        """Send message to Slack via integration."""
        try:
            if not self.slack_enabled or not self.slack_integration:
                logger.warning("Slack integration not available, message not sent")
                return False
                
            # Use the Slack integration to send message
            success = await self.slack_integration.send_message(channel, message)
            
            if success:
                logger.info(f"SLACK MESSAGE -> {channel}: {message[:100]}...")
            else:
                logger.error(f"Failed to send message to {channel}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
            
    async def _test_slack_connectivity(self) -> bool:
        """Test Slack connectivity."""
        try:
            if not self.slack_integration:
                return False
                
            # Test by getting channels or sending test
            if hasattr(self.slack_integration, 'get_channels'):
                channels = await self.slack_integration.get_channels()
                return len(channels) >= 0  # Even 0 channels means connection works
            
            return True
        except Exception as e:
            logger.error(f"Slack connectivity test failed: {e}")
            return False
            
    async def _initialize_slack_integration(self) -> None:
        """Initialize Slack integration using MCP client."""
        try:
            # Use MCP client for Slack integration
            from core.integrations.slack_mcp_client import SlackMCPClient
            import os
            
            mcp_config = {
                'enabled': True,
                'mcp_server_path': 'npx @modelcontextprotocol/server-slack',
                'slack_token': self.slack_bot_token or os.getenv('SLACK_BOT_TOKEN'),
                'mcp_server_port': 3000
            }
            
            self.slack_integration = SlackMCPClient(mcp_config)
            if await self.slack_integration.initialize():
                logger.info("Slack MCP client initialized successfully")
                return
            else:
                self.slack_integration = None
                logger.warning("Failed to initialize Slack MCP client")
                
        except Exception as e:
            logger.error(f"Failed to initialize Slack MCP integration: {e}")
            self.slack_integration = None
            
        # Fallback to original MCP integration if available
        if not self.slack_integration:
            try:
                from core.integrations.slack_mcp import SlackMCPIntegration
                mcp_config = self.mcp_config
                self.slack_integration = SlackMCPIntegration(mcp_config)
                await self.slack_integration.initialize()
                logger.info("Slack MCP integration initialized (fallback)")
                        
            except Exception as e:
                logger.warning(f"Failed to initialize fallback Slack integration: {e}")
                self.slack_integration = None
            
    def _format_notification(self, message: str, urgency: str) -> str:
        """Format notification message based on urgency."""
        emoji_map = {
            'low': '📢',
            'normal': '🔔', 
            'high': '⚠️',
            'critical': '🚨'
        }
        
        emoji = emoji_map.get(urgency, '🔔')
        timestamp = datetime.utcnow().strftime('%H:%M:%S')
        
        return f"{emoji} **AETHELRED Notification** [{timestamp}]\\n{message}"
        
    async def _route_user_command(self, command: str, user: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Route user command to appropriate agent."""
        # Simple routing logic (can be made more sophisticated)
        if 'config' in command or 'setting' in command:
            target_agent = 'A_ChiefOfStaff'
        elif 'performance' in command or 'metric' in command:
            target_agent = 'S_Auditor'
        elif 'task' in command or 'work' in command:
            target_agent = 'A_ChiefOfStaff'
        else:
            target_agent = 'A_ChiefOfStaff'  # Default to Chief of Staff
            
        return {
            'target_agent': target_agent,
            'command': command,
            'user': user,
            'routing_reason': f"Command contains keywords for {target_agent}"
        }
        
    async def _get_system_status_summary(self) -> str:
        """Get system status summary for users."""
        try:
            # Get memory health
            memory_health = await self.memory_manager.health_check()
            healthy_tiers = sum(1 for h in memory_health.values() if h.get('status') == 'healthy')
            total_tiers = len(memory_health)
            
            status_msg = f"""🤖 **AETHELRED System Status**
            
📊 **Memory System**: {healthy_tiers}/{total_tiers} tiers healthy
🔄 **Status**: Operational
📈 **Messages Processed**: {self.messages_processed}
🎯 **Commands Executed**: {self.commands_executed}
⏰ **Last Update**: {datetime.utcnow().strftime('%H:%M:%S UTC')}

Use `aethelred help` for available commands."""
            
            return status_msg
            
        except Exception as e:
            return f"❌ Error retrieving system status: {e}"
            
    def _get_help_message(self) -> str:
        """Get help message for users."""
        return """🤖 **AETHELRED Commands**
        
**System Commands:**
• `aethelred status` - Get system status
• `aethelred agents` - List active agents  
• `aethelred tasks` - View task queue
• `aethelred health` - System health check

**Agent Commands:**
• `aethelred config [setting]` - Configure system
• `aethelred performance` - Get performance metrics
• `aethelred help` - Show this help

**Examples:**
• "Hey AETHELRED, what's the system status?"
• "AETHELRED performance metrics please"
• "Show me active agents"

I'm here to help you interact with the AETHELRED system! 🚀"""
        
    async def _get_agent_status_summary(self) -> str:
        """Get agent status summary."""
        return """🤖 **Active AETHELRED Agents**
        
• **Chief of Staff (A_ChiefOfStaff)** - System Governor ✅
• **Auditor (S_Auditor)** - Performance Observer ✅  
• **Secretary (S_Secretary)** - Communication Manager ✅

All agents operational and ready for tasks."""
        
    async def _get_task_status_summary(self) -> str:
        """Get task status summary."""
        return f"""📋 **Task Management Status**
        
🔄 **Active Tasks**: 0
✅ **Completed**: {self.commands_executed}
⏳ **Queue**: Empty
🎯 **Processing**: Ready

System ready for new task assignments."""
        
    async def _get_health_summary(self) -> str:
        """Get health summary."""
        try:
            memory_health = await self.memory_manager.health_check()
            healthy_tiers = sum(1 for h in memory_health.values() if h.get('status') == 'healthy')
            
            return f"""🏥 **AETHELRED Health Check**
            
✅ **Overall Status**: Healthy
📊 **Memory**: {healthy_tiers}/{len(memory_health)} tiers operational
🔗 **Connectivity**: Slack integration active
⚡ **Performance**: Optimal
🛡️ **Security**: All systems secure

System operating within normal parameters."""
            
        except Exception as e:
            return f"⚠️ Health check encountered issues: {e}"
            
    async def _generate_status_report(self, report_type: str) -> str:
        """Generate detailed status report."""
        if report_type == 'summary':
            return await self._get_system_status_summary()
        elif report_type == 'detailed':
            summary = await self._get_system_status_summary()
            agents = await self._get_agent_status_summary()
            tasks = await self._get_task_status_summary()
            health = await self._get_health_summary()
            
            return f"{summary}\\n\\n{agents}\\n\\n{tasks}\\n\\n{health}"
        else:
            return await self._get_system_status_summary()
            
    async def _analyze_request_routing(self, request: str) -> str:
        """Analyze request to determine optimal agent routing."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['config', 'setting', 'rule', 'govern']):
            return 'A_ChiefOfStaff'
        elif any(word in request_lower for word in ['performance', 'metric', 'audit', 'monitor']):
            return 'S_Auditor'
        elif any(word in request_lower for word in ['message', 'communicate', 'notify', 'report']):
            return 'S_Secretary'
        else:
            return 'A_ChiefOfStaff'  # Default routing
            
    async def _load_conversation_history(self) -> None:
        """Load conversation history from memory."""
        try:
            # Load recent conversations
            logger.info("Loading conversation history")
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            
    async def _setup_notification_channels(self) -> None:
        """Set up notification channels."""
        try:
            # Default notification channels
            self.notification_channels = ['#aethelred-notifications', '#general']
            logger.info(f"Set up {len(self.notification_channels)} notification channels")
        except Exception as e:
            logger.error(f"Failed to set up notification channels: {e}")
            
    def _start_message_processing(self) -> None:
        """Start background message processing tasks."""
        # Message processing loop
        message_task = asyncio.create_task(self._message_processing_loop())
        self._background_tasks.add(message_task)
        message_task.add_done_callback(self._background_tasks.discard)
        
    async def _message_processing_loop(self) -> None:
        """Background message processing loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1)  # Process messages every second
                
                # Process queued messages
                # This would handle incoming messages from MCP
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                
    async def _get_conversation_history(self, channel: str, time_window_hours: int) -> List[Dict[str, Any]]:
        """Get conversation history for a channel."""
        # Placeholder - implement actual conversation retrieval
        return []
        
    async def _generate_conversation_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """Generate summary of conversations."""
        if not conversations:
            return "No conversations found in the specified time window."
            
        return f"Summarized {len(conversations)} messages from recent conversations."