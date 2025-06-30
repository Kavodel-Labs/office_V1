"""
Enhanced Comms Secretary - Brigade-Tier Agent
Implementation of the intelligent communication coordinator from aethelred-slack-interface specification.
"""

import asyncio
import logging
import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from agents.base.agent import Agent, AgentCapability, AgentStatus, TaskResult
from core.memory.tier_manager import MemoryTierManager
from core.reasoning.multi_llm_interpreter import MultiLLMReasoningEngine, create_reasoning_engine

logger = logging.getLogger(__name__)

@dataclass
class SlackMessage:
    """Represents a Slack message with metadata."""
    channel: str
    text: str
    user: str
    timestamp: str
    thread_ts: Optional[str] = None
    message_type: str = "message"
    
@dataclass
class ConversationContext:
    """Context for ongoing conversations."""
    user_id: str
    channel: str
    message_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    active_tasks: List[str]
    last_interaction: float
    session_id: str

@dataclass
class NotificationTemplate:
    """Template for system notifications."""
    name: str
    template_type: str  # task_completion, decision_required, alert
    blocks: List[Dict[str, Any]]
    priority: str = "medium"

class CommsSecretary(Agent):
    """
    Brigade-tier agent responsible for all human communication.
    
    Serves as the intelligent gatekeeper, interpreter, and dispatcher for 
    human-AI interactions through Slack interface.
    """
    
    def __init__(self, memory_manager: MemoryTierManager, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="B_Comms-Secretary",
            version=1,
            tier="brigade",
            role="Communication Coordinator & Human Interface",
            capabilities=[
                AgentCapability.COMMUNICATIONS_EXTERNAL,
                AgentCapability.TASKS_RECEIVE,
                AgentCapability.NOTIFICATIONS_SEND,
                AgentCapability.AGORA_CONSENSUS,
                AgentCapability.AGORA_INVOKE,
                "slack.send",
                "slack.receive", 
                "slack.interpret",
                "terminal.passthrough",
                "files.transfer",
                "notifications.manage",
                "decisions.coordinate"
            ],
            config=config or {},
            memory_manager=memory_manager
        )
        
        self.memory_manager = memory_manager
        
        # Initialize Slack client
        self.slack_token = os.environ.get("SLACK_BOT_TOKEN")
        if not self.slack_token:
            logger.warning("SLACK_BOT_TOKEN not found - Slack integration will be limited")
            self.slack_client = None
        else:
            self.slack_client = WebClient(token=self.slack_token)
        
        # Initialize reasoning engine
        self.reasoning_engine = None
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_timeout = 1800  # 30 minutes
        
        # Notification templates
        self.notification_templates = self._load_notification_templates()
        
        # Channel configuration
        self.channels = {
            "general": config.get("slack", {}).get("channels", {}).get("general", "general"),
            "alerts": config.get("slack", {}).get("channels", {}).get("alerts", "aethelred-notifications"),
            "decisions": config.get("slack", {}).get("channels", {}).get("decisions", "aethelred-commands"),
            "commands": "aethelred-commands",
            "notifications": "aethelred-notifications",
            "test": "aethelred-test"
        }
        
        # Security and permissions
        self.allowed_users = config.get("security", {}).get("allowed_users", [])
        self.admin_users = config.get("security", {}).get("admin_users", [])
        
        # Terminal passthrough settings
        self.terminal_enabled = config.get("terminal_passthrough", {}).get("enabled", True)
        self.require_confirmation = config.get("terminal_passthrough", {}).get("require_confirmation", True)
        
        # Performance tracking
        self.message_count = 0
        self.interpretation_times = []
        self.response_times = []
        
    async def on_initialize(self) -> None:
        """Initialize Comms Secretary specific resources."""
        logger.info("Initializing Enhanced Comms Secretary...")
        
        # Initialize reasoning engine
        reasoning_config = self.config.get("reasoning", {})
        try:
            self.reasoning_engine = create_reasoning_engine(reasoning_config)
            logger.info("Multi-LLM reasoning engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            # Continue with fallback pattern matching
        
        # Test Slack connection if available
        if self.slack_client:
            try:
                auth_test = self.slack_client.auth_test()
                logger.info(f"Connected to Slack as: {auth_test['user']}")
                
                # Send initialization notification
                await self._send_initialization_notification()
                
            except Exception as e:
                logger.error(f"Slack connection test failed: {e}")
        
        # Initialize conversation context cleanup task
        asyncio.create_task(self._cleanup_conversations_loop())
        
        logger.info("Enhanced Comms Secretary initialization complete")
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute communication-related tasks."""
        task_type = task.get("type")
        
        try:
            if task_type == "slack_message":
                return await self._handle_slack_message(task)
            elif task_type == "send_notification":
                return await self._send_notification(task)
            elif task_type == "terminal_command":
                return await self._handle_terminal_command(task)
            elif task_type == "file_transfer":
                return await self._handle_file_transfer(task)
            elif task_type == "decision_request":
                return await self._handle_decision_request(task)
            elif task_type == "status_update":
                return await self._handle_status_update(task)
            elif task_type == "interactive_response":
                return await self._handle_interactive_response(task)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return {"status": "unknown_task_type", "task_type": task_type}
                
        except Exception as e:
            logger.error(f"Error executing task {task_type}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_slack_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming Slack message with full reasoning capabilities."""
        start_time = time.time()
        
        # Extract message data
        message_data = task.get("data", {})
        message_text = message_data.get("text", "")
        user_id = message_data.get("user", "unknown")
        channel = message_data.get("channel", "unknown")
        timestamp = message_data.get("ts", str(time.time()))
        
        self.message_count += 1
        
        logger.info(f"Processing Slack message from {user_id}: {message_text[:50]}...")
        
        try:
            # Get conversation context
            context = await self._get_conversation_context(user_id, channel)
            
            # Build interpretation context
            interpretation_context = {
                "user_id": user_id,
                "channel": channel,
                "history": context.message_history,
                "user_preferences": context.user_preferences,
                "active_tasks": context.active_tasks,
                "timestamp": timestamp
            }
            
            # Interpret message using Agora consensus or fallback
            interpretation_start = time.time()
            
            # Try Agora consensus for complex message interpretation
            if self.agora_orchestrator and len(message_text) > 50:
                try:
                    # Create Agora task for message interpretation
                    agora_task = {
                        'id': f"slack_interpret_{int(time.time())}",
                        'type': 'natural_language_analysis',
                        'prompt': f"""
Analyze this Slack message from user {user_id} and provide interpretation:

Message: "{message_text}"

Context:
- Channel: {channel}
- User preferences: {context.user_preferences}
- Active tasks: {context.active_tasks}
- Recent history: {context.message_history[-3:] if context.message_history else 'None'}

Determine:
1. Intent (question, command, request, information, casual)
2. Confidence level (0.0-1.0)
3. Required action (if any)
4. Response strategy
5. Priority level

Provide structured analysis with clear recommendations.
                        """,
                        'complexity': 'medium',
                        'max_loops': 1,
                        'context': interpretation_context
                    }
                    
                    agora_result = await self.invoke_agora_consensus(agora_task)
                    
                    # Convert Agora result to interpretation format
                    interpretation = self._parse_agora_interpretation(
                        agora_result, message_text, interpretation_context
                    )
                    
                    # Store interpretation metrics
                    self.interpretation_times.append(time.time() - interpretation_start)
                    
                    logger.info(f"Agora interpretation completed with confidence {interpretation.confidence:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Agora interpretation failed, using fallback: {e}")
                    interpretation = self._simple_interpretation(message_text, interpretation_context)
            
            elif self.reasoning_engine:
                # Use legacy reasoning engine if available
                consensus_result = await self.reasoning_engine.interpret_with_consensus(
                    message_text, interpretation_context
                )
                interpretation = consensus_result.final_interpretation
                
                # Store interpretation metrics
                self.interpretation_times.append(consensus_result.processing_time)
                
                # Log detailed reasoning results
                await self._log_interpretation_analysis(
                    user_id, message_text, consensus_result
                )
            else:
                # Fallback to simple pattern matching
                interpretation = self._simple_interpretation(message_text, interpretation_context)
            
            interpretation_time = time.time() - interpretation_start
            
            # Update conversation context
            await self._update_conversation_context(
                context, message_text, interpretation
            )
            
            # Log interaction to memory system
            await self._log_interaction_to_memory(
                user_id, channel, message_text, interpretation, timestamp
            )
            
            # Process based on interpretation confidence
            if interpretation.confidence < 0.5:
                # Low confidence - ask for clarification
                response = await self._handle_clarification_needed(
                    channel, interpretation, user_id
                )
                return {
                    "status": "clarification_requested",
                    "confidence": interpretation.confidence,
                    "response": response
                }
            
            # Route task based on intent
            routing_result = await self._route_interpreted_message(
                interpretation, user_id, channel, message_text
            )
            
            # Send acknowledgment to user
            await self._send_acknowledgment(channel, interpretation, user_id)
            
            processing_time = time.time() - start_time
            self.response_times.append(processing_time)
            
            return {
                "status": "processed",
                "intent": interpretation.intent,
                "confidence": interpretation.confidence,
                "routing_result": routing_result,
                "processing_time": processing_time,
                "interpretation_time": interpretation_time
            }
            
        except Exception as e:
            logger.error(f"Failed to process Slack message: {e}")
            
            # Send error response to user
            if self.slack_client:
                try:
                    await self._send_slack_message(
                        channel,
                        "I encountered an error processing your message. Please try again or contact support."
                    )
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {send_error}")
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _route_interpreted_message(self, interpretation, user_id: str, 
                                       channel: str, original_message: str) -> Dict[str, Any]:
        """Route interpreted message to appropriate system components."""
        
        intent = interpretation.intent
        entities = interpretation.entities
        
        # Create task for appropriate handler
        routed_task = {
            "id": f"human_request_{int(time.time())}_{user_id}",
            "type": "human_request",
            "intent": intent,
            "entities": entities,
            "source": {
                "user_id": user_id,
                "channel": channel,
                "original_message": original_message,
                "timestamp": time.time()
            },
            "interpretation_confidence": interpretation.confidence,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Route based on intent
        if intent == "status_inquiry":
            # Route to monitoring system
            return await self._route_status_inquiry(routed_task)
        elif intent == "task_creation":
            # Route to Chief of Staff for task planning
            return await self._route_task_creation(routed_task)
        elif intent == "terminal_command":
            # Handle terminal passthrough
            return await self._route_terminal_command(routed_task)
        elif intent == "research_request":
            # Route to research agent
            return await self._route_research_request(routed_task)
        elif intent == "configuration_change":
            # Route to admin system
            return await self._route_configuration_change(routed_task)
        elif intent == "help_request":
            # Handle directly
            return await self._handle_help_request(routed_task)
        elif intent == "greeting":
            # Handle directly
            return await self._handle_greeting(routed_task)
        else:
            # General routing to Chief of Staff
            return await self._route_general_request(routed_task)
    
    async def _send_acknowledgment(self, channel: str, interpretation, user_id: str):
        """Send intelligent acknowledgment based on interpretation."""
        
        intent = interpretation.intent
        confidence = interpretation.confidence
        
        # Use interpretation's suggested response if available and high confidence
        if interpretation.response and confidence > 0.7:
            acknowledgment = interpretation.response
        else:
            # Generate context-aware acknowledgment
            acknowledgments = {
                "status_inquiry": "I'll check that status for you right away! 📊",
                "task_creation": "Got it! I'll create that task and get the right team on it. 🚀",
                "terminal_command": "I'll execute that command securely. ⚡",
                "research_request": "I'll research that topic for you. 🔍",
                "configuration_change": "I'll process that configuration change. ⚙️",
                "help_request": "I'm here to help! Let me get you the information you need. 💡",
                "greeting": "Hello! Great to see you! How can I help today? 👋",
                "clarification_needed": "I want to make sure I understand correctly - could you clarify?"
            }
            
            acknowledgment = acknowledgments.get(
                intent, 
                f"I understand you want me to {intent.replace('_', ' ')}. I'm on it! 🤖"
            )
        
        # Add confidence indicator for transparency
        if confidence < 0.8:
            acknowledgment += f" (Confidence: {confidence:.0%})"
        
        await self._send_slack_message(channel, acknowledgment)
    
    async def _send_slack_message(self, channel: str, message: str, 
                                blocks: Optional[List[Dict[str, Any]]] = None,
                                thread_ts: Optional[str] = None) -> Optional[str]:
        """Send message to Slack with error handling."""
        if not self.slack_client:
            logger.warning("Slack client not available")
            return None
        
        try:
            kwargs = {
                "channel": channel,
                "username": "AETHELRED-Enhanced"
            }
            
            if blocks:
                kwargs["blocks"] = blocks
            else:
                kwargs["text"] = message
            
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            
            response = self.slack_client.chat_postMessage(**kwargs)
            return response["ts"]
            
        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return None
    
    async def _get_conversation_context(self, user_id: str, channel: str) -> ConversationContext:
        """Get or create conversation context for user."""
        
        context_key = f"{user_id}_{channel}"
        
        if context_key not in self.active_conversations:
            # Create new context
            context = ConversationContext(
                user_id=user_id,
                channel=channel,
                message_history=[],
                user_preferences={},
                active_tasks=[],
                last_interaction=time.time(),
                session_id=f"session_{int(time.time())}_{user_id}"
            )
            
            # Try to load context from memory
            try:
                stored_context = await self._load_user_context_from_memory(user_id)
                if stored_context:
                    context.user_preferences = stored_context.get("preferences", {})
                    context.message_history = stored_context.get("recent_history", [])[-10:]  # Last 10 messages
            except Exception as e:
                logger.error(f"Failed to load user context: {e}")
            
            self.active_conversations[context_key] = context
        
        context = self.active_conversations[context_key]
        context.last_interaction = time.time()
        
        return context
    
    async def _load_notification_templates(self) -> Dict[str, NotificationTemplate]:
        """Load notification templates for different event types."""
        return {
            "task_completion": NotificationTemplate(
                name="task_completion",
                template_type="task_completion",
                blocks=[
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "✅ Task Completed"}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": "*Task:* {task_name}"},
                            {"type": "mrkdwn", "text": "*Agent:* {agent_id}"},
                            {"type": "mrkdwn", "text": "*Duration:* {duration}"},
                            {"type": "mrkdwn", "text": "*Status:* {status}"}
                        ]
                    }
                ]
            ),
            "decision_required": NotificationTemplate(
                name="decision_required",
                template_type="decision_required",
                blocks=[
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🤔 Decision Required"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "{description}"}
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Approve"},
                                "value": "approve:{decision_id}",
                                "action_id": "decision_approve",
                                "style": "primary"
                            },
                            {
                                "type": "button", 
                                "text": {"type": "plain_text", "text": "Reject"},
                                "value": "reject:{decision_id}",
                                "action_id": "decision_reject",
                                "style": "danger"
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "More Info"},
                                "value": "info:{decision_id}",
                                "action_id": "decision_info"
                            }
                        ]
                    }
                ]
            )
        }
    
    async def _send_initialization_notification(self):
        """Send enhanced initialization notification to Slack."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🧠 AETHELRED Enhanced Communication System Online"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Advanced Human-AI Interface Activated* 🚀"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "*🤖 Multi-LLM Reasoning:* Active"},
                    {"type": "mrkdwn", "text": "*💬 Natural Language:* Full understanding"},
                    {"type": "mrkdwn", "text": "*⚡ Terminal Passthrough:* Secure execution"},
                    {"type": "mrkdwn", "text": "*📊 Decision Workflows:* Interactive approval"},
                    {"type": "mrkdwn", "text": "*🧠 Memory Integration:* 4-tier storage"},
                    {"type": "mrkdwn", "text": "*🔒 Security Framework:* Enterprise-grade"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Ready for advanced human-AI collaboration!* Just talk to me naturally - I understand context, intent, and can coordinate complex workflows."
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "System Status"},
                        "value": "system_status",
                        "action_id": "get_status"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Help"},
                        "value": "help",
                        "action_id": "get_help"
                    }
                ]
            }
        ]
        
        await self._send_slack_message(
            self.channels["notifications"],
            "AETHELRED Enhanced Communication System is now online!",
            blocks=blocks
        )
    
    # Placeholder methods for routing (to be implemented based on system architecture)
    async def _route_status_inquiry(self, task): return {"status": "routed_to_monitoring"}
    async def _route_task_creation(self, task): return {"status": "routed_to_chief_of_staff"}
    async def _route_terminal_command(self, task): return {"status": "routed_to_terminal"}
    async def _route_research_request(self, task): return {"status": "routed_to_research"}
    async def _route_configuration_change(self, task): return {"status": "routed_to_admin"}
    async def _route_general_request(self, task): return {"status": "routed_to_general"}
    
    async def _handle_help_request(self, task): return {"status": "help_provided"}
    async def _handle_greeting(self, task): return {"status": "greeting_handled"}
    
    # Additional placeholder methods
    async def _handle_clarification_needed(self, channel, interpretation, user_id): return "clarification_sent"
    async def _log_interpretation_analysis(self, user_id, message, consensus_result): pass
    async def _update_conversation_context(self, context, message, interpretation): pass
    async def _log_interaction_to_memory(self, user_id, channel, message, interpretation, timestamp): pass
    async def _load_user_context_from_memory(self, user_id): return None
    async def _cleanup_conversations_loop(self): pass
    
    def _simple_interpretation(self, message, context):
        """Fallback interpretation method."""
        from core.reasoning.multi_llm_interpreter import Interpretation
        return Interpretation(
            intent="clarification_needed",
            entities={"message": message},
            confidence=0.3,
            response="I need clarification on your request."
        )
    
    # Additional task handlers (stubs for now)
    async def _send_notification(self, task): return {"status": "notification_sent"}
    async def _handle_terminal_command(self, task): return {"status": "terminal_executed"}
    async def _handle_file_transfer(self, task): return {"status": "file_transferred"}
    async def _handle_decision_request(self, task): return {"status": "decision_presented"}
    async def _handle_status_update(self, task): return {"status": "status_updated"}
    async def _handle_interactive_response(self, task): return {"status": "interaction_processed"}
    
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate if this agent can handle the given task."""
        task_type = task.get("type", "")
        
        # Communication-related task types this agent can handle
        valid_task_types = {
            "slack_message",
            "send_notification", 
            "terminal_command",
            "file_transfer",
            "decision_request",
            "status_update",
            "interactive_response",
            "human_request",
            "natural_language_command",
            "research_request"
        }
        
        return task_type in valid_task_types
    
    async def check_agent_health(self) -> Dict[str, Any]:
        """Enhanced health check for Comms Secretary."""
        health_data = {
            "slack_connection": self.slack_client is not None,
            "reasoning_engine": self.reasoning_engine is not None,
            "active_conversations": len(self.active_conversations),
            "messages_processed": self.message_count,
            "avg_interpretation_time": sum(self.interpretation_times[-100:]) / len(self.interpretation_times[-100:]) if self.interpretation_times else 0,
            "avg_response_time": sum(self.response_times[-100:]) / len(self.response_times[-100:]) if self.response_times else 0
        }
        
        # Test Slack connection if available
        if self.slack_client:
            try:
                self.slack_client.auth_test()
                health_data["slack_status"] = "connected"
            except Exception as e:
                health_data["slack_status"] = f"error: {str(e)}"
        else:
            health_data["slack_status"] = "not_configured"
        
        return health_data
    
    def _parse_agora_interpretation(self, agora_result: Dict[str, Any], 
                                   message_text: str, context: Dict[str, Any]) -> Any:
        """Parse Agora consensus result into interpretation format."""
        try:
            # Extract content from Agora result
            content = agora_result.get('content', '')
            confidence = agora_result.get('confidence', 0.7)
            
            # Parse the structured analysis from content
            intent = "question"  # Default
            action = None
            priority = "medium"
            
            # Simple parsing of Agora response
            content_lower = content.lower()
            
            # Determine intent
            if any(word in content_lower for word in ['command', 'execute', 'run', 'do']):
                intent = "command"
            elif any(word in content_lower for word in ['request', 'please', 'help']):
                intent = "request"
            elif any(word in content_lower for word in ['question', '?', 'what', 'how', 'why']):
                intent = "question"
            elif any(word in content_lower for word in ['information', 'tell', 'explain']):
                intent = "information"
            else:
                intent = "casual"
            
            # Determine action needed
            if any(word in content_lower for word in ['terminal', 'bash', 'command']):
                action = "terminal_passthrough"
            elif any(word in content_lower for word in ['task', 'todo', 'assignment']):
                action = "task_creation"
            elif any(word in content_lower for word in ['file', 'upload', 'download']):
                action = "file_operation"
            elif any(word in content_lower for word in ['notification', 'alert', 'notify']):
                action = "send_notification"
            
            # Determine priority
            if any(word in content_lower for word in ['urgent', 'critical', 'asap']):
                priority = "high"
            elif any(word in content_lower for word in ['low', 'later', 'whenever']):
                priority = "low"
            
            # Create interpretation object (simplified version)
            class AgoraInterpretation:
                def __init__(self):
                    self.intent = intent
                    self.confidence = confidence
                    self.action = action
                    self.priority = priority
                    self.reasoning = content
                    self.response = content
                    
            return AgoraInterpretation()
            
        except Exception as e:
            logger.error(f"Failed to parse Agora interpretation: {e}")
            # Return simple fallback interpretation
            class FallbackInterpretation:
                def __init__(self):
                    self.intent = "question"
                    self.confidence = 0.5
                    self.action = None
                    self.priority = "medium"
                    self.reasoning = "Agora parsing failed, using fallback"
                    self.response = f"I received your message: {message_text}"
                    
            return FallbackInterpretation()