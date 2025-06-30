"""
Interactive Slack Components for AETHELRED
Rich UI components for human-AI interaction workflows.
Implementation from aethelred-slack-interface specification.
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of interactive components."""
    BUTTON = "button"
    SELECT = "select"
    TEXT_INPUT = "text_input"
    MODAL = "modal"
    WORKFLOW = "workflow"

class InteractionState(Enum):
    """States of interactive workflows."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass
class InteractionContext:
    """Context for interactive workflows."""
    interaction_id: str
    user_id: str
    channel: str
    workflow_type: str
    state: InteractionState
    data: Dict[str, Any]
    created_at: float
    expires_at: Optional[float] = None
    completed_at: Optional[float] = None

class SlackBlockBuilder:
    """Builder for Slack Block Kit components."""
    
    @staticmethod
    def header(text: str) -> Dict[str, Any]:
        """Create header block."""
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text
            }
        }
    
    @staticmethod
    def section(text: str, markdown: bool = True, 
               fields: Optional[List[str]] = None,
               accessory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create section block."""
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn" if markdown else "plain_text",
                "text": text
            }
        }
        
        if fields:
            block["fields"] = [
                {"type": "mrkdwn", "text": field} for field in fields
            ]
        
        if accessory:
            block["accessory"] = accessory
            
        return block
    
    @staticmethod
    def divider() -> Dict[str, Any]:
        """Create divider block."""
        return {"type": "divider"}
    
    @staticmethod
    def context(elements: List[str]) -> Dict[str, Any]:
        """Create context block."""
        return {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": element} for element in elements
            ]
        }
    
    @staticmethod
    def actions(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create actions block."""
        return {
            "type": "actions",
            "elements": elements
        }
    
    @staticmethod
    def button(text: str, value: str, action_id: str,
              style: Optional[str] = None, url: Optional[str] = None,
              confirm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create button element."""
        button = {
            "type": "button",
            "text": {"type": "plain_text", "text": text},
            "action_id": action_id
        }
        
        if value:
            button["value"] = value
        if url:
            button["url"] = url
        if style:
            button["style"] = style  # primary, danger
        if confirm:
            button["confirm"] = confirm
            
        return button
    
    @staticmethod
    def select_menu(placeholder: str, action_id: str, options: List[Dict[str, str]],
                   initial_option: Optional[str] = None) -> Dict[str, Any]:
        """Create select menu."""
        select = {
            "type": "static_select",
            "placeholder": {"type": "plain_text", "text": placeholder},
            "action_id": action_id,
            "options": [
                {
                    "text": {"type": "plain_text", "text": opt["text"]},
                    "value": opt["value"]
                }
                for opt in options
            ]
        }
        
        if initial_option:
            select["initial_option"] = {
                "text": {"type": "plain_text", "text": initial_option},
                "value": initial_option
            }
            
        return select
    
    @staticmethod
    def confirmation_dialog(title: str, text: str, 
                          confirm_text: str = "Confirm",
                          deny_text: str = "Cancel") -> Dict[str, Any]:
        """Create confirmation dialog."""
        return {
            "title": {"type": "plain_text", "text": title},
            "text": {"type": "mrkdwn", "text": text},
            "confirm": {"type": "plain_text", "text": confirm_text},
            "deny": {"type": "plain_text", "text": deny_text}
        }

class WorkflowTemplates:
    """Pre-built workflow templates for common interactions."""
    
    @staticmethod
    def task_approval(task_data: Dict[str, Any], interaction_id: str) -> List[Dict[str, Any]]:
        """Task approval workflow."""
        task_name = task_data.get("name", "Unknown Task")
        task_description = task_data.get("description", "No description")
        estimated_time = task_data.get("estimated_time", "Unknown")
        priority = task_data.get("priority", "medium")
        
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
        
        return [
            SlackBlockBuilder.header("🤔 Task Approval Required"),
            SlackBlockBuilder.section(
                f"*Task:* {task_name}\n*Description:* {task_description}",
                fields=[
                    f"*Priority:* {priority_emoji} {priority.title()}",
                    f"*Estimated Time:* {estimated_time}",
                    f"*Requested:* {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"*ID:* `{interaction_id}`"
                ]
            ),
            SlackBlockBuilder.divider(),
            SlackBlockBuilder.actions([
                SlackBlockBuilder.button(
                    "✅ Approve", f"approve:{interaction_id}", "task_approve", "primary"
                ),
                SlackBlockBuilder.button(
                    "❌ Reject", f"reject:{interaction_id}", "task_reject", "danger",
                    confirm=SlackBlockBuilder.confirmation_dialog(
                        "Reject Task",
                        "Are you sure you want to reject this task?",
                        "Reject", "Cancel"
                    )
                ),
                SlackBlockBuilder.button(
                    "ℹ️ More Info", f"info:{interaction_id}", "task_info"
                )
            ]),
            SlackBlockBuilder.context([
                "Use the buttons above to approve or reject this task. You can also request more information."
            ])
        ]
    
    @staticmethod
    def command_confirmation(command_data: Dict[str, Any], interaction_id: str) -> List[Dict[str, Any]]:
        """Terminal command confirmation workflow."""
        command = command_data.get("command", "")
        user = command_data.get("user_name", "Unknown User")
        risk_level = command_data.get("risk_level", "medium")
        
        risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        risk_emoji = risk_colors.get(risk_level, "⚪")
        
        return [
            SlackBlockBuilder.header("⚡ Terminal Command Approval"),
            SlackBlockBuilder.section(
                f"*User:* {user}\n*Command:* ```{command}```",
                fields=[
                    f"*Risk Level:* {risk_emoji} {risk_level.title()}",
                    f"*Requested:* {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"*Session:* `{interaction_id}`"
                ]
            ),
            SlackBlockBuilder.divider(),
            SlackBlockBuilder.actions([
                SlackBlockBuilder.button(
                    "✅ Execute", f"execute:{interaction_id}", "cmd_execute", "primary"
                ),
                SlackBlockBuilder.button(
                    "❌ Deny", f"deny:{interaction_id}", "cmd_deny", "danger"
                ),
                SlackBlockBuilder.button(
                    "⏸️ Hold", f"hold:{interaction_id}", "cmd_hold"
                )
            ]),
            SlackBlockBuilder.context([
                "⚠️ This will execute on the user's local machine. Review carefully before approving."
            ])
        ]
    
    @staticmethod
    def system_status_dashboard(status_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """System status dashboard."""
        system_health = status_data.get("health", "unknown")
        uptime = status_data.get("uptime", "unknown")
        active_agents = status_data.get("active_agents", 0)
        
        health_emoji = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(system_health, "⚪")
        
        return [
            SlackBlockBuilder.header("📊 AETHELRED System Dashboard"),
            SlackBlockBuilder.section(
                f"*System Status:* {health_emoji} {system_health.title()}",
                fields=[
                    f"*Uptime:* {uptime}",
                    f"*Active Agents:* {active_agents}",
                    f"*Memory Usage:* {status_data.get('memory_usage', 'N/A')}",
                    f"*Last Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ]
            ),
            SlackBlockBuilder.divider(),
            SlackBlockBuilder.actions([
                SlackBlockBuilder.button(
                    "🔄 Refresh", "refresh_status", "status_refresh"
                ),
                SlackBlockBuilder.button(
                    "📈 Metrics", "view_metrics", "status_metrics"
                ),
                SlackBlockBuilder.button(
                    "🔧 Settings", "system_settings", "status_settings"
                )
            ])
        ]
    
    @staticmethod
    def decision_workflow(decision_data: Dict[str, Any], interaction_id: str) -> List[Dict[str, Any]]:
        """General decision workflow."""
        title = decision_data.get("title", "Decision Required")
        description = decision_data.get("description", "A decision is needed")
        options = decision_data.get("options", ["Approve", "Reject"])
        urgency = decision_data.get("urgency", "normal")
        
        urgency_emoji = {"urgent": "🚨", "high": "⚠️", "normal": "ℹ️", "low": "📝"}.get(urgency, "❓")
        
        # Create buttons for each option
        buttons = []
        for i, option in enumerate(options):
            style = "primary" if i == 0 else ("danger" if "reject" in option.lower() else None)
            buttons.append(
                SlackBlockBuilder.button(
                    option, f"decision:{interaction_id}:{i}", f"decision_option_{i}", style
                )
            )
        
        return [
            SlackBlockBuilder.header(f"{urgency_emoji} {title}"),
            SlackBlockBuilder.section(description),
            SlackBlockBuilder.divider(),
            SlackBlockBuilder.actions(buttons),
            SlackBlockBuilder.context([
                f"Urgency: {urgency.title()} • ID: {interaction_id}"
            ])
        ]

class InteractiveComponentManager:
    """Manager for interactive Slack components and workflows."""
    
    def __init__(self, slack_client: WebClient):
        self.slack_client = slack_client
        self.active_interactions: Dict[str, InteractionContext] = {}
        self.interaction_handlers: Dict[str, Callable] = {}
        self.workflow_timeouts: Dict[str, int] = {
            "task_approval": 3600,  # 1 hour
            "command_confirmation": 300,  # 5 minutes
            "decision": 1800,  # 30 minutes
            "default": 900  # 15 minutes
        }
        
    def register_interaction_handler(self, action_id: str, handler: Callable):
        """Register handler for interactive component actions."""
        self.interaction_handlers[action_id] = handler
        logger.info(f"Registered interaction handler for: {action_id}")
    
    async def create_task_approval(self, channel: str, task_data: Dict[str, Any], 
                                 requester_id: str) -> str:
        """Create task approval interaction."""
        interaction_id = str(uuid.uuid4())
        
        # Create interaction context
        context = InteractionContext(
            interaction_id=interaction_id,
            user_id=requester_id,
            channel=channel,
            workflow_type="task_approval",
            state=InteractionState.PENDING,
            data=task_data,
            created_at=time.time(),
            expires_at=time.time() + self.workflow_timeouts["task_approval"]
        )
        
        self.active_interactions[interaction_id] = context
        
        # Generate blocks
        blocks = WorkflowTemplates.task_approval(task_data, interaction_id)
        
        # Send to Slack
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text="Task approval required",
                blocks=blocks
            )
            
            # Store message info
            context.data["message_ts"] = response["ts"]
            context.data["channel"] = channel
            
            logger.info(f"Created task approval interaction: {interaction_id}")
            return interaction_id
            
        except SlackApiError as e:
            logger.error(f"Failed to create task approval: {e}")
            raise
    
    async def create_command_confirmation(self, channel: str, command_data: Dict[str, Any],
                                        requester_id: str) -> str:
        """Create command confirmation interaction."""
        interaction_id = str(uuid.uuid4())
        
        context = InteractionContext(
            interaction_id=interaction_id,
            user_id=requester_id,
            channel=channel,
            workflow_type="command_confirmation",
            state=InteractionState.PENDING,
            data=command_data,
            created_at=time.time(),
            expires_at=time.time() + self.workflow_timeouts["command_confirmation"]
        )
        
        self.active_interactions[interaction_id] = context
        
        blocks = WorkflowTemplates.command_confirmation(command_data, interaction_id)
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text="Terminal command approval required",
                blocks=blocks
            )
            
            context.data["message_ts"] = response["ts"]
            context.data["channel"] = channel
            
            logger.info(f"Created command confirmation: {interaction_id}")
            return interaction_id
            
        except SlackApiError as e:
            logger.error(f"Failed to create command confirmation: {e}")
            raise
    
    async def create_system_dashboard(self, channel: str, status_data: Dict[str, Any]) -> str:
        """Create system status dashboard."""
        blocks = WorkflowTemplates.system_status_dashboard(status_data)
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text="AETHELRED System Dashboard",
                blocks=blocks
            )
            
            logger.info(f"Created system dashboard in {channel}")
            return response["ts"]
            
        except SlackApiError as e:
            logger.error(f"Failed to create dashboard: {e}")
            raise
    
    async def create_decision_workflow(self, channel: str, decision_data: Dict[str, Any],
                                     requester_id: str) -> str:
        """Create general decision workflow."""
        interaction_id = str(uuid.uuid4())
        
        context = InteractionContext(
            interaction_id=interaction_id,
            user_id=requester_id,
            channel=channel,
            workflow_type="decision",
            state=InteractionState.PENDING,
            data=decision_data,
            created_at=time.time(),
            expires_at=time.time() + self.workflow_timeouts["decision"]
        )
        
        self.active_interactions[interaction_id] = context
        
        blocks = WorkflowTemplates.decision_workflow(decision_data, interaction_id)
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text="Decision required",
                blocks=blocks
            )
            
            context.data["message_ts"] = response["ts"]
            context.data["channel"] = channel
            
            logger.info(f"Created decision workflow: {interaction_id}")
            return interaction_id
            
        except SlackApiError as e:
            logger.error(f"Failed to create decision workflow: {e}")
            raise
    
    async def handle_interaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactive component action."""
        action = payload.get("actions", [{}])[0]
        action_id = action.get("action_id", "")
        action_value = action.get("value", "")
        user = payload.get("user", {})
        user_id = user.get("id", "")
        
        logger.info(f"Handling interaction: {action_id} by {user_id}")
        
        # Parse interaction ID from value
        if ":" in action_value:
            parts = action_value.split(":")
            if len(parts) >= 2:
                interaction_id = parts[1]
                
                # Find interaction context
                if interaction_id in self.active_interactions:
                    context = self.active_interactions[interaction_id]
                    
                    # Update state
                    context.state = InteractionState.IN_PROGRESS
                    
                    # Route to specific handler
                    return await self._route_interaction(action_id, action_value, payload, context)
        
        # Check for registered handlers
        if action_id in self.interaction_handlers:
            return await self.interaction_handlers[action_id](payload)
        
        # Default response
        return {"status": "unhandled", "action_id": action_id}
    
    async def _route_interaction(self, action_id: str, action_value: str,
                               payload: Dict[str, Any], context: InteractionContext) -> Dict[str, Any]:
        """Route interaction to appropriate handler."""
        
        action_type = action_value.split(":")[0]
        user_id = payload.get("user", {}).get("id", "")
        
        if context.workflow_type == "task_approval":
            return await self._handle_task_approval_action(action_type, payload, context)
        elif context.workflow_type == "command_confirmation":
            return await self._handle_command_confirmation_action(action_type, payload, context)
        elif context.workflow_type == "decision":
            return await self._handle_decision_action(action_type, payload, context)
        else:
            logger.warning(f"No handler for workflow type: {context.workflow_type}")
            return {"status": "no_handler", "workflow_type": context.workflow_type}
    
    async def _handle_task_approval_action(self, action_type: str, payload: Dict[str, Any],
                                         context: InteractionContext) -> Dict[str, Any]:
        """Handle task approval action."""
        user_id = payload.get("user", {}).get("id", "")
        
        if action_type == "approve":
            context.state = InteractionState.COMPLETED
            context.completed_at = time.time()
            context.data["approved_by"] = user_id
            context.data["decision"] = "approved"
            
            # Update message
            await self._update_approval_message(context, "✅ Task Approved", "primary")
            
            return {"status": "approved", "approved_by": user_id}
            
        elif action_type == "reject":
            context.state = InteractionState.COMPLETED
            context.completed_at = time.time()
            context.data["rejected_by"] = user_id
            context.data["decision"] = "rejected"
            
            await self._update_approval_message(context, "❌ Task Rejected", "danger")
            
            return {"status": "rejected", "rejected_by": user_id}
            
        elif action_type == "info":
            # Send detailed info as ephemeral message
            await self._send_task_info(context, user_id)
            return {"status": "info_sent"}
        
        return {"status": "unknown_action", "action_type": action_type}
    
    async def _handle_command_confirmation_action(self, action_type: str, payload: Dict[str, Any],
                                                context: InteractionContext) -> Dict[str, Any]:
        """Handle command confirmation action."""
        user_id = payload.get("user", {}).get("id", "")
        
        if action_type == "execute":
            context.state = InteractionState.COMPLETED
            context.completed_at = time.time()
            context.data["approved_by"] = user_id
            context.data["decision"] = "execute"
            
            await self._update_approval_message(context, "⚡ Command Approved for Execution", "primary")
            
            return {"status": "execute_approved", "approved_by": user_id}
            
        elif action_type == "deny":
            context.state = InteractionState.COMPLETED
            context.completed_at = time.time()
            context.data["denied_by"] = user_id
            context.data["decision"] = "denied"
            
            await self._update_approval_message(context, "🚫 Command Execution Denied", "danger")
            
            return {"status": "execute_denied", "denied_by": user_id}
            
        elif action_type == "hold":
            context.data["held_by"] = user_id
            # Keep as pending but mark as held
            
            await self._update_approval_message(context, "⏸️ Command Held for Review", "")
            
            return {"status": "held", "held_by": user_id}
        
        return {"status": "unknown_action", "action_type": action_type}
    
    async def _handle_decision_action(self, action_type: str, payload: Dict[str, Any],
                                    context: InteractionContext) -> Dict[str, Any]:
        """Handle general decision action."""
        user_id = payload.get("user", {}).get("id", "")
        
        if action_type == "decision":
            # Extract option index from value
            value_parts = payload.get("actions", [{}])[0].get("value", "").split(":")
            if len(value_parts) >= 3:
                option_index = int(value_parts[2])
                options = context.data.get("options", [])
                
                if 0 <= option_index < len(options):
                    selected_option = options[option_index]
                    
                    context.state = InteractionState.COMPLETED
                    context.completed_at = time.time()
                    context.data["decided_by"] = user_id
                    context.data["decision"] = selected_option
                    context.data["option_index"] = option_index
                    
                    await self._update_approval_message(
                        context, f"✅ Decision: {selected_option}", "primary"
                    )
                    
                    return {
                        "status": "decision_made",
                        "decided_by": user_id,
                        "decision": selected_option,
                        "option_index": option_index
                    }
        
        return {"status": "unknown_action", "action_type": action_type}
    
    async def _update_approval_message(self, context: InteractionContext, 
                                     status_text: str, color: str):
        """Update approval message with result."""
        try:
            # Create updated blocks showing the result
            blocks = [
                SlackBlockBuilder.header(status_text),
                SlackBlockBuilder.section(
                    f"*Decision made by:* <@{context.data.get('approved_by', context.data.get('rejected_by', context.data.get('decided_by', 'unknown')))}>",
                    fields=[
                        f"*Completed:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"*Workflow:* {context.workflow_type.replace('_', ' ').title()}",
                        f"*ID:* `{context.interaction_id}`"
                    ]
                )
            ]
            
            # Update the original message
            self.slack_client.chat_update(
                channel=context.data.get("channel"),
                ts=context.data.get("message_ts"),
                text=status_text,
                blocks=blocks
            )
            
        except SlackApiError as e:
            logger.error(f"Failed to update approval message: {e}")
    
    async def _send_task_info(self, context: InteractionContext, user_id: str):
        """Send detailed task information as ephemeral message."""
        task_data = context.data
        
        blocks = [
            SlackBlockBuilder.header("📋 Detailed Task Information"),
            SlackBlockBuilder.section(
                f"*Full Description:* {task_data.get('description', 'N/A')}",
                fields=[
                    f"*Estimated Duration:* {task_data.get('estimated_time', 'Unknown')}",
                    f"*Required Resources:* {task_data.get('resources', 'None specified')}",
                    f"*Dependencies:* {task_data.get('dependencies', 'None')}",
                    f"*Assigned Agent:* {task_data.get('assigned_agent', 'To be determined')}"
                ]
            )
        ]
        
        try:
            self.slack_client.chat_postEphemeral(
                channel=context.data.get("channel"),
                user=user_id,
                text="Detailed task information",
                blocks=blocks
            )
        except SlackApiError as e:
            logger.error(f"Failed to send task info: {e}")
    
    async def cleanup_expired_interactions(self):
        """Clean up expired interactions."""
        current_time = time.time()
        expired_interactions = []
        
        for interaction_id, context in self.active_interactions.items():
            if (context.expires_at and 
                current_time > context.expires_at and 
                context.state in [InteractionState.PENDING, InteractionState.IN_PROGRESS]):
                
                expired_interactions.append(interaction_id)
        
        for interaction_id in expired_interactions:
            context = self.active_interactions[interaction_id]
            context.state = InteractionState.EXPIRED
            
            # Update message to show expiration
            await self._update_approval_message(context, "⏰ Interaction Expired", "")
            
            del self.active_interactions[interaction_id]
            logger.info(f"Cleaned up expired interaction: {interaction_id}")
    
    def get_interaction_status(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an interaction."""
        if interaction_id in self.active_interactions:
            context = self.active_interactions[interaction_id]
            return {
                "interaction_id": interaction_id,
                "workflow_type": context.workflow_type,
                "state": context.state.value,
                "created_at": context.created_at,
                "expires_at": context.expires_at,
                "completed_at": context.completed_at,
                "data": context.data
            }
        return None
    
    def get_active_interactions_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active interactions for a user."""
        user_interactions = []
        for context in self.active_interactions.values():
            if context.user_id == user_id and context.state in [InteractionState.PENDING, InteractionState.IN_PROGRESS]:
                user_interactions.append({
                    "interaction_id": context.interaction_id,
                    "workflow_type": context.workflow_type,
                    "state": context.state.value,
                    "created_at": context.created_at,
                    "expires_at": context.expires_at
                })
        return user_interactions