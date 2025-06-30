"""
Slack MCP Integration for Project Aethelred.

This module provides integration with Slack via Model Context Protocol (MCP)
leveraging the existing Cursor IDE MCP setup.
"""

import asyncio
import json
import logging
import subprocess
import websockets
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SlackMessage:
    """Represents a Slack message."""
    channel: str
    text: str
    user: Optional[str] = None
    timestamp: Optional[str] = None
    thread_ts: Optional[str] = None
    message_type: str = "message"


@dataclass
class SlackChannel:
    """Represents a Slack channel."""
    id: str
    name: str
    is_private: bool = False
    is_archived: bool = False


class SlackMCPIntegration:
    """
    Slack MCP Integration for AETHELRED.
    
    Provides bidirectional communication with Slack through MCP protocol,
    leveraging your existing Cursor IDE Slack integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.cursor_mcp_path = config.get('cursor_mcp_path', '/usr/local/lib/node_modules/@modelcontextprotocol/server-slack')
        
        # Message handling
        self.message_handlers: List[Callable] = []
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        self.inbound_queue: asyncio.Queue = asyncio.Queue()
        
        # Connection state
        self.connected = False
        self.channels: Dict[str, SlackChannel] = {}
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_attempts = 0
        
    async def initialize(self) -> bool:
        """Initialize Slack MCP integration."""
        if not self.enabled:
            logger.info("Slack MCP integration disabled")
            return True
            
        try:
            logger.info("Initializing Slack MCP integration...")
            
            # Test MCP connectivity
            mcp_available = await self._test_mcp_connectivity()
            if not mcp_available:
                logger.warning("MCP not available, running in mock mode")
                return False
                
            # Load channels
            await self._load_channels()
            
            # Start message processing
            self._start_message_processing()
            
            self.connected = True
            logger.info("Slack MCP integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack MCP integration: {e}")
            return False
            
    async def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None) -> bool:
        """
        Send message to Slack channel.
        
        Args:
            channel: Slack channel ID or name
            text: Message text
            thread_ts: Optional thread timestamp for replies
            
        Returns:
            True if message sent successfully
        """
        try:
            if not self.enabled:
                logger.info(f"MOCK SLACK -> {channel}: {text}")
                return True
                
            message = SlackMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Add to outbound queue
            await self.outbound_queue.put(message)
            
            # For now, use direct MCP call
            success = await self._send_via_mcp(message)
            
            if success:
                self.messages_sent += 1
                logger.info(f"Message sent to Slack channel {channel}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
            
    async def get_channels(self) -> List[SlackChannel]:
        """Get list of available Slack channels."""
        try:
            if not self.enabled:
                # Return mock channels
                return [
                    SlackChannel("C1234567", "general"),
                    SlackChannel("C2345678", "aethelred-notifications"),
                    SlackChannel("C3456789", "development")
                ]
                
            channels = await self._fetch_channels_via_mcp()
            return channels
            
        except Exception as e:
            logger.error(f"Failed to get Slack channels: {e}")
            return []
            
    async def get_channel_history(self, channel: str, limit: int = 100) -> List[SlackMessage]:
        """Get message history for a channel."""
        try:
            if not self.enabled:
                # Return mock history
                return []
                
            history = await self._fetch_history_via_mcp(channel, limit)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get channel history: {e}")
            return []
            
    def add_message_handler(self, handler: Callable[[SlackMessage], None]) -> None:
        """Add a message handler for incoming Slack messages."""
        self.message_handlers.append(handler)
        
    async def start_listening(self) -> None:
        """Start listening for incoming Slack messages."""
        if not self.enabled:
            logger.info("Slack listening disabled (mock mode)")
            return
            
        try:
            # Start MCP message listener
            await self._start_mcp_listener()
            
        except Exception as e:
            logger.error(f"Failed to start Slack listener: {e}")
            
    async def stop(self) -> None:
        """Stop Slack MCP integration."""
        self.connected = False
        logger.info("Slack MCP integration stopped")
        
    # Private methods
    
    async def _test_mcp_connectivity(self) -> bool:
        """Test if MCP Slack integration is available."""
        try:
            # Check if MCP server is available
            # This would typically test the connection to your Cursor IDE MCP setup
            
            # For now, assume it's available if the path exists
            import os
            if os.path.exists('/usr/local/bin/mcp-client'):
                return True
                
            # Alternative: check if running in Cursor IDE environment
            cursor_env = os.environ.get('CURSOR_IDE')
            if cursor_env:
                return True
                
            logger.warning("MCP not detected, running in mock mode")
            return False
            
        except Exception as e:
            logger.error(f"MCP connectivity test failed: {e}")
            return False
            
    async def _send_via_mcp(self, message: SlackMessage) -> bool:
        """Send message via MCP protocol."""
        try:
            # This would integrate with your actual MCP Slack setup
            # For now, we'll use a subprocess call as an example
            
            mcp_command = [
                'node',
                self.cursor_mcp_path,
                'send-message',
                '--channel', message.channel,
                '--text', message.text
            ]
            
            if message.thread_ts:
                mcp_command.extend(['--thread', message.thread_ts])
                
            # Execute MCP command
            result = subprocess.run(
                mcp_command,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.debug(f"MCP command successful: {result.stdout}")
                return True
            else:
                logger.error(f"MCP command failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("MCP command timed out")
            return False
        except Exception as e:
            logger.error(f"MCP send failed: {e}")
            return False
            
    async def _load_channels(self) -> None:
        """Load available Slack channels."""
        try:
            channels = await self.get_channels()
            self.channels = {ch.id: ch for ch in channels}
            logger.info(f"Loaded {len(self.channels)} Slack channels")
            
        except Exception as e:
            logger.error(f"Failed to load channels: {e}")
            
    async def _fetch_channels_via_mcp(self) -> List[SlackChannel]:
        """Fetch channels via MCP."""
        try:
            # MCP command to get channels
            result = subprocess.run(
                ['node', self.cursor_mcp_path, 'list-channels'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                channels_data = json.loads(result.stdout)
                channels = []
                
                for ch_data in channels_data:
                    channel = SlackChannel(
                        id=ch_data['id'],
                        name=ch_data['name'],
                        is_private=ch_data.get('is_private', False),
                        is_archived=ch_data.get('is_archived', False)
                    )
                    channels.append(channel)
                    
                return channels
                
        except Exception as e:
            logger.error(f"Failed to fetch channels via MCP: {e}")
            
        return []
        
    async def _fetch_history_via_mcp(self, channel: str, limit: int) -> List[SlackMessage]:
        """Fetch message history via MCP."""
        try:
            result = subprocess.run(
                ['node', self.cursor_mcp_path, 'get-history', '--channel', channel, '--limit', str(limit)],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                history_data = json.loads(result.stdout)
                messages = []
                
                for msg_data in history_data:
                    message = SlackMessage(
                        channel=channel,
                        text=msg_data['text'],
                        user=msg_data.get('user'),
                        timestamp=msg_data.get('ts'),
                        thread_ts=msg_data.get('thread_ts')
                    )
                    messages.append(message)
                    
                return messages
                
        except Exception as e:
            logger.error(f"Failed to fetch history via MCP: {e}")
            
        return []
        
    def _start_message_processing(self) -> None:
        """Start background message processing."""
        # Outbound message processor
        outbound_task = asyncio.create_task(self._process_outbound_messages())
        
        # Inbound message processor  
        inbound_task = asyncio.create_task(self._process_inbound_messages())
        
    async def _process_outbound_messages(self) -> None:
        """Process outbound message queue."""
        while self.connected:
            try:
                # Wait for messages to send
                message = await asyncio.wait_for(self.outbound_queue.get(), timeout=1.0)
                
                # Send via MCP
                await self._send_via_mcp(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Outbound message processing error: {e}")
                
    async def _process_inbound_messages(self) -> None:
        """Process inbound message queue."""
        while self.connected:
            try:
                # Wait for incoming messages
                message = await asyncio.wait_for(self.inbound_queue.get(), timeout=1.0)
                
                # Notify handlers
                for handler in self.message_handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
                        
                self.messages_received += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Inbound message processing error: {e}")
                
    async def _start_mcp_listener(self) -> None:
        """Start MCP message listener."""
        try:
            # This would establish a persistent connection to receive messages
            # For now, we'll simulate periodic polling
            
            while self.connected:
                # Poll for new messages every 5 seconds
                await asyncio.sleep(5)
                
                # Check for new messages in configured channels
                for channel_id in self.channels:
                    await self._poll_channel_messages(channel_id)
                    
        except Exception as e:
            logger.error(f"MCP listener error: {e}")
            
    async def _poll_channel_messages(self, channel: str) -> None:
        """Poll for new messages in a channel."""
        try:
            # Get recent messages
            messages = await self._fetch_history_via_mcp(channel, 10)
            
            # Add new messages to inbound queue
            for message in messages:
                await self.inbound_queue.put(message)
                
        except Exception as e:
            logger.error(f"Failed to poll channel {channel}: {e}")


# Factory function for easy initialization
def create_slack_integration(config: Dict[str, Any]) -> SlackMCPIntegration:
    """Create and configure Slack MCP integration."""
    return SlackMCPIntegration(config)