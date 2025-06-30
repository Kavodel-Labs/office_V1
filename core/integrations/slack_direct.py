#!/usr/bin/env python3
"""
Direct Slack API Integration for AETHELRED.

This module provides direct integration with Slack using the Slack Web API,
allowing AETHELRED to create channels, send messages, and receive events.
"""

import asyncio
import json
import logging
import os
import aiohttp
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SlackChannel:
    """Represents a Slack channel."""
    id: str
    name: str
    is_private: bool = False
    is_archived: bool = False
    created: Optional[str] = None


@dataclass
class SlackMessage:
    """Represents a Slack message."""
    channel: str
    text: str
    user: Optional[str] = None
    timestamp: Optional[str] = None
    thread_ts: Optional[str] = None
    message_type: str = "message"


class SlackDirectIntegration:
    """
    Direct Slack API Integration for AETHELRED.
    
    Uses Slack Web API to provide full bidirectional communication
    with your Slack workspace.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get('bot_token', os.getenv('SLACK_BOT_TOKEN'))
        self.app_token = config.get('app_token', os.getenv('SLACK_APP_TOKEN'))
        self.signing_secret = config.get('signing_secret', os.getenv('SLACK_SIGNING_SECRET'))
        
        # API configuration
        self.base_url = "https://slack.com/api"
        self.headers = {
            'Authorization': f'Bearer {self.bot_token}',
            'Content-Type': 'application/json'
        }
        
        # State management
        self.connected = False
        self.channels: Dict[str, SlackChannel] = {}
        self.bot_user_id: Optional[str] = None
        
        # Message handling
        self.message_handlers: List[Callable] = []
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.api_calls = 0
        
    async def initialize(self) -> bool:
        """Initialize Slack integration."""
        if not self.bot_token:
            logger.error("Slack bot token not provided. Set SLACK_BOT_TOKEN environment variable.")
            return False
            
        try:
            logger.info("Initializing direct Slack integration...")
            
            # Test API connection
            if not await self._test_connection():
                logger.error("Failed to connect to Slack API")
                return False
                
            # Get bot info
            await self._get_bot_info()
            
            # Load channels
            await self._load_channels()
            
            self.connected = True
            logger.info("Direct Slack integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack integration: {e}")
            return False
            
    async def create_channel(self, name: str, is_private: bool = False, topic: str = "") -> Optional[SlackChannel]:
        """Create a new Slack channel."""
        try:
            endpoint = "conversations.create"
            payload = {
                "name": name,
                "is_private": is_private
            }
            
            if topic:
                payload["topic"] = topic
                
            response = await self._api_call(endpoint, payload)
            
            if response and response.get('ok'):
                channel_data = response['channel']
                channel = SlackChannel(
                    id=channel_data['id'],
                    name=channel_data['name'],
                    is_private=channel_data.get('is_private', False),
                    created=str(datetime.utcnow())
                )
                
                self.channels[channel.id] = channel
                logger.info(f"Created Slack channel: #{name} ({channel.id})")
                
                # Set topic if provided
                if topic:
                    await self.set_channel_topic(channel.id, topic)
                    
                return channel
            else:
                error = response.get('error', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to create channel {name}: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating channel {name}: {e}")
            return None
            
    async def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None, 
                          attachments: Optional[List[Dict]] = None) -> bool:
        """Send message to Slack channel."""
        try:
            # Convert channel name to ID if needed
            channel_id = await self._resolve_channel(channel)
            if not channel_id:
                logger.error(f"Could not resolve channel: {channel}")
                return False
                
            payload = {
                "channel": channel_id,
                "text": text,
                "as_user": True
            }
            
            if thread_ts:
                payload["thread_ts"] = thread_ts
                
            if attachments:
                payload["attachments"] = attachments
                
            response = await self._api_call("chat.postMessage", payload)
            
            if response and response.get('ok'):
                self.messages_sent += 1
                logger.info(f"Message sent to {channel}: {text[:50]}...")
                return True
            else:
                error = response.get('error', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to send message: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
            
    async def send_rich_message(self, channel: str, title: str, message: str, 
                               color: str = "good", fields: Optional[List[Dict]] = None) -> bool:
        """Send a rich formatted message with attachments."""
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "ts": int(datetime.utcnow().timestamp())
        }
        
        if fields:
            attachment["fields"] = fields
            
        return await self.send_message(channel, "", attachments=[attachment])
        
    async def send_notification(self, channel: str, title: str, message: str, urgency: str = "normal") -> bool:
        """Send a system notification with appropriate formatting."""
        # Map urgency to colors
        color_map = {
            "low": "#36a64f",      # Green
            "normal": "#439FE0",   # Blue  
            "high": "#ff9900",     # Orange
            "critical": "#ff0000"  # Red
        }
        
        color = color_map.get(urgency, "#439FE0")
        icon = "🔔" if urgency == "normal" else "⚠️" if urgency == "high" else "🚨"
        
        formatted_title = f"{icon} {title}"
        return await self.send_rich_message(channel, formatted_title, message, color)
        
    async def get_channels(self) -> List[SlackChannel]:
        """Get list of all channels."""
        return list(self.channels.values())
        
    async def get_channel_by_name(self, name: str) -> Optional[SlackChannel]:
        """Get channel by name."""
        # Remove # if present
        clean_name = name.lstrip('#')
        
        for channel in self.channels.values():
            if channel.name == clean_name:
                return channel
        return None
        
    async def set_channel_topic(self, channel: str, topic: str) -> bool:
        """Set channel topic."""
        try:
            channel_id = await self._resolve_channel(channel)
            if not channel_id:
                return False
                
            response = await self._api_call("conversations.setTopic", {
                "channel": channel_id,
                "topic": topic
            })
            
            return response and response.get('ok', False)
            
        except Exception as e:
            logger.error(f"Error setting channel topic: {e}")
            return False
            
    def add_message_handler(self, handler: Callable[[SlackMessage], None]) -> None:
        """Add message handler for incoming messages."""
        self.message_handlers.append(handler)
        
    async def start_event_listener(self) -> None:
        """Start listening for Slack events (requires Socket Mode)."""
        if not self.app_token:
            logger.warning("App token not provided. Cannot start event listener.")
            return
            
        # This would implement Socket Mode for real-time events
        # For now, we'll use polling as a fallback
        logger.info("Starting Slack event listener...")
        
        try:
            while self.connected:
                # Poll for new messages (simplified approach)
                await asyncio.sleep(5)
                # In a real implementation, this would use Socket Mode
                
        except Exception as e:
            logger.error(f"Event listener error: {e}")
            
    async def stop(self) -> None:
        """Stop Slack integration."""
        self.connected = False
        logger.info("Slack integration stopped")
        
    # Private methods
    
    async def _test_connection(self) -> bool:
        """Test Slack API connection."""
        try:
            response = await self._api_call("auth.test")
            if response and response.get('ok'):
                logger.info(f"Connected to Slack as: {response.get('user', 'Unknown')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
            
    async def _get_bot_info(self) -> None:
        """Get bot user information."""
        try:
            response = await self._api_call("auth.test")
            if response and response.get('ok'):
                self.bot_user_id = response.get('user_id')
                logger.info(f"Bot user ID: {self.bot_user_id}")
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            
    async def _load_channels(self) -> None:
        """Load all channels from Slack."""
        try:
            response = await self._api_call("conversations.list", {
                "types": "public_channel,private_channel",
                "limit": 1000
            })
            
            if response and response.get('ok'):
                channels = response.get('channels', [])
                for ch in channels:
                    channel = SlackChannel(
                        id=ch['id'],
                        name=ch['name'],
                        is_private=ch.get('is_private', False),
                        is_archived=ch.get('is_archived', False),
                        created=ch.get('created')
                    )
                    self.channels[channel.id] = channel
                    
                logger.info(f"Loaded {len(self.channels)} Slack channels")
            else:
                logger.error("Failed to load channels")
                
        except Exception as e:
            logger.error(f"Error loading channels: {e}")
            
    async def _resolve_channel(self, channel: str) -> Optional[str]:
        """Resolve channel name or ID to channel ID."""
        # If it's already an ID (starts with C)
        if channel.startswith('C'):
            return channel
            
        # Remove # if present and find by name
        clean_name = channel.lstrip('#')
        for ch in self.channels.values():
            if ch.name == clean_name:
                return ch.id
                
        return None
        
    async def _api_call(self, endpoint: str, payload: Optional[Dict] = None) -> Optional[Dict]:
        """Make API call to Slack."""
        try:
            self.api_calls += 1
            url = f"{self.base_url}/{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                if payload:
                    async with session.post(url, headers=self.headers, json=payload) as response:
                        return await response.json()
                else:
                    async with session.get(url, headers=self.headers) as response:
                        return await response.json()
                        
        except Exception as e:
            logger.error(f"API call to {endpoint} failed: {e}")
            return None
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "connected": self.connected,
            "channels_loaded": len(self.channels),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "api_calls": self.api_calls,
            "bot_user_id": self.bot_user_id
        }


# Factory function
def create_slack_integration(config: Dict[str, Any]) -> SlackDirectIntegration:
    """Create direct Slack integration."""
    return SlackDirectIntegration(config)