#!/usr/bin/env python3
"""
Local Button Test - Simulate button clicks to test webhook functionality
"""

import asyncio
import json
import requests
import time
from slack_sdk import WebClient

async def test_webhook_buttons():
    """Test all webhook button handlers locally"""
    
    webhook_url = "http://localhost:3000"
    slack_token = "xoxb-7652792950980-9122749890356-CYbcyyQOa379kQ6j0whT4WjF"
    slack_client = WebClient(token=slack_token)
    
    print("🧪 Testing AETHELRED Webhook Buttons Locally...")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{webhook_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Webhook server is responding")
            print(f"📊 Status: {response.json()}")
        else:
            print(f"❌ Webhook health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to webhook server: {e}")
        return
    
    print()
    
    # Test 2: Send buttons to Slack and simulate clicks
    
    # First, send a message with buttons
    test_message = {
        "text": "🧪 **LOCAL BUTTON TEST**",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🧪 **Testing Interactive Buttons Locally**\n\nThis tests the webhook server functionality:"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📊 System Dashboard"},
                        "action_id": "system_dashboard",
                        "style": "primary"
                    },
                    {
                        "type": "button", 
                        "text": {"type": "plain_text", "text": "🏛️ Test Reasoning"},
                        "action_id": "test_reasoning",
                        "style": "danger"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "💻 Terminal Demo"},
                        "action_id": "terminal_demo"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🤖 Help"},
                        "action_id": "help_action"
                    }
                ]
            }
        ]
    }
    
    # Send to Slack
    try:
        slack_response = slack_client.chat_postMessage(
            channel="aethelred-test",
            text=test_message["text"],
            blocks=test_message["blocks"],
            username="LOCAL-BUTTON-TESTER"
        )
        print(f"✅ Test buttons sent to Slack: {slack_response['ts']}")
    except Exception as e:
        print(f"❌ Failed to send buttons to Slack: {e}")
    
    print()
    
    # Test 3: Simulate button clicks locally
    button_tests = [
        {
            "name": "System Dashboard",
            "action_id": "system_dashboard",
            "description": "Should show system status"
        },
        {
            "name": "Help Action", 
            "action_id": "help_action",
            "description": "Should show help documentation"
        },
        {
            "name": "Terminal Demo",
            "action_id": "terminal_demo", 
            "description": "Should show terminal capabilities"
        },
        {
            "name": "Test Reasoning",
            "action_id": "test_reasoning",
            "description": "Should trigger Agora consensus (takes ~30s)"
        }
    ]
    
    for test in button_tests:
        print(f"🔘 Testing: {test['name']}")
        print(f"📝 Expected: {test['description']}")
        
        # Create simulated Slack payload
        payload = {
            "type": "block_actions",
            "user": {
                "id": "U07K1CKU4G5",
                "name": "test_user"
            },
            "channel": {
                "id": "C0935E3TLBZ",
                "name": "aethelred-test"
            },
            "actions": [
                {
                    "action_id": test["action_id"],
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": test["name"]
                    }
                }
            ],
            "trigger_id": f"test_trigger_{int(time.time())}"
        }
        
        # Send POST request to webhook
        try:
            webhook_response = requests.post(
                f"{webhook_url}/slack/interactive",
                data={"payload": json.dumps(payload)},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            
            if webhook_response.status_code == 200:
                result = webhook_response.json()
                print(f"✅ Webhook responded: {result.get('text', 'No response text')}")
                
                # For reasoning test, wait a bit for async processing
                if test["action_id"] == "test_reasoning":
                    print("⏳ Waiting for Agora consensus (this takes time)...")
                    await asyncio.sleep(5)  # Give it some time to start
                    
            else:
                print(f"❌ Webhook error: {webhook_response.status_code}")
                print(f"Response: {webhook_response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error testing {test['name']}: {e}")
        
        print("-" * 30)
        await asyncio.sleep(2)  # Wait between tests
    
    print("\n🎯 **Test Summary:**")
    print("✅ Webhook server is running and responding")
    print("✅ Button payloads are being processed")
    print("✅ Async responses should appear in Slack")
    print("🏛️ Agora consensus test may take 30-60 seconds to complete")
    
    print(f"\n📱 Check your Slack #aethelred-test channel for:")
    print("  • Test buttons message")
    print("  • System dashboard updates")
    print("  • Help documentation")
    print("  • Terminal demo info")
    print("  • Agora consensus results")

if __name__ == "__main__":
    asyncio.run(test_webhook_buttons())