#!/usr/bin/env python
"""Simple test for the butler agent that doesn't require external agents"""

import asyncio
import json
import httpx

async def test_butler_capabilities():
    """Test butler's capability reporting without calling external agents"""
    
    butler_url = "http://localhost:10000"
    
    # Simple capability query that shouldn't trigger external agent calls
    import uuid
    query = {
        "jsonrpc": "2.0",
        "id": "test-cap-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text", 
                    "text": "Butler, please describe your orchestration capabilities without calling other agents."
                }],
                "messageId": uuid.uuid4().hex
            }
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Testing butler agent capabilities...")
        
        # First check if butler is running
        try:
            card_response = await client.get(f"{butler_url}/.well-known/agent.json")
            if card_response.status_code == 200:
                print("✓ Butler agent is running")
                card = card_response.json()
                print(f"✓ Agent: {card['name']} v{card['version']}")
                print(f"✓ Skills: {len(card['skills'])} available")
            else:
                print("✗ Butler agent not responding")
                return
        except Exception as e:
            print(f"✗ Cannot connect to butler: {e}")
            return
        
        # Send the capability query
        try:
            print("\nSending capability query...")
            response = await client.post(butler_url, json=query)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Received response")
                
                # Extract the message
                if 'result' in result and 'message' in result.get('result', {}):
                    message = result['result']['message']
                    if 'parts' in message:
                        for part in message['parts']:
                            if part.get('kind') == 'text':
                                print("\nButler response:")
                                print("-" * 50)
                                print(part.get('text', ''))
                                print("-" * 50)
                else:
                    print(f"Response structure: {json.dumps(result, indent=2)}")
            else:
                print(f"✗ Error response: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"✗ Error sending query: {e}")

if __name__ == "__main__":
    asyncio.run(test_butler_capabilities())