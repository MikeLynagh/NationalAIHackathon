#!/usr/bin/env python3
"""
Test Smart Plug Integration
"""

import asyncio
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from smart_plug_agent import SmartPlugAgent

async def test_smart_plug_integration():
    """Test the smart plug integration"""
    print("🧪 Testing Smart Plug Integration")
    print("=" * 50)
    
    # Initialize the agent
    agent = SmartPlugAgent()
    
    # Test commands
    test_commands = [
        "turn on dishwasher",
        "set dishwasher at 14:30", 
        "turn off dishwasher"
    ]
    
    for command in test_commands:
        print(f"\n🎯 Testing command: '{command}'")
        try:
            result = await agent.process_llm_command(command)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test system status
    print(f"\n📊 System Status:")
    status = agent.get_system_status()
    print(f"Connected devices: {len(status['connected_devices'])}")
    print(f"Active devices: {status['active_devices']}")
    
    # Test direct device control
    print(f"\n🔌 Testing direct dishwasher control:")
    try:
        success = await agent.turn_on_device_directly("dishwasher")
        print(f"Direct turn on result: {success}")
        
        success = await agent.turn_off_device_directly("dishwasher") 
        print(f"Direct turn off result: {success}")
    except Exception as e:
        print(f"❌ Direct control error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Smart Plug Integration Test")
    asyncio.run(test_smart_plug_integration())
    print("\n✅ Test completed!")
