"""
Demo and Testing Script for Smart Plug Agent System

This script demonstrates various use cases and tests the Smart Plug Agent System
with different types of natural language commands.
"""

import asyncio
import json
from smart_plug_agent import SmartPlugAgent
from datetime import datetime, timedelta

async def demo_basic_commands():
    """Demo basic smart plug commands"""
    print("=" * 60)
    print("🏠 SMART PLUG AGENT SYSTEM DEMO")
    print("=" * 60)
    
    agent = SmartPlugAgent()
    
    # Test commands from LLM
    test_commands = [
        "Set dishwasher at 14:00",
        "Turn on dryer at 16:00", 
        "Set heater at 17:00",
        "Set EV charge at 22:00",
        "Turn off heater in 30 minutes",
        "Turn on coffee maker at 7:00 AM",
        "Set lights at 8:00 PM",
        "Turn off TV in 2 hours",
        "Activate air conditioner at 15:30",
        "Start washing machine in 5 minutes"
    ]
    
    print("\n🎯 Testing Natural Language Commands:")
    print("-" * 40)
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n[{i}] Processing: '{command}'")
        result = await agent.process_llm_command(command)
        
        # Pretty print the result
        if result['success']:
            if result['action'] == 'scheduled':
                print(f"   ✅ Scheduled for {result['scheduled_for']}")
                print(f"   📋 Job ID: {result['job_id']}")
            else:
                print(f"   ✅ Executed immediately")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay between commands
        await asyncio.sleep(0.5)
    
    return agent

async def demo_system_status(agent):
    """Demo system status reporting"""
    print("\n" + "=" * 60)
    print("📊 SYSTEM STATUS REPORT")
    print("=" * 60)
    
    status = agent.get_system_status()
    
    print(f"\n📱 Total Devices: {status['total_devices']}")
    print(f"🔋 Active Devices: {status['active_devices']}")
    print(f"⏰ Scheduled Jobs: {len(status['scheduled_jobs'])}")
    print(f"🕒 Report Time: {status['timestamp']}")
    
    print("\n🏠 Device Status:")
    print("-" * 30)
    for device in status['device_states']:
        status_icon = "🟢" if device['is_on'] else "🔴"
        print(f"  {status_icon} {device['device_id']} ({device['location']}) - {'ON' if device['is_on'] else 'OFF'}")
    
    if status['scheduled_jobs']:
        print("\n📅 Scheduled Jobs:")
        print("-" * 30)
        for job in status['scheduled_jobs']:
            status_icon = {"pending": "⏳", "executing": "🔄", "completed": "✅", "failed": "❌", "cancelled": "🚫"}.get(job['status'], "❓")
            scheduled_time = datetime.fromisoformat(job['scheduled_for']).strftime("%H:%M")
            print(f"  {status_icon} {job['device']} - {job['action']} at {scheduled_time} ({job['status']})")

async def demo_immediate_actions(agent):
    """Demo immediate device control"""
    print("\n" + "=" * 60)
    print("⚡ IMMEDIATE DEVICE CONTROL DEMO")
    print("=" * 60)
    
    immediate_commands = [
        "Turn on lights",
        "Turn off heater", 
        "Activate coffee maker",
        "Turn off dryer"
    ]
    
    for command in immediate_commands:
        print(f"\n🎯 Executing: '{command}'")
        result = await agent.process_llm_command(command)
        
        if result['success']:
            print(f"   ✅ {result['device']} action completed")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        await asyncio.sleep(1)

async def demo_edge_cases(agent):
    """Demo edge cases and error handling"""
    print("\n" + "=" * 60)
    print("🧪 EDGE CASES AND ERROR HANDLING")
    print("=" * 60)
    
    edge_cases = [
        "Turn on unknown device",  # Unknown device
        "Set something at yesterday",  # Invalid time
        "Do something with lights",  # Ambiguous action
        "",  # Empty command
        "Just some random text",  # No device or action
        "Turn on dishwasher at 25:00",  # Invalid time format
    ]
    
    for command in edge_cases:
        print(f"\n🧪 Testing: '{command}'")
        result = await agent.process_llm_command(command)
        
        if result['success']:
            print(f"   ✅ Handled successfully")
        else:
            print(f"   ❌ Expected failure: {result.get('error', 'Unknown error')}")

async def demo_time_formats(agent):
    """Demo different time format handling"""
    print("\n" + "=" * 60)
    print("🕐 TIME FORMAT HANDLING DEMO")
    print("=" * 60)
    
    time_commands = [
        "Turn on lights at 9:30",  # 24-hour format
        "Set coffee maker at 7:00 AM",  # 12-hour format
        "Turn on heater in 15 minutes",  # Relative time
        "Set dishwasher in 2 hours",  # Relative time
        "Turn on TV at 20:00",  # Evening time
    ]
    
    for command in time_commands:
        print(f"\n🕐 Testing: '{command}'")
        result = await agent.process_llm_command(command)
        
        if result['success'] and result['action'] == 'scheduled':
            scheduled_time = datetime.fromisoformat(result['scheduled_for'])
            time_str = scheduled_time.strftime("%Y-%m-%d %H:%M")
            print(f"   ✅ Scheduled for: {time_str}")
        elif result['success']:
            print(f"   ✅ Executed immediately")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

async def demo_device_management(agent):
    """Demo direct device management"""
    print("\n" + "=" * 60)
    print("🔧 DIRECT DEVICE MANAGEMENT")
    print("=" * 60)
    
    print("\n🔍 Available Devices:")
    for friendly_name, device_id in agent.parser.device_mappings.items():
        print(f"  📱 {friendly_name} → {device_id}")
    
    print("\n🎮 Direct Control Examples:")
    
    # Turn on some devices directly
    devices_to_test = ['lights', 'coffee maker', 'heater']
    
    for device in devices_to_test:
        print(f"\n🔌 Turning ON {device}...")
        success = await agent.turn_on_device_directly(device)
        print(f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}")
        
        await asyncio.sleep(1)
        
        print(f"🔌 Turning OFF {device}...")
        success = await agent.turn_off_device_directly(device)
        print(f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}")

async def interactive_demo():
    """Interactive demo mode"""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE MODE")
    print("=" * 60)
    print("Enter natural language commands to control smart plugs.")
    print("Type 'status' to see system status, 'help' for examples, or 'quit' to exit.")
    
    agent = SmartPlugAgent()
    
    example_commands = [
        "Turn on lights",
        "Set dishwasher at 14:00",
        "Turn off heater in 30 minutes",
        "Activate coffee maker at 7:00 AM"
    ]
    
    while True:
        try:
            user_input = input("\n🎯 Enter command: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'status':
                status = agent.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Active devices: {status['active_devices']}/{status['total_devices']}")
                print(f"   Scheduled jobs: {len(status['scheduled_jobs'])}")
                continue
            elif user_input.lower() == 'help':
                print("\n💡 Example commands:")
                for cmd in example_commands:
                    print(f"   • {cmd}")
                continue
            elif not user_input:
                continue
            
            result = await agent.process_llm_command(user_input)
            
            if result['success']:
                if result['action'] == 'scheduled':
                    scheduled_time = datetime.fromisoformat(result['scheduled_for']).strftime("%H:%M")
                    print(f"   ✅ Scheduled {result['device']} for {scheduled_time}")
                else:
                    print(f"   ✅ {result['device']} action completed")
            else:
                print(f"   ❌ {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Main demo function"""
    print("🚀 Starting Smart Plug Agent System Demo...")
    
    # Run comprehensive demo
    agent = await demo_basic_commands()
    await demo_system_status(agent)
    await demo_immediate_actions(agent)
    await demo_time_formats(agent)
    await demo_edge_cases(agent)
    await demo_device_management(agent)
    
    # Final system status
    await demo_system_status(agent)
    
    print("\n" + "=" * 60)
    print("✨ DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Optionally run interactive mode
    run_interactive = input("\n🎮 Would you like to try interactive mode? (y/n): ").lower()
    if run_interactive == 'y':
        await interactive_demo()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
