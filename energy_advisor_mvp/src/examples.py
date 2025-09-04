"""
Example usage of the Smart Plug Agent System
This file demonstrates various ways to use the system
"""

import asyncio
from datetime import datetime
from main import EnhancedSmartPlugAgent

async def example_basic_usage():
    """Basic usage examples"""
    print("📝 Example 1: Basic Usage")
    print("-" * 30)
    
    # Initialize the agent
    agent = EnhancedSmartPlugAgent()
    
    # Process some basic commands
    commands = [
        "Turn on the lights",
        "Set dishwasher at 14:00", 
        "Turn off heater in 30 minutes"
    ]
    
    for cmd in commands:
        print(f"Command: '{cmd}'")
        result = await agent.process_natural_language_command(cmd)
        
        if result['success']:
            action = result['action']
            device = result['device']
            
            if action == 'scheduled':
                time_str = datetime.fromisoformat(result['scheduled_for']).strftime("%H:%M")
                print(f"  ✅ Scheduled {device} for {time_str}")
            else:
                print(f"  ✅ {device} turned {'on' if 'on' in cmd.lower() else 'off'} immediately")
        else:
            print(f"  ❌ Failed: {result['error']}")
        print()

async def example_with_llm():
    """Example using LLM processing"""
    print("📝 Example 2: With LLM Processing")
    print("-" * 30)
    
    agent = EnhancedSmartPlugAgent()
    
    # These commands would work better with real LLM
    advanced_commands = [
        "Can you start the coffee maker for my morning routine?",
        "I need the EV charged by tomorrow morning",
        "Make sure the heater is off before I go to bed"
    ]
    
    for cmd in advanced_commands:
        print(f"Command: '{cmd}'")
        # Use LLM processing (will use mock in demo)
        result = await agent.process_natural_language_command(cmd, use_llm=True)
        
        if result['success']:
            confidence = result.get('llm_confidence', 0)
            print(f"  ✅ Understood with {confidence:.1%} confidence")
            print(f"  📋 Action: {result['action']} on {result['device']}")
        else:
            print(f"  ❌ Could not understand: {result['error']}")
        print()

async def example_system_monitoring():
    """Example of system monitoring"""
    print("📝 Example 3: System Monitoring")
    print("-" * 30)
    
    agent = EnhancedSmartPlugAgent()
    
    # Get comprehensive status
    status = await agent.get_comprehensive_status()
    
    print("System Overview:")
    print(f"  📱 Total devices: {status['summary']['total_devices']}")
    print(f"  🔋 Active devices: {status['summary']['active_devices']}")
    print(f"  ⚡ Power consumption: {status['summary']['total_power_consumption']:.1f}W")
    print(f"  📅 Pending jobs: {status['summary']['pending_jobs']}")
    
    print("\nDevices by Location:")
    for location, devices in status['devices_by_location'].items():
        print(f"  🏠 {location.title()}:")
        for device in devices:
            status_icon = "🟢" if device['is_on'] else "🔴"
            print(f"    {status_icon} {device['friendly_name']} ({device['power_consumption']:.0f}W)")
    
    # Get smart recommendations
    recommendations = await agent.get_device_recommendations()
    if recommendations:
        print("\nSmart Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  💡 {i}. {rec}")

async def example_scheduling():
    """Example of advanced scheduling"""
    print("📝 Example 4: Advanced Scheduling")
    print("-" * 30)
    
    agent = EnhancedSmartPlugAgent()
    
    # Schedule multiple devices
    schedule_commands = [
        "Set coffee maker at 6:30 AM",  # Morning routine
        "Turn on dishwasher at 2 PM",   # Afternoon cleaning
        "Start EV charging at 11 PM",   # Night charging
        "Turn off all lights at midnight"  # Night routine
    ]
    
    scheduled_jobs = []
    
    for cmd in schedule_commands:
        print(f"Scheduling: '{cmd}'")
        result = await agent.process_natural_language_command(cmd)
        
        if result['success'] and result['action'] == 'scheduled':
            job_id = result['job_id']
            scheduled_time = datetime.fromisoformat(result['scheduled_for'])
            scheduled_jobs.append(job_id)
            
            print(f"  ✅ Job {job_id} scheduled for {scheduled_time.strftime('%H:%M')}")
        else:
            print(f"  ❌ Scheduling failed")
    
    # Show scheduled jobs
    status = await agent.get_comprehensive_status()
    if status['scheduled_jobs']:
        print(f"\nTotal scheduled jobs: {len(status['scheduled_jobs'])}")
        for job in status['scheduled_jobs']:
            time_str = datetime.fromisoformat(job['scheduled_for']).strftime("%H:%M")
            print(f"  📅 {job['device']} - {job['action']} at {time_str} ({job['status']})")

async def example_error_handling():
    """Example of error handling"""
    print("📝 Example 5: Error Handling")
    print("-" * 30)
    
    agent = EnhancedSmartPlugAgent()
    
    # Commands that should fail gracefully
    error_commands = [
        "Turn on the refrigerator",  # Unknown device
        "Set dishwasher at 25:00",   # Invalid time
        "",                          # Empty command
        "Do something random",       # No clear intent
    ]
    
    for cmd in error_commands:
        print(f"Testing: '{cmd}'")
        result = await agent.process_natural_language_command(cmd)
        
        if result['success']:
            print(f"  ⚠️ Unexpectedly succeeded: {result['action']}")
        else:
            print(f"  ✅ Handled gracefully: {result['error']}")
        print()

async def example_emergency_features():
    """Example of emergency features"""
    print("📝 Example 6: Emergency Features")
    print("-" * 30)
    
    agent = EnhancedSmartPlugAgent()
    
    # First, turn on some devices
    print("Setting up scenario with active devices...")
    await agent.process_natural_language_command("Turn on lights")
    await agent.process_natural_language_command("Turn on heater")
    
    # Check status before emergency
    status = await agent.get_comprehensive_status()
    print(f"Active devices before emergency: {status['summary']['active_devices']}")
    
    # Emergency shutdown
    print("\nExecuting emergency shutdown...")
    result = await agent.emergency_shutdown()
    
    if result['success']:
        print("✅ Emergency shutdown completed")
        print(f"📊 Devices shut down: {len(result['shutdown_results'])}")
        print(f"📅 Jobs cancelled: {result['cancelled_jobs']}")
        
        # Verify all devices are off
        status = await agent.get_comprehensive_status()
        print(f"Active devices after emergency: {status['summary']['active_devices']}")
    else:
        print(f"❌ Emergency shutdown failed: {result['error']}")

async def run_all_examples():
    """Run all examples"""
    print("🚀 Smart Plug Agent System - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_with_llm,
        example_system_monitoring,
        example_scheduling,
        example_error_handling,
        example_emergency_features
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            await example_func()
            if i < len(examples):
                print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            continue
    
    print("\n🎉 All examples completed!")
    print("\n💡 Next steps:")
    print("  • Run 'python main.py' for interactive mode")
    print("  • Run 'python demo.py' for comprehensive demo")
    print("  • Configure real devices in config.py")
    print("  • Add LLM API keys for advanced processing")

if __name__ == "__main__":
    asyncio.run(run_all_examples())
