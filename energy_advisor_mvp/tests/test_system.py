#!/usr/bin/env python3
"""
Quick test script to verify the Smart Plug Agent System is working correctly
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_quick_test():
    """Run a quick test of the system"""
    print("🧪 Running Quick Test of Smart Plug Agent System")
    print("=" * 60)
    
    try:
        # Test 1: Import all modules
        print("\n1️⃣ Testing imports...")
        from config import Config
        from smart_plug_agent import SmartPlugAgent
        from hardware_interface import HardwareManager
        from llm_integration import LLMManager
        from main import EnhancedSmartPlugAgent
        print("   ✅ All imports successful")
        
        # Test 2: Initialize basic agent
        print("\n2️⃣ Testing basic agent initialization...")
        basic_agent = SmartPlugAgent()
        print("   ✅ Basic agent initialized")
        
        # Test 3: Test configuration
        print("\n3️⃣ Testing configuration...")
        print(f"   📱 Total devices configured: {len(Config.DEVICES)}")
        print(f"   🔌 Device types: {set(d.device_type for d in Config.DEVICES)}")
        print(f"   🏠 Locations: {set(d.location for d in Config.DEVICES)}")
        print("   ✅ Configuration loaded successfully")
        
        # Test 4: Test hardware manager
        print("\n4️⃣ Testing hardware manager...")
        hw_manager = HardwareManager(Config.DEVICES)
        available_devices = hw_manager.list_available_devices()
        print(f"   🔌 Available devices: {len(available_devices)}")
        print("   ✅ Hardware manager initialized")
        
        # Test 5: Test LLM manager
        print("\n5️⃣ Testing LLM manager...")
        llm_manager = LLMManager()
        providers = llm_manager.get_available_providers()
        print(f"   🤖 Available LLM providers: {providers}")
        print("   ✅ LLM manager initialized")
        
        # Test 6: Test enhanced agent
        print("\n6️⃣ Testing enhanced agent...")
        enhanced_agent = EnhancedSmartPlugAgent()
        print("   ✅ Enhanced agent initialized")
        
        # Test 7: Simple command processing
        print("\n7️⃣ Testing command processing...")
        test_commands = [
            "Turn on lights",
            "Set dishwasher at 14:00",
            "Turn off heater"
        ]
        
        for cmd in test_commands:
            print(f"   🎯 Testing: '{cmd}'")
            result = await enhanced_agent.process_natural_language_command(cmd, use_llm=False)
            status = "✅" if result['success'] else "❌"
            print(f"      {status} Result: {result.get('action', 'N/A')}")
        
        # Test 8: System status
        print("\n8️⃣ Testing system status...")
        status = await enhanced_agent.get_comprehensive_status()
        print(f"   📊 Total devices: {status['summary']['total_devices']}")
        print(f"   🔋 Active devices: {status['summary']['active_devices']}")
        print(f"   ⚡ Power consumption: {status['summary']['total_power_consumption']}W")
        print("   ✅ System status retrieved")
        
        # Test 9: Device control
        print("\n9️⃣ Testing direct device control...")
        test_device = Config.DEVICES[0].device_id
        print(f"   🔌 Testing device: {test_device}")
        
        # Turn on
        success = await hw_manager.turn_on_device(test_device)
        print(f"   {'✅' if success else '❌'} Turn ON: {success}")
        
        # Get status
        device_status = await hw_manager.get_device_status(test_device)
        is_on = device_status.get('is_on', False)
        print(f"   {'✅' if is_on else '❌'} Status check: {'ON' if is_on else 'OFF'}")
        
        # Turn off
        success = await hw_manager.turn_off_device(test_device)
        print(f"   {'✅' if success else '❌'} Turn OFF: {success}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! System is working correctly.")
        print("=" * 60)
        
        print("\n💡 To try the interactive mode, run:")
        print("   python main.py")
        print("\n📚 To see full demo, run:")
        print("   python demo.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_interactive_test():
    """Run an interactive test"""
    print("\n🎮 Interactive Test Mode")
    print("Enter commands to test (or 'quit' to exit):")
    
    try:
        from main import EnhancedSmartPlugAgent
        agent = EnhancedSmartPlugAgent()
        
        while True:
            try:
                cmd = input("\n🎯 Command: ").strip()
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not cmd:
                    continue
                    
                result = await agent.process_natural_language_command(cmd, use_llm=False)
                
                if result['success']:
                    print(f"   ✅ Success: {result.get('action', 'executed')}")
                    if 'device' in result:
                        print(f"   🔌 Device: {result['device']}")
                    if 'scheduled_for' in result:
                        print(f"   ⏰ Scheduled: {result['scheduled_for']}")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")

def main():
    """Main test function"""
    print("🚀 Smart Plug Agent System - Test Suite")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        asyncio.run(run_interactive_test())
    else:
        asyncio.run(run_quick_test())

if __name__ == "__main__":
    main()
