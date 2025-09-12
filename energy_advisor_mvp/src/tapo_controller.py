#!/usr/bin/env python3
"""
Working Tapo P100 Smart Plug Controller
Uses proper Kasa library authentication methods
"""

import asyncio
from kasa import Discover, Credentials
import sys
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def control_tapo_device(action, ip="172.20.10.4", username=None, password=None):
    """
    Control Tapo P100 using proper authentication
    """
    # Get credentials from environment if not provided
    if not username:
        username = os.getenv('TAPO_USERNAME')
    if not password:
        password = os.getenv('TAPO_PASSWORD')
    
    device = None
    try:
        print(f"🔍 Discovering Tapo P100 at {ip}...")
        
        # Create credentials object
        if username and password:
            credentials = Credentials(username=username, password=password)
            print(f"🔐 Using credentials for user: {username}")
        else:
            credentials = None
            print("⚠️  No credentials provided")
        
        # Discover the device
        device = await Discover.discover_single(ip, credentials=credentials)
        
        if not device:
            print(f"❌ No device found at {ip}")
            return False
        
        print(f"✅ Found device: {device.model}")
        
        # Connect to the device
        try:
            await device.update()
            print(f"🔗 Connected to device: {device.alias or 'Unnamed Device'}")
        except Exception as update_error:
            print(f"⚠️  Connection issue: {update_error}")
            print("Attempting to continue...")
        
        # Perform the action
        if action == "status":
            await show_device_status(device)
        elif action == "on":
            await turn_on_device(device)
        elif action == "off":
            await turn_off_device(device)
        else:
            print(f"❌ Unknown action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if device:
            try:
                await device.disconnect()
            except:
                pass

async def show_device_status(device):
    """Display comprehensive device status"""
    print("\n" + "="*55)
    print("🔌 TAPO P100 SMART PLUG STATUS")
    print("="*55)
    
    try:
        # Basic information
        print(f"Device Name:     {getattr(device, 'alias', 'Unknown')}")
        print(f"Model:           {getattr(device, 'model', 'Unknown')}")
        print(f"Device Type:     {getattr(device, 'device_type', 'Unknown')}")
        print(f"IP Address:      {getattr(device, 'host', 'Unknown')}")
        print(f"MAC Address:     {getattr(device, 'mac', 'Unknown')}")
        
        # Power status
        try:
            is_on = device.is_on
            status_icon = "🟢 ON" if is_on else "🔴 OFF"
            print(f"Power Status:    {status_icon}")
        except Exception as status_error:
            print(f"Power Status:    ❓ Cannot determine ({status_error})")
        
        # Additional features
        features = []
        if hasattr(device, 'turn_on'):
            features.append("✅ Power Control")
        if hasattr(device, 'emeter'):
            features.append("✅ Energy Monitoring")
        if hasattr(device, 'has_emeter') and device.has_emeter:
            features.append("✅ Energy Meter")
        
        if features:
            print(f"Features:        {' | '.join(features)}")
        
        # Device information
        try:
            hw_info = getattr(device, 'hw_info', {})
            if hw_info:
                print(f"Hardware Info:   {hw_info}")
        except:
            pass
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    print("="*55)

async def turn_on_device(device):
    """Turn on the smart plug"""
    try:
        print("🔄 Turning device ON...")
        await device.turn_on()
        
        # Wait a moment and verify
        await asyncio.sleep(1)
        await device.update()
        
        if device.is_on:
            print("✅ Device successfully turned ON! 🟢")
        else:
            print("⚠️  Device may not have turned on. Check manually.")
            
    except Exception as e:
        print(f"❌ Failed to turn device ON: {e}")
        print("💡 This might be due to authentication or network issues.")

async def turn_off_device(device):
    """Turn off the smart plug"""
    try:
        print("🔄 Turning device OFF...")
        await device.turn_off()
        
        # Wait a moment and verify
        await asyncio.sleep(1)
        await device.update()
        
        if not device.is_on:
            print("✅ Device successfully turned OFF! 🔴")
        else:
            print("⚠️  Device may not have turned off. Check manually.")
            
    except Exception as e:
        print(f"❌ Failed to turn device OFF: {e}")
        print("💡 This might be due to authentication or network issues.")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Control your Tapo P100 Smart Plug',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python working_tapo.py status          # Check device status
  python working_tapo.py on              # Turn on the plug
  python working_tapo.py off             # Turn off the plug
  
  # Use different credentials:
  python working_tapo.py on -u your@email.com -p yourpassword
        """
    )
    
    parser.add_argument('action', choices=['on', 'off', 'status'], 
                       help='Action to perform on the smart plug')
    parser.add_argument('--ip', default='172.20.10.4', 
                       help='IP address of the smart plug (default: 172.20.10.4)')
    parser.add_argument('--username', '-u', 
                       help='Tapo account username/email (uses env var if not provided)')
    parser.add_argument('--password', '-p', 
                       help='Tapo account password (uses env var if not provided)')
    
    # Handle no arguments case
    if len(sys.argv) == 1:
        print("🔌 Tapo P100 Smart Plug Controller")
        print("="*45)
        print("\nQuick test - checking device status...")
        print("Use 'python working_tapo.py --help' for full options\n")
        
        # Run a quick status check
        asyncio.run(control_tapo_device('status'))
        return
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.action.upper()} operation...")
    
    # Execute the command
    success = asyncio.run(control_tapo_device(
        action=args.action,
        ip=args.ip,
        username=args.username,
        password=args.password
    ))
    
    # Final status
    if success:
        print(f"\n🎉 {args.action.upper()} operation completed successfully!")
    else:
        print(f"\n💥 {args.action.upper()} operation failed!")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure the device is powered on and connected")
        print("2. Verify you're on the same network as the device")
        print("3. Check that your Tapo credentials are correct")
        print("4. Try resetting the device if issues persist")

if __name__ == "__main__":
    main()
