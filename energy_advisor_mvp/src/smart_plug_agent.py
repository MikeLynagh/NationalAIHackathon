"""
Smart Plug Control Agent System

This system creates an intelligent agent that can interpret natural language commands
from an LLM and translate them into smart plug control actions.

Architecture Overview:
1. Natural Language Parser - Extracts device, action, and time from LLM output
2. Device Manager - Maps device names to smart plug identifiers
3. Scheduler - Handles time-based automation
4. Smart Plug Controller - Interfaces with actual hardware
5. Action Executor - Coordinates the entire workflow

Example Commands:
- "Set dishwasher at 14:00"
- "Turn on dryer at 16:00" 
- "Set heater at 17:00"
- "Set EV charge at 22:00"
"""

import re
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import os
from threading import Thread
from dotenv import load_dotenv
from kasa import Discover, Credentials

# Import LLM integration
try:
    from llm_integration import LLMManager, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Load environment variables from both possible locations
load_dotenv()  # Current directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))  # src directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))  # parent directory

class ActionType(Enum):
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off" 
    SET_SCHEDULE = "set_schedule"
    CANCEL_SCHEDULE = "cancel_schedule"

@dataclass
class SmartPlugCommand:
    """Represents a parsed command for smart plug control"""
    device_name: str
    action: ActionType
    scheduled_time: Optional[datetime] = None
    duration: Optional[int] = None  # in minutes
    
class NaturalLanguageParser:
    """Parses natural language commands into structured data"""
    
    def __init__(self):
        # Device name mappings (expandable)
        self.device_mappings = {
            'dishwasher': 'kitchen_dishwasher',
            'dryer': 'laundry_dryer',
            'heater': 'living_room_heater',
            'ev charge': 'garage_ev_charger',
            'ev charger': 'garage_ev_charger',
            'washing machine': 'laundry_washer',
            'coffee maker': 'kitchen_coffee_maker',
            'air conditioner': 'bedroom_ac',
            'ac': 'bedroom_ac',
            'lights': 'living_room_lights',
            'fan': 'bedroom_fan',
            'tv': 'living_room_tv',
            'microwave': 'kitchen_microwave'
        }
        
        # Time parsing patterns
        self.time_patterns = [
            r'at (\d{1,2}):(\d{2})',  # at 14:00
            r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)',  # at 2:00 PM
            r'in (\d+) (minutes?|hours?)',  # in 30 minutes
            r'(tomorrow|today) at (\d{1,2}):(\d{2})',  # tomorrow at 14:00
        ]
        
        # Action patterns
        self.action_patterns = {
            ActionType.TURN_ON: [r'turn on', r'start', r'activate', r'switch on', r'power on'],
            ActionType.TURN_OFF: [r'turn off', r'stop', r'deactivate', r'switch off', r'power off'],
            ActionType.SET_SCHEDULE: [r'set', r'schedule', r'program']
        }
    
    def parse_command(self, llm_output: str) -> Optional[SmartPlugCommand]:
        """
        Parse natural language command into SmartPlugCommand
        
        Args:
            llm_output: Raw text from LLM like "Set dishwasher at 14:00"
            
        Returns:
            SmartPlugCommand object or None if parsing fails
        """
        try:
            llm_output = llm_output.lower().strip()
            print(f"Parsing command: '{llm_output}'")
            
            # Extract device name
            device_name = self._extract_device(llm_output)
            if not device_name:
                print(f"Warning: Could not identify device in command: '{llm_output}'")
                return None
                
            # Extract action
            action = self._extract_action(llm_output)
            if not action:
                action = ActionType.SET_SCHEDULE  # Default for scheduled commands
                
            # Extract time
            scheduled_time = self._extract_time(llm_output)
            
            command = SmartPlugCommand(
                device_name=device_name,
                action=action,
                scheduled_time=scheduled_time
            )
            
            print(f"Parsed command: Device={device_name}, Action={action.value}, Time={scheduled_time}")
            return command
            
        except Exception as e:
            print(f"Error parsing command '{llm_output}': {e}")
            return None
    
    def _extract_device(self, text: str) -> Optional[str]:
        """Extract device name from text"""
        for device_key, device_id in self.device_mappings.items():
            if device_key in text:
                return device_id
        return None
    
    def _extract_action(self, text: str) -> Optional[ActionType]:
        """Extract action type from text"""
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return action_type
        return None
    
    def _extract_time(self, text: str) -> Optional[datetime]:
        """Extract scheduled time from text"""
        # Pattern: at HH:MM (24-hour format)
        match = re.search(r'at (\d{1,2}):(\d{2})', text)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            
            # Validate time format
            if hour > 23 or minute > 59:
                print(f"Warning: Invalid time format {hour}:{minute:02d} - hour must be 0-23, minute must be 0-59")
                return None
                
            now = datetime.now()
            try:
                scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)
                    
                return scheduled_time
            except ValueError as e:
                print(f"Warning: Error creating datetime with hour={hour}, minute={minute}: {e}")
                return None
        
        # Pattern: at H:MM AM/PM
        match = re.search(r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)', text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3).lower()
            
            # Validate basic ranges
            if hour < 1 or hour > 12 or minute > 59:
                print(f"Warning: Invalid 12-hour time format {hour}:{minute:02d} {period.upper()}")
                return None
            
            # Convert to 24-hour format
            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0
                
            now = datetime.now()
            try:
                scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)
                    
                return scheduled_time
            except ValueError as e:
                print(f"Warning: Error creating datetime with hour={hour}, minute={minute}: {e}")
                return None
        
        # Pattern: in X minutes/hours
        match = re.search(r'in (\d+) (minutes?|hours?)', text)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            # Validate reasonable ranges
            if amount <= 0:
                print(f"Warning: Invalid time amount: {amount}")
                return None
            if 'minute' in unit and amount > 1440:  # More than 24 hours in minutes
                print(f"Warning: Time amount too large: {amount} minutes")
                return None
            if 'hour' in unit and amount > 168:  # More than a week in hours
                print(f"Warning: Time amount too large: {amount} hours")
                return None
            
            try:
                if 'minute' in unit:
                    return datetime.now() + timedelta(minutes=amount)
                elif 'hour' in unit:
                    return datetime.now() + timedelta(hours=amount)
            except OverflowError:
                print(f"Warning: Time calculation overflow with {amount} {unit}")
                return None
        
        return None

class SmartPlugController:
    """Interfaces with actual smart plug hardware (Tapo P100 and Kasa devices)"""
    
    def __init__(self):
        # Mock smart plug states
        self.plug_states = {}
        self.device_registry = {
            'kitchen_dishwasher': {'ip': '172.20.10.4', 'type': 'tapo', 'location': 'kitchen'},
            'laundry_dryer': {'ip': '192.168.1.101', 'type': 'kasa', 'location': 'laundry'},
            'living_room_heater': {'ip': '192.168.1.102', 'type': 'kasa', 'location': 'living_room'},
            'garage_ev_charger': {'ip': '192.168.1.103', 'type': 'kasa', 'location': 'garage'},
            'laundry_washer': {'ip': '192.168.1.104', 'type': 'kasa', 'location': 'laundry'},
            'kitchen_coffee_maker': {'ip': '192.168.1.105', 'type': 'kasa', 'location': 'kitchen'},
            'bedroom_ac': {'ip': '192.168.1.106', 'type': 'kasa', 'location': 'bedroom'},
            'living_room_lights': {'ip': '192.168.1.107', 'type': 'kasa', 'location': 'living_room'},
            'bedroom_fan': {'ip': '192.168.1.108', 'type': 'kasa', 'location': 'bedroom'},
            'living_room_tv': {'ip': '192.168.1.109', 'type': 'kasa', 'location': 'living_room'},
            'kitchen_microwave': {'ip': '192.168.1.110', 'type': 'kasa', 'location': 'kitchen'}
        }
        
        # Initialize all devices as off
        for device_id in self.device_registry.keys():
            self.plug_states[device_id] = False
        
        # Get credentials from environment
        self.tapo_username = os.getenv('TAPO_USERNAME')
        self.tapo_password = os.getenv('TAPO_PASSWORD')
    
    async def _control_tapo_device(self, ip: str, action: str) -> bool:
        """
        Control Tapo P100 using proper authentication
        """
        device = None
        try:
            print(f"🔍 Discovering Tapo P100 at {ip}...")
            
            # Create credentials object
            if self.tapo_username and self.tapo_password:
                credentials = Credentials(username=self.tapo_username, password=self.tapo_password)
                print(f"🔐 Using credentials for user: {self.tapo_username}")
            else:
                print("⚠️  No Tapo credentials found in environment")
                return False
            
            # Discover the device
            device = await Discover.discover_single(ip, credentials=credentials)
            
            if not device:
                print(f"❌ No Tapo device found at {ip}")
                return False
            
            print(f"✅ Found Tapo device: {device.model}")
            
            # Connect to the device
            try:
                await device.update()
                print(f"� Connected to Tapo device: {device.alias or 'Unnamed Device'}")
            except Exception as update_error:
                print(f"⚠️  Tapo connection issue: {update_error}")
                print("Attempting to continue...")
            
            # Perform the action
            if action == "on":
                await device.turn_on()
                await asyncio.sleep(1)  # Wait for state change
                await device.update()
                return device.is_on
            elif action == "off":
                await device.turn_off()
                await asyncio.sleep(1)  # Wait for state change
                await device.update()
                return not device.is_on
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error controlling Tapo device at {ip}: {e}")
            return False
        finally:
            if device:
                try:
                    await device.disconnect()
                except:
                    pass
    
    async def turn_on_device(self, device_id: str) -> bool:
        """Turn on a smart plug device"""
        try:
            if device_id not in self.device_registry:
                print(f"Error: Device {device_id} not found in registry")
                return False
                
            device_info = self.device_registry[device_id]
            device_type = device_info.get('type', 'kasa')
            ip = device_info['ip']
            
            print(f"🔌 Turning ON {device_id} ({device_type}) at {ip}")
            
            if device_type == 'tapo':
                # Use Tapo P100 control
                success = await self._control_tapo_device(ip, "on")
            else:
                # For other devices, simulate the action (can be extended for actual Kasa devices)
                print(f"🔄 Simulating turn ON for {device_type} device...")
                await asyncio.sleep(0.5)  # Simulate network delay
                success = True
            
            if success:
                self.plug_states[device_id] = True
                print(f"✅ Successfully turned ON {device_id}")
            else:
                print(f"❌ Failed to turn ON {device_id}")
                
            return success
            
        except Exception as e:
            print(f"❌ Error turning on {device_id}: {e}")
            return False
    
    async def turn_off_device(self, device_id: str) -> bool:
        """Turn off a smart plug device"""
        try:
            if device_id not in self.device_registry:
                print(f"Error: Device {device_id} not found in registry")
                return False
                
            device_info = self.device_registry[device_id]
            device_type = device_info.get('type', 'kasa')
            ip = device_info['ip']
            
            print(f"🔌 Turning OFF {device_id} ({device_type}) at {ip}")
            
            if device_type == 'tapo':
                # Use Tapo P100 control
                success = await self._control_tapo_device(ip, "off")
            else:
                # For other devices, simulate the action (can be extended for actual Kasa devices)
                print(f"🔄 Simulating turn OFF for {device_type} device...")
                await asyncio.sleep(0.5)  # Simulate network delay
                success = True
            
            if success:
                self.plug_states[device_id] = False
                print(f"✅ Successfully turned OFF {device_id}")
            else:
                print(f"❌ Failed to turn OFF {device_id}")
                
            return success
            
        except Exception as e:
            print(f"❌ Error turning off {device_id}: {e}")
            return False
    
    def get_device_status(self, device_id: str) -> Dict:
        """Get current status of a device"""
        if device_id not in self.device_registry:
            return {'error': f'Device {device_id} not found'}
            
        device_info = self.device_registry[device_id]
        return {
            'device_id': device_id,
            'is_on': self.plug_states.get(device_id, False),
            'location': device_info.get('location', 'unknown'),
            'type': device_info.get('type', 'unknown'),
            'ip': device_info.get('ip', 'unknown'),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_all_devices_status(self) -> List[Dict]:
        """Get status of all devices"""
        return [self.get_device_status(device_id) for device_id in self.device_registry.keys()]

class SmartPlugScheduler:
    """Handles scheduling of smart plug actions"""
    
    def __init__(self, controller: SmartPlugController):
        self.controller = controller
        self.scheduled_jobs = {}
        self.running = False
        
    def schedule_action(self, command: SmartPlugCommand) -> str:
        """Schedule a smart plug action"""
        job_id = f"{command.device_name}_{int(time.time())}"
        
        if command.scheduled_time:
            # Calculate delay until scheduled time
            delay = (command.scheduled_time - datetime.now()).total_seconds()
            
            if delay > 0:
                print(f"⏰ Scheduling {command.action.value} for {command.device_name} in {delay:.0f} seconds")
                
                # Schedule the job
                def job_function():
                    asyncio.run(self._execute_scheduled_action(command, job_id))
                
                # Using threading timer for simplicity
                timer = Thread(target=self._delayed_execution, 
                             args=(delay, job_function))
                timer.daemon = True
                timer.start()
                
                self.scheduled_jobs[job_id] = {
                    'command': command,
                    'scheduled_for': command.scheduled_time,
                    'timer': timer,
                    'status': 'pending'
                }
                
                print(f"📅 Scheduled {command.action.value} for {command.device_name} at {command.scheduled_time}")
                return job_id
            else:
                print(f"⚠️ Cannot schedule action in the past. Time: {command.scheduled_time}")
        
        return None
    
    def _delayed_execution(self, delay: float, job_function):
        """Execute job after delay"""
        time.sleep(delay)
        job_function()
    
    async def _execute_scheduled_action(self, command: SmartPlugCommand, job_id: str):
        """Execute the scheduled action"""
        print(f"🚀 Executing scheduled action: {command.action.value} for {command.device_name}")
        
        # Update job status
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id]['status'] = 'executing'
        
        success = False
        try:
            if command.action == ActionType.TURN_ON or command.action == ActionType.SET_SCHEDULE:
                success = await self.controller.turn_on_device(command.device_name)
            elif command.action == ActionType.TURN_OFF:
                success = await self.controller.turn_off_device(command.device_name)
            
            # Update job status
            if job_id in self.scheduled_jobs:
                self.scheduled_jobs[job_id]['status'] = 'completed' if success else 'failed'
                self.scheduled_jobs[job_id]['executed_at'] = datetime.now()
                
        except Exception as e:
            print(f"❌ Error executing scheduled action: {e}")
            if job_id in self.scheduled_jobs:
                self.scheduled_jobs[job_id]['status'] = 'failed'
                self.scheduled_jobs[job_id]['error'] = str(e)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id]['status'] = 'cancelled'
            print(f"🚫 Cancelled job {job_id}")
            return True
        return False
    
    def list_scheduled_jobs(self) -> List[Dict]:
        """List all scheduled jobs"""
        jobs = []
        for job_id, job_data in self.scheduled_jobs.items():
            job_info = {
                'job_id': job_id,
                'device': job_data['command'].device_name,
                'action': job_data['command'].action.value,
                'scheduled_for': job_data['scheduled_for'].isoformat(),
                'status': job_data.get('status', 'unknown')
            }
            
            if 'executed_at' in job_data:
                job_info['executed_at'] = job_data['executed_at'].isoformat()
            if 'error' in job_data:
                job_info['error'] = job_data['error']
                
            jobs.append(job_info)
        return jobs

class SmartPlugAgent:
    """Main agent that orchestrates the entire system"""
    
    def __init__(self):
        self.parser = NaturalLanguageParser()
        self.controller = SmartPlugController()
        self.scheduler = SmartPlugScheduler(self.controller)
        
        # Initialize LLM if available and API key is set
        self.llm_manager = None
        self.use_llm = False
        
        if LLM_AVAILABLE:
            try:
                # Check for API keys with proper prefix validation
                openai_key = os.getenv('OPENAI_API_KEY')
                anthropic_key = os.getenv('ANTHROPIC_API_KEY')
                gemini_key = os.getenv('GEMINI_API_KEY')
                
                # Validate API keys with proper prefixes
                valid_openai = (openai_key and openai_key.startswith('sk-') and len(openai_key) > 10)
                valid_anthropic = (anthropic_key and 
                                 (anthropic_key.startswith('sk-ant-') or 
                                  (len(anthropic_key) > 20 and not anthropic_key.startswith('your') and 
                                   anthropic_key != 'your_anthropic_key_here')))
                valid_gemini = (gemini_key and gemini_key.startswith('AIza') and len(gemini_key) > 10)
                
                if valid_openai or valid_anthropic or valid_gemini:
                    from llm_integration import LLMManager
                    self.llm_manager = LLMManager()
                    self.use_llm = True
                    
                    # Determine which provider to use
                    if valid_gemini:
                        print("🧠 LLM integration enabled with Gemini API")
                    elif valid_openai:
                        print("🧠 LLM integration enabled with OpenAI API")
                    elif valid_anthropic:
                        print("🧠 LLM integration enabled with Anthropic API")
                else:
                    print("⚠️ No valid API keys found - using direct pattern matching")
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e} - using direct pattern matching")
        
        print("🤖 Smart Plug Agent initialized")
        
    async def process_llm_command(self, llm_output: str) -> Dict:
        """
        Process a command from LLM output or direct user input
        
        Args:
            llm_output: Natural language command like "Set dishwasher at 14:00"
            
        Returns:
            Dict with execution result
        """
        print(f"\n🎯 Processing command: '{llm_output}'")
        
        # First try LLM if available and enabled
        if self.use_llm and self.llm_manager:
            try:
                print("🧠 Using LLM for command interpretation...")
                
                # Use LLM to interpret the command
                llm_response = await self.llm_manager.process_command(llm_output)
                
                if llm_response.success:
                    # Parse LLM response into SmartPlugCommand
                    command = self._parse_llm_response(llm_response, llm_output)
                    if command:
                        return await self._execute_command(command, llm_output, use_llm=True, llm_confidence=llm_response.confidence)
                    else:
                        print("⚠️ LLM response could not be converted to command, falling back to direct parsing")
                else:
                    print("⚠️ LLM processing failed, falling back to direct parsing")
            except Exception as e:
                print(f"⚠️ LLM processing error: {e}, falling back to direct parsing")
        
        # Fallback to direct parsing
        print("🔍 Using direct pattern matching...")
        command = self.parser.parse_command(llm_output)
        
        if not command:
            error_msg = f"Could not parse command: '{llm_output}'"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'original_command': llm_output,
                'parsing_method': 'direct',
                'llm_confidence': 0.0
            }
        
        return await self._execute_command(command, llm_output, use_llm=False, llm_confidence=0.0)
    
    def _parse_llm_response(self, llm_response, original_command: str) -> Optional[SmartPlugCommand]:
        """Convert LLM response to SmartPlugCommand"""
        try:
            intent = llm_response.intent.lower() if llm_response.intent else ''
            device_name = llm_response.device.lower() if llm_response.device else ''
            time_str = llm_response.time
            
            # Map device name to our internal mapping
            device_id = None
            if device_name:
                device_id = self.parser.device_mappings.get(device_name)
                if not device_id:
                    # Try fuzzy matching
                    for key, value in self.parser.device_mappings.items():
                        if device_name in key or key in device_name:
                            device_id = value
                            break
            
            if not device_id:
                print(f"❌ Device '{device_name}' not found in device mappings")
                return None
            
            # Determine action type
            action = None
            if intent == 'turn_on':
                action = ActionType.TURN_ON
            elif intent == 'turn_off':
                action = ActionType.TURN_OFF
            elif intent == 'schedule':
                action = ActionType.SET_SCHEDULE
            else:
                action = ActionType.SET_SCHEDULE  # Default for commands with time
            
            # Parse time if provided
            scheduled_time = None
            if time_str:
                scheduled_time = self._parse_time_from_llm(time_str)
            
            return SmartPlugCommand(
                device_name=device_id,
                action=action,
                scheduled_time=scheduled_time
            )
            
        except Exception as e:
            print(f"❌ Error parsing LLM response: {e}")
            return None
    
    def _parse_time_from_llm(self, time_str: str) -> Optional[datetime]:
        """Parse time string from LLM response"""
        try:
            time_str = time_str.lower().strip()
            now = datetime.now()
            
            # Handle various time formats
            if ':' in time_str:
                # Format like "14:00" or "2:00"
                time_parts = time_str.replace('pm', '').replace('am', '').strip().split(':')
                if len(time_parts) == 2:
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    
                    # Handle AM/PM
                    if 'pm' in time_str and hour != 12:
                        hour += 12
                    elif 'am' in time_str and hour == 12:
                        hour = 0
                    
                    scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If time has passed today, schedule for tomorrow
                    if scheduled_time <= now:
                        scheduled_time += timedelta(days=1)
                    
                    return scheduled_time
            
            return None
        except Exception as e:
            print(f"❌ Error parsing time '{time_str}': {e}")
            return None
    
    async def _execute_command(self, command: SmartPlugCommand, original_command: str, use_llm: bool = False, llm_confidence: float = 0.0) -> Dict:
        """Execute the parsed command"""
        parsing_method = 'llm' if use_llm else 'direct'
        
        # Execute based on whether it's immediate or scheduled
        if command.scheduled_time and command.scheduled_time > datetime.now():
            # Schedule for future execution
            job_id = self.scheduler.schedule_action(command)
            if job_id:
                return {
                    'success': True,
                    'action': 'scheduled',
                    'job_id': job_id,
                    'device': command.device_name,
                    'scheduled_for': command.scheduled_time.isoformat(),
                    'original_command': original_command,
                    'parsing_method': parsing_method,
                    'llm_confidence': llm_confidence
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to schedule action',
                    'original_command': original_command,
                    'parsing_method': parsing_method,
                    'llm_confidence': llm_confidence
                }
        else:
            # Execute immediately
            print("⚡ Executing command immediately")
            if command.action == ActionType.TURN_ON or command.action == ActionType.SET_SCHEDULE:
                success = await self.controller.turn_on_device(command.device_name)
            elif command.action == ActionType.TURN_OFF:
                success = await self.controller.turn_off_device(command.device_name)
            else:
                success = False
            
            return {
                'success': success,
                'action': 'executed_immediately',
                'device': command.device_name,
                'original_command': original_command,
                'parsing_method': parsing_method,
                'llm_confidence': llm_confidence
            }
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'connected_devices': list(self.controller.device_registry.keys()),
            'device_states': self.controller.get_all_devices_status(),
            'scheduled_jobs': self.scheduler.list_scheduled_jobs(),
            'timestamp': datetime.now().isoformat(),
            'total_devices': len(self.controller.device_registry),
            'active_devices': sum(1 for state in self.controller.plug_states.values() if state)
        }
    
    def get_device_by_name(self, device_name: str) -> Optional[str]:
        """Get device ID by friendly name"""
        return self.parser.device_mappings.get(device_name.lower())
    
    async def turn_on_device_directly(self, device_name: str) -> bool:
        """Turn on device directly by name"""
        device_id = self.get_device_by_name(device_name)
        if device_id:
            return await self.controller.turn_on_device(device_id)
        return False
    
    async def turn_off_device_directly(self, device_name: str) -> bool:
        """Turn off device directly by name"""
        device_id = self.get_device_by_name(device_name)
        if device_id:
            return await self.controller.turn_off_device(device_id)
        return False
