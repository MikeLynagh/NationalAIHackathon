"""
Enhanced Smart Plug Agent with LLM Integration and Hardware Interface
This is the main entry point that combines all components
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import Config
from hardware_interface import HardwareManager
from llm_integration import LLMManager, LLMResponse
from smart_plug_agent import SmartPlugCommand, ActionType, SmartPlugScheduler

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedSmartPlugAgent:
    """Enhanced Smart Plug Agent with LLM integration and hardware interface"""
    
    def __init__(self, llm_provider: str = "openai"):
        logger.info("Initializing Enhanced Smart Plug Agent...")
        
        # Initialize components
        self.hardware_manager = HardwareManager(Config.DEVICES)
        self.llm_manager = LLMManager(primary_provider=llm_provider)
        self.scheduler = SmartPlugScheduler(self.hardware_manager)
        
        # Device name mappings for backward compatibility
        self.device_mappings = {device.friendly_name: device.device_id for device in Config.DEVICES}
        
        # Add aliases
        for friendly_name, aliases in Config.DEVICE_ALIASES.items():
            device_id = self.device_mappings.get(friendly_name)
            if device_id:
                for alias in aliases:
                    self.device_mappings[alias] = device_id
        
        logger.info(f"Agent initialized with {len(Config.DEVICES)} devices")
    
    async def process_natural_language_command(self, user_input: str, use_llm: bool = True) -> Dict:
        """
        Process natural language command using LLM or fallback parsing
        
        Args:
            user_input: Natural language command from user
            use_llm: Whether to use LLM for processing (True) or fallback parser (False)
            
        Returns:
            Dict with execution result
        """
        logger.info(f"Processing command: '{user_input}' (LLM: {use_llm})")
        
        try:
            if use_llm:
                # Use LLM for intent extraction
                llm_response = await self.llm_manager.process_command(user_input)
                
                if not llm_response.success:
                    logger.warning(f"LLM processing failed: {llm_response.error}")
                    # Fall back to simple parsing
                    return await self._fallback_processing(user_input)
                
                # Convert LLM response to SmartPlugCommand
                command = self._llm_response_to_command(llm_response, user_input)
            else:
                # Use fallback simple parsing
                return await self._fallback_processing(user_input)
            
            if not command:
                return {
                    'success': False,
                    'error': 'Could not understand the command',
                    'original_command': user_input,
                    'llm_confidence': getattr(llm_response, 'confidence', 0.0) if use_llm else 0.0
                }
            
            # Execute the command
            result = await self._execute_command(command)
            result['original_command'] = user_input
            result['llm_confidence'] = getattr(llm_response, 'confidence', 0.0) if use_llm else 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing command '{user_input}': {e}")
            return {
                'success': False,
                'error': str(e),
                'original_command': user_input
            }
    
    def _llm_response_to_command(self, llm_response: LLMResponse, original_input: str) -> Optional[SmartPlugCommand]:
        """Convert LLM response to SmartPlugCommand"""
        try:
            # Map LLM device name to actual device ID
            device_id = None
            if llm_response.device:
                device_id = self.device_mappings.get(llm_response.device.lower())
                if not device_id:
                    # Try fuzzy matching
                    for mapped_name, mapped_id in self.device_mappings.items():
                        if llm_response.device.lower() in mapped_name or mapped_name in llm_response.device.lower():
                            device_id = mapped_id
                            break
            
            if not device_id:
                logger.warning(f"Could not map device '{llm_response.device}' to known device")
                return None
            
            # Map LLM intent to action type
            action_map = {
                'turn_on': ActionType.TURN_ON,
                'turn_off': ActionType.TURN_OFF,
                'schedule': ActionType.SET_SCHEDULE
            }
            
            action = action_map.get(llm_response.intent, ActionType.SET_SCHEDULE)
            
            # Parse time if provided
            scheduled_time = None
            if llm_response.time:
                scheduled_time = self._parse_time_from_llm(llm_response.time)
            
            return SmartPlugCommand(
                device_name=device_id,
                action=action,
                scheduled_time=scheduled_time
            )
            
        except Exception as e:
            logger.error(f"Error converting LLM response to command: {e}")
            return None
    
    def _parse_time_from_llm(self, time_str: str) -> Optional[datetime]:
        """Parse time string from LLM response"""
        try:
            import re
            time_str = time_str.lower().strip()
            
            # Pattern: HH:MM (24-hour)
            match = re.search(r'(\d{1,2}):(\d{2})', time_str)
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                now = datetime.now()
                scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)
                return scheduled_time
            
            # Pattern: relative time "in X minutes/hours"
            match = re.search(r'in (\d+) (minutes?|hours?)', time_str)
            if match:
                amount = int(match.group(1))
                unit = match.group(2)
                
                if 'minute' in unit:
                    return datetime.now() + timedelta(minutes=amount)
                elif 'hour' in unit:
                    return datetime.now() + timedelta(hours=amount)
            
            # Pattern: "2 PM", "14:00", etc.
            # This would need more sophisticated parsing
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing time '{time_str}': {e}")
            return None
    
    async def _fallback_processing(self, user_input: str) -> Dict:
        """Fallback processing using simple pattern matching"""
        logger.info("Using fallback processing")
        
        # Use the original smart_plug_agent parsing logic
        from smart_plug_agent import NaturalLanguageParser
        parser = NaturalLanguageParser()
        command = parser.parse_command(user_input)
        
        if not command:
            return {
                'success': False,
                'error': 'Could not parse command with fallback method',
                'original_command': user_input
            }
        
        return await self._execute_command(command)
    
    async def _execute_command(self, command: SmartPlugCommand) -> Dict:
        """Execute a SmartPlugCommand"""
        try:
            # Check if device exists
            if command.device_name not in [device.device_id for device in Config.DEVICES]:
                return {
                    'success': False,
                    'error': f'Device {command.device_name} not found'
                }
            
            # Check power consumption safety
            if command.action == ActionType.TURN_ON or command.action == ActionType.SET_SCHEDULE:
                current_status = await self.hardware_manager.get_all_devices_status()
                active_devices = [status['device_id'] for status in current_status if status.get('is_on', False)]
                active_devices.append(command.device_name)  # Add the device we're turning on
                
                if not Config.validate_power_consumption(active_devices):
                    return {
                        'success': False,
                        'error': 'Power consumption would exceed safety limits'
                    }
            
            # Execute based on timing
            if command.scheduled_time and command.scheduled_time > datetime.now():
                # Schedule for future execution
                job_id = self.scheduler.schedule_action(command)
                if job_id:
                    return {
                        'success': True,
                        'action': 'scheduled',
                        'job_id': job_id,
                        'device': command.device_name,
                        'scheduled_for': command.scheduled_time.isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Failed to schedule action'
                    }
            else:
                # Execute immediately
                if command.action == ActionType.TURN_ON or command.action == ActionType.SET_SCHEDULE:
                    success = await self.hardware_manager.turn_on_device(command.device_name)
                elif command.action == ActionType.TURN_OFF:
                    success = await self.hardware_manager.turn_off_device(command.device_name)
                else:
                    success = False
                
                return {
                    'success': success,
                    'action': 'executed_immediately',
                    'device': command.device_name
                }
                
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            # Get hardware status
            device_statuses = await self.hardware_manager.get_all_devices_status()
            total_power = await self.hardware_manager.get_total_power_consumption()
            
            # Get scheduled jobs
            scheduled_jobs = self.scheduler.list_scheduled_jobs()
            
            # Calculate summary statistics
            total_devices = len(Config.DEVICES)
            active_devices = sum(1 for status in device_statuses if status.get('is_on', False))
            pending_jobs = sum(1 for job in scheduled_jobs if job.get('status') == 'pending')
            
            # Group devices by location
            devices_by_location = {}
            for status in device_statuses:
                device_config = Config.get_device_by_id(status['device_id'])
                if device_config:
                    location = device_config.location
                    if location not in devices_by_location:
                        devices_by_location[location] = []
                    devices_by_location[location].append({
                        'device_id': status['device_id'],
                        'friendly_name': device_config.friendly_name,
                        'is_on': status.get('is_on', False),
                        'power_consumption': status.get('power_consumption', 0)
                    })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_devices': total_devices,
                    'active_devices': active_devices,
                    'pending_jobs': pending_jobs,
                    'total_power_consumption': total_power,
                    'power_limit': Config.MAX_POWER_TOTAL * Config.POWER_SAFETY_MARGIN
                },
                'devices_by_location': devices_by_location,
                'device_statuses': device_statuses,
                'scheduled_jobs': scheduled_jobs,
                'available_providers': self.llm_manager.get_available_providers(),
                'primary_llm_provider': self.llm_manager.primary_provider
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def emergency_shutdown(self) -> Dict:
        """Emergency shutdown of all devices"""
        logger.warning("Emergency shutdown initiated")
        
        try:
            # Cancel all scheduled jobs
            for job in self.scheduler.scheduled_jobs:
                self.scheduler.cancel_job(job)
            
            # Turn off all devices
            shutdown_results = await self.hardware_manager.emergency_shutdown()
            
            return {
                'success': True,
                'action': 'emergency_shutdown',
                'timestamp': datetime.now().isoformat(),
                'shutdown_results': shutdown_results,
                'cancelled_jobs': len(self.scheduler.scheduled_jobs)
            }
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_device_recommendations(self, context: str = "") -> List[str]:
        """Get smart recommendations based on context"""
        recommendations = []
        
        try:
            current_hour = datetime.now().hour
            device_statuses = await self.hardware_manager.get_all_devices_status()
            
            # Time-based recommendations
            if 6 <= current_hour <= 9:  # Morning
                if not any(s['device_id'] == 'kitchen_coffee_maker' and s.get('is_on') for s in device_statuses):
                    recommendations.append("Consider turning on the coffee maker for your morning routine")
            
            elif 17 <= current_hour <= 19:  # Evening
                if not any(s['device_id'] == 'living_room_lights' and s.get('is_on') for s in device_statuses):
                    recommendations.append("You might want to turn on the living room lights")
            
            elif 22 <= current_hour or current_hour <= 6:  # Night/Late evening
                # Suggest EV charging
                if not any(s['device_id'] == 'garage_ev_charger' and s.get('is_on') for s in device_statuses):
                    recommendations.append("Good time to charge your EV with lower electricity rates")
            
            # Energy efficiency recommendations
            total_power = await self.hardware_manager.get_total_power_consumption()
            if total_power > Config.MAX_POWER_TOTAL * 0.7:
                recommendations.append("High power usage detected. Consider turning off non-essential devices")
            
            # Device-specific recommendations
            active_high_power = [s for s in device_statuses 
                               if s.get('is_on') and s.get('power_consumption', 0) > 1000]
            if len(active_high_power) > 2:
                recommendations.append("Multiple high-power devices are active. Monitor energy usage")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to system error"]

# Command-line interface
async def interactive_mode():
    """Interactive command-line interface"""
    print("🏠 Enhanced Smart Plug Agent")
    print("=" * 50)
    print("Available commands:")
    print("  • Natural language: 'Turn on lights', 'Set dishwasher at 14:00'")
    print("  • 'status' - Show system status")
    print("  • 'recommendations' - Get smart recommendations")
    print("  • 'emergency' - Emergency shutdown")
    print("  • 'help' - Show this help")
    print("  • 'quit' - Exit")
    print()
    
    agent = EnhancedSmartPlugAgent()
    
    while True:
        try:
            user_input = input("🎯 Command: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'status':
                status = await agent.get_comprehensive_status()
                print(f"\n📊 System Status:")
                print(f"  Active devices: {status['summary']['active_devices']}/{status['summary']['total_devices']}")
                print(f"  Power usage: {status['summary']['total_power_consumption']:.1f}W")
                print(f"  Pending jobs: {status['summary']['pending_jobs']}")
                print(f"  LLM Provider: {status['primary_llm_provider']}")
                continue
            
            elif user_input.lower() == 'recommendations':
                recommendations = await agent.get_device_recommendations()
                print(f"\n💡 Smart Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
                continue
            
            elif user_input.lower() == 'emergency':
                result = await agent.emergency_shutdown()
                if result['success']:
                    print("🚨 Emergency shutdown completed")
                else:
                    print(f"❌ Emergency shutdown failed: {result['error']}")
                continue
            
            elif user_input.lower() == 'help':
                print("\n💡 Example commands:")
                examples = [
                    "Turn on lights",
                    "Set dishwasher at 14:00",
                    "Turn off heater in 30 minutes",
                    "Activate coffee maker at 7:00 AM",
                    "Schedule EV charger at 10 PM"
                ]
                for cmd in examples:
                    print(f"  • {cmd}")
                continue
            
            elif not user_input:
                continue
            
            # Process natural language command
            result = await agent.process_natural_language_command(user_input)
            
            if result['success']:
                if result['action'] == 'scheduled':
                    scheduled_time = datetime.fromisoformat(result['scheduled_for']).strftime("%H:%M")
                    print(f"   ✅ Scheduled {result['device']} for {scheduled_time}")
                else:
                    print(f"   ✅ {result['device']} action completed")
                
                # Show confidence if available
                if 'llm_confidence' in result and result['llm_confidence'] > 0:
                    print(f"   🎯 Confidence: {result['llm_confidence']:.1%}")
            else:
                print(f"   ❌ {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_mode())
