"""
Configuration settings for the Smart Plug Agent System
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DeviceConfig:
    """Configuration for a smart plug device"""
    device_id: str
    friendly_name: str
    ip_address: str
    device_type: str
    location: str
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    max_power: int = 1500  # watts
    
class Config:
    """Main configuration class"""
    
    # Device Registry - Add your actual devices here
    DEVICES = [
        DeviceConfig(
            device_id="kitchen_dishwasher",
            friendly_name="dishwasher", 
            ip_address="192.168.1.100",
            device_type="kasa",
            location="kitchen",
            manufacturer="TP-Link",
            model="HS110",
            max_power=1800
        ),
        DeviceConfig(
            device_id="laundry_dryer",
            friendly_name="dryer",
            ip_address="192.168.1.101", 
            device_type="kasa",
            location="laundry",
            manufacturer="TP-Link",
            model="HS110",
            max_power=2400
        ),
        DeviceConfig(
            device_id="living_room_heater",
            friendly_name="heater",
            ip_address="192.168.1.102",
            device_type="kasa", 
            location="living_room",
            manufacturer="TP-Link",
            model="HS110",
            max_power=1500
        ),
        DeviceConfig(
            device_id="garage_ev_charger",
            friendly_name="ev charge",
            ip_address="192.168.1.103",
            device_type="kasa",
            location="garage", 
            manufacturer="TP-Link",
            model="HS220",
            max_power=7200
        ),
        DeviceConfig(
            device_id="laundry_washer",
            friendly_name="washing machine",
            ip_address="192.168.1.104",
            device_type="kasa",
            location="laundry",
            manufacturer="TP-Link", 
            model="HS110",
            max_power=500
        ),
        DeviceConfig(
            device_id="kitchen_coffee_maker",
            friendly_name="coffee maker",
            ip_address="192.168.1.105",
            device_type="kasa",
            location="kitchen",
            manufacturer="TP-Link",
            model="HS105",
            max_power=1000
        ),
        DeviceConfig(
            device_id="bedroom_ac",
            friendly_name="air conditioner", 
            ip_address="192.168.1.106",
            device_type="kasa",
            location="bedroom",
            manufacturer="TP-Link",
            model="HS110",
            max_power=1200
        ),
        DeviceConfig(
            device_id="living_room_lights",
            friendly_name="lights",
            ip_address="192.168.1.107",
            device_type="kasa",
            location="living_room",
            manufacturer="TP-Link",
            model="HS105", 
            max_power=100
        ),
        DeviceConfig(
            device_id="bedroom_fan",
            friendly_name="fan",
            ip_address="192.168.1.108",
            device_type="kasa",
            location="bedroom",
            manufacturer="TP-Link",
            model="HS105",
            max_power=75
        ),
        DeviceConfig(
            device_id="living_room_tv",
            friendly_name="tv",
            ip_address="192.168.1.109", 
            device_type="kasa",
            location="living_room",
            manufacturer="TP-Link",
            model="HS105",
            max_power=200
        ),
        DeviceConfig(
            device_id="kitchen_microwave",
            friendly_name="microwave",
            ip_address="192.168.1.110",
            device_type="kasa",
            location="kitchen",
            manufacturer="TP-Link",
            model="HS110",
            max_power=1100
        )
    ]
    
    # Alternative device names for better NLP recognition
    DEVICE_ALIASES = {
        'dishwasher': ['dish washer', 'dishes'],
        'dryer': ['clothes dryer', 'tumble dryer'],
        'heater': ['space heater', 'electric heater', 'heating'],
        'ev charge': ['ev charger', 'electric vehicle charger', 'car charger', 'tesla charger'],
        'washing machine': ['washer', 'clothes washer', 'laundry machine'],
        'coffee maker': ['coffee machine', 'coffee pot', 'espresso machine'],
        'air conditioner': ['ac', 'aircon', 'cooling', 'air conditioning'],
        'lights': ['lighting', 'lamp', 'bulbs'],
        'fan': ['ceiling fan', 'electric fan'],
        'tv': ['television', 'smart tv'],
        'microwave': ['microwave oven', 'micro']
    }
    
    # Time parsing configuration
    TIME_PATTERNS = {
        '24_hour': r'at (\d{1,2}):(\d{2})',  # at 14:00
        '12_hour': r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)',  # at 2:00 PM
        'relative_minutes': r'in (\d+) (minutes?)',  # in 30 minutes  
        'relative_hours': r'in (\d+) (hours?)',  # in 2 hours
        'relative_day': r'(tomorrow|today) at (\d{1,2}):(\d{2})',  # tomorrow at 14:00
    }
    
    # Action keywords
    ACTION_KEYWORDS = {
        'turn_on': ['turn on', 'start', 'activate', 'switch on', 'power on', 'enable'],
        'turn_off': ['turn off', 'stop', 'deactivate', 'switch off', 'power off', 'disable'],
        'schedule': ['set', 'schedule', 'program', 'plan']
    }
    
    # Scheduling configuration
    MAX_SCHEDULE_DAYS = 7  # Maximum days in advance to schedule
    MAX_CONCURRENT_JOBS = 50  # Maximum number of scheduled jobs
    SCHEDULE_PRECISION_MINUTES = 1  # Minimum scheduling precision
    
    # Hardware interface configuration
    DEVICE_TIMEOUT = 5.0  # seconds
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # seconds
    
    # Safety limits
    MAX_POWER_TOTAL = 15000  # watts - total max power for all devices
    POWER_SAFETY_MARGIN = 0.8  # Use 80% of max power as safety limit
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "smart_plug_agent.log"
    
    # API configuration (for future web interface)
    API_HOST = "localhost"
    API_PORT = 8000
    API_DEBUG = True
    
    # LLM Integration configuration
    LLM_PROVIDERS = {
        'openai': {
            'api_key_env': 'OPENAI_API_KEY',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 200
        },
        'anthropic': {
            'api_key_env': 'ANTHROPIC_API_KEY', 
            'model': 'claude-3-sonnet-20240229',
            'temperature': 0.1,
            'max_tokens': 200
        },
        'gemini': {
            'api_key_env': 'GEMINI_API_KEY',
            'model': 'gemini-1.5-pro',
            'temperature': 0.1,
            'max_tokens': 200
        }
    }
    
    # Default LLM provider - Now using Gemini
    DEFAULT_LLM_PROVIDER = 'gemini'
    
    # API Keys (fallback - NOT recommended for production)
    # OPENAI_API_KEY = "your-key-here"  # Uncomment and add your key
    # ANTHROPIC_API_KEY = "your-key-here"  # Uncomment and add your key
    # GEMINI_API_KEY = "your-key-here"  # Uncomment and add your key
    
    # Natural language processing configuration
    NLP_CONFIDENCE_THRESHOLD = 0.7
    NLP_MAX_TOKENS = 100
    
    @classmethod
    def get_device_by_id(cls, device_id: str) -> DeviceConfig:
        """Get device configuration by ID"""
        for device in cls.DEVICES:
            if device.device_id == device_id:
                return device
        return None
    
    @classmethod
    def get_device_by_name(cls, friendly_name: str) -> DeviceConfig:
        """Get device configuration by friendly name"""
        friendly_name = friendly_name.lower()
        for device in cls.DEVICES:
            if device.friendly_name.lower() == friendly_name:
                return device
        return None
    
    @classmethod
    def get_all_device_names(cls) -> List[str]:
        """Get all device friendly names including aliases"""
        names = [device.friendly_name for device in cls.DEVICES]
        for aliases in cls.DEVICE_ALIASES.values():
            names.extend(aliases)
        return names
    
    @classmethod
    def get_devices_by_location(cls, location: str) -> List[DeviceConfig]:
        """Get all devices in a specific location"""
        return [device for device in cls.DEVICES if device.location.lower() == location.lower()]
    
    @classmethod
    def get_total_max_power(cls) -> int:
        """Get total maximum power consumption of all devices"""
        return sum(device.max_power for device in cls.DEVICES)
    
    @classmethod
    def validate_power_consumption(cls, active_devices: List[str]) -> bool:
        """Check if activating devices would exceed power limits"""
        total_power = 0
        for device_id in active_devices:
            device = cls.get_device_by_id(device_id)
            if device:
                total_power += device.max_power
        
        max_allowed = cls.MAX_POWER_TOTAL * cls.POWER_SAFETY_MARGIN
        return total_power <= max_allowed
