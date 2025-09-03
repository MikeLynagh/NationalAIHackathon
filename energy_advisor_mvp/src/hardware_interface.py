"""
Hardware interface layer for different smart plug manufacturers
Supports TP-Link Kasa, Amazon Smart Plug, and other common brands
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from datetime import datetime
import logging
from config import DeviceConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartPlugInterface(ABC):
    """Abstract base class for smart plug interfaces"""
    
    @abstractmethod
    async def turn_on(self) -> bool:
        """Turn on the smart plug"""
        pass
    
    @abstractmethod
    async def turn_off(self) -> bool:
        """Turn off the smart plug"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict:
        """Get current status of the smart plug"""
        pass
    
    @abstractmethod
    async def get_power_consumption(self) -> float:
        """Get current power consumption in watts"""
        pass

class KasaSmartPlug(SmartPlugInterface):
    """TP-Link Kasa smart plug interface"""
    
    def __init__(self, device_config: DeviceConfig):
        self.config = device_config
        self.ip = device_config.ip_address
        self._is_on = False
        self._power_consumption = 0.0
        self._last_update = datetime.now()
    
    async def turn_on(self) -> bool:
        """Turn on the Kasa smart plug"""
        try:
            logger.info(f"Turning ON Kasa device at {self.ip}")
            
            # In a real implementation, you would use the python-kasa library:
            # from kasa import SmartPlug
            # plug = SmartPlug(self.ip)
            # await plug.update()
            # await plug.turn_on()
            
            # For demo purposes, simulate the action
            await asyncio.sleep(0.5)  # Simulate network delay
            self._is_on = True
            self._power_consumption = self.config.max_power * 0.8  # Simulate 80% power usage
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned ON {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn ON {self.config.device_id}: {e}")
            return False
    
    async def turn_off(self) -> bool:
        """Turn off the Kasa smart plug"""
        try:
            logger.info(f"Turning OFF Kasa device at {self.ip}")
            
            # Real implementation:
            # plug = SmartPlug(self.ip) 
            # await plug.update()
            # await plug.turn_off()
            
            # Demo simulation
            await asyncio.sleep(0.5)
            self._is_on = False
            self._power_consumption = 0.0
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned OFF {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn OFF {self.config.device_id}: {e}")
            return False
    
    async def get_status(self) -> Dict:
        """Get current status of the Kasa smart plug"""
        try:
            # Real implementation:
            # plug = SmartPlug(self.ip)
            # await plug.update()
            # return {
            #     'is_on': plug.is_on,
            #     'power': plug.current_consumption(),
            #     'voltage': plug.voltage,
            #     'current': plug.current
            # }
            
            # Demo simulation
            return {
                'device_id': self.config.device_id,
                'is_on': self._is_on,
                'power_consumption': self._power_consumption,
                'voltage': 120.0 if self._is_on else 0.0,
                'current': self._power_consumption / 120.0 if self._is_on else 0.0,
                'last_update': self._last_update.isoformat(),
                'manufacturer': self.config.manufacturer,
                'model': self.config.model
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for {self.config.device_id}: {e}")
            return {'error': str(e)}
    
    async def get_power_consumption(self) -> float:
        """Get current power consumption in watts"""
        return self._power_consumption

class AmazonSmartPlug(SmartPlugInterface):
    """Amazon Smart Plug interface"""
    
    def __init__(self, device_config: DeviceConfig):
        self.config = device_config
        self.ip = device_config.ip_address
        self._is_on = False
        self._power_consumption = 0.0
        self._last_update = datetime.now()
    
    async def turn_on(self) -> bool:
        """Turn on the Amazon smart plug"""
        try:
            logger.info(f"Turning ON Amazon device at {self.ip}")
            
            # In a real implementation, you would use Amazon's Alexa Voice Service API
            # or a third-party library for Amazon smart plugs
            
            await asyncio.sleep(0.3)  # Simulate network delay
            self._is_on = True
            self._power_consumption = self.config.max_power * 0.75
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned ON {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn ON {self.config.device_id}: {e}")
            return False
    
    async def turn_off(self) -> bool:
        """Turn off the Amazon smart plug"""
        try:
            logger.info(f"Turning OFF Amazon device at {self.ip}")
            
            await asyncio.sleep(0.3)
            self._is_on = False
            self._power_consumption = 0.0
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned OFF {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn OFF {self.config.device_id}: {e}")
            return False
    
    async def get_status(self) -> Dict:
        """Get current status of the Amazon smart plug"""
        return {
            'device_id': self.config.device_id,
            'is_on': self._is_on,
            'power_consumption': self._power_consumption,
            'last_update': self._last_update.isoformat(),
            'manufacturer': 'Amazon',
            'model': 'Smart Plug'
        }
    
    async def get_power_consumption(self) -> float:
        """Get current power consumption in watts"""
        return self._power_consumption

class WyzeSmartPlug(SmartPlugInterface):
    """Wyze smart plug interface"""
    
    def __init__(self, device_config: DeviceConfig):
        self.config = device_config
        self.ip = device_config.ip_address
        self._is_on = False
        self._power_consumption = 0.0
        self._last_update = datetime.now()
    
    async def turn_on(self) -> bool:
        """Turn on the Wyze smart plug"""
        try:
            logger.info(f"Turning ON Wyze device at {self.ip}")
            
            # Real implementation would use Wyze API or python-wyze library
            await asyncio.sleep(0.4)
            self._is_on = True
            self._power_consumption = self.config.max_power * 0.85
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned ON {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn ON {self.config.device_id}: {e}")
            return False
    
    async def turn_off(self) -> bool:
        """Turn off the Wyze smart plug"""
        try:
            logger.info(f"Turning OFF Wyze device at {self.ip}")
            
            await asyncio.sleep(0.4)
            self._is_on = False
            self._power_consumption = 0.0
            self._last_update = datetime.now()
            
            logger.info(f"Successfully turned OFF {self.config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to turn OFF {self.config.device_id}: {e}")
            return False
    
    async def get_status(self) -> Dict:
        """Get current status of the Wyze smart plug"""
        return {
            'device_id': self.config.device_id,
            'is_on': self._is_on,
            'power_consumption': self._power_consumption,
            'last_update': self._last_update.isoformat(),
            'manufacturer': 'Wyze',
            'model': 'Smart Plug'
        }
    
    async def get_power_consumption(self) -> float:
        """Get current power consumption in watts"""
        return self._power_consumption

class SmartPlugFactory:
    """Factory class to create appropriate smart plug interfaces"""
    
    @staticmethod
    def create_smart_plug(device_config: DeviceConfig) -> SmartPlugInterface:
        """Create a smart plug interface based on device type"""
        device_type = device_config.device_type.lower()
        
        if device_type == 'kasa':
            return KasaSmartPlug(device_config)
        elif device_type == 'amazon':
            return AmazonSmartPlug(device_config)
        elif device_type == 'wyze':
            return WyzeSmartPlug(device_config)
        else:
            # Default to Kasa for unknown types
            logger.warning(f"Unknown device type '{device_type}', defaulting to Kasa")
            return KasaSmartPlug(device_config)

class HardwareManager:
    """Manages all smart plug hardware interfaces"""
    
    def __init__(self, device_configs: List[DeviceConfig]):
        self.devices = {}
        self.device_configs = {config.device_id: config for config in device_configs}
        self._initialize_devices()
    
    def _initialize_devices(self):
        """Initialize all device interfaces"""
        for config in self.device_configs.values():
            try:
                interface = SmartPlugFactory.create_smart_plug(config)
                self.devices[config.device_id] = interface
                logger.info(f"Initialized {config.device_type} device: {config.device_id}")
            except Exception as e:
                logger.error(f"Failed to initialize device {config.device_id}: {e}")
    
    async def turn_on_device(self, device_id: str) -> bool:
        """Turn on a specific device"""
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        return await self.devices[device_id].turn_on()
    
    async def turn_off_device(self, device_id: str) -> bool:
        """Turn off a specific device"""
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        return await self.devices[device_id].turn_off()
    
    async def get_device_status(self, device_id: str) -> Dict:
        """Get status of a specific device"""
        if device_id not in self.devices:
            return {'error': f'Device {device_id} not found'}
        
        return await self.devices[device_id].get_status()
    
    async def get_all_devices_status(self) -> List[Dict]:
        """Get status of all devices"""
        statuses = []
        for device_id in self.devices:
            status = await self.get_device_status(device_id)
            statuses.append(status)
        return statuses
    
    async def get_total_power_consumption(self) -> float:
        """Get total power consumption of all active devices"""
        total_power = 0.0
        for device in self.devices.values():
            power = await device.get_power_consumption()
            total_power += power
        return total_power
    
    async def get_devices_by_location(self, location: str) -> List[str]:
        """Get device IDs for a specific location"""
        return [
            device_id for device_id, config in self.device_configs.items()
            if config.location.lower() == location.lower()
        ]
    
    async def emergency_shutdown(self) -> Dict:
        """Emergency shutdown of all devices"""
        logger.warning("Emergency shutdown initiated")
        results = {}
        
        for device_id in self.devices:
            try:
                success = await self.turn_off_device(device_id)
                results[device_id] = 'success' if success else 'failed'
            except Exception as e:
                results[device_id] = f'error: {e}'
                logger.error(f"Emergency shutdown failed for {device_id}: {e}")
        
        return results
    
    def get_device_config(self, device_id: str) -> Optional[DeviceConfig]:
        """Get device configuration"""
        return self.device_configs.get(device_id)
    
    def list_available_devices(self) -> List[str]:
        """List all available device IDs"""
        return list(self.devices.keys())

# Example usage
async def test_hardware_manager():
    """Test the hardware manager"""
    from config import Config
    
    # Initialize hardware manager with config
    hw_manager = HardwareManager(Config.DEVICES)
    
    print("Testing Hardware Manager...")
    
    # Test turning devices on/off
    test_device = Config.DEVICES[0].device_id
    print(f"\nTesting device: {test_device}")
    
    # Turn on
    success = await hw_manager.turn_on_device(test_device)
    print(f"Turn ON result: {success}")
    
    # Get status
    status = await hw_manager.get_device_status(test_device)
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Turn off
    success = await hw_manager.turn_off_device(test_device)
    print(f"Turn OFF result: {success}")
    
    # Get total power consumption
    total_power = await hw_manager.get_total_power_consumption()
    print(f"Total power consumption: {total_power}W")
    
    # Get all devices status
    all_status = await hw_manager.get_all_devices_status()
    print(f"All devices status: {len(all_status)} devices")

if __name__ == "__main__":
    asyncio.run(test_hardware_manager())
