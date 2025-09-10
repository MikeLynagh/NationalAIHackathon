# Smart Plug Control Agent System

A comprehensive intelligent agent system that controls smart plugs based on natural language commands from Large Language Models (LLMs).

## Features

- **Natural Language Processing**: Understands commands like "Set dishwasher at 14:00" or "Turn on heater in 30 minutes"
- **Multiple LLM Support**: Integrates with OpenAI GPT and Anthropic Claude
- **Hardware Abstraction**: Supports TP-Link Kasa, Amazon Smart Plug, Wyze, and other brands
- **Smart Scheduling**: Advanced scheduling with timezone handling and conflict resolution
- **Safety Features**: Power consumption monitoring and safety limits
- **Interactive Interface**: Command-line interface for testing and demonstration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  LLM Provider   │───▶│ Intent Parsing  │
│ "Set dryer 2PM" │    │ (GPT/Claude)    │    │ Extract: device │
└─────────────────┘    └─────────────────┘    │ action, time    │
                                               └─────────┬───────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼───────┐
│ Hardware Layer  │◀───│   Scheduler     │◀───│ Command Parser  │
│ (Kasa/Amazon)   │    │ (Time-based)    │    │ Device Mapping  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Natural Language Parser (`smart_plug_agent.py`)
- Extracts device names, actions, and timing from natural language
- Supports multiple time formats (24-hour, 12-hour, relative)
- Handles device aliases and fuzzy matching

### 2. LLM Integration (`llm_integration.py`)
- OpenAI GPT and Anthropic Claude support
- Structured intent extraction with confidence scoring
- Fallback to simple parsing if LLM fails

### 3. Hardware Interface (`hardware_interface.py`)
- Abstract interface for different smart plug brands
- TP-Link Kasa, Amazon Smart Plug, Wyze support
- Mock implementations for testing

### 4. Smart Scheduler (`smart_plug_agent.py`)
- Time-based action scheduling
- Job management and cancellation
- Conflict resolution and retry logic

### 5. Configuration Management (`config.py`)
- Device registry and settings
- Power consumption limits
- Customizable device aliases

### 6. Main Agent (`main.py`)
- Orchestrates all components
- Comprehensive status reporting
- Emergency shutdown capabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-plug-agent
```

2. Install dependencies:
```bash
pip install asyncio python-kasa openai anthropic
```

3. Configure devices in `config.py`:
```python
DEVICES = [
    DeviceConfig(
        device_id="kitchen_dishwasher",
        friendly_name="dishwasher",
        ip_address="192.168.1.100",
        device_type="kasa",
        location="kitchen"
    ),
    # Add your devices...
]
```

4. Set up API keys (optional for demo):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Demo Mode
```bash
python demo.py
```

### Python API
```python
from main import EnhancedSmartPlugAgent

agent = EnhancedSmartPlugAgent()
result = await agent.process_natural_language_command("Turn on lights at 7 PM")
```

## Example Commands

| Command | Device | Action | Time |
|---------|--------|--------|------|
| "Set dishwasher at 14:00" | dishwasher | schedule | 14:00 today |
| "Turn on dryer at 16:00" | dryer | schedule | 16:00 today |
| "Set heater at 17:00" | heater | schedule | 17:00 today |
| "Set EV charge at 22:00" | ev_charger | schedule | 22:00 today |
| "Turn off heater in 30 minutes" | heater | schedule | +30 min |
| "Activate coffee maker at 7 AM" | coffee_maker | schedule | 07:00 tomorrow |

## Configuration

### Device Configuration
```python
@dataclass
class DeviceConfig:
    device_id: str          # Unique identifier
    friendly_name: str      # Name for voice commands
    ip_address: str         # Network address
    device_type: str        # kasa, amazon, wyze
    location: str           # kitchen, bedroom, etc.
    max_power: int = 1500   # Maximum watts
```

### Safety Limits
```python
MAX_POWER_TOTAL = 15000      # Total max power (watts)
POWER_SAFETY_MARGIN = 0.8   # Use 80% as safety limit
```

## Hardware Integration

### TP-Link Kasa
```python
# Real implementation (commented in code):
from kasa import SmartPlug
plug = SmartPlug("192.168.1.100")
await plug.update()
await plug.turn_on()
```

### Amazon Smart Plug
```python
# Requires Alexa Voice Service API
# or third-party integration
```

### Wyze Smart Plug
```python
# Requires Wyze API or python-wyze library
```

## LLM Integration

### OpenAI GPT
```python
# Set environment variable:
export OPENAI_API_KEY="your-key"

# Or in code:
agent = EnhancedSmartPlugAgent(llm_provider="openai")
```

### Anthropic Claude
```python
# Set environment variable:
export ANTHROPIC_API_KEY="your-key"

# Or in code:
agent = EnhancedSmartPlugAgent(llm_provider="anthropic")
```

## Advanced Features

### Power Management
- Monitors total power consumption
- Prevents exceeding safety limits
- Provides energy usage recommendations

### Smart Scheduling
- Handles timezone conversions
- Conflict detection and resolution
- Automatic retry on failures

### Device Grouping
- Group devices by location
- Bulk operations by room
- Scene-based control

### Status Monitoring
```python
status = await agent.get_comprehensive_status()
print(f"Active devices: {status['summary']['active_devices']}")
print(f"Power usage: {status['summary']['total_power_consumption']}W")
```

### Emergency Features
```python
# Emergency shutdown all devices
result = await agent.emergency_shutdown()
```

## Development

### Adding New Device Types
1. Create new class inheriting from `SmartPlugInterface`
2. Implement required methods: `turn_on()`, `turn_off()`, `get_status()`
3. Add to `SmartPlugFactory`

### Adding New LLM Providers
1. Create new class inheriting from `LLMProvider`
2. Implement `process_command()` and `generate_response()`
3. Add to `LLMManager`

### Testing
```bash
# Run comprehensive demo
python demo.py

# Test individual components
python hardware_interface.py
python llm_integration.py
```

## Security Considerations

- Input validation and sanitization
- Rate limiting for API calls
- Secure device communication
- Audit logging for all actions
- User authentication (for web interface)

## Troubleshooting

### Common Issues

1. **Device not responding**
   - Check IP address and network connectivity
   - Verify device is powered on
   - Check firewall settings

2. **LLM parsing errors**
   - Verify API keys are set correctly
   - Check internet connectivity
   - Review command syntax

3. **Scheduling failures**
   - Verify system time is correct
   - Check for conflicting schedules
   - Review power consumption limits

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Web-based dashboard
- [ ] Mobile app integration
- [ ] Voice control (Alexa/Google)
- [ ] Machine learning for usage patterns
- [ ] Integration with weather APIs
- [ ] Energy cost optimization
- [ ] Home automation scenes
- [ ] Multi-home support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the example configurations
