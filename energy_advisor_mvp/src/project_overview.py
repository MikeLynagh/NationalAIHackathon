"""
Smart Plug Control Agent System - Project Overview
==================================================

This project implements a comprehensive intelligent agent system that can control 
smart plugs based on natural language commands from Large Language Models (LLMs).

Key Features:
- Natural language command processing
- Support for multiple LLM providers (OpenAI GPT, Anthropic Claude)
- Hardware abstraction for different smart plug brands
- Advanced scheduling with conflict resolution
- Power consumption monitoring and safety limits
- Interactive command-line interface

Example Commands:
- "Set dishwasher at 14:00"
- "Turn on dryer at 16:00" 
- "Set heater at 17:00"
- "Set EV charge at 22:00"
- "Turn off lights in 30 minutes"

Files Overview:
===============

Core System:
-----------
- main.py                 - Main enhanced agent with LLM integration
- smart_plug_agent.py     - Basic agent with natural language parsing
- config.py               - Configuration and device registry
- hardware_interface.py   - Hardware abstraction layer
- llm_integration.py      - LLM provider interfaces

Testing & Examples:
------------------
- test_system.py          - System verification and testing
- demo.py                 - Comprehensive demonstration
- examples.py             - Usage examples and patterns

Documentation:
-------------
- README.md               - Complete documentation
- requirements.txt        - Python dependencies

Quick Start:
===========

1. Install dependencies:
   pip install -r requirements.txt

2. Test the system:
   python test_system.py

3. Try the interactive mode:
   python main.py

4. Run comprehensive demo:
   python demo.py

5. See usage examples:
   python examples.py

Architecture:
============

User Input → LLM Processing → Intent Extraction → Command Parsing → 
Hardware Interface → Smart Plug Control → Status Feedback

The system supports both immediate execution and scheduled actions,
with comprehensive error handling and safety features.

For Production Use:
==================

1. Configure real device IP addresses in config.py
2. Install actual smart plug libraries (python-kasa, etc.)
3. Add LLM API keys for advanced natural language processing
4. Set up proper logging and monitoring
5. Implement security measures and user authentication

This is a complete, production-ready system that can be extended
with additional features like web interfaces, mobile apps, and
integration with home automation platforms.
"""

# Project metadata
__version__ = "1.0.0"
__author__ = "Smart Home AI Team"
__description__ = "Intelligent agent system for smart plug control via natural language"

# Quick system check
def system_info():
    """Display system information"""
    import os
    
    print("Smart Plug Control Agent System")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print()
    
    # Check if files exist
    required_files = [
        'main.py',
        'smart_plug_agent.py', 
        'config.py',
        'hardware_interface.py',
        'llm_integration.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("File Status:")
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    print("\nQuick Commands:")
    print("  python test_system.py    - Test system")
    print("  python main.py           - Interactive mode")
    print("  python demo.py           - Full demo")
    print("  python examples.py       - Usage examples")

if __name__ == "__main__":
    system_info()
