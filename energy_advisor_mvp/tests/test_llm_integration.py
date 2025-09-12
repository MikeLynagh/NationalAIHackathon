#!/usr/bin/env python3
"""
Test LLM Integration for Smart Plug Agent

This test verifies that:
1. API key detection works correctly
2. LLM vs Direct parsing is used appropriately  
3. Device commands are parsed correctly
4. Fallback mechanisms work
"""

import os
import sys
import asyncio
import unittest
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'src', '.env'))

from smart_plug_agent import SmartPlugAgent

class TestLLMIntegration(unittest.TestCase):
    """Test cases for LLM integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = SmartPlugAgent()
    
    def test_api_key_detection(self):
        """Test that API key detection works correctly with proper prefixes"""
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        # Check for valid keys with proper prefixes
        valid_openai = (openai_key and openai_key.startswith('sk-') and len(openai_key) > 10)
        valid_anthropic = (anthropic_key and 
                         (anthropic_key.startswith('sk-ant-') or 
                          (len(anthropic_key) > 20 and not anthropic_key.startswith('your') and 
                           anthropic_key != 'your_anthropic_key_here')))
        valid_gemini = (gemini_key and gemini_key.startswith('AIza') and len(gemini_key) > 10)
        
        has_any_valid_key = valid_openai or valid_anthropic or valid_gemini
        
        # Agent should enable LLM only if there's a valid key
        self.assertEqual(self.agent.use_llm, has_any_valid_key,
                        f"Agent LLM usage should match API key availability. "
                        f"Valid keys: OpenAI={valid_openai}, Anthropic={valid_anthropic}, Gemini={valid_gemini}")
    
    async def test_dishwasher_commands(self):
        """Test dishwasher commands work with both LLM and direct parsing"""
        test_commands = [
            "turn on dishwasher",
            "turn off dishwasher",
            "set dishwasher at 14:00"
        ]
        
        for command in test_commands:
            with self.subTest(command=command):
                result = await self.agent.process_llm_command(command)
                
                # Command should succeed
                self.assertTrue(result.get('success'), 
                              f"Command '{command}' should succeed")
                
                # Should have a parsing method
                parsing_method = result.get('parsing_method')
                self.assertIn(parsing_method, ['llm', 'direct'],
                            f"Command '{command}' should have valid parsing method")
                
                # Should identify dishwasher device
                device = result.get('device')
                self.assertIn('dishwasher', device.lower(),
                            f"Command '{command}' should identify dishwasher device")
    
    async def test_llm_vs_direct_parsing(self):
        """Test that LLM parsing is used when available, direct when not"""
        command = "turn on lights"
        result = await self.agent.process_llm_command(command)
        
        parsing_method = result.get('parsing_method')
        
        if self.agent.use_llm:
            # If LLM is available, should try LLM first
            # (might fallback to direct if LLM fails)
            self.assertIn(parsing_method, ['llm', 'direct'],
                         "Should use LLM or fallback to direct")
        else:
            # If no LLM, should use direct
            self.assertEqual(parsing_method, 'direct',
                           "Should use direct parsing when no LLM available")
    
    async def test_unknown_commands(self):
        """Test handling of unknown commands"""
        unknown_commands = [
            "make me a sandwich",
            "what's the weather",
            "turn on the spaceship"
        ]
        
        for command in unknown_commands:
            with self.subTest(command=command):
                result = await self.agent.process_llm_command(command)
                
                # Should fail gracefully
                self.assertFalse(result.get('success'),
                               f"Unknown command '{command}' should fail")
                
                # Should have error message
                self.assertIn('error', result,
                            f"Unknown command '{command}' should have error")

async def run_async_tests():
    """Run async tests"""
    print("🧪 Running LLM Integration Tests")
    print("=" * 50)
    
    # Create test instance
    test_instance = TestLLMIntegration()
    test_instance.setUp()
    
    # Test API key detection
    print("1. Testing API key detection...")
    try:
        test_instance.test_api_key_detection()
        print("   ✅ API key detection test passed")
    except Exception as e:
        print(f"   ❌ API key detection test failed: {e}")
    
    # Test dishwasher commands
    print("\n2. Testing dishwasher commands...")
    try:
        await test_instance.test_dishwasher_commands()
        print("   ✅ Dishwasher commands test passed")
    except Exception as e:
        print(f"   ❌ Dishwasher commands test failed: {e}")
    
    # Test LLM vs direct parsing
    print("\n3. Testing LLM vs direct parsing...")
    try:
        await test_instance.test_llm_vs_direct_parsing()
        print("   ✅ LLM vs direct parsing test passed")
    except Exception as e:
        print(f"   ❌ LLM vs direct parsing test failed: {e}")
    
    # Test unknown commands
    print("\n4. Testing unknown commands...")
    try:
        await test_instance.test_unknown_commands()
        print("   ✅ Unknown commands test passed")
    except Exception as e:
        print(f"   ❌ Unknown commands test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🧪 Tests completed!")
    
    # Show agent configuration
    agent = test_instance.agent
    print(f"\n🔧 Agent Configuration:")
    print(f"   LLM Available: {hasattr(agent, 'llm_manager') and agent.llm_manager is not None}")
    print(f"   Use LLM: {getattr(agent, 'use_llm', False)}")
    
    # Show API key status
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    print("\n🔑 API Key Status:")
    
    # Use same validation logic as the agent
    valid_openai = (openai_key and openai_key.startswith('sk-') and len(openai_key) > 10)
    valid_anthropic = (anthropic_key and 
                     (anthropic_key.startswith('sk-ant-') or 
                      (len(anthropic_key) > 20 and not anthropic_key.startswith('your') and 
                       anthropic_key != 'your_anthropic_key_here')))
    valid_gemini = (gemini_key and gemini_key.startswith('AIza') and len(gemini_key) > 10)
    
    print(f"   OpenAI: {'✅ Valid (sk-)' if valid_openai else '❌ Invalid/Missing (needs sk- prefix)'}")
    print(f"   Anthropic: {'✅ Valid (sk-ant- or proper key)' if valid_anthropic else '❌ Invalid/Missing (needs sk-ant- prefix or proper key)'}")
    print(f"   Gemini: {'✅ Valid (AIza prefix)' if valid_gemini else '❌ Invalid/Missing (needs AIza prefix)'}")

if __name__ == "__main__":
    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
