"""
LLM Integration module for Smart Plug Agent System
Supports OpenAI GPT and Anthropic Claude for natural language processing
"""

import os
import json
import asyncio
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from dotenv import load_dotenv
load_dotenv()

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM processing"""
    success: bool
    intent: str
    device: Optional[str] = None
    action: Optional[str] = None
    time: Optional[str] = None
    confidence: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def process_command(self, user_input: str, context: Dict = None) -> LLMResponse:
        """Process user command and extract intent"""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for natural language processing"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Using mock responses.")
        else:
            try:
                # In a real implementation, you would import openai here
                # import openai
                # self.client = openai.AsyncOpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized (mock)")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    async def process_command(self, user_input: str, context: Dict = None) -> LLMResponse:
        """Process user command using OpenAI GPT"""
        try:
            # Create prompt for intent extraction
            prompt = self._create_intent_prompt(user_input, context)
            
            if self.client:
                # Real implementation would be:
                # response = await self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[{"role": "user", "content": prompt}],
                #     temperature=0.1,
                #     max_tokens=200
                # )
                # response_text = response.choices[0].message.content
                
                # Mock response for demo
                response_text = self._mock_openai_response(user_input)
            else:
                response_text = self._mock_openai_response(user_input)
            
            # Parse the structured response
            return self._parse_llm_response(response_text, user_input)
            
        except Exception as e:
            logger.error(f"OpenAI processing error: {e}")
            return LLMResponse(
                success=False,
                intent="error",
                error=str(e),
                raw_response=""
            )
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using OpenAI GPT"""
        try:
            if self.client:
                # Real implementation
                pass
            else:
                # Mock response
                return "This is a mock response from OpenAI GPT."
                
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error generating response: {e}"
    
    def _create_intent_prompt(self, user_input: str, context: Dict = None) -> str:
        """Create prompt for intent extraction"""
        available_devices = [
            "dishwasher", "dryer", "heater", "ev charger", "washing machine",
            "coffee maker", "air conditioner", "lights", "fan", "tv", "microwave"
        ]
        
        prompt = f"""
You are a smart home assistant that controls smart plugs. Analyze the user's command and extract:
1. Intent (turn_on, turn_off, schedule, status, unknown)
2. Device name (from available devices)
3. Action (on, off, schedule)
4. Time (if scheduling)

Available devices: {', '.join(available_devices)}

User command: "{user_input}"

Respond in JSON format:
{{
    "intent": "turn_on|turn_off|schedule|status|unknown",
    "device": "device_name_or_null",
    "action": "on|off|schedule|null", 
    "time": "time_string_or_null",
    "confidence": 0.0-1.0
}}

Examples:
- "Turn on the lights" → {{"intent": "turn_on", "device": "lights", "action": "on", "time": null, "confidence": 0.95}}
- "Set dishwasher at 2 PM" → {{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": "2 PM", "confidence": 0.90}}
"""
        return prompt
    
    def _mock_openai_response(self, user_input: str) -> str:
        """Mock OpenAI response for demo purposes"""
        user_input = user_input.lower()
        
        # Simple pattern matching for demo
        if "turn on" in user_input or "switch on" in user_input:
            if "lights" in user_input:
                return '{"intent": "turn_on", "device": "lights", "action": "on", "time": null, "confidence": 0.95}'
            elif "dishwasher" in user_input:
                return '{"intent": "turn_on", "device": "dishwasher", "action": "on", "time": null, "confidence": 0.95}'
            elif "heater" in user_input:
                return '{"intent": "turn_on", "device": "heater", "action": "on", "time": null, "confidence": 0.95}'
        
        elif "turn off" in user_input or "switch off" in user_input:
            if "lights" in user_input:
                return '{"intent": "turn_off", "device": "lights", "action": "off", "time": null, "confidence": 0.95}'
            elif "heater" in user_input:
                return '{"intent": "turn_off", "device": "heater", "action": "off", "time": null, "confidence": 0.95}'
        
        elif "set" in user_input or "schedule" in user_input:
            if "dishwasher" in user_input and ("14:00" in user_input or "2 pm" in user_input):
                return '{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": "14:00", "confidence": 0.90}'
            elif "dryer" in user_input and ("16:00" in user_input or "4 pm" in user_input):
                return '{"intent": "schedule", "device": "dryer", "action": "schedule", "time": "16:00", "confidence": 0.90}'
            elif "ev" in user_input and ("22:00" in user_input or "10 pm" in user_input):
                return '{"intent": "schedule", "device": "ev charger", "action": "schedule", "time": "22:00", "confidence": 0.90}'
        
        # Default unknown response
        return '{"intent": "unknown", "device": null, "action": null, "time": null, "confidence": 0.1}'
    
    def _parse_llm_response(self, response_text: str, original_input: str) -> LLMResponse:
        """Parse LLM JSON response"""
        try:
            data = json.loads(response_text)
            return LLMResponse(
                success=True,
                intent=data.get('intent', 'unknown'),
                device=data.get('device'),
                action=data.get('action'),
                time=data.get('time'),
                confidence=data.get('confidence', 0.0),
                raw_response=response_text
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return LLMResponse(
                success=False,
                intent="error",
                error=f"JSON parsing error: {e}",
                raw_response=response_text
            )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for natural language processing"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.client = None
        
        if not self.api_key:
            logger.warning("Anthropic API key not found. Using mock responses.")
        else:
            try:
                # In a real implementation:
                # import anthropic
                # self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized (mock)")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    async def process_command(self, user_input: str, context: Dict = None) -> LLMResponse:
        """Process user command using Anthropic Claude"""
        try:
            prompt = self._create_intent_prompt(user_input, context)
            
            if self.client:
                # Real implementation would be:
                # response = await self.client.messages.create(
                #     model=self.model,
                #     max_tokens=200,
                #     temperature=0.1,
                #     messages=[{"role": "user", "content": prompt}]
                # )
                # response_text = response.content[0].text
                
                # Mock response for demo
                response_text = self._mock_claude_response(user_input)
            else:
                response_text = self._mock_claude_response(user_input)
            
            return self._parse_llm_response(response_text, user_input)
            
        except Exception as e:
            logger.error(f"Anthropic processing error: {e}")
            return LLMResponse(
                success=False,
                intent="error",
                error=str(e),
                raw_response=""
            )
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Anthropic Claude"""
        return "This is a mock response from Anthropic Claude."
    
    def _create_intent_prompt(self, user_input: str, context: Dict = None) -> str:
        """Create prompt for intent extraction"""
        return f"""
I need you to analyze smart home commands and extract structured information.

User said: "{user_input}"

Extract these elements:
- Intent: What does the user want to do? (turn_on, turn_off, schedule, status, unknown)
- Device: Which device? (dishwasher, dryer, heater, lights, etc.)
- Action: What action? (on, off, schedule)
- Time: When to execute? (if scheduling)
- Confidence: How confident are you? (0.0 to 1.0)

Respond only with JSON in this format:
{{"intent": "...", "device": "...", "action": "...", "time": "...", "confidence": 0.0}}
"""
    
    def _mock_claude_response(self, user_input: str) -> str:
        """Mock Claude response - similar to OpenAI but with slightly different patterns"""
        user_input = user_input.lower()
        
        # Claude might be better at understanding context
        if any(word in user_input for word in ["activate", "start", "power on"]):
            if "coffee" in user_input:
                return '{"intent": "turn_on", "device": "coffee maker", "action": "on", "time": null, "confidence": 0.98}'
            elif "ac" in user_input or "air conditioning" in user_input:
                return '{"intent": "turn_on", "device": "air conditioner", "action": "on", "time": null, "confidence": 0.97}'
        
        elif "in" in user_input and ("minutes" in user_input or "hours" in user_input):
            if "30 minutes" in user_input:
                return '{"intent": "schedule", "device": "heater", "action": "off", "time": "in 30 minutes", "confidence": 0.85}'
            elif "2 hours" in user_input:
                return '{"intent": "schedule", "device": "tv", "action": "off", "time": "in 2 hours", "confidence": 0.85}'
        
        # Fall back to similar logic as OpenAI
        return self._mock_openai_response_fallback(user_input)
    
    def _mock_openai_response_fallback(self, user_input: str) -> str:
        """Fallback to OpenAI-style response"""
        if "dishwasher" in user_input and "14:00" in user_input:
            return '{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": "14:00", "confidence": 0.92}'
        elif "dryer" in user_input and "16:00" in user_input:
            return '{"intent": "schedule", "device": "dryer", "action": "schedule", "time": "16:00", "confidence": 0.92}'
        else:
            return '{"intent": "unknown", "device": null, "action": null, "time": null, "confidence": 0.2}'
    
    def _parse_llm_response(self, response_text: str, original_input: str) -> LLMResponse:
        """Parse LLM JSON response - same as OpenAI"""
        try:
            data = json.loads(response_text)
            return LLMResponse(
                success=True,
                intent=data.get('intent', 'unknown'),
                device=data.get('device'),
                action=data.get('action'),
                time=data.get('time'),
                confidence=data.get('confidence', 0.0),
                raw_response=response_text
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return LLMResponse(
                success=False,
                intent="error",
                error=f"JSON parsing error: {e}",
                raw_response=response_text
            )

class LLMManager:
    """Manages multiple LLM providers and provides unified interface"""
    
    def __init__(self, primary_provider: str = "openai"):
        self.providers = {}
        self.primary_provider = primary_provider
        
        # Initialize providers
        self.providers['openai'] = OpenAIProvider()
        self.providers['anthropic'] = AnthropicProvider()
        
        logger.info(f"LLM Manager initialized with primary provider: {primary_provider}")
    
    async def process_command(self, user_input: str, provider: str = None) -> LLMResponse:
        """Process command using specified or primary provider"""
        provider_name = provider or self.primary_provider
        
        if provider_name not in self.providers:
            logger.error(f"Provider '{provider_name}' not available")
            return LLMResponse(
                success=False,
                intent="error",
                error=f"Provider '{provider_name}' not available"
            )
        
        try:
            response = await self.providers[provider_name].process_command(user_input)
            logger.info(f"Processed command with {provider_name}: {response.intent} (confidence: {response.confidence})")
            return response
        except Exception as e:
            logger.error(f"Error processing command with {provider_name}: {e}")
            return LLMResponse(
                success=False,
                intent="error",
                error=str(e)
            )
    
    async def get_best_response(self, user_input: str) -> LLMResponse:
        """Get response from all providers and return the best one"""
        responses = []
        
        for provider_name in self.providers:
            try:
                response = await self.providers[provider_name].process_command(user_input)
                if response.success:
                    responses.append((provider_name, response))
            except Exception as e:
                logger.error(f"Error with provider {provider_name}: {e}")
        
        if not responses:
            return LLMResponse(
                success=False,
                intent="error",
                error="All providers failed"
            )
        
        # Return response with highest confidence
        best_provider, best_response = max(responses, key=lambda x: x[1].confidence)
        logger.info(f"Best response from {best_provider} with confidence {best_response.confidence}")
        return best_response
    
    def switch_primary_provider(self, provider: str):
        """Switch primary provider"""
        if provider in self.providers:
            self.primary_provider = provider
            logger.info(f"Switched primary provider to {provider}")
        else:
            logger.error(f"Provider '{provider}' not available")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())

# Example usage and testing
async def test_llm_integration():
    """Test LLM integration"""
    llm_manager = LLMManager()
    
    test_commands = [
        "Turn on the lights",
        "Set dishwasher at 14:00",
        "Turn off heater in 30 minutes",
        "Start the coffee maker",
        "Schedule dryer at 4 PM",
        "Unknown device command"
    ]
    
    print("Testing LLM Integration...")
    print("=" * 50)
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        
        # Test with primary provider
        response = await llm_manager.process_command(command)
        print(f"  Intent: {response.intent}")
        print(f"  Device: {response.device}")
        print(f"  Action: {response.action}")
        print(f"  Time: {response.time}")
        print(f"  Confidence: {response.confidence}")
        
        if not response.success:
            print(f"  Error: {response.error}")

if __name__ == "__main__":
    asyncio.run(test_llm_integration())
