"""
LLM Integration module for Smart Plug Agent System
Supports OpenAI GPT and Anthropic Claude for natural language processing
"""

import os
import json
import asyncio
import requests
import socket
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

def check_network_connectivity(host="8.8.8.8", port=53, timeout=3):
    """Check if network connectivity is available"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def check_dns_resolution(hostname="google.com", timeout=3):
    """Check if DNS resolution is working"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        return True
    except socket.gaierror:
        return False

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

class GeminiProvider(LLMProvider):
    """Google Gemini provider for natural language processing"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model
        self.client = None
        self.connection_timeout = 10  # seconds
        self.max_retries = 3
        
        if not self.api_key:
            logger.warning("Gemini API key not found. Using mock responses.")
        else:
            # Check network connectivity first
            if not check_network_connectivity():
                logger.warning("No network connectivity detected. Gemini will use mock responses.")
                self.client = None
            elif not check_dns_resolution():
                logger.warning("DNS resolution issues detected. Gemini will use mock responses.")
                self.client = None
            else:
                try:
                    # Import and initialize Gemini client
                    from google import genai
                    os.environ['GEMINI_API_KEY'] = self.api_key
                    self.client = genai.Client()
                    logger.info("Gemini client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    self.client = None
    
    async def process_command(self, user_input: str, context: Dict = None) -> LLMResponse:
        """Process user command using Google Gemini with network resilience"""
        try:
            # Create prompt for intent extraction
            prompt = self._create_intent_prompt(user_input, context)
            
            if self.client:
                # Try real Gemini API call with retries
                response_text = await self._call_gemini_with_retries(prompt)
                if response_text is None:
                    logger.warning("Gemini API call failed, falling back to mock response")
                    response_text = self._mock_gemini_response(user_input)
            else:
                # Fallback to mock response
                response_text = self._mock_gemini_response(user_input)
            
            # Parse the structured response
            return self._parse_llm_response(response_text, user_input)
            
        except Exception as e:
            logger.error(f"Gemini processing error: {e}")
            # Return a fallback response based on simple pattern matching
            return self._create_fallback_response(user_input, str(e))
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Google Gemini with network resilience"""
        try:
            if self.client:
                # Try real Gemini API call with retries
                response_text = await self._call_gemini_with_retries(prompt)
                if response_text is not None:
                    return response_text
                else:
                    logger.warning("Gemini API call failed, using fallback response")
                    return "I'm having trouble connecting to the AI service. Using local processing instead."
            else:
                # Mock response
                return "This is a mock response from Google Gemini."
                
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Unable to generate response due to network issues: {e}"
    
    def _create_intent_prompt(self, user_input: str, context: Dict = None) -> str:
        """Create prompt for intent extraction - same format as other providers"""
        devices = context.get('devices', []) if context else []
        device_list = ', '.join(devices) if devices else "dishwasher, dryer, heater, ev_charger, washer, coffee_maker, ac, lights, fan, tv, microwave"
        
        prompt = f"""Analyze this smart home command and extract the intent as JSON:

Command: "{user_input}"
Available devices: {device_list}

Return ONLY a JSON object with these fields:
- intent: "turn_on", "turn_off", "schedule", "status", "unknown"
- device: exact device name from available devices (or null)
- action: "on", "off", "schedule", "status" (or null)
- time: extracted time in HH:MM format (or null)
- confidence: confidence score 0.0-1.0

Examples:
"Turn on the lights" → {{"intent": "turn_on", "device": "lights", "action": "on", "time": null, "confidence": 0.95}}
"Schedule dishwasher at 2pm" → {{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": "14:00", "confidence": 0.92}}
"Turn off heater" → {{"intent": "turn_off", "device": "heater", "action": "off", "time": null, "confidence": 0.93}}

Respond with JSON only, no explanations."""

        return prompt
    
    async def _call_gemini_with_retries(self, prompt: str) -> Optional[str]:
        """Call Gemini API with retry logic for network resilience"""
        import asyncio
        import socket
        
        for attempt in range(self.max_retries):
            try:
                # Use timeout to prevent hanging
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._make_gemini_call, prompt),
                    timeout=self.connection_timeout
                )
                return response
                
            except (socket.gaierror, ConnectionError, OSError) as e:
                # Network connectivity issues
                if "getaddrinfo failed" in str(e) or isinstance(e, socket.gaierror):
                    logger.warning(f"Network connectivity issue (attempt {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed. Using fallback processing.")
                    return None
            
            except Exception as e:
                # Handle SSL and other API errors
                error_str = str(e).lower()
                if any(ssl_error in error_str for ssl_error in ["ssl", "certificate", "tls"]):
                    logger.warning(f"SSL/Certificate error (attempt {attempt + 1}/{self.max_retries}): {e}")
                elif "getaddrinfo failed" in error_str:
                    logger.warning(f"DNS resolution error (attempt {attempt + 1}/{self.max_retries}): {e}")
                else:
                    logger.warning(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed. Using fallback processing.")
                    return None
                    
            except asyncio.TimeoutError:
                logger.warning(f"API call timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("API calls timed out. Using fallback processing.")
                    return None
        
        return None
    
    def _make_gemini_call(self, prompt: str) -> str:
        """Make the actual Gemini API call (synchronous)"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()
    
    def _create_fallback_response(self, user_input: str, error: str) -> LLMResponse:
        """Create a fallback response when network fails"""
        # Use simple pattern matching for basic commands
        user_input_lower = user_input.lower()
        
        # Handle "stop all" or similar commands
        if any(phrase in user_input_lower for phrase in ["stop all", "turn off all", "shut down all"]):
            return LLMResponse(
                success=True,
                intent="turn_off",
                device="all",
                action="off",
                confidence=0.8,
                raw_response=f"Network error occurred ({error}), using fallback processing"
            )
        
        # Use the mock response as fallback
        response_text = self._mock_gemini_response(user_input)
        result = self._parse_llm_response(response_text, user_input)
        result.error = f"Network connectivity issue: {error}"
        result.confidence = max(0.1, result.confidence - 0.3)  # Lower confidence for fallback
        return result
    
    def _mock_gemini_response(self, user_input: str) -> str:
        """Mock response for when API is not available - enhanced with more patterns"""
        user_input = user_input.lower()
        
        # Handle "stop all" or "turn off all" commands
        if any(phrase in user_input for phrase in ["stop all", "turn off all", "shut down all", "off all"]):
            return '{"intent": "turn_off", "device": "all", "action": "off", "time": null, "confidence": 0.9}'
        
        if "turn on" in user_input or "switch on" in user_input or "start" in user_input:
            if "lights" in user_input:
                return '{"intent": "turn_on", "device": "lights", "action": "on", "time": null, "confidence": 0.95}'
            elif "dishwasher" in user_input:
                return '{"intent": "turn_on", "device": "dishwasher", "action": "on", "time": null, "confidence": 0.95}'
            elif "coffee" in user_input:
                return '{"intent": "turn_on", "device": "coffee maker", "action": "on", "time": null, "confidence": 0.94}'
            elif "heater" in user_input:
                return '{"intent": "turn_on", "device": "heater", "action": "on", "time": null, "confidence": 0.93}'
            elif "dryer" in user_input:
                return '{"intent": "turn_on", "device": "dryer", "action": "on", "time": null, "confidence": 0.93}'
            elif "ev" in user_input or "charger" in user_input:
                return '{"intent": "turn_on", "device": "ev charger", "action": "on", "time": null, "confidence": 0.93}'
        elif "turn off" in user_input or "switch off" in user_input or "stop" in user_input:
            if "lights" in user_input:
                return '{"intent": "turn_off", "device": "lights", "action": "off", "time": null, "confidence": 0.95}'
            elif "dishwasher" in user_input:
                return '{"intent": "turn_off", "device": "dishwasher", "action": "off", "time": null, "confidence": 0.95}'
            elif "heater" in user_input:
                return '{"intent": "turn_off", "device": "heater", "action": "off", "time": null, "confidence": 0.93}'
            elif "dryer" in user_input:
                return '{"intent": "turn_off", "device": "dryer", "action": "off", "time": null, "confidence": 0.93}'
        elif "set" in user_input or "schedule" in user_input:
            if "dishwasher" in user_input:
                if any(time_phrase in user_input for time_phrase in ["14:00", "2pm", "2 pm"]):
                    return '{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": "14:00", "confidence": 0.92}'
                else:
                    return '{"intent": "schedule", "device": "dishwasher", "action": "schedule", "time": null, "confidence": 0.85}'
            elif "coffee" in user_input:
                if any(time_phrase in user_input for time_phrase in ["7:00", "7am", "7 am"]):
                    return '{"intent": "schedule", "device": "coffee maker", "action": "schedule", "time": "7:00", "confidence": 0.92}'
                else:
                    return '{"intent": "schedule", "device": "coffee maker", "action": "schedule", "time": null, "confidence": 0.85}'
            elif "dryer" in user_input:
                if any(time_phrase in user_input for time_phrase in ["16:00", "4pm", "4 pm"]):
                    return '{"intent": "schedule", "device": "dryer", "action": "schedule", "time": "16:00", "confidence": 0.92}'
                else:
                    return '{"intent": "schedule", "device": "dryer", "action": "schedule", "time": null, "confidence": 0.85}'
            elif "heater" in user_input:
                if any(time_phrase in user_input for time_phrase in ["18:00", "6pm", "6 pm"]):
                    return '{"intent": "schedule", "device": "heater", "action": "schedule", "time": "18:00", "confidence": 0.92}'
                else:
                    return '{"intent": "schedule", "device": "heater", "action": "schedule", "time": null, "confidence": 0.85}'
        
        return '{"intent": "unknown", "device": null, "action": null, "time": null, "confidence": 0.2}'
    
    def _parse_llm_response(self, response_text: str, original_input: str) -> LLMResponse:
        """Parse LLM JSON response - same as other providers"""
        try:
            # Clean the response text - remove code blocks if present
            clean_response = response_text.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            data = json.loads(clean_response)
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
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Raw response: {response_text}")
            return LLMResponse(
                success=False,
                intent="error",
                error=f"JSON parsing error: {e}",
                raw_response=response_text
            )

class LLMManager:
    """Manages multiple LLM providers and provides unified interface"""
    
    def __init__(self, primary_provider: str = "gemini"):
        self.providers = {}
        self.primary_provider = primary_provider
        
        # Initialize providers
        self.providers['openai'] = OpenAIProvider()
        self.providers['anthropic'] = AnthropicProvider()
        self.providers['gemini'] = GeminiProvider()
        
        logger.info(f"LLM Manager initialized with primary provider: {primary_provider}")
        
        # Log which providers have valid API keys with proper prefix validation
        for name, provider in self.providers.items():
            has_valid_key = self._validate_api_key(name, provider)
            if has_valid_key:
                logger.info(f"✅ {name.title()} provider ready with API key")
            else:
                logger.warning(f"⚠️  {name.title()} provider using mock responses (no API key)")
        
        # Verify primary provider is available
        if primary_provider not in self.providers:
            logger.warning(f"Primary provider '{primary_provider}' not found, falling back to first available")
            self.primary_provider = list(self.providers.keys())[0]
    
    def _validate_api_key(self, provider_name: str, provider) -> bool:
        """Validate API key format for each provider"""
        if not hasattr(provider, 'api_key') or not provider.api_key:
            return False
        
        api_key = provider.api_key.strip()
        
        if provider_name == 'openai':
            # OpenAI keys start with sk- (historical) or sk-proj- (newer project keys)
            return api_key.startswith('sk-') and len(api_key) > 10
        elif provider_name == 'anthropic':
            # Anthropic keys start with sk-ant- (admin keys documented)
            # For regular keys, we'll accept any non-placeholder key that's substantial
            return (api_key.startswith('sk-ant-') or 
                   (len(api_key) > 20 and not api_key.startswith('your') and api_key != 'your_anthropic_key_here'))
        elif provider_name == 'gemini':
            # Google/Gemini API keys commonly start with AIza
            return api_key.startswith('AIza') and len(api_key) > 10
        else:
            # For unknown providers, just check it's not a placeholder
            return (len(api_key) > 10 and 
                   not api_key.startswith('your') and 
                   api_key not in ['your_openai_key_here', 'your_anthropic_key_here', 'your_gemini_key_here'])
    
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
