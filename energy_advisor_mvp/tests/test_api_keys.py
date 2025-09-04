#!/usr/bin/env python3
"""
Test script to verify OpenAI API key configuration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from src directory
env_path = Path(__file__).parent.parent / "src" / ".env"
load_dotenv(dotenv_path=env_path)

def test_api_keys():
    """Test if API keys are properly configured"""
    print("🔑 API Key Configuration Test")
    print("=" * 40)
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    print(openai_key)
    if openai_key:
        if openai_key.startswith('sk-'):
            print("✅ OpenAI API key found and looks valid")
            print(f"   Key preview: {openai_key[:10]}...{openai_key[-4:]}")
        else:
            print("⚠  OpenAI API key found but may be invalid (should start with 'sk-')")
            print(f"   Key preview: {openai_key[:10]}...")
    else:
        print("❌ OpenAI API key not found")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Check Anthropic API key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        if anthropic_key.startswith('sk-ant-'):
            print("✅ Anthropic API key found and looks valid")
            print(f"   Key preview: {anthropic_key[:10]}...{anthropic_key[-4:]}")
        else:
            print("⚠  Anthropic API key found but may be invalid")
            print(f"   Key preview: {anthropic_key[:10]}...")
    else:
        print("❌ Anthropic API key not found")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        if len(gemini_key) > 10:  # Gemini keys don't have a standard prefix
            print("✅ Gemini API key found")
            print(f"   Key preview: {gemini_key[:10]}...{gemini_key[-4:]}")
        else:
            print("⚠  Gemini API key found but may be invalid (too short)")
            print(f"   Key preview: {gemini_key[:10]}...")
    else:
        print("❌ Gemini API key not found")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
    
    print("\n💡 How to set API keys:")
    print("   1. Environment variable (recommended):")
    print("      export OPENAI_API_KEY='your-key-here'")
    print("      export GEMINI_API_KEY='your-key-here'")
    print("   2. .env file (create in project directory):")
    print("      OPENAI_API_KEY=your-key-here")
    print("      GEMINI_API_KEY=your-key-here")
    print("   3. Add to ~/.zshrc or ~/.bash_profile for persistence")
    
    return openai_key or anthropic_key or gemini_key

async def test_openai_connection():
    """Test actual connection to OpenAI (requires real API key)"""
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\n❌ Cannot test OpenAI connection - no API key found")
        return False
    
    try:
        # Try to import openai
        import openai
        print("\n🧪 Testing OpenAI connection...")
        
        # Create client
        client = openai.OpenAI(api_key=openai_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello' if you can hear me."}],
            max_tokens=10
        )
        
        print("✅ OpenAI connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
        
    except ImportError:
        print("⚠  OpenAI library not installed. Install with: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_gemini_connection():
    """Test actual connection to Google Gemini (requires real API key)"""
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("\n❌ Cannot test Gemini connection - no API key found")
        return False
    
    try:
        # Try to import google.generativeai
        print("\n🧪 Testing Gemini connection...")
        
        # Set the API key as environment variable for the google client
        os.environ['GEMINI_API_KEY'] = gemini_key
        
        # Try the new genai client approach
        try:
            from google import genai
            
            # The client gets the API key from the environment variable GEMINI_API_KEY
            client = genai.Client()
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp", 
                contents="Say 'Hello' if you can hear me."
            )
            
            print("✅ Gemini connection successful!")
            print(f"   Response: {response.text[:100]}...")
            return True
            
        except ImportError:
            # Fallback to older google-generativeai library
            import google.generativeai as genai
            
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content("Say 'Hello' if you can hear me.")
            
            print("✅ Gemini connection successful!")
            print(f"   Response: {response.text[:100]}...")
            return True
        
    except ImportError:
        print("⚠  Google Generative AI library not installed.")
        print("   Install with: pip install google-generativeai")
        print("   Or for new client: pip install google-genai")
        return False
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

async def test_system_with_llm():
    """Test the smart plug system with LLM"""
    try:
        import sys
        import os
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        if src_path not in sys.path:
            sys.path.insert(0,src_path)
            
        from main import EnhancedSmartPlugAgent
        
        print("\n🤖 Testing Smart Plug Agent with LLM...")
        agent = EnhancedSmartPlugAgent()
        
        # Test with LLM enabled
        result = await agent.process_natural_language_command(
            "Turn on the lights", 
            use_llm=True
        )
        
        if result['success']:
            confidence = result.get('llm_confidence', 0)
            print(f"✅ LLM processing successful (confidence: {confidence:.1%})")
        else:
            print(f"⚠  LLM processing failed, using fallback: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Smart Plug Agent - API Key Test")
    print()
    
    # Test API key configuration
    has_keys = test_api_keys()
    
    if has_keys:
        print("\n🔄 Would you like to test the actual API connection? (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                import asyncio
                asyncio.run(test_openai_connection())
                test_gemini_connection()
                asyncio.run(test_system_with_llm())
        except KeyboardInterrupt:
            print("\n\nTest cancelled.")
    else:
        print("\n💡 Add your API keys and run this test again!")
    
    print("\n📚 Getting API Keys:")
    print("   • OpenAI: https://platform.openai.com/api-keys")
    print("   • Anthropic: https://console.anthropic.com/")
    print("   • Google Gemini: https://aistudio.google.com/app/apikey")

if __name__ == "__main__":
    main()