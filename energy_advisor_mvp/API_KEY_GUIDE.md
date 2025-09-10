# API Key Configuration Guide

## 🔑 How to Add Your OpenAI API Key

### Method 1: Environment Variable (Recommended)

#### For current session only:
```bash
export OPENAI_API_KEY='sk-your-actual-api-key-here'
```

#### For permanent setup (add to your shell profile):
```bash
# For zsh (macOS default)
echo 'export OPENAI_API_KEY="sk-your-actual-api-key-here"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export OPENAI_API_KEY="sk-your-actual-api-key-here"' >> ~/.bash_profile
source ~/.bash_profile
```

### Method 2: .env File (Project-specific)

1. Edit the `.env` file in the project directory:
```bash
# Open the .env file
nano .env
```

2. Replace the placeholder with your actual key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

### Method 3: Direct Code Configuration (Not Recommended)

In `config.py`, uncomment and add your key:
```python
# API Keys (fallback - NOT recommended for production)
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```

## 🔍 Getting Your API Keys

### OpenAI API Key:
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. **Important**: Save it immediately - you won't be able to see it again!

### Anthropic API Key (optional):
1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Navigate to API keys section
4. Create a new key
5. Copy the key (starts with `sk-ant-`)

## 🧪 Testing Your Configuration

Run the test script to verify your setup:
```bash
python test_api_keys.py
```

This will:
- ✅ Check if API keys are found
- ✅ Validate key format
- 🧪 Test actual API connection (optional)
- 🤖 Test the smart plug system with LLM

## 🚀 Using the System with API Keys

Once configured, you can use advanced LLM features:

```python
from main import EnhancedSmartPlugAgent

# Initialize agent
agent = EnhancedSmartPlugAgent()

# Process commands with LLM understanding
result = await agent.process_natural_language_command(
    "Can you turn on the coffee maker for my morning routine?",
    use_llm=True  # Enable LLM processing
)
```

## 🔒 Security Best Practices

1. **Never commit API keys to git**:
   ```bash
   # Add .env to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables for production**

3. **Rotate keys regularly**

4. **Set usage limits** in your OpenAI/Anthropic dashboard

5. **Monitor usage** to detect unexpected charges

## 🛠️ Troubleshooting

### "API key not found" error:
```bash
# Check if environment variable is set
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY='your-key-here'
```

### "Invalid API key" error:
- Verify the key starts with `sk-` for OpenAI
- Check for extra spaces or quotes
- Ensure the key hasn't been revoked

### "Rate limit exceeded":
- You've hit API usage limits
- Wait or upgrade your OpenAI plan
- Check your usage dashboard

### "Module not found" errors:
```bash
# Install required packages
pip install openai anthropic python-dotenv
```

## 📊 System Behavior

### With API Keys:
- ✅ Advanced natural language understanding
- ✅ Higher confidence in command parsing
- ✅ Better handling of complex/ambiguous commands
- ✅ Context-aware responses

### Without API Keys:
- ✅ Basic pattern matching still works
- ✅ All core functionality available
- ⚠️ Limited to simple command formats
- ⚠️ Lower confidence in complex parsing

## 💡 Example Commands

### With LLM (requires API key):
- "Can you help me set up the dishwasher for when I get home at 6 PM?"
- "I want the coffee ready for my morning routine tomorrow"
- "Make sure the heater is off before I go to bed"

### Without LLM (pattern matching):
- "Set dishwasher at 18:00"
- "Turn on coffee maker at 7:00 AM"
- "Turn off heater in 2 hours"

Both approaches work, but LLM provides more natural interaction!
