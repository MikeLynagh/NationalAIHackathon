#!/bin/bash

# Energy Advisor MVP - Setup Script
# This script sets up the development environment

echo "🚀 Setting up Energy Advisor MVP Development Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file from template
if [ ! -f .env ]; then
    echo "🔑 Creating .env file from template..."
    cat > .env << EOF
# Energy Advisor MVP - Environment Variables
# Fill in your actual API keys

# AI API Configuration
# Choose one: OpenAI OR Anthropic (Claude)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# AI Model Selection
AI_MODEL=gpt-4

# API Configuration
AI_MAX_TOKENS=4000
AI_TEMPERATURE=0.3

# Fallback Configuration
ENABLE_AI_FALLBACK=true
AI_TIMEOUT_SECONDS=30

# Development Settings
DEBUG_MODE=false
LOG_LEVEL=INFO

# Data Processing
MAX_FILE_SIZE_MB=50
MAX_ROWS_PROCESS=100000
EOF
    echo "✅ .env file created. Please edit it with your API keys."
else
    echo "✅ .env file already exists."
fi

# Create data directory structure
echo "📁 Setting up data directories..."
mkdir -p data
mkdir -p tests

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run tests: python -m pytest tests/ -v"
echo "4. Start development: python -m streamlit run streamlit_app.py"
echo ""
echo "Happy coding! 🚀" 