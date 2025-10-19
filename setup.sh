#!/bin/bash

# Setup script for the Intelligent RAG Q&A System

echo "=========================================="
echo "Intelligent RAG Q&A System Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create .env file from example
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please edit it with your configuration."
else
    echo ""
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data
mkdir -p logs
mkdir -p test_results
mkdir -p chroma_db

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your LLM provider configuration"
echo "2. Place your PDF knowledge base in data/knowledge_base.pdf"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run the system: python src/qa_system.py"
echo ""
echo "For Ollama (recommended for local/free):"
echo "  - Install Ollama from https://ollama.ai"
echo "  - Run: ollama pull llama3.2"
echo "  - Set LLM_PROVIDER=ollama in .env"
echo ""
echo "For Groq (recommended for cloud/free):"
echo "  - Get API key from https://console.groq.com"
echo "  - Set GROQ_API_KEY in .env"
echo "  - Set LLM_PROVIDER=groq in .env"
echo ""
