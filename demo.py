#!/usr/bin/env python3
"""Demo script for the Intelligent RAG Q&A system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.qa_system import QASystem
from src.utils.logger import app_logger


def print_header():
    """Print demo header."""
    print("\n" + "=" * 80)
    print(" " * 20 + "INTELLIGENT RAG Q&A DEMO")
    print(" " * 15 + "with Dynamic Web Fallback")
    print("=" * 80 + "\n")


def run_demo_questions(qa_system: QASystem, auto_continue: bool = False):
    """Run predefined demo questions.
    
    Args:
        qa_system: The QA system instance
        auto_continue: If True, skip input prompts between questions
    """
    demo_questions = [
        {
            "question": "What is the main topic of the document?",
            "description": "Testing RAG with knowledge base",
            "expected": "knowledge_base"
        },
        {
            "question": "What is the current weather in Tokyo?",
            "description": "Testing web fallback (current information)",
            "expected": "web"
        },
        {
            "question": "What is Python programming language?",
            "description": "Testing general knowledge (may use web)",
            "expected": "web"
        }
    ]
    
    print("Running demo questions...")
    print("-" * 80 + "\n")
    
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n{'='*80}")
        print(f"DEMO QUESTION {i}/{ len(demo_questions)}")
        print(f"Description: {demo['description']}")
        print(f"Expected Source: {demo['expected']}")
        print('=' * 80)
        
        try:
            response = qa_system.ask(demo['question'])
            qa_system.display_response(response)
            
            # Check if source matches expectation
            if response['source_type'] == demo['expected']:
                print("✓ Source selection: CORRECT")
            else:
                print(f"✗ Source selection: Expected {demo['expected']}, got {response['source_type']}")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        print("\n")
        if not auto_continue:
            input("Press Enter to continue to next question...")


def main():
    """Main demo function."""
    import sys
    
    # Check for --no-interactive flag
    skip_interactive = '--no-interactive' in sys.argv
    
    print_header()
    
    print("This demo showcases the Intelligent RAG Q&A system with:")
    print("  1. Retrieval from private knowledge base (PDF)")
    print("  2. Intelligent fallback detection")
    print("  3. Dynamic web search when needed")
    print("  4. Transparent user notification")
    print("  5. Source citations\n")
    
    # Check if knowledge base exists
    from config.settings import settings
    kb_path = settings.KNOWLEDGE_BASE_PATH
    
    if not kb_path.exists():
        print("⚠️  WARNING: Knowledge base PDF not found!")
        print(f"   Expected location: {kb_path}")
        print("\n   To add your PDF:")
        print("   1. Place your PDF at data/knowledge_base.pdf")
        print("   2. Or set KNOWLEDGE_BASE_PATH in .env\n")
        print("   The demo will continue with web-only mode.\n")
        if not skip_interactive:
            input("Press Enter to continue...")
    
    print("Initializing system...")
    print("-" * 80)
    
    try:
        # Initialize system
        qa_system = QASystem()
        qa_system.setup()
        
        # Show system info
        info = qa_system.get_system_info()
        print("\n✓ System initialized successfully!")
        print(f"\n📊 Configuration:")
        print(f"   LLM Provider: {info['llm_provider']}")
        print(f"   LLM Model: {info['llm_model']}")
        print(f"   Embeddings: {info['embedding_model']}")
        print(f"   Knowledge Base: {'✓ Loaded' if info['knowledge_base_exists'] else '✗ Not Found'}")
        
        if info['vector_store_stats'].get('document_count'):
            print(f"   Documents in KB: {info['vector_store_stats']['document_count']}")
        
        print("\n")
        if not skip_interactive:
            input("Press Enter to start demo questions...")
        
        # Run demo questions
        run_demo_questions(qa_system, auto_continue=skip_interactive)
        
        # Interactive mode (skip if --no-interactive flag is present)
        if not skip_interactive:
            print("\n" + "=" * 80)
            print("INTERACTIVE MODE")
            print("=" * 80)
            print("\nYou can now ask your own questions!")
            print("Type 'quit' or 'exit' to end the demo.\n")
            
            while True:
                try:
                    question = input("\n❓ Your question: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 Thanks for trying the demo!")
                        break
                    
                    if not question:
                        continue
                    
                    response = qa_system.ask(question)
                    qa_system.display_response(response)
                
                except KeyboardInterrupt:
                    print("\n\n👋 Demo interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
                    app_logger.error(f"Demo error: {str(e)}")
        else:
            print("\n" + "=" * 80)
            print("✓ Demo completed successfully!")
            print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Failed to initialize system: {str(e)}")
        print("\nPlease check:")
        print("  1. .env file is configured correctly")
        print("  2. Virtual environment is activated")
        print("  3. All dependencies are installed")
        print("\nRun: ./setup.sh to set up the environment")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
