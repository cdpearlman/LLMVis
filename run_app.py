"""
Entry point for running the transformer visualization dashboard.
"""

import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point."""
    try:
        print("🚀 Starting Transformer Visualization Dashboard...")
        print("📋 Loading components and services...")
        
        from app.app import run_app
        
        print("✅ All components loaded successfully!")
        print("🌐 Dashboard will be available at: http://127.0.0.1:8050")
        print("📖 Instructions:")
        print("   1. Select a model (gpt2 or Qwen/Qwen2.5-0.5B)")
        print("   2. Enter a prompt")
        print("   3. Click 'Find Module Names'")
        print("   4. Select attention/MLP patterns and parameters")
        print("   5. Click 'Visualize' to generate the dashboard")
        print("\n" + "="*50)
        
        # Run the app
        run_app(debug=True)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
