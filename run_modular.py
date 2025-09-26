#!/usr/bin/env python3
"""
Development runner for Hallucinations.cloud modular version
Use this to run the clean, modular architecture for development and testing
"""

import subprocess
import sys
import os

def main():
    """Run the modular Streamlit application"""

    print("🚀 Starting Hallucinations.cloud (Modular Version)")
    print("=" * 50)
    print()
    print("Features available in modular version:")
    print("✅ Clean modular architecture")
    print("✅ Demo authentication (5 queries)")
    print("✅ AI model integration (8 models)")
    print("✅ H-Score analysis engine")
    print("✅ Contradiction detection")
    print("✅ Results export capabilities")
    print()
    print("⚠️  For production features, use: streamlit run Hallucinations_9_23.py")
    print("   Production includes: Stripe payments, Twilio SMS, Supabase database")
    print()
    print("Starting modular version...")
    print("-" * 30)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "main_modular.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Hallucinations.cloud")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have streamlit installed: pip install streamlit")
        print("2. Check your .env file has the required API keys")
        print("3. Make sure you're in the correct directory")

if __name__ == "__main__":
    main()