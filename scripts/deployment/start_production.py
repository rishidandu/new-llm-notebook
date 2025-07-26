#!/usr/bin/env python3
"""
Production startup script for ASU RAG API
Uses Gunicorn for production deployment
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the production server using Gunicorn"""
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Check if we're in the right directory
    if not (project_root / "src" / "rag" / "api_server.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check for required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "TWILIO_ACCOUNT_SID", 
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        sys.exit(1)
    
    print("🚀 Starting ASU RAG API in production mode...")
    print("📡 Using Gunicorn with optimized settings")
    print("🔗 Server will be available at http://0.0.0.0:3000")
    
    # Start Gunicorn
    try:
        subprocess.run([
            "gunicorn",
            "--config", "gunicorn.conf.py",
            "--chdir", str(project_root),
            "src.rag.api_server:create_api_server()"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Gunicorn not found. Please install it with: pip install gunicorn")
        sys.exit(1)

if __name__ == "__main__":
    main() 