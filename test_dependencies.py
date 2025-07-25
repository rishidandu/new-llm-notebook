#!/usr/bin/env python3
"""
Test script to verify all dependencies can be imported correctly
"""

def test_imports():
    """Test importing all required dependencies"""
    print("🧪 Testing dependency imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"❌ chromadb import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence_transformers import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ flask imported successfully")
    except ImportError as e:
        print(f"❌ flask import failed: {e}")
        return False
    
    try:
        import flask_cors
        print("✅ flask_cors imported successfully")
    except ImportError as e:
        print(f"❌ flask_cors import failed: {e}")
        return False
    
    try:
        import twilio
        print("✅ twilio imported successfully")
    except ImportError as e:
        print(f"❌ twilio import failed: {e}")
        return False
    
    try:
        import praw
        print("✅ praw imported successfully")
    except ImportError as e:
        print(f"❌ praw import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        import trafilatura
        print("✅ trafilatura imported successfully")
    except ImportError as e:
        print(f"❌ trafilatura import failed: {e}")
        return False
    
    try:
        import tqdm
        print("✅ tqdm imported successfully")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    try:
        import gunicorn
        print("✅ gunicorn imported successfully")
    except ImportError as e:
        print("⚠️  gunicorn not installed (only needed for production)")
    
    print("🎉 All dependencies imported successfully!")
    return True

def test_project_imports():
    """Test importing project modules"""
    print("\n🧪 Testing project module imports...")
    
    try:
        from config.settings import Config
        print("✅ config.settings imported successfully")
    except ImportError as e:
        print(f"❌ config.settings import failed: {e}")
        return False
    
    try:
        from src.rag.rag_system import ASURAGSystem
        print("✅ src.rag.rag_system imported successfully")
    except ImportError as e:
        print(f"❌ src.rag.rag_system import failed: {e}")
        return False
    
    try:
        from src.utils.asu_grades_processor import ASUGradesProcessor
        print("✅ src.utils.asu_grades_processor imported successfully")
    except ImportError as e:
        print(f"❌ src.utils.asu_grades_processor import failed: {e}")
        return False
    
    try:
        from src.rag.api_server import create_api_server
        print("✅ src.rag.api_server imported successfully")
    except ImportError as e:
        print(f"❌ src.rag.api_server import failed: {e}")
        return False
    
    print("🎉 All project modules imported successfully!")
    return True

if __name__ == "__main__":
    print("🚀 ASU RAG System - Dependency Test")
    print("=" * 50)
    
    deps_ok = test_imports()
    project_ok = test_project_imports()
    
    if deps_ok and project_ok:
        print("\n✅ All tests passed! Ready for deployment.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1) 