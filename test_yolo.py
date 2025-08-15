#!/usr/bin/env python3
"""
Simple test script for YOLO poker detector
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from yolo import PokerDetector, PokerOCR, CLASSES, MODEL_PATH, CONFIDENCE_THRESHOLD
    print("✅ Successfully imported YOLO classes")
    
    # Test constructor
    try:
        ocr = PokerOCR()
        print("✅ PokerOCR initialized successfully")
    except Exception as e:
        print(f"❌ PokerOCR initialization failed: {e}")
    
    # Test detector constructor
    try:
        detector = PokerDetector(MODEL_PATH, CLASSES, CONFIDENCE_THRESHOLD, ocr_engine=ocr)
        print("✅ PokerDetector initialized successfully")
    except Exception as e:
        print(f"❌ PokerDetector initialization failed: {e}")
    
    # Check environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API key found: {api_key[:10]}...")
    else:
        print("⚠ OPENAI_API_KEY not set in environment")
        print("💡 Set it with: set OPENAI_API_KEY=your_key_here")
    
    print("\n🎯 Ready to run! Use: python yolo.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
