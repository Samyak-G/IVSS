#!/usr/bin/env python3
"""
Security Monitoring System - Main Entry Point
=============================================
This is the main entry point for the security monitoring system.

Usage:
    python main.py [options]

For web interface:
    python src/web/app.py

For training:
    python src/training/train_cpu_optimized.py
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.main import main

if __name__ == "__main__":
    main()
