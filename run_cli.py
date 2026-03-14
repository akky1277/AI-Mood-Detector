"""
================================================================
  AI MOOD DETECTOR - Command Line Version
  File: run_cli.py

  Run this if you just want the terminal version (no browser).
  Usage: python run_cli.py
================================================================
"""

from mood_engine import run_cli, train_and_save
import os

if __name__ == "__main__":
    if not os.path.exists("model.pkl"):
        train_and_save()
    run_cli()
