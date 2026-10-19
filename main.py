"""
Sleep-EDF Single-Channel Frontal EEG Sleep Staging Pipeline
Author: Rishi Maruvada
"""

import os
import sys

print("=" * 60)
print("SLEEP STAGE CLASSIFIER: SINGLE-CHANNEL FRONTAL EEG (FPZ-CZ)")
print("=" * 60)

# Run feature extraction and visual generation
os.system(f"{sys.executable} src/generate_visuals.py")

print("\n[+] Pipeline execution complete. Check outputs/ for figures.")