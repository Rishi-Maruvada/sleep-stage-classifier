"""
Sleep-EDF Single-Channel Frontal EEG Sleep Staging Pipeline
Author: Rishi Maruvada
"""

import subprocess
import sys

print("=" * 60)
print("SLEEP STAGE CLASSIFIER: SINGLE-CHANNEL FRONTAL EEG (FPZ-CZ)")
print("=" * 60)

# Run feature extraction and visual generation
subprocess.run([sys.executable, "src/generate_visuals.py"], check=True)

print("\n[+] Pipeline execution complete. Check outputs/ for figures.")