import mne
from mne.datasets.sleep_physionet.age import fetch_data
from collections import Counter

print("Loading the brainwaves and the doctor's answer sheet...")

# Get the file paths (already cached on your laptop, so this takes 1 second)
[paths] = fetch_data(subjects=[0], recording=[1])
psg_file, hypno_file = paths[0], paths[1]

# 1. Read the doctor's labels (Hypnogram)
annotations = mne.read_annotations(hypno_file)

print("\n--- THE SLEEP DOCTOR'S LABELS ---")
print(f"Total labeled events: {len(annotations)}")

# Count how long the patient spent in each stage
stage_durations = Counter()
for annot in annotations:
    stage_durations[annot['description']] += annot['duration'] / 3600  # Convert seconds to hours

for stage, hours in sorted(stage_durations.items()):
    print(f"  {stage:<16} : {hours:.2f} hours")