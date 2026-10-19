import mne
import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data

print("Loading and preparing data...")
[paths] = fetch_data(subjects=[0], recording=[1])
psg_file, hypno_file = paths[0], paths[1]

# 1. Load raw data and attach the doctor's labels
raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
annotations = mne.read_annotations(hypno_file)
raw.set_annotations(annotations, emit_warning=False)
raw.annotations.rename({'Sleep stage 4': 'Sleep stage 3'})

# 2. Keep ONLY the forehead channel (Jack's headband location)
raw.pick(["EEG Fpz-Cz"])

# 3. Clean the signal: Keep only 0.5 Hz to 30 Hz
# (Removes sweat drift < 0.5 Hz and wall-socket hum > 30 Hz)
print("Applying bandpass filter (0.5 Hz to 30 Hz)...")
raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)

# 4. Standard AASM 5-Stage mapping (combining Stage 3 & 4 into class 3)
stage_dict = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage R': 4
}
stage_names = ['Wake', 'N1 (Drowsy)', 'N2 (Light)', 'N3 (Deep)', 'REM (Dreaming)']

# 5. Crop the excess 15 hours of daytime wakefulness!
# Keep only: 30 mins before sleep onset -> 30 mins after final morning awakening
sleep_annots = [a for a in raw.annotations if a['description'] in stage_dict]
first_sleep = min(a['onset'] for a in sleep_annots if a['description'] != 'Sleep stage W')
last_sleep = max(a['onset'] + a['duration'] for a in sleep_annots if a['description'] != 'Sleep stage W')

t_start = max(0, first_sleep - 30 * 60)
t_end = min(raw.times[-1], last_sleep + 30 * 60)
raw.crop(tmin=t_start, tmax=t_end)

# 6. Slice into standard 30-second chunks (Epochs)
events, _ = mne.events_from_annotations(raw, event_id=stage_dict, chunk_duration=30.0, verbose=False)
epochs = mne.Epochs(
    raw=raw, 
    events=events, 
    event_id=stage_dict, 
    tmin=0.0, 
    tmax=30.0 - 0.01,  # Exactly 3000 points per chunk @ 100Hz
    baseline=None, 
    preload=True, 
    verbose=False
)

print("\n" + "=" * 50)
print(f"CLEAN DATASET READY!")
print(f"Total 30-second chunks: {len(epochs)}")
print(f"Data tensor shape: {epochs.get_data().shape} (chunks, channel, samples)")
print("=" * 50)

# Print the new balanced class counts
labels = epochs.events[:, 2]
for class_id in range(5):
    count = np.sum(labels == class_id)
    pct = (count / len(labels)) * 100
    print(f"  Class {class_id} ({stage_names[class_id]:<14}): {count:4d} chunks ({pct:5.1f}%)")