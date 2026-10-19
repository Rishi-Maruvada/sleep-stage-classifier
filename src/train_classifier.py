import mne
import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("1. Loading and filtering data...")
[paths] = fetch_data(subjects=[0], recording=[1])
raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
raw.set_annotations(mne.read_annotations(paths[1]), emit_warning=False)
raw.pick(["EEG Fpz-Cz"])
raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)

stage_dict = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}
stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Crop daytime wake padding
sleep_annots = [a for a in raw.annotations if a['description'] in stage_dict]
first_sleep = min(a['onset'] for a in sleep_annots if a['description'] != 'Sleep stage W')
last_sleep = max(a['onset'] + a['duration'] for a in sleep_annots if a['description'] != 'Sleep stage W')
raw.crop(tmin=max(0, first_sleep - 1800), tmax=min(raw.times[-1], last_sleep + 1800))

# Slice into 30-second epochs
events, _ = mne.events_from_annotations(raw, event_id=stage_dict, chunk_duration=30.0, verbose=False)
epochs = mne.Epochs(raw=raw, events=events, event_id=stage_dict, tmin=0.0, tmax=29.99, baseline=None, preload=True, verbose=False)

labels = epochs.events[:, 2]

print("2. Extracting the 5 'Volume Knobs' (Frequency Bands) for each chunk...")
# Compute Power Spectral Density (Welch method)
spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=30.0, n_fft=400, verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)
psds = psds.squeeze(axis=1)  # Shape: (841, frequencies)

# Define our 5 biological bands
bands = {
    'Delta (0.5-4Hz)':  (0.5, 4.0),
    'Theta (4-8Hz)':    (4.0, 8.0),
    'Alpha (8-12Hz)':   (8.0, 12.0),
    'Sigma (12-16Hz)':  (12.0, 16.0),
    'Beta (16-30Hz)':   (16.0, 30.0)
}

# Sum up the power in each band for every single chunk
features = []
for band_name, (fmin, fmax) in bands.items():
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    # Sum power in this band
    band_power = np.sum(psds[:, idx_band], axis=1)
    features.append(band_power)

# Shape: (841 chunks, 5 features)
X = np.column_stack(features)
# Convert to log scale (normalizes extreme power spikes)
X = np.log10(X + 1e-10)
y = labels

print(f"Feature matrix built! Shape: {X.shape} (841 chunks x 5 biological bands)")

# 3. Split data: 75% for training the AI, 25% for testing it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("\n3. Training Random Forest Sleep Stage Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict on the unseen test chunks
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=" * 60)
print(f"MODEL TRAINING COMPLETE!")
print(f"OVERALL ACCURACY ON UNSEEN DATA: {acc * 100:.1f}%")
print("=" * 60)

print("\nDetailed Performance per Sleep Stage:")
print(classification_report(y_test, y_pred, target_names=stage_names, zero_division=0))

print("Confusion Matrix (Rows = Real Stage, Columns = AI Prediction):")
cm = confusion_matrix(y_test, y_pred)
print("       Wake  N1   N2   N3  REM")
for idx, row in enumerate(cm):
    print(f"{stage_names[idx]:<5}  " + "  ".join([f"{val:3d}" for val in row]))