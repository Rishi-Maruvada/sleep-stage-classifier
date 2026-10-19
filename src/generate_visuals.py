"""
Generate Visual Artifacts: Hypnogram & Confusion Matrix Heatmap
Author: Rishi Maruvada
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Ensure outputs directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[*] Loading dataset and extracting features...")
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

# Crop wake padding
sleep_annots = [a for a in raw.annotations if a['description'] in stage_dict]
first_sleep = min(a['onset'] for a in sleep_annots if a['description'] != 'Sleep stage W')
last_sleep = max(a['onset'] + a['duration'] for a in sleep_annots if a['description'] != 'Sleep stage W')
raw.crop(tmin=max(0, first_sleep - 1800), tmax=min(raw.times[-1], last_sleep + 1800))

# Slice epochs
events, _ = mne.events_from_annotations(raw, event_id=stage_dict, chunk_duration=30.0, verbose=False)
epochs = mne.Epochs(raw=raw, events=events, event_id=stage_dict, tmin=0.0, tmax=29.99, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, 2]

# Compute PSD and band powers
spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=30.0, n_fft=400, verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)
psds = psds.squeeze(axis=1)

bands = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 12.0),
    'Sigma': (12.0, 16.0),
    'Beta': (16.0, 30.0)
}

features = []
for _, (fmin, fmax) in bands.items():
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    features.append(np.sum(psds[:, idx], axis=1))

X = np.log10(np.column_stack(features) + 1e-10)
y = labels

# Split with indices preserved to reconstruct time sequence
indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.25, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Predict on FULL timeline for hypnogram reconstruction
y_pred_full = clf.predict(X)

# =====================================================================
# PLOT 1: CONFUSION MATRIX HEATMAP
# =====================================================================
print("[*] Generating Confusion Matrix Plot...")
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=stage_names, 
    yticklabels=stage_names,
    title="Normalized Confusion Matrix (Frontal EEG Fpz-Cz)",
    ylabel="Clinical Ground Truth",
    xlabel="AI Model Prediction"
)

# Add text values
thresh = cm_normalized.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)",
            ha="center", va="center",
            color="white" if cm_normalized[i, j] > thresh else "black",
            fontsize=9
        )

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"[+] Saved {cm_path}")

# =====================================================================
# PLOT 2: THE HYPNOGRAM (TIME-SERIES SLEEP ARCHITECTURE)
# =====================================================================
print("[*] Generating Full Night Hypnogram Comparison...")
# In clinical hypnograms, Wake is top (0), N1 is 1, N2 is 2, N3 is 3, REM is usually mapped between W and N1 or separate
time_hours = (np.arange(len(y)) * 30.0) / 3600.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True, sharey=True)

# Ground Truth
ax1.step(time_hours, y, where='post', color='#1f77b4', linewidth=1.5)
ax1.set_title("Ground Truth Hypnogram (Sleep Physician Scoring)", fontsize=12, fontweight='bold')
ax1.set_yticks(range(5))
ax1.set_yticklabels(stage_names)
ax1.set_ylabel("Stage")
ax1.grid(True, linestyle='--', alpha=0.5)

# AI Prediction
ax2.step(time_hours, y_pred_full, where='post', color='#ff7f0e', linewidth=1.5)
ax2.set_title("AI Model Predictions (Random Forest on Welch Spectral Features)", fontsize=12, fontweight='bold')
ax2.set_yticks(range(5))
ax2.set_yticklabels(stage_names)
ax2.set_xlabel("Time from Sleep Onset (Hours)", fontsize=11)
ax2.set_ylabel("Stage")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
hypno_path = os.path.join(OUTPUT_DIR, "hypnogram_comparison.png")
plt.savefig(hypno_path)
print(f"[+] Saved {hypno_path}")
print("[+] Visuals generated successfully!")