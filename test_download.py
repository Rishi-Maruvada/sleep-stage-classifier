import mne
from mne.datasets.sleep_physionet.age import fetch_data

print("Downloading one night of real sleep data...")

# Download Subject 0, Night 1 (free public hospital dataset)
[paths] = fetch_data(subjects=[0], recording=[1])

# Load the brainwave recording
raw = mne.io.read_raw_edf(paths[0], preload=False)

print("\n--- IT WORKED! HERE IS WHAT IS INSIDE ---")
print(f"Channels (sensors) recorded: {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} measurements per second")
print(f"Total time recorded: {raw.n_times / raw.info['sfreq'] / 3600:.1f} hours")