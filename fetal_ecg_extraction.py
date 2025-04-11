import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 1. Define the path to your dataset folder
dataset_folder = r"C:\Users\user\Desktop\dataset"  # Raw string (avoid escape issues)
file_name = "r01.edf"  # Change to "r10.edf" for the other recording
file_path = os.path.join(dataset_folder, file_name)  # Full path to the file

# 2. Load the EDF file
try:
    raw = mne.io.read_raw_edf(file_path, preload=True)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Check the path or filename!")
    exit()

# 3. Extract ECG channel (assuming channel 0 is abdominal ECG)
fs = int(raw.info['sfreq'])  # Sampling frequency (Hz)
ecg_data = raw.get_data()[0, :]  # Get all samples from the first channel
time = np.arange(len(ecg_data)) / fs  # Time axis in seconds

# 4. Filter the signal (bandpass: 1-40 Hz for fetal ECG)
b, a = signal.butter(4, [1.0, 40.0], btype='bandpass', fs=fs)
filtered_ecg = signal.filtfilt(b, a, ecg_data)

# 5. Plot raw vs. filtered signal
plt.figure(figsize=(12, 6))
plt.plot(time, ecg_data, 'b-', alpha=0.5, label="Raw Abdominal ECG")
plt.plot(time, filtered_ecg, 'r-', linewidth=1, label="Filtered (Fetal ECG?)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title(f"Fetal ECG Extraction from {file_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dataset_folder, "fetal_ecg_extraction.png"))  # Save plot to your dataset folder
plt.show()

# 6. (Optional) Detect fetal QRS peaks (simplified)
peaks, _ = signal.find_peaks(filtered_ecg, height=np.percentile(filtered_ecg, 90), distance=fs*0.5)
plt.plot(time[peaks], filtered_ecg[peaks], 'go', label="Detected Peaks")
plt.legend()
plt.show()

# 7. (Optional) Print estimated fetal heart rate
if len(peaks) > 1:
    rr_intervals = np.diff(peaks) / fs
    fetal_hr = 60 / np.mean(rr_intervals)
    print(f"Estimated Fetal Heart Rate: {fetal_hr:.2f} BPM")
    # Add to your code (before plt.show())
    plt.figure(figsize=(12, 4))
    plt.magnitude_spectrum(filtered_ecg, Fs=fs, scale='dB', color='purple')
    plt.title("Frequency Spectrum of Filtered Fetal ECG")
    plt.grid()
    plt.show()
