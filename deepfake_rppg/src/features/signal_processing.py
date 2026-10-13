import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def normalize_signal(sig):
    """
    Normalize signal to zero mean and unit variance.
    """
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

def generate_spectrogram(sig, fs=30, nperseg=64, noverlap=48):
    """
    Generate a spectrogram from the 1D rPPG signal with improved frequency resolution.
    
    Args:
        sig: 1D numpy array.
        fs: Sampling frequency (FPS of video).
        nperseg: Length of each segment (lower for better temporal resolution)
        noverlap: Number of points to overlap (higher for smoother transitions)
        
    Returns:
        Sxx: Spectrogram matrix (Frequency, Time).
    """
    if len(sig) < nperseg:
        # Pad signal if too short
        sig = np.pad(sig, (0, nperseg - len(sig)), mode='reflect')
    
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 1. Log scaling (Decibels) - use log1p for numerical stability
    Sxx = 10 * np.log10(Sxx + 1e-10)
    
    # 2. Min-Max Normalization to [0, 1] for CNN input
    Sxx_min = np.min(Sxx)
    Sxx_max = np.max(Sxx)
    if Sxx_max > Sxx_min:
        Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min + 1e-6)
    else:
        Sxx = np.zeros_like(Sxx)
        
    return Sxx


def filter_signal(sig, fs=30, low=0.7, high=4.0):
    """
    Bandpass filter the signal (HB range: 42-240 BPM).
    Uses higher order filter for better frequency separation.
    """
    # Ensure frequencies are valid
    nyquist = fs / 2
    if high >= nyquist:
        high = nyquist - 0.1
    if low <= 0:
        low = 0.1
    
    if low >= high:
        return sig  # Invalid band, return unfiltered
    
    try:
        b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
        return signal.filtfilt(b, a, sig)
    except:
        return sig  # If filtering fails, return original

def smooth_signal(sig, window_size=5):
    """Smooth signal using moving average to reduce noise."""
    if len(sig) < window_size:
        return sig
    return np.convolve(sig, np.ones(window_size) / window_size, mode='same')

def interpolate_missing_frames(sig, max_gap=10):
    """Interpolate missing frames (marked as 0) to handle detection failures."""
    sig = sig.copy()
    zero_indices = np.where(sig == 0)[0]
    
    if len(zero_indices) > 0 and len(zero_indices) < len(sig) // 2:
        # Only interpolate if gaps are small
        for idx in zero_indices:
            left = max(0, idx - 1)
            right = min(len(sig) - 1, idx + 1)
            while left > 0 and sig[left] == 0:
                left -= 1
            while right < len(sig) - 1 and sig[right] == 0:
                right += 1
            if left < right and sig[left] != 0 and sig[right] != 0:
                sig[idx] = (sig[left] + sig[right]) / 2
    
    return sig
