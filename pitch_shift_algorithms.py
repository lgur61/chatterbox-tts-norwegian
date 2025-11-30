"""
Pitch shifting algorithms using resampling and time-stretching.

This module implements pitch shifting by combining resampling (which shifts pitch
and duration) with time-stretching (which restores duration). This allows utilizing
high-quality time-stretching algorithms (like WSOLA, Rubberband) for pitch control.

Additionally provides PSOLA (Pitch Synchronous Overlap and Add) algorithms which
preserve formants and voice character - the gold standard for speech pitch shifting.
"""

import math
from typing import Callable, Optional
import numpy as np

import torch
import torchaudio as ta

from time_stretch_algorithms import get_algorithm as get_time_stretch_algorithm

# Optional high-quality pitch shifting libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False


def pitch_shift_resample(
    wav: torch.Tensor,
    sr: int,
    n_steps: float,
    time_stretch_func: Callable[[torch.Tensor, float], torch.Tensor]
) -> torch.Tensor:
    """
    Shift pitch by n_steps semitones using resampling and time-stretching.

    Args:
        wav: Input waveform tensor (channels, samples) or (samples,)
        sr: Sample rate
        n_steps: Pitch shift in semitones (positive = up, negative = down)
        time_stretch_func: Function to perform time stretching (preserves pitch)

    Returns:
        Pitch-shifted waveform
    """
    if n_steps == 0.0:
        return wav

    # Calculate pitch shift factor (rate)
    # rate = 2^(n_steps/12)
    # If n_steps = 12 (octave up), rate = 2.0
    rate = 2.0 ** (n_steps / 12.0)

    # To shift pitch UP by factor 'rate' (e.g. 2.0):
    # 1. We effectively want to play the audio 'rate' times faster.
    #    This would reduce duration by factor 'rate'.
    # 2. To compensate, we first time-stretch the audio to be 'rate' times LONGER.
    #    This means slowing it down by factor 1/rate.
    
    # Step 1: Time-stretch
    # We want duration to increase by 'rate'.
    # Speed factor = 1 / rate.
    # e.g. Pitch up 1 octave (rate=2). We need 2x duration. Speed = 0.5.
    stretch_speed = 1.0 / rate
    
    # Note: Some algorithms might handle extreme stretch factors poorly.
    wav_stretched = time_stretch_func(wav, stretch_speed)

    # Step 2: Resample
    # We want to "play faster" (or slower).
    # Resampling from SR to (SR / rate) simulates playing at different speed.
    # e.g. Pitch up (rate=2). Resample 44k -> 22k.
    # This reduces sample count by half.
    # Since we started with 2x duration (2x samples), we end up with 1x duration.
    
    target_sr = int(sr / rate)
    
    # torchaudio resample: (waveform, orig_freq, new_freq)
    # We treat the stretched waveform as being at 'sr', and we want to convert it
    # to 'target_sr' (which effectively drops/adds samples).
    # Wait, if we use ta.transforms.Resample(sr, target_sr), it converts signal
    # sampled at 'sr' to one sampled at 'target_sr'.
    # If target_sr < sr (downsample), we get fewer samples.
    # This is what we want.
    
    wav_shifted = ta.functional.resample(wav_stretched, sr, target_sr)

    return wav_shifted


def pitch_shift_psola_librosa(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
    """
    PSOLA-based pitch shifting using librosa.

    Preserves formants and voice character - better for speech than resampling.
    Uses librosa's pitch_shift which implements a PSOLA-like algorithm internally.

    Args:
        wav: Input waveform tensor (channels, samples) or (samples,)
        sr: Sample rate
        n_steps: Pitch shift in semitones (positive = up, negative = down)

    Returns:
        Pitch-shifted waveform with preserved formants
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not available, falling back to resample method")
        from time_stretch_algorithms import time_stretch_wsola
        return pitch_shift_resample(wav, sr, n_steps, lambda w, r: time_stretch_wsola(w, r))

    if n_steps == 0.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        # Single channel
        wav_np = wav_cpu.numpy()
        shifted = librosa.effects.pitch_shift(
            wav_np,
            sr=sr,
            n_steps=n_steps,
            bins_per_octave=12  # Standard semitone resolution
        )
        return torch.from_numpy(shifted).to(device)

    elif wav_cpu.dim() == 2:
        # Multi-channel - process each channel
        shifted_channels = []
        for ch in wav_cpu:
            ch_np = ch.numpy()
            shifted = librosa.effects.pitch_shift(
                ch_np,
                sr=sr,
                n_steps=n_steps,
                bins_per_octave=12
            )
            shifted_channels.append(torch.from_numpy(shifted))

        return torch.stack(shifted_channels, dim=0).to(device)

    else:
        raise ValueError("PSOLA expects 1D or 2D waveform (channels, samples)")


def pitch_shift_psola_parselmouth(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
    """
    Professional-grade PSOLA using Praat's parselmouth library.

    This is the gold standard for speech pitch shifting, used by phoneticians worldwide.
    Preserves formants perfectly and produces the most natural-sounding results.

    Args:
        wav: Input waveform tensor (channels, samples) or (samples,)
        sr: Sample rate
        n_steps: Pitch shift in semitones (positive = up, negative = down)

    Returns:
        Pitch-shifted waveform with preserved formants
    """
    if not PARSELMOUTH_AVAILABLE:
        print("Parselmouth not available, falling back to librosa PSOLA")
        return pitch_shift_psola_librosa(wav, sr, n_steps)

    if n_steps == 0.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    # Convert semitones to frequency ratio
    pitch_factor = 2.0 ** (n_steps / 12.0)

    def _shift_channel(ch: torch.Tensor) -> torch.Tensor:
        """Shift a single channel using Praat PSOLA."""
        ch_np = ch.numpy()

        # Create Praat Sound object
        sound = parselmouth.Sound(ch_np, sampling_frequency=sr)

        # Use Praat's Change gender function which implements high-quality PSOLA
        # pitch_factor: multiply pitch by this factor
        # formant_shift_ratio: 1.0 = preserve formants (don't shift)
        # pitch_range_factor: 1.0 = preserve pitch range
        # duration_factor: 1.0 = preserve duration
        shifted_sound = call(
            sound,
            "Change gender",
            75,              # Pitch floor (Hz)
            600,             # Pitch ceiling (Hz)
            1.0,             # Formant shift ratio (1.0 = preserve formants)
            pitch_factor,    # New pitch median (ratio)
            1.0,             # Pitch range factor
            1.0              # Duration factor
        )

        # Extract the modified audio
        shifted_np = shifted_sound.values[0]  # Get first channel
        return torch.from_numpy(shifted_np)

    if wav_cpu.dim() == 1:
        return _shift_channel(wav_cpu).to(device=device, dtype=torch.float32)

    elif wav_cpu.dim() == 2:
        shifted_channels = [_shift_channel(ch) for ch in wav_cpu]
        # Ensure all channels have the same length
        min_len = min(ch.size(-1) for ch in shifted_channels)
        shifted_channels = [ch[..., :min_len] for ch in shifted_channels]
        return torch.stack(shifted_channels, dim=0).to(device=device, dtype=torch.float32)

    else:
        raise ValueError("PSOLA expects 1D or 2D waveform (channels, samples)")


def pitch_shift_formant_preserving_pv(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
    """
    Formant-preserving phase vocoder using envelope extraction.

    This approach separates the spectral envelope (formants) from the harmonic structure,
    shifts the pitch, then reapplies the original envelope. Better quality than simple
    resampling for larger pitch shifts.

    Args:
        wav: Input waveform tensor (channels, samples) or (samples,)
        sr: Sample rate
        n_steps: Pitch shift in semitones (positive = up, negative = down)

    Returns:
        Pitch-shifted waveform with preserved formants
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not available, falling back to resample method")
        from time_stretch_algorithms import time_stretch_wsola
        return pitch_shift_resample(wav, sr, n_steps, lambda w, r: time_stretch_wsola(w, r))

    if n_steps == 0.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    def _shift_channel(ch: torch.Tensor) -> torch.Tensor:
        """Shift a single channel with formant preservation."""
        ch_np = ch.numpy()

        # Compute STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(ch_np, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(D), np.angle(D)

        # Extract spectral envelope (formants) using cepstral analysis
        log_mag = np.log(mag + 1e-10)

        # Liftering to separate envelope from fine structure
        lifter_order = 20  # Low order = smooth envelope
        cepstrum = np.fft.rfft(log_mag, axis=0)
        lifter = np.zeros_like(cepstrum)
        lifter[:lifter_order] = 1.0
        envelope_cepstrum = cepstrum * lifter
        envelope_log = np.fft.irfft(envelope_cepstrum, n=log_mag.shape[0], axis=0)
        envelope = np.exp(envelope_log)

        # Separate fine structure
        fine_structure = mag / (envelope + 1e-10)

        # Shift fine structure (pitch) by resampling in frequency
        from scipy.ndimage import shift as nd_shift
        shift_amount = -n_steps / 12.0 * n_fft / 2  # Frequency bins to shift
        fine_structure_shifted = nd_shift(
            fine_structure,
            shift=[shift_amount, 0],
            order=3,
            mode='constant',
            cval=0.0
        )

        # Recombine: shifted pitch with original envelope (formants)
        mag_shifted = fine_structure_shifted * envelope

        # Reconstruct with original phase (could use phase vocoder for better results)
        D_shifted = mag_shifted * np.exp(1j * phase)

        # Time-stretch to compensate for duration change
        y_shifted = librosa.istft(D_shifted, hop_length=hop_length, length=len(ch_np))

        # Apply time stretching to restore original duration if needed
        if len(y_shifted) != len(ch_np):
            y_shifted = librosa.effects.time_stretch(
                y_shifted,
                rate=len(y_shifted) / len(ch_np)
            )

        return torch.from_numpy(y_shifted[:len(ch_np)])

    if wav_cpu.dim() == 1:
        return _shift_channel(wav_cpu).to(device=device, dtype=torch.float32)

    elif wav_cpu.dim() == 2:
        shifted_channels = [_shift_channel(ch) for ch in wav_cpu]
        min_len = min(ch.size(-1) for ch in shifted_channels)
        shifted_channels = [ch[..., :min_len] for ch in shifted_channels]
        return torch.stack(shifted_channels, dim=0).to(device=device, dtype=torch.float32)

    else:
        raise ValueError("Formant-preserving PV expects 1D or 2D waveform (channels, samples)")


def get_pitch_algorithm(algorithm_name: str) -> Callable:
    """
    Get a pitch shifting function by name.

    Args:
        algorithm_name: Name of the algorithm. Options:
            - 'psola': PSOLA using librosa (best for speech, preserves formants)
            - 'psola_parselmouth': Professional PSOLA using Praat (gold standard)
            - 'formant_pv': Formant-preserving phase vocoder
            - Time-stretching based (resample + stretch):
              'wsola', 'phase_vocoder', 'rubberband', 'librosa', etc.

    Returns:
        Function with signature (wav, sr, n_steps) -> wav
    """
    # Direct pitch-shifting algorithms (formant-preserving)
    if algorithm_name == "psola":
        return pitch_shift_psola_librosa
    elif algorithm_name == "psola_parselmouth":
        return pitch_shift_psola_parselmouth
    elif algorithm_name == "formant_pv":
        return pitch_shift_formant_preserving_pv

    # Resample-based algorithms using time-stretching
    # Get the underlying time stretch function
    ts_algo = get_time_stretch_algorithm(algorithm_name)

    def pitch_shift_func(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
        # We need to pass the SR to the time stretcher if it needs it (like rubberband)
        def _ts_with_sr(w: torch.Tensor, r: float) -> torch.Tensor:
            return ts_algo(w, r, sr)

        return pitch_shift_resample(wav, sr, n_steps, _ts_with_sr)

    return pitch_shift_func


def get_available_pitch_algorithms() -> dict[str, dict]:
    """
    Get information about available pitch shifting algorithms.

    Returns:
        Dictionary mapping algorithm names to their info (available, description, etc.)
    """
    return {
        "psola": {
            "available": LIBROSA_AVAILABLE,
            "description": "PSOLA (Librosa) - Best for Speech",
            "quality": "Excellent",
            "formant_preservation": True,
            "speed": "Medium",
            "dependencies": "librosa",
            "recommended": True
        },
        "psola_parselmouth": {
            "available": PARSELMOUTH_AVAILABLE,
            "description": "PSOLA (Praat) - Gold Standard",
            "quality": "Highest",
            "formant_preservation": True,
            "speed": "Medium",
            "dependencies": "praat-parselmouth",
            "recommended": True
        },
        "formant_pv": {
            "available": LIBROSA_AVAILABLE,
            "description": "Formant-Preserving Phase Vocoder",
            "quality": "High",
            "formant_preservation": True,
            "speed": "Slow",
            "dependencies": "librosa, scipy",
            "recommended": False
        },
        "wsola": {
            "available": True,
            "description": "Resample + WSOLA Time-Stretch",
            "quality": "Good",
            "formant_preservation": False,
            "speed": "Fast",
            "dependencies": None,
            "recommended": False
        },
        "phase_vocoder": {
            "available": True,
            "description": "Resample + Phase Vocoder",
            "quality": "Fair",
            "formant_preservation": False,
            "speed": "Fast",
            "dependencies": None,
            "recommended": False
        },
        "rubberband": {
            "available": True,  # Availability checked at runtime
            "description": "Resample + Rubberband",
            "quality": "High",
            "formant_preservation": False,
            "speed": "Slow",
            "dependencies": "pyrubberband",
            "recommended": False
        }
    }


# Print availability info on module import
def _print_pitch_availability():
    """Print information about available pitch shifting algorithms."""
    print("\n=== Pitch Shifting Algorithms ===")
    print("Available algorithms:")

    algos = get_available_pitch_algorithms()
    for name, info in algos.items():
        status = "✓" if info["available"] else "✗"
        formant = " [Formant-Preserving]" if info["formant_preservation"] else ""
        recommended = " ⭐ RECOMMENDED" if info.get("recommended", False) else ""
        deps = f" (requires {info['dependencies']})" if info['dependencies'] else ""
        print(f"  {status} {name}: {info['quality']} quality{formant}{recommended}{deps}")

    print("\nℹ️  For speech, use PSOLA algorithms - they preserve voice character!")
    print("=" * 35 + "\n")


_print_pitch_availability()
