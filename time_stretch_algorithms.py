"""
Time-stretching algorithms for pitch-preserving speed adjustment.

This module provides multiple high-quality time-stretching algorithms optimized
for speech and audio processing. All algorithms preserve pitch while changing speed.
"""

import math
import shutil
import os
import numpy as np
from typing import Callable

import torch
import torchaudio as ta

# Optional high-quality libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
except ImportError:
    PYRUBBERBAND_AVAILABLE = False

try:
    from audiotsm2 import wsola as tsm_wsola  # type: ignore
    from audiotsm2.io.array import ArrReader, ArrWriter  # type: ignore
    AUDIOTSM_AVAILABLE = True
except ImportError:
    tsm_wsola = None
    ArrReader = None  # type: ignore
    ArrWriter = None  # type: ignore
    AUDIOTSM_AVAILABLE = False


def _sox_available() -> bool:
    """Detect if torchaudio SoX effects are usable; allow manual override."""
    if os.environ.get("FORCE_SOX", "").lower() in {"1", "true", "yes"}:
        print("FORCE_SOX=1 set: attempting to use SoX effects.")
        return True
    try:
        _ = ta.utils.sox_utils.list_effects()
        return True
    except Exception as exc:
        print(f"SoX effects unavailable: {exc}")
        if shutil.which("sox"):
            print("SoX CLI found on PATH but torchaudio SoX backend not available.")
        return False


USE_SOX = _sox_available()


def time_stretch_phase_vocoder(wav: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Phase-vocoder stretch that preserves pitch without requiring SoX.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)

    Returns:
        Time-stretched waveform tensor
    """
    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        channels = [wav_cpu]
        squeeze_back = True
    elif wav_cpu.dim() == 2:
        channels = [ch for ch in wav_cpu]
        squeeze_back = False
    else:
        raise ValueError("Phase vocoder expects 1D or 2D waveform (channels, samples)")

    def _pv_channel(channel: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        spec = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        )
        phase_advance = torch.linspace(
            0, math.pi * hop_length, spec.size(0), device=spec.device
        )[..., None]

        stretched = ta.functional.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)
        new_length = int(channel.size(-1) / rate)

        return torch.istft(
            stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=new_length,
        )

    stretched_channels = [
        _pv_channel(ch, n_fft=1024, hop_length=256, win_length=1024) for ch in channels
    ]
    if squeeze_back:
        return stretched_channels[0].to(device)
    return torch.stack(stretched_channels, dim=0).to(device)


def time_stretch_phase_vocoder_multiband(wav: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Multi-band phase vocoder: blend short/long windows to preserve transients and tone.
    """
    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        channels = [wav_cpu]
        squeeze_back = True
    elif wav_cpu.dim() == 2:
        channels = [ch for ch in wav_cpu]
        squeeze_back = False
    else:
        raise ValueError("Phase vocoder expects 1D or 2D waveform (channels, samples)")

    def _pv_channel(channel: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        spec = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        )
        phase_advance = torch.linspace(
            0, math.pi * hop_length, spec.size(0), device=spec.device
        )[..., None]
        stretched = ta.functional.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)
        new_length = int(channel.size(-1) / rate)
        return torch.istft(
            stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=new_length,
        )

    stretched_channels = []
    for ch in channels:
        fast = _pv_channel(ch, n_fft=512, hop_length=128, win_length=512)
        smooth = _pv_channel(ch, n_fft=2048, hop_length=512, win_length=2048)

        # Simple energy-based blend
        min_len = min(fast.size(-1), smooth.size(-1))
        fast = fast[..., :min_len]
        smooth = smooth[..., :min_len]
        alpha = 0.6  # favor transient-preserving small window slightly
        blended = alpha * fast + (1 - alpha) * smooth
        stretched_channels.append(blended)

    if squeeze_back:
        return stretched_channels[0].to(device)
    return torch.stack(stretched_channels, dim=0).to(device)


def time_stretch_hpss_phase_vocoder(wav: torch.Tensor, rate: float, sr: int) -> torch.Tensor:
    """
    HPSS + phase vocoder: split harmonic/percussive, stretch, then recombine.
    Requires librosa.
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not available, falling back to phase vocoder")
        return time_stretch_phase_vocoder(wav, rate)

    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    def _stretch_channel(ch_np: np.ndarray) -> np.ndarray:
        harm, perc = librosa.effects.hpss(ch_np)
        harm_s = librosa.effects.time_stretch(harm, rate=rate)
        perc_s = librosa.effects.time_stretch(perc, rate=rate)
        min_len = min(harm_s.shape[-1], perc_s.shape[-1])
        return harm_s[:min_len] + perc_s[:min_len]

    if wav_cpu.dim() == 1:
        stretched = _stretch_channel(wav_cpu.numpy())
        return torch.from_numpy(stretched).to(device)
    elif wav_cpu.dim() == 2:
        stretched_channels = [
            torch.from_numpy(_stretch_channel(ch.numpy())) for ch in wav_cpu
        ]
        min_len = min(ch.size(-1) for ch in stretched_channels)
        stacked = torch.stack([ch[..., :min_len] for ch in stretched_channels], dim=0)
        return stacked.to(device)
    else:
        raise ValueError("HPSS phase vocoder expects 1D or 2D waveform (channels, samples)")


def time_stretch_wsola(wav: torch.Tensor, rate: float) -> torch.Tensor:
    """
    WSOLA (Waveform Similarity Overlap-Add) algorithm.

    Excellent quality for speech with minimal artifacts. This is a pure PyTorch
    implementation that requires no external dependencies.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)

    Returns:
        Time-stretched waveform tensor
    """
    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        channels = [wav_cpu]
        squeeze_back = True
    elif wav_cpu.dim() == 2:
        channels = [ch for ch in wav_cpu]
        squeeze_back = False
    else:
        raise ValueError("WSOLA expects 1D or 2D waveform (channels, samples)")

    window_size = 1024
    tolerance = 512
    hop_synthesis = window_size // 2
    hop_analysis = int(hop_synthesis * rate)
    hop_analysis = max(1, hop_analysis)
    window = torch.hann_window(window_size, device=wav_cpu.device)

    def _stretch_channel(w: torch.Tensor) -> torch.Tensor:
        output_length = int(w.size(-1) / rate)
        output = torch.zeros(output_length, device=w.device)
        pos_output = 0

        while pos_output + window_size <= output_length:
            natural_pos = int(pos_output * rate)
            search_start = max(0, natural_pos - tolerance)
            search_end = min(w.size(-1) - window_size, natural_pos + tolerance)

            if search_end <= search_start:
                break

            best_corr = -float("inf")
            best_pos = natural_pos

            if pos_output > 0:
                prev_output = output[max(0, pos_output - window_size):pos_output]
                overlap_size = min(prev_output.size(0), window_size)

                if overlap_size > 0:
                    for search_pos in range(search_start, search_end):
                        candidate = w[search_pos:search_pos + window_size]
                        if candidate.size(0) < window_size:
                            continue

                        corr = torch.dot(
                            prev_output[-overlap_size:],
                            candidate[:overlap_size]
                        )
                        if corr > best_corr:
                            best_corr = corr
                            best_pos = search_pos

            if best_pos + window_size <= w.size(-1):
                segment = w[best_pos:best_pos + window_size] * window
                end_pos = min(pos_output + window_size, output_length)
                segment_len = end_pos - pos_output
                output[pos_output:end_pos] += segment[:segment_len]

            pos_output += hop_synthesis

        if output.abs().max() > 0:
            output = output / output.abs().max()
        return output

    stretched_channels = [_stretch_channel(ch) for ch in channels]
    if squeeze_back:
        return stretched_channels[0].to(device)
    return torch.stack(stretched_channels, dim=0).to(device)


def time_stretch_ola(wav: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Simple Overlap-Add algorithm.

    Fast but lower quality, may have artifacts. Good for quick previews.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)

    Returns:
        Time-stretched waveform tensor
    """
    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        channels = [wav_cpu]
        squeeze_back = True
    elif wav_cpu.dim() == 2:
        channels = [ch for ch in wav_cpu]
        squeeze_back = False
    else:
        raise ValueError("OLA expects 1D or 2D waveform (channels, samples)")

    window_size = 1024
    hop_synthesis = window_size // 2
    hop_analysis = int(hop_synthesis * rate)
    hop_analysis = max(1, hop_analysis)
    window = torch.hann_window(window_size, device=wav_cpu.device)

    def _stretch_channel(w: torch.Tensor) -> torch.Tensor:
        output_length = int(w.size(-1) / rate)
        output = torch.zeros(output_length, device=w.device)

        pos_input = 0
        pos_output = 0

        while pos_input + window_size <= w.size(-1) and pos_output + window_size <= output_length:
            segment = w[pos_input:pos_input + window_size] * window
            output[pos_output:pos_output + window_size] += segment

            pos_input += hop_analysis
            pos_output += hop_synthesis

        if output.abs().max() > 0:
            output = output / output.abs().max()
        return output

    stretched_channels = [_stretch_channel(ch) for ch in channels]
    if squeeze_back:
        return stretched_channels[0].to(device)
    return torch.stack(stretched_channels, dim=0).to(device)


def time_stretch_librosa(wav: torch.Tensor, rate: float, sr: int) -> torch.Tensor:
    """
    Librosa time stretch with advanced phase vocoder.

    High quality with good defaults. Requires librosa to be installed.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)
        sr: Sample rate

    Returns:
        Time-stretched waveform tensor
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not available, falling back to phase vocoder")
        return time_stretch_phase_vocoder(wav, rate)

    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    if wav_cpu.dim() == 1:
        channels = [wav_cpu]
        squeeze_back = True
    elif wav_cpu.dim() == 2:
        channels = [ch for ch in wav_cpu]
        squeeze_back = False
    else:
        raise ValueError("Librosa expects 1D or 2D waveform (channels, samples)")

    stretched_channels = []
    for ch in channels:
        stretched_np = librosa.effects.time_stretch(ch.numpy(), rate=rate)
        stretched_channels.append(torch.from_numpy(stretched_np))

    if squeeze_back:
        return stretched_channels[0].to(device)
    return torch.stack(stretched_channels, dim=0).to(device)


def time_stretch_rubberband(wav: torch.Tensor, rate: float, sr: int) -> torch.Tensor:
    """
    Rubber Band library - industry standard, highest quality.

    Professional-grade time stretching. Requires pyrubberband and the
    Rubber Band system library to be installed.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)
        sr: Sample rate

    Returns:
        Time-stretched waveform tensor
    """
    if not PYRUBBERBAND_AVAILABLE:
        print("pyrubberband not available, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)

    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_cpu = wav.detach().cpu()

    try:
        if wav_cpu.dim() == 1:
            stretched = pyrb.time_stretch(wav_cpu.numpy(), sr, rate)
            return torch.from_numpy(stretched).to(device)
        elif wav_cpu.dim() == 2:
            stretched_channels = [
                torch.from_numpy(pyrb.time_stretch(ch.numpy(), sr, rate))
                for ch in wav_cpu
            ]
            return torch.stack(stretched_channels, dim=0).to(device)
        else:
            raise ValueError("Rubberband expects 1D or 2D waveform (channels, samples)")
    except Exception as e:
        print(f"Rubber Band failed: {e}, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)


def time_stretch_audiotsm(wav: torch.Tensor, rate: float, sr: int) -> torch.Tensor:
    """
    AudioTSM WSOLA implementation - optimized for quality.

    Enhanced WSOLA implementation from the audiotsm2 library.
    Requires audiotsm2 to be installed.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)
        sr: Sample rate

    Returns:
        Time-stretched waveform tensor
    """
    if not AUDIOTSM_AVAILABLE:
        print("audiotsm2 not available, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)

    rate = float(rate)
    if rate == 1.0:
        return wav

    device = wav.device
    wav_np = wav.detach().cpu().numpy()
    transposed = False

    # Ensure shape is (frames, channels) for AudioTSM
    if wav_np.ndim == 1:
        wav_np = wav_np[:, None]
        transposed = True
    elif wav_np.shape[0] <= wav_np.shape[1]:
        wav_np = wav_np.T
        transposed = True

    channels = wav_np.shape[1]

    try:
        reader = ArrReader(wav_np.astype(np.float32, copy=False), channels=channels, samplerate=sr, samplewidth=2)  # type: ignore[arg-type]

        class _FloatWriter:
            """Collect audiotsm output into float32 without quantizing to int16."""

            def __init__(self, channels: int):
                self.channels = channels
                self.chunks: list[np.ndarray] = []

            def write(self, buffer: np.ndarray) -> int:
                frames = buffer.T.astype(np.float32, copy=False)
                self.chunks.append(frames.copy())
                return frames.shape[0]

            def data(self) -> np.ndarray:
                if not self.chunks:
                    return np.zeros((0, self.channels), dtype=np.float32)
                return np.concatenate(self.chunks, axis=0)

        writer = _FloatWriter(channels)
        tsm = tsm_wsola(channels, speed=rate)
        tsm.run(reader, writer)

        stretched = writer.data()
        if wav.dim() == 1:
            stretched = stretched[:, 0]  # Back to 1D
        elif transposed:
            stretched = stretched.T  # Back to (channels, frames)

        return torch.from_numpy(stretched).to(device)
    except Exception as e:
        print(f"AudioTSM failed: {e}, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)


def time_stretch_sox(wav: torch.Tensor, rate: float, sr: int) -> torch.Tensor:
    """
    SoX tempo effect - high quality when available.

    Uses SoX backend through torchaudio. Requires SoX to be installed
    on the system.

    Args:
        wav: Input waveform tensor
        rate: Speed factor (< 1.0 slows down, > 1.0 speeds up)
        sr: Sample rate

    Returns:
        Time-stretched waveform tensor
    """
    if not USE_SOX:
        print("SoX not available, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)

    rate = float(rate)
    if rate == 1.0:
        return wav

    try:
        squeeze_back = False
        wav_in = wav
        if wav.dim() == 1:
            wav_in = wav.unsqueeze(0)
            squeeze_back = True
        elif wav.dim() != 2:
            raise ValueError("SoX expects 1D or 2D waveform (channels, samples)")

        effects = [["tempo", "-s", f"{rate}"]]
        stretched, _ = ta.sox_effects.apply_effects_tensor(wav_in, sr, effects)
        if squeeze_back:
            stretched = stretched.squeeze(0)
        return stretched.to(wav.device)
    except Exception as e:
        print(f"SoX failed: {e}, falling back to WSOLA")
        return time_stretch_wsola(wav, rate)


def get_algorithm(algorithm: str) -> Callable:
    """
    Get the time-stretching algorithm function by name.

    Args:
        algorithm: Name of the algorithm

    Returns:
        Time-stretching function that takes (wav, rate) or (wav, rate, sr)
    """
    algorithms = {
        "wsola": lambda w, r, sr=None: time_stretch_wsola(w, r),
        "phase_vocoder": lambda w, r, sr=None: time_stretch_phase_vocoder(w, r),
        "phase_vocoder_multiband": lambda w, r, sr=None: time_stretch_phase_vocoder_multiband(w, r),
        "hpss_phase_vocoder": lambda w, r, sr: time_stretch_hpss_phase_vocoder(w, r, sr),
        "librosa": lambda w, r, sr: time_stretch_librosa(w, r, sr),
        "rubberband": lambda w, r, sr: time_stretch_rubberband(w, r, sr),
        "audiotsm": lambda w, r, sr: time_stretch_audiotsm(w, r, sr),
        "sox": lambda w, r, sr: time_stretch_sox(w, r, sr),
        "ola": lambda w, r, sr=None: time_stretch_ola(w, r),
    }

    return algorithms.get(algorithm, lambda w, r, sr=None: time_stretch_wsola(w, r))


def get_available_algorithms() -> dict[str, dict]:
    """
    Get information about available algorithms and their dependencies.

    Returns:
        Dictionary mapping algorithm names to their info (available, description)
    """
    return {
        "wsola": {
            "available": True,
            "description": "WSOLA - Best for Speech (Default)",
            "quality": "Excellent",
            "speed": "Fast",
            "dependencies": None
        },
        "phase_vocoder": {
            "available": True,
            "description": "Phase Vocoder - Basic, Good All-Around",
            "quality": "Good",
            "speed": "Fast",
            "dependencies": None
        },
        "phase_vocoder_multiband": {
            "available": True,
            "description": "Phase Vocoder (Multi-Band Blend)",
            "quality": "Good",
            "speed": "Medium",
            "dependencies": None
        },
        "hpss_phase_vocoder": {
            "available": LIBROSA_AVAILABLE,
            "description": "HPSS + Phase Vocoder",
            "quality": "High",
            "speed": "Medium",
            "dependencies": "librosa"
        },
        "librosa": {
            "available": LIBROSA_AVAILABLE,
            "description": "Librosa - High Quality",
            "quality": "High",
            "speed": "Medium",
            "dependencies": "librosa"
        },
        "rubberband": {
            "available": PYRUBBERBAND_AVAILABLE,
            "description": "Rubber Band - Industry Standard",
            "quality": "Highest",
            "speed": "Slow",
            "dependencies": "pyrubberband"
        },
        "audiotsm": {
            "available": AUDIOTSM_AVAILABLE,
            "description": "AudioTSM - Optimized WSOLA",
            "quality": "Excellent",
            "speed": "Fast",
            "dependencies": "audiotsm2"
        },
        "sox": {
            "available": USE_SOX,
            "description": "SoX Tempo - High Quality",
            "quality": "High",
            "speed": "Fast",
            "dependencies": "SoX"
        },
        "ola": {
            "available": True,
            "description": "OLA - Fast but Lower Quality",
            "quality": "Fair",
            "speed": "Very Fast",
            "dependencies": None
        }
    }


# Print availability info on module import
def _print_availability():
    """Print information about available algorithms."""
    print("\n=== Time-Stretching Algorithms ===")
    print("Available algorithms:")

    algos = get_available_algorithms()
    for name, info in algos.items():
        status = "✓" if info["available"] else "✗"
        deps = f" (requires {info['dependencies']})" if info['dependencies'] else ""
        print(f"  {status} {name}: {info['quality']} quality, {info['speed']}{deps}")

    print("=" * 35 + "\n")


_print_availability()
