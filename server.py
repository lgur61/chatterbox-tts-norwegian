from pathlib import Path
import io
from typing import Literal, Optional

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from chatterbox.tts import ChatterboxTTS

# Import time-stretching algorithms
from time_stretch_algorithms import get_algorithm
from pitch_shift_algorithms import get_pitch_algorithm

REPO_ID = "akhbar/chatterbox-tts-norwegian"
MODEL_FILES = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]
BASE_DIR = Path(__file__).parent
DEMO_HTML = BASE_DIR / "demo.html"
ALLOWED_SAMPLE_RATES = (16000, 22050, 24000, 44100, 48000)


def load_model():
    local_dir = None
    for fname in MODEL_FILES:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=fname)
        local_dir = Path(local_path).parent

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Chatterbox TTS on device: {device}")
    model = ChatterboxTTS.from_local(local_dir, device=device)
    return model


app = FastAPI(title="Norwegian TTS API")
tts_model = load_model()


def apply_speed_and_pitch(
    wav: torch.Tensor,
    sr: int,
    pitch: float,
    speed: float,
    speed_algorithm: Optional[str] = None,
    pitch_algorithm: Optional[str] = None
) -> tuple[torch.Tensor, int]:
    """
    Apply speed and pitch transformations using selected algorithms.

    Args:
        wav: Input waveform tensor
        sr: Sample rate
        pitch: Pitch shift in semitones 
        speed: Speed factor 
        speed_algorithm: Time-stretching algorithm for speed changes
        pitch_algorithm: Pitch shifting algorithm (formant-preserving recommended)

    Available speed algorithms:
        - wsola: WSOLA - Best for speech 
        - phase_vocoder: Basic phase vocoder - Good all-around
        - librosa: Librosa time stretch - High quality
        - rubberband: Rubber Band library - Industry standard (requires installation)
        - audiotsm: AudioTSM WSOLA - Optimized implementation
        - sox: SoX tempo - High quality (requires SoX)
        - ola: Simple overlap-add - Fast but lower quality

    Available pitch algorithms:
        - psola: PSOLA via librosa - Best for speech, preserves voice character 
        - psola_parselmouth: PSOLA via Praat - Gold standard (requires praat-parselmouth)
        - formant_pv: Formant-preserving phase vocoder - High quality
    """
    pitch = max(-12.0, min(12.0, pitch))
    speed = max(0.5, min(1.5, speed))

    # Apply time stretching if speed != 1.0
    if speed != 1.0:
        algo = speed_algorithm or "wsola"  # Default to WSOLA for speed
        stretch_func = get_algorithm(algo)
        wav = stretch_func(wav, speed, sr)

    # Apply pitch shift if pitch != 0.0
    if pitch != 0.0:
        if pitch_algorithm:
            # Use specified pitch algorithm (formant-preserving recommended)
            pitch_func = get_pitch_algorithm(pitch_algorithm)
            wav = pitch_func(wav, sr, pitch)
        else:
            # Default to PSOLA if available, otherwise fallback
            try:
                pitch_func = get_pitch_algorithm("psola")
                wav = pitch_func(wav, sr, pitch)
            except Exception:
                # Final fallback to torchaudio
                wav = ta.functional.pitch_shift(wav, sr, n_steps=pitch)

    return wav, sr


SpeedAlgorithmType = Literal[
    "wsola",
    "phase_vocoder",
    "phase_vocoder_multiband",
    "hpss_phase_vocoder",
    "librosa",
    "rubberband",
    "audiotsm",
    "sox",
    "ola"
]

PitchAlgorithmType = Literal[
    "psola",                    
    "psola_parselmouth",        
    "formant_pv",               
]


class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 1.0
    cfg_weight: float = 0.5
    temperature: float = 0.4
    speed: float = 1.0
    pitch: float = 0.0
    speed_algorithm: Optional[SpeedAlgorithmType] = None
    pitch_algorithm: Optional[PitchAlgorithmType] = None
    sample_rate: Optional[int] = None



@app.post("/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Debug: log incoming request parameters (truncate text to 80 chars)
    print(
        "[/tts] Incoming",
        {
            "text_preview": text[:80],
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "temperature": req.temperature,
            "speed": req.speed,
            "pitch": req.pitch,
            "speed_algorithm": req.speed_algorithm,
            "pitch_algorithm": req.pitch_algorithm,
            "sample_rate": req.sample_rate,
        },
    )

    try:
        wav = tts_model.generate(
            text,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
        )
        sr = tts_model.sr

        # Handle backward compatibility with old 'algorithm' parameter
        speed_algo = req.speed_algorithm
        pitch_algo = req.pitch_algorithm

        
        wav, sr = apply_speed_and_pitch(
            wav, sr, req.pitch, req.speed,
            speed_algorithm=speed_algo,
            pitch_algorithm=pitch_algo
        )

        target_sr = req.sample_rate if req.sample_rate is not None else sr
        if req.sample_rate is not None and target_sr not in ALLOWED_SAMPLE_RATES:
            raise HTTPException(
                status_code=400,
                detail=f"sample_rate must be one of {ALLOWED_SAMPLE_RATES}"
            )
        if target_sr != sr:
            wav = ta.functional.resample(wav, sr, target_sr)
            sr = target_sr
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    buffer = io.BytesIO()
    ta.save(buffer, wav, sr, format="wav")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts.wav"'},
    )


@app.get("/", response_class=HTMLResponse)
def demo():
    if not DEMO_HTML.exists():
        raise HTTPException(status_code=500, detail="Demo page missing")
    return HTMLResponse(content=DEMO_HTML.read_text(encoding="utf-8"))
