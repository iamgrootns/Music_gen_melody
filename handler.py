import os
import torch
import torchaudio
import runpod
import base64
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from scipy import signal
import traceback

# --- Path for Initialization Error Logging ---
INIT_ERROR_FILE = "/tmp/init_error.log"

# --- Global Variables & Model Loading with Error Catching ---
model = None
try:
    # Clear any previous error logs on successful start
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
        
    print("Loading MusicGen melody model...")
    from audiocraft.models import MusicGen # Moved import inside try block for robust error catching
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
    print("✅ Model loaded successfully.")

except Exception as e:
    # If loading fails, write the full error traceback to a file
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    model = None # Ensure model is None if loading failed

# --- Helper Functions ---
def upsample_audio(input_wav_bytes, target_sr=48000):
    """Upsamples audio from an in-memory WAV byte stream."""
    try:
        with BytesIO(input_wav_bytes) as in_io:
            sr, audio = wavfile.read(in_io)

        print(f"Upsampling audio from {sr} Hz to {target_sr} Hz")
        up_factor = target_sr / sr
        upsampled_audio = signal.resample(audio, int(len(audio) * up_factor))

        if audio.dtype == np.int16:
            upsampled_audio = upsampled_audio.astype(np.int16)

        with BytesIO() as out_io:
            wavfile.write(out_io, target_sr, upsampled_audio)
            return out_io.getvalue(), target_sr

    except Exception as e:
        print(f"Upsampling failed: {str(e)}")
        return input_wav_bytes, sr

# --- Runpod Handler ---
def handler(event):
    """
    The handler function that will be called by Runpod for each job.
    """
    # --- Check for Initialization Error First ---
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_message = f.read()
        return {"error": f"Worker initialization failed: {error_message}"}

    if model is None:
        return {"error": "Model is not loaded, and no specific initialization error was found."}
        
    # --- Input Validation ---
    job_input = event.get("input", {})
    text = job_input.get("text")
    duration = job_input.get("duration", 60)
    sample_rate = job_input.get("sample_rate", 32000)

    if not text:
        return {"error": "No text prompt provided."}
    
    if sample_rate not in [32000, 48000]:
        return {"error": "Unsupported sample_rate. Only 32000 and 48000 are allowed."}

    # --- Music Generation ---
    try:
        print(f"Generating {duration}s audio with prompt: '{text}'")
        model.set_generation_params(duration=duration)
        res = model.generate([text])
        audio_tensor = res[0].cpu()
        base_sr = model.sample_rate

        buffer = BytesIO()
        torchaudio.save(buffer, audio_tensor, base_sr, format="wav")
        raw_wav_bytes = buffer.getvalue()
        
        final_wav_bytes, final_sr = raw_wav_bytes, base_sr

        if sample_rate == 48000:
            print("Upsampling to 48kHz requested.")
            final_wav_bytes, final_sr = upsample_audio(raw_wav_bytes, target_sr=48000)

        audio_base64 = base64.b64encode(final_wav_bytes).decode('utf-8')

        print("✅ Generation complete.")
        return {
            "audio_base64": audio_base64,
            "sample_rate": final_sr,
            "format": "wav"
        }

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return {"error": f"An error occurred during generation: {traceback.format_exc()}"}

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})
