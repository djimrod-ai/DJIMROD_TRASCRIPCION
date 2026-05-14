import os
import sys
from types import ModuleType

# 1. Solución para Pyannote en Windows
fake_torchcodec = ModuleType("torchcodec")
sys.modules["torchcodec"] = fake_torchcodec

import shutil
import torch
import librosa
import numpy as np
import time
import streamlit as st
from faster_whisper la import WhisperModel # Error corregido abajo
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from moviepy import VideoFileClip

# --- CONFIGURACIÓN ---
HF_TOKEN = "TU_TOKEN_DE_HUGGING_FACE_AQUI" 
MODEL_SIZE = "small" 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

st.set_page_config(page_title="Transcriptor Pro IA", page_icon="🎙️", layout="wide")

st.title("🎙️ Transcriptor de Alta Fidelidad")
st.markdown("Sube tu video y obtén la transcripción completa del audio sin etiquetas de hablantes.")

@st.cache_resource
def load_models():
    whisper = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=4)
    diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    return whisper, diarization

try:
    whisper_model, diarization_pipeline = load_models()
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()

uploaded_file = st.file_uploader("Sube tu video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    temp_video = f"temp_{uploaded_file.name}"
    temp_audio = "temp_audio.wav"
    
    if st.button("🚀 Iniciar Transcripción"):
        try:
            with st.status("Procesando archivo...", expanded=True) as status:
                # 1. Guardado optimizado
                st.write("📦 Guardando archivo...")
                with open(temp_video, "wb") as f:
                    while True:
                        chunk = uploaded_file.read(1024 * 1024)
                        if not chunk: break
                        f.write(chunk)

                # 2. Extracción de audio
                st.write("🔊 Extrayendo audio...")
                video_clip = VideoFileClip(temp_video)
                video_clip.audio.write_audiofile(temp_audio, codec='pcm_s16le', fps=16000)
                video_clip.close()

                # 3. Diarización (Solo para organizar el flujo del texto)
                st.write("🔍 Analizando estructura del audio...")
                audio_data, sample_rate = librosa.load(temp_audio, sr=16000)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
                diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

                turns = []
                if hasattr(diarization_result, 'itertracks'):
                    turns = list(diarization_result.itertracks(yield_label=True))
                elif hasattr(diarization_result, '__iter__'):
                    turns = list(diarization_result)

                # 4. Transcripción sin etiquetas de hablante
                st.write("✍️ Transcribiendo texto...")
                final_transcript = []
                current_speaker = None
                current_text_block = []
                
                if not turns:
                    segments, _ = whisper_model.transcribe(temp_audio, language="es", vad_filter=True)
                    for s in segments:
                        if s.text.strip(): 
                            final_transcript.append(s.text.strip())
                else:
                    total_turns = len(turns)
                    progress_bar = st.progress(0)

                    for i, turn_data in enumerate(turns):
                        if isinstance(turn_data, tuple):
                            turn, _, speaker = turn_data
                        else:
                            turn = turn_data[0] if hasattr(turn_data, '__getitem__') else turn_data
                            speaker = getattr(turn_data, 'label', 'Hablante')

                        if (turn.end - turn.start) < 0.1: continue

                        segments, _ = whisper_model.transcribe(
                            temp_audio, language="es", clip_timestamps=(turn.start, turn.end),
                            beam_size=5, temperature=0, vad_filter=True
                        )
                        text = " ".join([s.text for s in segments]).strip()
                        
                        if text:
                            # Si es la misma persona, acumulamos el texto para crear párrafos
                            if speaker == current_speaker:
                                current_text_block.append(text)
                            else:
                                # Si cambia la persona, guardamos el párrafo anterior y empezamos uno nuevo
                                if current_speaker is not None:
                                    final_transcript.append(" ".join(current_text_block))
                                current_speaker = speaker
                                current_text_block = [text]
                        
                        progress_bar.progress((i + 1) / total_turns)

                    if current_speaker is not None:
                        final_transcript.append(" ".join(current_text_block))

                status.update(label="¡Completado!", state="complete", expanded=False)

            # RESULTADO FINAL (Texto puro)
            st.subheader("📝 Transcripción Final")
            # Unimos los bloques con doble salto de línea para mantener la estructura de párrafos
            result_text = "\n\n".join(final_transcript)
            st.markdown(result_text)

            st.download_button(
                label="📥 Descargar archivo .txt",
                data=result_text,
                file_name=f"transcripcion_{uploaded_file.name}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            if os.path.exists(temp_video): 
                try: os.remove(temp_video)
                except: pass
            if os.path.exists(temp_audio): 
                try: os.remove(temp_audio)
                except: pass
