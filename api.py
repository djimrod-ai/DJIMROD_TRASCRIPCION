import os
import sys
from types import ModuleType
import shutil
import torch
import librosa
import numpy as np
import time
import streamlit as st
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from moviepy import VideoFileClip

# Fix para Pyannote en Windows
fake_torchcodec = ModuleType("torchcodec")
sys.modules["torchcodec"] = fake_torchcodec

# --- CONFIGURACIÓN ---
HF_TOKEN = "IIIII" 
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

st.set_page_config(page_title="Transcriptor de videos", page_icon="🎙️", layout="wide")

st.title("🎙️ Transcriptor de videos")
st.markdown("Convierte tu video en texto.")

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
                # GUARDADO POR TROZOS (Soporte para 2GB+)
                st.write("📦 Guardando archivo en servidor...")
                with open(temp_video, "wb") as f:
                    while True:
                        chunk = uploaded_file.read(1024 * 1024)
                        if not chunk: break
                        f.write(chunk)

                # EXTRACCIÓN de audio
                st.write("🔊 Extrayendo audio...")
                video_clip = VideoFileClip(temp_video)
                video_clip.audio.write_audiofile(temp_audio, codec='pcm_s16le', fps=16000)
                video_clip.close()

                # DIARIZACIÓN
                st.write("🔍 Analizando voces...")
                audio_data, sample_rate = librosa.load(temp_audio, sr=16000)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
                diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

                turns = []
                if hasattr(diarization_result, 'itertracks'):
                    turns = list(diarization_result.itertracks(yield_label=True))
                elif hasattr(diarization_result, '__iter__'):
                    turns = list(diarization_result)

                # TRANSCRIPCIÓN Y FUSIÓN
                st.write("✍️ Transcribiendo texto...")
                final_transcript = []
                current_speaker = None
                current_text_block = []
                
                if not turns:
                    st.warning("No se detectaron hablantes. Transcribiendo audio completo...")
                    segments, _ = whisper_model.transcribe(temp_audio, language="es", vad_filter=True)
                    for s in segments:
                        if s.text.strip(): final_transcript.append(f"Hablante: {s.text.strip()}")
                else:
                    total_turns = len(turns)
                    progress_bar = st.progress(0)

                    for i, turn_data in enumerate(turns):
                        if isinstance(turn_data, tuple): turn, _, speaker = turn_data
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
                            if speaker == current_speaker:
                                current_text_block.append(text)
                            else:
                                if current_speaker is not None:
                                    final_transcript.append(f"**{current_speaker}**: {' '.join(current_text_block)}")
                                current_speaker = speaker
                                current_text_block = [text]
                        
                        progress_bar.progress((i + 1) / total_turns)

                    if current_speaker is not None:
                        final_transcript.append(f"**{current_speaker}**: {' '.join(current_text_block)}")

                status.update(label="¡Completado!", state="complete", expanded=False)

            st.subheader("📝 Resultado Final")
            result_text = "\n\n".join(final_transcript)
            st.markdown(result_text)
            st.download_button("📥 Descargar .txt", data=result_text, file_name="transcripcion.txt")

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            if os.path.exists(temp_video): 
                try: os.remove(temp_video)
                except: pass
            if os.path.exists(temp_audio): 
                try: os.remove(temp_audio)
                except: pass


