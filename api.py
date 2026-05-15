# api.py

import gc
import os
import time
import streamlit as st
import torch

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip

# =========================================================
# CONFIGURACIÓN
# =========================================================

MODEL_SIZE = "small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Token desde Streamlit Secrets o variable de entorno
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Transcriptor de Videos",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Transcriptor de Videos")
st.markdown("Convierte videos en texto con diarización de hablantes.")

# =========================================================
# CARGA DE MODELOS
# =========================================================

@st.cache_resource
def load_models():

    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=4
    )

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    return whisper_model, diarization_pipeline


try:
    whisper_model, diarization_pipeline = load_models()

except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.stop()

# =========================================================
# SUBIDA DE VIDEO
# =========================================================

uploaded_file = st.file_uploader(
    "Sube tu video",
    type=["mp4", "mov", "avi", "mkv"]
)

# =========================================================
# PROCESAMIENTO
# =========================================================

if uploaded_file is not None:

    temp_video = f"temp_{uploaded_file.name}"
    temp_audio = "temp_audio.wav"

    if st.button("🚀 Iniciar Transcripción"):

        try:

            with st.status("Procesando archivo...", expanded=True) as status:

                # =====================================================
                # GUARDAR VIDEO
                # =====================================================

                st.write("📦 Guardando archivo...")

                with open(temp_video, "wb") as f:

                    while True:

                        chunk = uploaded_file.read(1024 * 1024)

                        if not chunk:
                            break

                        f.write(chunk)

                # =====================================================
                # EXTRAER AUDIO
                # =====================================================

                st.write("🔊 Extrayendo audio...")

                video_clip = VideoFileClip(temp_video)

                video_clip.audio.write_audiofile(
                    temp_audio,
                    codec="pcm_s16le",
                    fps=16000,
                    logger=None
                )

                video_clip.close()

                # =====================================================
                # DIARIZACIÓN
                # =====================================================

                st.write("🔍 Analizando hablantes...")

                diarization = diarization_pipeline(temp_audio)

                speaker_segments = []

                for turn, _, speaker in diarization.itertracks(yield_label=True):

                    speaker_segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })

                # =====================================================
                # TRANSCRIPCIÓN COMPLETA (UNA SOLA VEZ)
                # =====================================================

                st.write("✍️ Transcribiendo audio...")

                segments, info = whisper_model.transcribe(
                    temp_audio,
                    language="es",
                    beam_size=5,
                    vad_filter=True
                )

                # =====================================================
                # FUSIÓN SPEAKER + TEXTO
                # =====================================================

                st.write("🧠 Fusionando resultados...")

                final_transcript = []

                current_speaker = None
                current_text = []

                segments = list(segments)

                progress_bar = st.progress(0)

                total_segments = len(segments)

                for i, segment in enumerate(segments):

                    seg_start = segment.start
                    seg_end = segment.end
                    text = segment.text.strip()

                    if not text:
                        continue

                    detected_speaker = "Hablante"

                    for spk in speaker_segments:

                        overlap = (
                            seg_start >= spk["start"]
                            and seg_start <= spk["end"]
                        )

                        if overlap:
                            detected_speaker = spk["speaker"]
                            break

                    # Fusionar bloques del mismo hablante
                    if detected_speaker == current_speaker:

                        current_text.append(text)

                    else:

                        if current_speaker is not None:

                            final_transcript.append(
                                f"**{current_speaker}**: {' '.join(current_text)}"
                            )

                        current_speaker = detected_speaker
                        current_text = [text]

                    progress_bar.progress((i + 1) / total_segments)

                # Último bloque
                if current_speaker is not None:

                    final_transcript.append(
                        f"**{current_speaker}**: {' '.join(current_text)}"
                    )

                # =====================================================
                # FINALIZAR
                # =====================================================

                status.update(
                    label="✅ Transcripción completada",
                    state="complete",
                    expanded=False
                )

            # =========================================================
            # RESULTADO
            # =========================================================

            st.subheader("📝 Resultado Final")

            result_text = "\n\n".join(final_transcript)

            st.markdown(result_text)

            st.download_button(
                "📥 Descargar TXT",
                data=result_text,
                file_name="transcripcion.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(f"❌ Error: {e}")

        finally:

            gc.collect()
            time.sleep(1)

            if os.path.exists(temp_video):

                try:
                    os.remove(temp_video)
                except:
                    pass

            if os.path.exists(temp_audio):

                try:
                    os.remove(temp_audio)
                except:
                    pass


