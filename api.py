import os
import streamlit as st

from faster_whisper import WhisperModel
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# --- CONFIGURACIÓN ---
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

st.set_page_config(
    page_title="Transcriptor IA",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Transcriptor de Audio a Texto")
st.markdown("Sube un video y obtén la transcripción completa en texto.")

@st.cache_resource
def load_model():
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=4
    )
    return model

try:
    whisper_model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Sube tu video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:

    temp_video = f"temp_{uploaded_file.name}"
    temp_audio = "temp_audio.wav"

    if st.button("🚀 Iniciar Transcripción"):

        try:
            with st.status("Procesando archivo...", expanded=True) as status:

                # 1. Guardar video
                st.write("📦 Guardando archivo...")

                with open(temp_video, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. Extraer audio
                st.write("🔊 Extrayendo audio...")

                video_clip = VideoFileClip(temp_video)

                video_clip.audio.write_audiofile(
                    temp_audio,
                    codec="pcm_s16le",
                    fps=16000
                )

                video_clip.close()

                # 3. Transcripción
                st.write("✍️ Transcribiendo audio...")

                segments, info = whisper_model.transcribe(
                    temp_audio,
                    language="es",
                    beam_size=5,
                    vad_filter=True
                )

                final_text = ""

                for segment in segments:
                    final_text += segment.text.strip() + " "

                status.update(
                    label="✅ Transcripción completada",
                    state="complete",
                    expanded=False
                )

            # RESULTADO
            st.subheader("📝 Texto Transcrito")

            st.markdown(final_text)

            st.download_button(
                label="📥 Descargar TXT",
                data=final_text,
                file_name=f"transcripcion_{uploaded_file.name}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")

        finally:
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


