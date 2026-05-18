import streamlit as st
import os
import gc
import time
import torch
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
import tempfile

# =============================================================================
# 1. CONFIGURACIÓN Y ESTÉTICA
# =============================================================================
st.set_page_config(
    page_title="Professional Transcriber",
    page_icon="🎙️",
    layout="wide"
)

# CSS para mejorar la apariencia
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎙️ Transcriptor de Alta Precisión</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Conversión de video a texto con optimización de fidelidad auditiva.</p>", unsafe_allow_html=True)

# =============================================================================
# 2. CARGA DE MODELOS (Optimizado para Precisión)
# =============================================================================

# Opciones de modelos: Tiny (Rápido), Base, Small, Medium (Equilibrado), Large (Máxima Precisión)
MODEL_OPTIONS = {
    "Básico (Rápido)": "tiny",
    "Estándar": "base",
    "Equilibrado": "small",
    "Alta Precisión": "medium",
    "Máxima Fidelidad": "large-v3"
}

# Configuración de Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

@st.cache_resource
def load_whisper(model_size):
    # Cargamos el modelo según la elección del usuario
    return WhisperModel(
        model_size, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE, 
        cpu_threads=4
    )

# =============================================================================
# 3. INTERFAZ DE USUARIO (Sidebar)
# =============================================================================
st.sidebar.title("⚙️ Configuración")

# Selector de Precisión
selected_model_label = st.sidebar.selectbox(
    "Selecciona el Nivel de Precisión", 
    options=list(MODEL_OPTIONS.keys()), 
    index=2 # Predeterminado: Equilibrado (small)
)
model_size = MODEL_OPTIONS[selected_model_label]

st.sidebar.info(f"Modelo actual: **{model_size}**\n\nCualquier modelo superior a 'small' aumentará la precisión pero requerirá más RAM.")

# Subida de archivo
uploaded_file = st.file_uploader(
    "Sube tu archivo de video",
    type=["mp4", "mov", "avi", "mkv"]
)

# =============================================================================
# 4. PROCESAMIENTO DE AUDIO Y TEXTO
# =============================================================================

if uploaded_file is not None:
    # Usamos archivos temporales para evitar llenar el disco del servidor
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_video:
        tmp_video_path = tmp_video.name
        st.write("📦 Guardando video temporalmente...")
        st.session_state.tmp_video = tmp_video_path
        
        # Escribir el archivo subido al disco
        uploaded_file.seek(0)
        tmp_video.write(uploaded_file.read())

    temp_audio = "temp_audio_optimized.wav"

    if st.button("🚀 Iniciar Transcripción de Alta Fidelidad"):
        try:
            with st.status("Procesando audio y texto...", expanded=True) as status:
                
                # 1. Extracción de audio optimizada
                st.write("🔊 Extrayendo audio en alta calidad...")
                video_clip = VideoFileClip(st.session_state.tmp_video)
                video_clip.audio.write_audiofile(
                    temp_audio,
                    codec="pcm_s16le",
                    fps=16000, # Frecuencia estándar para Whisper
                    logger=None
                )
                video_clip.close()

                # 2. Carga del modelo seleccionado
                st.write(f"🧠 Cargando modelo {model_size}...")
                model = load_whisper(model_size)

                # 3. Transcripción con parámetros de Alta Precisión
                st.write("✍️ Transcribiendo con análisis profundo...")
                # beam_size=5 aumenta la precisión buscando más caminos de palabras
                # vad_filter elimina silencios y ruidos molestos
                segments, info = model.transcribe(
                    temp_audio, 
                    language="es", 
                    beam_size=5, 
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                # Convertir el generador de segmentos en una lista de texto
                full_text = []
                for segment in segments:
                    full_text.append(segment.text.strip())
                
                final_result = " ".join(full_text)

                status.update(label="✅ Transcripción Completada", state="complete", expanded=False)

            # =============================================================================
            # RESULTADOS FINAL
            # =============================================================================
            st.subheader("📝 Texto Transcrito")
            
            # Mostrar el resultado en una caja elegante
            st.markdown(f"""
                <div class="result-box">
                    {final_result}
                </div>
                """, unsafe_allow_html=True)

            # Botón de descarga
            st.download_button(
                "📥 Descargar Transcripción (.txt)",
                data=final_result,
                file_name=f"transcripcion_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error durante la transcripción: {e}")

        finally:
            # Limpieza de archivos temporales para no saturar el servidor
            try:
                if os.path.exists(temp_audio): os.remove(temp_audio)
                if os.path.exists(st.session_state.tmp_video): os.remove(st.session_state.tmp_video)
            except: pass
            gc.collect()


