import os

FFMPEG_PATH = r"C:\Users\hp\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")


import argparse
import os
import tempfile
from typing import Optional

import streamlit as st
from PIL import Image

from schema import Myntra
from vector_search import run_vector_search


# Set up a wide, app-like layout similar to Myntra
st.set_page_config(
    page_title="Myntra-like Fashion Search",
    page_icon="🛍️",
    layout="wide",
)

# Global styles inspired by Myntra UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f6;
    }
    .myntra-nav {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
        padding: 0.75rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .myntra-brand {
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 0.08em;
        color: #ff3f6c;
    }
    .myntra-nav-links span {
        margin: 0 0.75rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #282c3f;
    }
    .myntra-hero {
        padding: 1.5rem 2rem 0.5rem 2rem;
    }
    .myntra-hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #282c3f;
        margin-bottom: 0.25rem;
    }
    .myntra-hero-subtitle {
        font-size: 0.95rem;
        color: #7e818c;
    }
    .myntra-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.15rem;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #282c3f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_whisper_model():
    """
    Lazily load the Whisper model once and cache it across Streamlit reruns.

    Returns:
        The loaded Whisper model.

    Notes:
        - Requires the `whisper` package: `pip install -U openai-whisper`
        - Also requires `ffmpeg` to be available on the system PATH.
    """
    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Audio search requires the `whisper` package. "
            "Install it with `pip install -U openai-whisper`."
        ) from exc

    # Use the 'base' model as a reasonable default between speed and quality.
    return whisper.load_model("base")


def extract_frame_from_video(video_bytes: bytes, frame_choice: str = "middle") -> Optional[Image.Image]:
    """
    Extract a single frame from video bytes for use as search query.

    Args:
        video_bytes: Raw video file content.
        frame_choice: One of "first", "middle", "last" to pick which frame to use.

    Returns:
        PIL Image of the chosen frame, or None if extraction fails.
    """
    try:
        import cv2
    except ImportError:
        st.error(
            "Video search requires OpenCV. Install with: `pip install opencv-python-headless`"
        )
        return None

    tmp_path = None
    try:
        suffix = ".mp4"  # default; cv2 will accept other formats from extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open the video file.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.error("Video has no frames.")
            cap.release()
            return None

        if frame_choice == "first":
            frame_idx = 0
        elif frame_choice == "last":
            frame_idx = total_frames - 1
        else:
            frame_idx = total_frames // 2  # middle

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            st.error("Could not read a frame from the video.")
            return None

        # BGR -> RGB and convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    except Exception as e:
        st.error(f"Error extracting video frame: {str(e)}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def transcribe_audio_bytes(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe raw audio bytes to text using Whisper.

    Args:
        audio_bytes: The contents of the uploaded audio file.

    Returns:
        The transcribed text, or None if transcription failed.
    """
    try:
        model = get_whisper_model()
    except RuntimeError as exc:
        # Surface a friendly error in the UI instead of crashing the app.
        st.error(str(exc))
        return None

    tmp_path = None
    try:
        # Persist the uploaded audio to a temporary file for Whisper to consume.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Run transcription
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
        if not text:
            st.warning("Could not recognize any speech in the uploaded audio.")
            return None

        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors


def main(args):
    # Top navigation / header (Myntra-like)
    st.markdown(
        """
        <div class="myntra-nav">
            <div class="myntra-brand">FASHION FINDER</div>
            <div class="myntra-nav-links">
                <span>MEN</span>
                <span>WOMEN</span>
                <span>KIDS</span>
                <span>HOME & LIVING</span>
                <span>BEAUTY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero section
    st.markdown(
        """
        <div class="myntra-hero">
            <div class="myntra-hero-title">Search fashion like on Myntra</div>
            <div class="myntra-hero-subtitle">
                Use text, images, videos, or voice to find visually similar products.
            </div>
            <div style="margin-top: 0.75rem;">
                <span class="myntra-chip">Kurta</span>
                <span class="myntra-chip">Jeans</span>
                <span class="myntra-chip">Sneakers</span>
                <span class="myntra-chip">Saree</span>
                <span class="myntra-chip">T-Shirts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: search controls
    st.sidebar.header("Search Controls")
    table_name = st.sidebar.text_input("Name of the table", args.table_name)
    search_query = st.sidebar.text_input("Search query", args.search_query)
    limit = st.sidebar.slider(
        "Limit the number of results",
        args.limit_min,
        args.limit_max,
        args.limit_default,
    )
    output_folder = st.sidebar.text_input("Output folder path", args.output_folder)

    # Image Based Search
    # Add an option for uploading an image for query
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
        # Set the search query as the uploaded image (CLIP handles images directly)
        search_query = image

    # Video Based Search
    st.sidebar.markdown("### 🎬 Video Search")
    uploaded_video = st.sidebar.file_uploader(
        "Upload a video (search by frame)",
        type=["mp4", "webm", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        video_id = f"{uploaded_video.name}_{len(video_bytes)}"
        frame_choice = st.sidebar.radio(
            "Use which frame for search?",
            ["middle", "first", "last"],
            horizontal=True,
            key="video_frame_choice",
        )
        cache_key = f"{video_id}_{frame_choice}"
        if st.session_state.get("video_cache_key") != cache_key:
            with st.spinner("Extracting frame..."):
                frame_img = extract_frame_from_video(video_bytes, frame_choice=frame_choice)
                if frame_img is not None:
                    st.session_state["video_frame"] = frame_img
                    st.session_state["video_cache_key"] = cache_key
        if "video_frame" in st.session_state:
            st.sidebar.image(
                st.session_state["video_frame"],
                caption="Frame used for search",
                use_container_width=True,
            )
            search_query = st.session_state["video_frame"]
    else:
        if "video_frame" in st.session_state:
            del st.session_state["video_frame"]
        if "video_cache_key" in st.session_state:
            del st.session_state["video_cache_key"]

    # Audio Based Search
    st.sidebar.markdown("### 🎤 Audio Search")
    
    # Option 1: Record audio directly in the browser
    recorded_audio = None
    try:
        st.sidebar.markdown("**Record Audio:**")
        recorded_audio = st.sidebar.audio_input("Record your query")
    except AttributeError:
        # st.audio_input is only available in Streamlit 1.28.0+
        st.sidebar.info("💡 **Tip:** Update Streamlit to 1.28.0+ for audio recording: `pip install --upgrade streamlit`")
    
    # Option 2: Upload an audio file (as fallback)
    st.sidebar.markdown("**Or Upload Audio File:**")
    uploaded_audio = st.sidebar.file_uploader(
        "Upload an audio query",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        label_visibility="collapsed"
    )
    
    # Process recorded audio (priority over uploaded file)
    audio_to_process = None
    audio_source = None
    
    if recorded_audio is not None:
        audio_to_process = recorded_audio
        audio_source = "recorded"
        # Clear uploaded audio cache when recording new audio
        if "uploaded_audio_id" in st.session_state:
            del st.session_state["uploaded_audio_id"]
            if "transcribed_audio" in st.session_state:
                del st.session_state["transcribed_audio"]
            if "audio_bytes" in st.session_state:
                del st.session_state["audio_bytes"]
    elif uploaded_audio is not None:
        audio_to_process = uploaded_audio
        audio_source = "uploaded"
    
    # Process audio if available
    if audio_to_process is not None:
        # Read audio bytes
        if audio_source == "recorded":
            # Recorded audio is already bytes
            audio_bytes = audio_to_process.read()
            audio_id = f"recorded_{len(audio_bytes)}"
        else:
            # Uploaded audio file
            audio_bytes = audio_to_process.read()
            audio_id = f"{uploaded_audio.name}_{uploaded_audio.size}"
        
        # Check if we've already transcribed this audio
        if "transcribed_audio" not in st.session_state or st.session_state.get("last_audio_id") != audio_id:
            # Show audio player
            st.sidebar.audio(audio_bytes, format="audio/wav")
            
            # Transcribe audio (this will be cached in session state)
            with st.spinner("Transcribing audio..."):
                transcribed_text = transcribe_audio_bytes(audio_bytes)
                st.session_state["transcribed_audio"] = transcribed_text
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["last_audio_id"] = audio_id
        else:
            # Use cached audio bytes and transcription
            st.sidebar.audio(st.session_state["audio_bytes"], format="audio/wav")
            transcribed_text = st.session_state["transcribed_audio"]
        
        if transcribed_text:
            st.sidebar.success(f"**Recognized query:** {transcribed_text}")
            # Override any previous text/image search query with the transcribed text.
            search_query = transcribed_text
    else:
        # Clear cached transcription when no audio is available
        if "transcribed_audio" in st.session_state:
            del st.session_state["transcribed_audio"]
        if "audio_bytes" in st.session_state:
            del st.session_state["audio_bytes"]
        if "last_audio_id" in st.session_state:
            del st.session_state["last_audio_id"]

    # Run the vector search when the button is clicked
    if st.sidebar.button("Run Vector Search"):
        run_vector_search(
            "~/.lancedb", table_name, Myntra, search_query, limit, output_folder
        )

    # Initialize session state for image index if it doesn't exist
    if "current_image_index" not in st.session_state:
        st.session_state.current_image_index = 0

    # Display images in output folder with interactive views
    if os.path.exists(output_folder):
        image_files = sorted(
            [
                f
                for f in os.listdir(output_folder)
                if f.lower().endswith(".jpg") or f.lower().endswith(".png")
            ]
        )
        if image_files:
            st.markdown("### Results")

            # Let the user switch between a grid view and a carousel view
            view_mode = st.radio(
                "View mode",
                ["Grid", "Carousel"],
                horizontal=True,
                key="results_view_mode",
            )

            if view_mode == "Grid":
                cols_per_row = 4
                for i in range(0, len(image_files), cols_per_row):
                    row_files = image_files[i : i + cols_per_row]
                    cols = st.columns(len(row_files))
                    for col, img_name in zip(cols, row_files):
                        with col:
                            img_path = os.path.join(output_folder, img_name)
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.write("Could not load image.")
                            pretty_name = (
                                os.path.splitext(img_name)[0].replace("_", " ").title()
                            )
                            st.caption(pretty_name)
            else:
                # Carousel-style view with previous/next buttons
                num_images = len(image_files)
                st.session_state.current_image_index %= num_images
                current_name = image_files[st.session_state.current_image_index]
                current_path = os.path.join(output_folder, current_name)

                try:
                    current_img = Image.open(current_path)
                    st.image(current_img, use_container_width=True)
                except Exception:
                    st.write("Could not load image.")

                st.caption(
                    os.path.splitext(current_name)[0].replace("_", " ").title()
                )

                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.button("◀ Previous"):
                        st.session_state.current_image_index = (
                            st.session_state.current_image_index - 1
                        ) % num_images
                with col_next:
                    if st.button("Next ▶"):
                        st.session_state.current_image_index = (
                            st.session_state.current_image_index + 1
                        ) % num_images
        else:
            st.write("No images found in the output folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Search")
    parser.add_argument(
        "--table_name", type=str, default="myntra", help="Name of the table"
    )
    parser.add_argument(
        "--search_query", type=str, default="kurta", help="Search query"
    )
    parser.add_argument(
        "--limit_min", type=int, default=1, help="Minimum limit for number of results"
    )
    parser.add_argument(
        "--limit_max", type=int, default=10, help="Maximum limit for number of results"
    )
    parser.add_argument(
        "--limit_default",
        type=int,
        default=3,
        help="Default limit for number of results",
    )
    parser.add_argument(
        "--output_folder", type=str, default="output", help="Output folder path"
    )
    args = parser.parse_args()
    main(args)
