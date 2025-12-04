import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.recommender.recommendation_pipeline import (
    recommend_by_mood,
    recommend_similar_song,
)

# ---------------------------
#  CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="EmoTune – Emotion-Aware Music",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"
AUDIO_DIR = Path("data/audio")


# ---------------------------
#  DARK THEME CSS
# ---------------------------
st.markdown(
    """
<style>
:root {
    --accent: #4C8CFF;
    --accent-soft: rgba(76,140,255,0.12);
    --bg-main: #05070C;
    --bg-panel: #111827;
    --bg-panel-soft: #111827;
    --border-subtle: #1f2937;
    --text-main: #E5E7EB;
    --text-muted: #9CA3AF;
}

/* Main background */
.stApp {
    background-color: var(--bg-main) !important;
}

/* Remove default padding at top */
.block-container {
    padding-top: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid var(--border-subtle);
}

/* Generic text colours */
h1, h2, h3, h4, h5, h6, label, span, p {
    color: var(--text-main) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.95rem;
    padding: 0.85rem 1.1rem;
    color: var(--text-muted) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-main) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--accent);
    color: white;
    border-radius: 999px;
    padding: 0.5rem 1.4rem;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #3A73F1;
}

/* Slider label */
div[data-baseweb="slider"] p {
    color: var(--text-main) !important;
}

/* Inputs */
div[data-baseweb="input"] > div {
    background-color: var(--bg-panel);
    border-radius: 0.6rem;
}
textarea {
    background-color: var(--bg-panel) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: var(--bg-panel-soft);
    border-radius: 0.75rem;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.9rem;
    color: var(--text-muted) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
#  SIDEBAR
# ---------------------------
st.sidebar.title("EmoTune")

st.sidebar.subheader("Mood")
selected_mood = st.sidebar.selectbox(
    "",
    ["happy", "sad", "energetic", "calm", "angry", "romantic", "mellow"],
    label_visibility="collapsed",
)

num_songs = st.sidebar.slider("Number of songs", 5, 20, 10)

st.sidebar.markdown("---")
with st.sidebar.expander("EmoTune subscription plans", expanded=False):
    st.markdown(
        """
**Free**
- Mood-based playlists  
- Text → playlist  
- Basic recommendations  

**Plus**
- Everything in Free  
- Voice-based playlists  
- Longer playlists  

**Pro**
- Everything in Plus  
- Priority emotion analysis  
- Advanced analytics view  
"""
    )


# ---------------------------
#  HEADER
# ---------------------------
st.markdown(
    """
<div style="
    background: radial-gradient(circle at 0% 0%, rgba(76,140,255,0.18), transparent 55%),
                radial-gradient(circle at 100% 0%, rgba(96,165,250,0.12), transparent 55%);
    padding: 1.4rem 1.6rem;
    border-radius: 1.2rem;
    border: 1px solid rgba(55,65,81,0.8);
    margin-bottom: 1.2rem;
">
  <h1 style="margin: 0; font-size: 2.1rem; letter-spacing: 0.04em;">EmoTune</h1>
  <p style="margin-top: 0.35rem; color: #9CA3AF; max-width: 640px;">
    An emotion-aware music experience. Build playlists from your mood, your words, or your voice.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------
#  HELPERS
# ---------------------------
def render_audio_preview_from_df(df: pd.DataFrame) -> None:
    """
    If the dataframe has a track_id column and local demo audio files exist
    under data/audio/<track_id>.mp3, let the user preview one.
    Always shows the demo section so the user knows the feature exists.
    """
    if df is None or df.empty or "track_id" not in df.columns:
        return

    # Build list of rows that have a matching local audio file
    available = []
    for _, row in df.iterrows():
        track_id = str(row["track_id"])
        audio_path = AUDIO_DIR / f"{track_id}.mp3"
        if audio_path.exists():
            label = f"{row.get('track_name', 'Unknown')} – {row.get('track_artist', '')}"
            available.append((label, audio_path))

    st.markdown("##### 🎧 Preview a track (demo)")

    if not available:
        st.info(
            "Audio preview is optional for the demo. "
            "Drop a few MP3 files in `data/audio/` named as `track_id.mp3` "
            "to enable playback here."
        )
        return

    labels = [lbl for lbl, _ in available]
    choice = st.selectbox(
        "Choose a track to play",
        labels,
        key=f"preview_{hash(df.head().to_csv())}",
    )

    chosen_path = dict(available)[choice]
    with open(chosen_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")


def call_backend_text(payload: dict) -> dict:
    resp = requests.post(f"{BACKEND_URL}/api/recommend/text", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_backend_audio(file) -> dict:
    files = {"file": (file.name, file, file.type or "audio/mpeg")}
    resp = requests.post(f"{BACKEND_URL}/api/recommend/audio", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------
#  TABS
# ---------------------------
tab_mood, tab_text, tab_voice, tab_similar = st.tabs(
    ["Mood → playlist", "Text → playlist", "Voice → playlist", "Similar song"]
)


# ---------------------------
#  TAB 1 – MOOD → PLAYLIST (LOCAL ENGINE)
# ---------------------------
with tab_mood:
    st.subheader("Mood-based playlist (local engine)")
    st.write(
        "Pick a mood in the sidebar and let EmoTune build a playlist using audio features "
        "from your songs dataset."
    )

    if st.button("Generate playlist", key="btn_mood"):
        with st.spinner("Finding tracks that match this mood..."):
            try:
                recs = recommend_by_mood(selected_mood, n=num_songs)
                if recs is None or len(recs) == 0:
                    st.warning("No songs found for this mood.")
                else:
                    st.success(f"Top {len(recs)} songs for mood: **{selected_mood}**")

                    display_cols = [
                        c
                        for c in ["track_name", "track_artist", "similarity", "explanation"]
                        if c in recs.columns
                    ]
                    st.dataframe(recs[display_cols], use_container_width=True)

                    # Optional local audio preview
                    render_audio_preview_from_df(recs)
            except Exception as e:
                st.error(f"Something went wrong while generating the playlist: {e}")


# ---------------------------
#  TAB 2 – TEXT → PLAYLIST (BACKEND)
# ---------------------------
with tab_text:
    st.subheader("Turn your words into a playlist")
    st.write("Describe how you feel, and EmoTune will detect the emotion and build a playlist.")

    user_text = st.text_area(
        "Describe your current mood",
        placeholder="Example: I'm tired but optimistic about our project presentation.",
        height=130,
    )

    if st.button("Create playlist from text", key="btn_text"):
        if not user_text.strip():
            st.warning("Please enter a short description first.")
        else:
            with st.spinner("Analyzing text and building a playlist..."):
                try:
                    data = call_backend_text({"text": user_text})

                    top_emotion = data.get("top_emotion", "Unknown")
                    top_score = data.get("top_score", 0.0)
                    st.success(f"Detected emotion: **{top_emotion}** (confidence ≈ {top_score:.2f})")

                    emotions = data.get("emotions", [])
                    if emotions:
                        with st.expander("Emotion breakdown"):
                            st.json(emotions)

                    recs = data.get("recommendations", [])
                    if recs:
                        df = pd.DataFrame(recs)
                        st.write("Playlist suggestion")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No songs were suggested.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend service: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ---------------------------
#  TAB 3 – VOICE → PLAYLIST (BACKEND)
# ---------------------------
with tab_voice:
    st.subheader("Let your voice pick the vibe")
    st.write(
        "Upload a short voice note or audio clip. EmoTune analyzes the emotional tone using "
        "the backend service and turns it into a playlist."
    )

    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
    )

    if st.button("Create playlist from voice", key="btn_voice"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Analyzing audio and building a playlist..."):
                try:
                    data = call_backend_audio(audio_file)

                    top_emotion = data.get("top_emotion", "Unknown")
                    top_score = data.get("top_score", 0.0)
                    st.success(f"Detected from voice: **{top_emotion}** (confidence ≈ {top_score:.2f})")

                    emotions = data.get("emotions", [])
                    if emotions:
                        with st.expander("Emotion breakdown"):
                            st.json(emotions)

                    recs = data.get("recommendations", [])
                    if recs:
                        df = pd.DataFrame(recs)
                        st.write("Playlist suggestion")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No songs were suggested.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend service: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ---------------------------
#  TAB 4 – SIMILAR SONG (LOCAL ENGINE)
# ---------------------------
with tab_similar:
    st.subheader("Find songs similar to a track")
    st.write(
        "Paste a `track_id` from your dataset and EmoTune will find nearby songs in the "
        "feature space using the local recommender."
    )

    track_id_input = st.text_input("Track ID")

    if st.button("Find similar songs", key="btn_similar"):
        if not track_id_input.strip():
            st.warning("Please enter a track ID.")
        else:
            with st.spinner("Searching for similar songs..."):
                try:
                    recs = recommend_similar_song(track_id_input.strip(), n=num_songs)
                    if recs is None or len(recs) == 0:
                        st.warning("No similar songs found.")
                    else:
                        st.success("Here are some similar tracks:")
                        display_cols = [
                            c
                            for c in ["track_name", "track_artist", "similarity", "explanation"]
                            if c in recs.columns
                        ]
                        st.dataframe(recs[display_cols], use_container_width=True)

                        render_audio_preview_from_df(recs)
                except Exception as e:
                    st.error(f"Something went wrong while searching for similar songs: {e}")