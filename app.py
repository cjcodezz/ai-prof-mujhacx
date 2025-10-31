# app.py - Complete Chat Interface with AI Avatar and Voice (updated TTS & avatar states)
import streamlit as st
import time
import json
import base64
import tempfile
import os
from typing import List, Dict, Optional
import requests
import io
import threading
from urllib.parse import urlparse

# Import backend functionality (your existing module)
from backend_rag import (
    chunk_by_topic, extract_text_from_file, scrape_url,
    upsert_chunks, answer, generate_sub_questions,
    EMBED_MODEL, CHAT_MODEL
)

# Optional TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Audio recording
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# ---------- Streamlit Configuration ----------
st.set_page_config(
    page_title="Ycotes RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Session State Initialization ----------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'chat_history': [],
        'uploaded_files': [],
        'socratic_questions': [],
        'selected_questions': [],
        'current_answer': "",
        'avatar_state': "idle",    # idle | thinking | speaking
        'audio_data': None,
        'is_speaking': False,
        'show_upload': False,
        'voice_enabled': True,
        'voice_transcript': "",
        'processing_state': "idle",
        'language': 'English',
        'response_style': 'Concise'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------- Custom CSS with Animations ----------
def local_css():
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 3s ease infinite;
        font-weight: 700;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    .avatar-container {
        display:flex; justify-content:center; align-items:center; margin:1rem 0; padding:1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius:20px; min-height:300px;
    }
    .avatar-gif { border-radius:15px; box-shadow:0 20px 40px rgba(0,0,0,0.3); max-width:100%; max-height:280px; }
    .chat-container { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius:20px; padding:2rem; margin:1rem 0; border:1px solid rgba(255,255,255,0.2); }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding:1.5rem; border-radius:20px; margin:1rem 0; margin-left:10%; animation: slideInRight 0.5s ease-out; }
    .assistant-message { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color:#333; padding:1.5rem; border-radius:20px; margin:1rem 0; margin-right:10%; animation: slideInLeft 0.5s ease-out; }
    @keyframes slideInRight { from { opacity:0; transform:translateX(30px);} to { opacity:1; transform:translateX(0);} }
    @keyframes slideInLeft  { from { opacity:0; transform:translateX(-30px);} to { opacity:1; transform:translateX(0);} }
    .voice-button { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border:none; border-radius:50%; width:80px; height:80px; font-size:1.5rem; cursor:pointer; transition:all 0.3s ease; animation:pulse 2s infinite; color:white; display:flex; align-items:center; justify-content:center; margin:0 auto; }
    .input-area { background: rgba(255,255,255,0.9); padding:2rem; border-radius:20px; margin:1rem 0; box-shadow:0 10px 30px rgba(0,0,0,0.1); }
    .thinking-animation { display:flex; justify-content:center; align-items:center; padding:1rem; color:#666; font-style:italic; }
    .dot-flashing { position:relative; width:10px; height:10px; border-radius:5px; background-color:#667eea; color:#667eea; animation:dotFlashing 1s infinite linear alternate; animation-delay:0.5s; margin:0 5px; }
    .socratic-question { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius:15px; padding:1.5rem; margin:1rem 0; color:white; border:none; box-shadow:0 8px 25px rgba(0,0,0,0.1); }
    .feature-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:20px; padding:2rem; margin:1rem 0; color:white; transition:all 0.4s cubic-bezier(0.175,0.885,0.32,1.275); border:none; box-shadow:0 15px 35px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ---------- Avatar GIF management ----------
def get_avatar_gif(state="idle"):
    """Return appropriate avatar GIF path based on state"""
    # Place these files in the same directory or adjust path
    avatars = {
        "idle": "male_smile.gif",
        "thinking": "male_smile.gif",
        "speaking": "male_speaking.gif"
    }
    return avatars.get(state, avatars["idle"])

def render_ai_avatar():
    """Render the AI avatar with current state"""
    gif = get_avatar_gif(st.session_state.avatar_state)
    # If gif file missing, show placeholder text
    if not os.path.exists(gif):
        st.markdown(f"<div class='avatar-container'><div style='color:white'>Avatar GIF not found: {gif}</div></div>", unsafe_allow_html=True)
        return
    st.markdown(f"<div class='avatar-container'><img src='{gif}' class='avatar-gif' alt='AI Avatar'></div>", unsafe_allow_html=True)

# ---------- TTS: pyttsx3 (offline) or gTTS fallback ----------
def synthesize_with_pyttsx3(text: str, lang_code: str = "en") -> Optional[bytes]:
    """Try pyttsx3 to synthesize speech into an MP3 temp file and return bytes.
       pyttsx3 may not directly save MP3 depending on platform; we use WAV then convert if needed.
    """
    if not PYTTSX3_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        # Try to set voice based on lang preference (best-effort)
        voices = engine.getProperty('voices')
        # Choose a voice that matches 'hi' if requested
        for v in voices:
            if lang_code.startswith("hi") and ('hi' in (v.id or '').lower() or 'hindi' in (v.name or '').lower()):
                engine.setProperty('voice', v.id)
                break
            if lang_code.startswith("en") and ('english' in (v.name or '').lower()):
                engine.setProperty('voice', v.id)
                break
        # Create temp WAV file (pyttsx3 can save to file)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        engine.save_to_file(text, tmp_wav.name)
        engine.runAndWait()
        # Try to convert WAV to MP3 using pydub if available, else return WAV bytes
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(tmp_wav.name)
            tmp_mp3 = tmp_wav.name + ".mp3"
            audio.export(tmp_mp3, format="mp3")
            data = open(tmp_mp3, "rb").read()
            # cleanup
            os.unlink(tmp_wav.name)
            os.unlink(tmp_mp3)
            return data
        except Exception:
            # fallback: return WAV bytes
            data = open(tmp_wav.name, "rb").read()
            os.unlink(tmp_wav.name)
            return data
    except Exception as e:
        # silent failure to let fallback handle
        st.write(f"pyttsx3 synth error: {e}")
        return None

def synthesize_with_gtts(text: str, lang_code: str = "en") -> Optional[bytes]:
    """Synthesize using gTTS (requires internet). Returns MP3 bytes."""
    if not GTTS_AVAILABLE:
        return None
    try:
        tlang = "hi" if lang_code.startswith("hi") else "en"
        tts = gTTS(text=text, lang=tlang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.close()
        tts.save(tmp.name)
        data = open(tmp.name, "rb").read()
        os.unlink(tmp.name)
        return data
    except Exception as e:
        st.write(f"gTTS synth error: {e}")
        return None

def synthesize_speech(text: str, language_code: str = "en-US") -> Optional[bytes]:
    """Unified TTS: try pyttsx3 (offline) first, then gTTS fallback."""
    # Normalize to 'hi' or 'en' token for our helpers
    lang_token = "hi" if language_code.lower().startswith("hi") else "en"
    # Try pyttsx3
    audio_bytes = None
    if PYTTSX3_AVAILABLE:
        audio_bytes = synthesize_with_pyttsx3(text, lang_token)
        if audio_bytes:
            return audio_bytes
    # Fallback to gTTS
    if GTTS_AVAILABLE:
        audio_bytes = synthesize_with_gtts(text, lang_token)
        if audio_bytes:
            return audio_bytes
    # if neither available return None
    return None

# ---------- Voice playback + avatar control ----------
def play_audio_and_set_avatar(audio_content: bytes):
    """Embed audio into Streamlit and manage avatar state during playback."""
    if not audio_content:
        return
    # set state
    st.session_state.avatar_state = "speaking"
    st.session_state.is_speaking = True

    # base64 encode and embed as autoplay audio tag
    b64 = base64.b64encode(audio_content).decode('ascii')
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <script>
            // Attempt to reset avatar after audio duration estimate sent back via streamlit events
        </script>
    """
    st.components.v1.html(audio_html, height=0)

    # approximate speaking time and reset avatar after that time
    # estimate: 0.28 - 0.35 sec per word; use 0.32 here
    # We don't have the text here; as caller will call play_audio_and_set_avatar immediately after generating audio,
    # we can't read the text inside. So the caller should pass an estimated words count if available.
    # We'll schedule a conservative  max wait (10s) fallback
    def reset_after_delay(delay_seconds=5):
        time.sleep(delay_seconds)
        # ensure thread-safe update
        try:
            st.session_state.avatar_state = "idle"
            st.session_state.is_speaking = False
        except Exception:
            pass

    # Start a short thread to reset avatar in ~5 seconds (the caller will also update if they compute a better estimate)
    threading.Thread(target=reset_after_delay, args=(5,), daemon=True).start()

# ---------- UI Components ----------
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        # Language selection
        language = st.radio(
            "🌐 Language",
            ["English", "Hindi"],
            index=0,
            key="language"
        )
        response_style = st.selectbox(
            "🎨 Response Style",
            ["Concise", "Detailed", "Technical", "Beginner Friendly"],
            index=0,
            key="response_style"
        )
        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")
        # Show which TTS engines are available
        st.write(f"pyttsx3: {'installed' if PYTTSX3_AVAILABLE else 'missing'}, gTTS: {'installed' if GTTS_AVAILABLE else 'missing'}")
        if PYTTSX3_AVAILABLE or GTTS_AVAILABLE:
            voice_enabled = st.checkbox(
                "Enable Voice Responses", 
                value=st.session_state.voice_enabled,
                key="voice_enabled_checkbox"
            )
            if voice_enabled != st.session_state.voice_enabled:
                st.session_state.voice_enabled = voice_enabled
                st.rerun()
        else:
            st.error("No TTS engines available (install pyttsx3 or gTTS). Voice will be disabled.")
            st.session_state.voice_enabled = False

        st.markdown("---")
        st.markdown("### 🧠 Model Info")
        st.write(f"**Chat Model:** {CHAT_MODEL}")
        st.write(f"**Embed Model:** {EMBED_MODEL}")

        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.avatar_state = "idle"
            st.rerun()
        if st.button("Upload Document", use_container_width=True):
            st.session_state.show_upload = True
            st.rerun()

def render_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🧠 Ycotes RAG Assistant</h1>', unsafe_allow_html=True)
        st.markdown("### Your Intelligent Multilingual Learning Companion")

def render_voice_input():
    """Render voice input section"""
    st.markdown("### 🎤 Voice Input")
    if AUDIO_RECORDER_AVAILABLE:
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#667eea",
            icon_name="microphone",
            icon_size="2x",
            key="audio_recorder"
        )
        if audio_bytes:
            with st.spinner("🔄 Processing voice..."):
                try:
                    # audio_bytes is in webm/opus format typically; backend_rag may provide transcription helper
                    # For now, store raw bytes for manual transcription steps
                    st.session_state.voice_transcript = None
                    st.success("Voice recorded — use a transcription endpoint or paste text below.")
                    return None
                except Exception as e:
                    st.error(f"❌ Transcription error (recorder): {e}")
    # Manual text input as fallback
    st.markdown("### 💬 Text Input")
    return None

def render_ai_avatar_block():
    render_ai_avatar()

def render_chat_interface():
    """Main chat interface"""
    render_ai_avatar_block()
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="assistant-message"><strong>Ycotes:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
            with col2:
                if st.session_state.voice_enabled and not st.session_state.is_speaking:
                    if st.button("🔊", key=f"speak_{i}"):
                        # speak this message
                        text_to_speak = chat["content"]
                        # estimate speaking duration
                        words = len(text_to_speak.split())
                        est_seconds = max(2, int(words * 0.32))
                        # synthesize
                        tts_bytes = synthesize_speech(text_to_speak, "hi-IN" if st.session_state.language == "Hindi" else "en-US")
                        if tts_bytes:
                            # set speaking state and create thread to reset after est_seconds
                            st.session_state.avatar_state = "speaking"
                            st.session_state.is_speaking = True
                            # embed audio and start reset thread
                            b64 = base64.b64encode(tts_bytes).decode('ascii')
                            audio_html = f"""
                                <audio autoplay>
                                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                                </audio>
                            """
                            st.components.v1.html(audio_html, height=0)
                            def reset_avatar_after(delay):
                                time.sleep(delay)
                                st.session_state.avatar_state = "idle"
                                st.session_state.is_speaking = False
                            threading.Thread(target=reset_avatar_after, args=(est_seconds,), daemon=True).start()
                        else:
                            st.error("No TTS engine available to speak.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    transcript = render_voice_input()

    question = st.text_area(
        "Or type your question:",
        value=st.session_state.get('voice_transcript', ''),
        placeholder="Ask me anything...",
        height=100,
        key="text_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Send Message", use_container_width=True, type="primary"):
            final_question = transcript or question
            if final_question:
                process_question(final_question, "standard")
                if 'voice_transcript' in st.session_state:
                    st.session_state.voice_transcript = ""
                st.rerun()
    with col2:
        if st.button("🧠 Socratic Mode", use_container_width=True):
            final_question = transcript or question
            if final_question:
                process_question(final_question, "socratic")
                if 'voice_transcript' in st.session_state:
                    st.session_state.voice_transcript = ""
                st.rerun()
    with col3:
        if st.button("📁 Upload Docs", use_container_width=True):
            st.session_state.show_upload = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_upload_interface():
    st.markdown("### 📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, DOCX, or CSV files",
        type=['pdf', 'txt', 'docx', 'csv'],
        accept_multiple_files=False,
        key="file_uploader"
    )
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Selected: **{uploaded_file.name}**")
        with col2:
            if st.button("🚀 Process Document", use_container_width=True):
                process_uploaded_file(uploaded_file)
                st.rerun()
    st.markdown("### 🌐 Web Content")
    url = st.text_input("Enter URL to scrape:", placeholder="https://example.com", key="url_input")
    if st.button("🌐 Scrape Website", use_container_width=True):
        process_url(url)
        st.rerun()
    if st.button("← Back to Chat", use_container_width=True):
        st.session_state.show_upload = False
        st.rerun()

def process_uploaded_file(uploaded_file):
    with st.spinner("🔄 Processing document..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            text = extract_text_from_file(tmp_path)
            if text.strip():
                chunks = chunk_by_topic(text)
                upsert_chunks(chunks, source=f"file_{uploaded_file.name}")
                st.success(f"✅ Successfully processed {len(chunks)} chunks from {uploaded_file.name}")
                st.session_state.uploaded_files.append(uploaded_file.name)
            else:
                st.error("❌ No text could be extracted from the file")
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def process_url(url):
    if url and url.startswith(('http://', 'https://')):
        with st.spinner("🔄 Scraping website content..."):
            try:
                text = scrape_url(url)
                if text.strip():
                    chunks = chunk_by_topic(text)
                    upsert_chunks(chunks, source=f"url_{urlparse(url).netloc}")
                    st.success(f"✅ Successfully processed {len(chunks)} chunks from {url}")
                else:
                    st.error("❌ No content could be scraped from the URL")
            except Exception as e:
                st.error(f"❌ Error scraping URL: {e}")
    else:
        st.error("❌ Please enter a valid URL")

def process_question(question: str, mode: str = "standard"):
    """Process user question"""
    lang_code = "hi" if st.session_state.language == "Hindi" else "en"
    style_map = {
        "Concise": "concise",
        "Detailed": "detailed",
        "Technical": "technical",
        "Beginner Friendly": "beginner"
    }
    style = style_map.get(st.session_state.response_style, "concise")
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": time.time()
    })
    st.session_state.avatar_state = "thinking"
    with st.spinner("🤔 Thinking..."):
        try:
            if mode == "standard":
                answer_text = answer(question, style=style, lang=lang_code)
                st.session_state.current_answer = answer_text
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "type": "answer",
                    "timestamp": time.time()
                })
                if st.session_state.voice_enabled:
                    # synthesize and play
                    lang_tag = "hi-IN" if st.session_state.language == "Hindi" else "en-US"
                    tts_bytes = synthesize_speech(answer_text, lang_tag)
                    if tts_bytes:
                        words = len(answer_text.split())
                        est_seconds = max(2, int(words * 0.32))
                        # embed audio
                        b64 = base64.b64encode(tts_bytes).decode('ascii')
                        audio_html = f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
                        st.components.v1.html(audio_html, height=0)
                        # set speaking avatar and schedule reset
                        st.session_state.avatar_state = "speaking"
                        st.session_state.is_speaking = True
                        def reset_avatar_after(delay):
                            time.sleep(delay)
                            st.session_state.avatar_state = "idle"
                            st.session_state.is_speaking = False
                        threading.Thread(target=reset_avatar_after, args=(est_seconds,), daemon=True).start()
                    else:
                        st.session_state.avatar_state = "idle"
                        st.error("TTS unavailable: install pyttsx3 or gTTS.")
                else:
                    st.session_state.avatar_state = "idle"
            else:
                st.session_state.socratic_questions = generate_sub_questions(question, lang_code)
                st.session_state.main_question = question
                st.session_state.socratic_lang = lang_code
                st.session_state.socratic_style = style
                st.session_state.selected_questions = []
                st.session_state.avatar_state = "idle"
        except Exception as e:
            st.error(f"❌ Error processing question: {e}")
            st.session_state.avatar_state = "idle"

def render_socratic_interface():
    st.markdown("### 🧠 Socratic Questions")
    st.write("Select which foundational questions you'd like explained:")
    for i, question in enumerate(st.session_state.socratic_questions):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="socratic-question">{question}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Explain", key=f"explain_{i}", use_container_width=True):
                explain_socratic_question(question, i)
                st.rerun()
    if st.session_state.get('selected_questions'):
        if st.button("🎯 Synthesize Final Answer", use_container_width=True, type="primary"):
            synthesize_socratic_answer()
            st.rerun()

def explain_socratic_question(question: str, index: int):
    if question not in st.session_state.selected_questions:
        st.session_state.selected_questions.append(question)
    st.session_state.avatar_state = "thinking"
    with st.spinner(f"Explaining: {question}"):
        try:
            answer_text = answer(question, style=st.session_state.socratic_style, lang=st.session_state.socratic_lang)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**{question}**\n\n{answer_text}",
                "type": "socratic_explanation",
                "timestamp": time.time()
            })
            if st.session_state.voice_enabled:
                tts_bytes = synthesize_speech(answer_text, "hi-IN" if st.session_state.socratic_lang == "hi" else "en-US")
                if tts_bytes:
                    words = len(answer_text.split())
                    est_seconds = max(2, int(words * 0.32))
                    b64 = base64.b64encode(tts_bytes).decode('ascii')
                    audio_html = f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
                    st.components.v1.html(audio_html, height=0)
                    def reset_avatar_after(delay):
                        time.sleep(delay)
                        st.session_state.avatar_state = "idle"
                    threading.Thread(target=reset_avatar_after, args=(est_seconds,), daemon=True).start()
                else:
                    st.session_state.avatar_state = "idle"
            else:
                st.session_state.avatar_state = "idle"
        except Exception as e:
            st.error(f"❌ Error explaining question: {e}")
            st.session_state.avatar_state = "idle"

def synthesize_socratic_answer():
    st.session_state.avatar_state = "thinking"
    with st.spinner("🎯 Synthesizing final answer..."):
        try:
            final_answer = answer(st.session_state.main_question, style=st.session_state.socratic_style, lang=st.session_state.socratic_lang)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Final Answer: {st.session_state.main_question}**\n\n{final_answer}",
                "type": "socratic_final",
                "timestamp": time.time()
            })
            st.session_state.current_answer = final_answer
            if st.session_state.voice_enabled:
                tts_bytes = synthesize_speech(final_answer, "hi-IN" if st.session_state.socratic_lang == "hi" else "en-US")
                if tts_bytes:
                    words = len(final_answer.split())
                    est_seconds = max(2, int(words * 0.32))
                    b64 = base64.b64encode(tts_bytes).decode('ascii')
                    audio_html = f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
                    st.components.v1.html(audio_html, height=0)
                    def reset_avatar_after(delay):
                        time.sleep(delay)
                        st.session_state.avatar_state = "idle"
                    threading.Thread(target=reset_avatar_after, args=(est_seconds,), daemon=True).start()
                else:
                    st.session_state.avatar_state = "idle"
            else:
                st.session_state.avatar_state = "idle"
            st.session_state.socratic_questions = []
            st.session_state.selected_questions = []
        except Exception as e:
            st.error(f"❌ Error synthesizing answer: {e}")
            st.session_state.avatar_state = "idle"

# ---------- Main App ----------
def main():
    initialize_session_state()
    local_css()
    render_header()
    render_sidebar()
    if st.session_state.show_upload:
        render_upload_interface()
    else:
        render_chat_interface()
        if st.session_state.socratic_questions:
            render_socratic_interface()

if __name__ == "__main__":
    main()
