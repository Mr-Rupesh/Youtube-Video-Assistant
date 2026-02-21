import os
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load env
load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("🎬 YouTube RAG Chatbot")
st.markdown("Chat with any YouTube video using AI")

# ==========================================
# SIDEBAR - VIDEO INPUT
# ==========================================
with st.sidebar:
    st.header("📹 Video Settings")
    video_id = st.text_input("YouTube Video ID", value="aircAruvnKk", help="The part after v= in YouTube URL")
    
    if st.button("🔄 Load Video"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ==========================================
# LOAD VIDEO (runs once)
# ==========================================
if st.session_state.vector_store is None:
    with st.spinner("🎥 Loading video transcript..."):
        try:
            # Fetch transcript
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            
            # Store with timestamps
            chunks_with_time = []
            for chunk in transcript_list:
                chunks_with_time.append({
                    "text": chunk.text,
                    "start": int(chunk.start),
                    "duration": chunk.duration
                })
            
            transcript = " ".join(c["text"] for c in chunks_with_time)
            
            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(transcript)
            
            # Create documents with metadata
            documents = []
            time_idx = 0
            for i, text in enumerate(texts):
                start_time = chunks_with_time[min(time_idx, len(chunks_with_time)-1)]["start"]
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "chunk_id": i,
                        "timestamp": start_time,
                        "source": f"{start_time//60}:{start_time%60:02d}"
                    }
                ))
                time_idx += len(text.split()) // 10
            
            # Vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
            
            st.success(f"✅ Loaded {len(documents)} chunks from video!")
            
        except TranscriptsDisabled:
            st.error(" No captions available for this video.")
        except Exception as e:
            st.error(f" Error: {str(e)}")

if st.session_state.vector_store:
    st.info("✅ Video loaded! Ready for chat integration.")
else:
    st.info("👈 Enter a YouTube Video ID and click 'Load Video' to start!")