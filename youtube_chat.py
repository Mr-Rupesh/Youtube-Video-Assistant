import os
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

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
    
    st.markdown("---")
    st.markdown("**Features Enabled:**")
    st.markdown("✅ Query Rewriting")
    st.markdown("✅ MMR Retrieval")
    st.markdown("✅ Cross-Encoder Reranking")

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "reranker" not in st.session_state:
    st.session_state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# LOAD VIDEO (runs once)
# ==========================================
if st.session_state.vector_store is None:
    with st.spinner("🎥 Loading video transcript..."):
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            
            chunks_with_time = []
            for chunk in transcript_list:
                chunks_with_time.append({
                    "text": chunk.text,
                    "start": int(chunk.start),
                    "duration": chunk.duration
                })
            
            transcript = " ".join(c["text"] for c in chunks_with_time)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(transcript)
            
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
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
            
            st.success(f"✅ Loaded {len(documents)} chunks from video!")
            
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==========================================
# REWRITE LLM SETUP
# ==========================================
@st.cache_resource
def get_rewrite_llm():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=100,
        temperature=0.3,
    )
    return llm

rewrite_llm = get_rewrite_llm()

# ==========================================
# QUERY REWRITING
# ==========================================
def rewrite_query(question):
    """Rewrite query for better retrieval"""
    prompt = f"""Rewrite this question to be more specific for searching a video transcript.
Original: {question}
Rewritten:"""
    try:
        rewritten = rewrite_llm.invoke(prompt)
        result = str(rewritten).strip().split("\n")[0]
        return result if len(result) > 10 else question
    except:
        return question

# ==========================================
# RERANKING
# ==========================================
def rerank_docs(query, docs):
    """Rerank using cross-encoder"""
    if not docs:
        return docs
    
    pairs = [[query, doc.page_content] for doc in docs]
    scores = st.session_state.reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:3]]

# ==========================================
# RETRIEVAL DEMO INTERFACE
# ==========================================
if st.session_state.vector_store:
    st.subheader("🔍 Test Retrieval Pipeline")
    
    query = st.text_input("Enter a test query:", placeholder="What does the video say about...")
    
    if query:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Step 1: Query Rewriting**")
            rewritten = rewrite_query(query)
            st.write(f"Original: {query}")
            st.write(f"Rewritten: {rewritten}")
        
        with col2:
            st.markdown("**Step 2: MMR Retrieval**")
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5}
            )
            raw_docs = retriever.invoke(rewritten)
            st.write(f"Retrieved {len(raw_docs)} chunks")
            for i, doc in enumerate(raw_docs[:3]):
                st.write(f"{i+1}. [{doc.metadata['source']}] {doc.page_content[:100]}...")
        
        st.markdown("**Step 3: Cross-Encoder Reranking**")
        reranked = rerank_docs(rewritten, raw_docs)
        for i, doc in enumerate(reranked):
            st.write(f"**Rank {i+1}** [{doc.metadata['source']}] {doc.page_content[:150]}...")

else:
    st.info("👈 Enter a YouTube Video ID and click 'Load Video' to start!")