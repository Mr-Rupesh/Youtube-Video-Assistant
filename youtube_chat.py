import os
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load env
load_dotenv()

# ==========================================
# INITIALIZE SESSION STATE FIRST
# ==========================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "reranker" not in st.session_state:
    from sentence_transformers import CrossEncoder
    st.session_state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("🎬 YouTube RAG Chatbot")
st.markdown("Chat with any YouTube video or just hang out!")

# ==========================================
# SIDEBAR - VIDEO INPUT & MEMORY
# ==========================================
with st.sidebar:
    st.header("📹 Video Settings")
    
    # Use current video_id from state or default
    current_video_id = st.session_state.video_id if st.session_state.video_id else "aircAruvnKk"
    video_id = st.text_input("YouTube Video ID", value=current_video_id, help="The part after v= in YouTube URL")
    
    if st.button("🔄 Load Video"):
        # Clear video data but keep chat history
        st.session_state.vector_store = None
        st.session_state.video_id = None
        st.rerun()
    
    st.markdown("---")
    
    # Status - check AFTER initialization
    st.markdown("**Status:**")
    if st.session_state.vector_store is not None and st.session_state.video_id == video_id:
        st.markdown(f"✅ Video loaded: `{st.session_state.video_id}`")
    else:
        st.markdown("⏳ No video loaded")
    
    # Memory management
    st.markdown("---")
    st.header("🧠 Memory")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
    
    msg_count = len(st.session_state.conversation_memory)
    st.markdown(f"Messages in memory: {msg_count}")

# ==========================================
# LOAD VIDEO (runs when needed)
# ==========================================
needs_loading = (
    st.session_state.vector_store is None or 
    st.session_state.video_id != video_id
)

if needs_loading:
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
            st.session_state.video_id = video_id
            
            # Add system message about video
            video_msg = f"📺 Loaded video: {video_id} ({len(documents)} chunks)"
            st.session_state.chat_history.append({"role": "assistant", "content": video_msg})
            
            st.sidebar.success(f"✅ Loaded {len(documents)} chunks!")
            st.rerun()  # Refresh to update sidebar status
            
        except TranscriptsDisabled:
            st.sidebar.error("❌ No captions available")
            st.session_state.video_id = None
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.session_state.video_id = None

# ==========================================
# LLM SETUP
# ==========================================
@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3",
        task="conversational",
        temperature=0.8,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)

@st.cache_resource
def get_rewrite_llm():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=100,
        temperature=0.3,
    )
    return llm

llm = get_llm()
rewrite_llm = get_rewrite_llm()

# ==========================================
# MEMORY & RETRIEVAL FUNCTIONS
# ==========================================
def get_conversation_context(max_messages=6):
    """Get recent conversation history for context"""
    recent = st.session_state.conversation_memory[-max_messages:]
    context = ""
    for msg in recent:
        role = "User" if msg["type"] == "human" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    return context

def add_to_memory(user_msg, assistant_msg):
    """Add exchange to conversation memory"""
    st.session_state.conversation_memory.append({
        "type": "human",
        "content": user_msg
    })
    st.session_state.conversation_memory.append({
        "type": "ai",
        "content": assistant_msg
    })

def rewrite_query(question):
    """Rewrite query considering conversation history"""
    context = get_conversation_context(max_messages=4)
    
    if not context:
        prompt = f"""Rewrite this question to be more specific for searching a video transcript.
Original: {question}
Rewritten:"""
    else:
        prompt = f"""Given this conversation history:
{context}

Rewrite the user's latest question to be more specific for searching a video transcript, considering the context.
Original: {question}
Rewritten:"""
    
    try:
        rewritten = rewrite_llm.invoke(prompt)
        result = str(rewritten).strip().split("\n")[0]
        return result if len(result) > 10 else question
    except:
        return question

def rerank_docs(query, docs):
    """Rerank using cross-encoder"""
    if not docs:
        return docs
    
    pairs = [[query, doc.page_content] for doc in docs]
    scores = st.session_state.reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:3]]

def format_docs(docs):
    """Format with source citations"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        text = doc.page_content
        formatted.append(f"[{source}] {text}")
    return "\n\n".join(formatted)

def is_video_question(question):
    """Detect if user is asking about the video or just chatting"""
    video_keywords = [
        "video", "say", "mention", "talk about", "discuss", "explain", 
        "what does", "how does", "why does", "when did", "who is", 
        "transcript", "speaker", "author", "content", "topic", "learn",
        "it", "that", "this", "those", "these"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in video_keywords)

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_question = st.chat_input("Type anything... ask about the video or just chat!")

if user_question:
    # Show user message
    st.chat_message("user").markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    with st.spinner(""):
        
        # Get conversation context
        conversation_context = get_conversation_context(max_messages=6)
        
        # Check if video is loaded and question is video-related
        video_context = ""
        sources = []
        use_video = False
        
        if st.session_state.vector_store:
            is_followup = any(word in user_question.lower() for word in ["it", "that", "this", "those", "these", "he", "she", "they"])
            
            if is_video_question(user_question) or is_followup:
                use_video = True
                rewritten = rewrite_query(user_question)
                
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5}
                )
                raw_docs = retriever.invoke(rewritten)
                reranked_docs = rerank_docs(rewritten, raw_docs)
                
                if reranked_docs:
                    video_context = format_docs(reranked_docs)
                    sources = reranked_docs
        
        # Build dynamic prompt with memory
        if video_context and conversation_context:
            prompt = f"""You're chatting with a user. You remember the conversation and have access to a YouTube video transcript.

Conversation history:
{conversation_context}

Video context:
{video_context}

User's latest message: {user_question}

Respond naturally, referencing the conversation history when relevant. Use video info with timestamps [MM:SS] when discussing content. Be conversational and friendly."""
        
        elif video_context:
            prompt = f"""You're chatting with a user about a YouTube video.

Video context:
{video_context}

User: {user_question}

Respond naturally using the video info. Add timestamps [MM:SS] when referencing specific parts. Be conversational and friendly."""
        
        elif conversation_context:
            prompt = f"""You're having a conversation with a user. You remember what was discussed.

Conversation history:
{conversation_context}

User's latest message: {user_question}

Respond naturally, referencing previous parts of the conversation when relevant. Be friendly and engaging."""
        
        else:
            prompt = f"""You're having a casual conversation with a user.

User: {user_question}

Respond naturally and conversationally. Be friendly and engaging."""
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Optional: subtle source indicator only for video questions
            if sources and use_video:
                with st.expander("📚 Sources", expanded=False):
                    for doc in sources:
                        st.write(f"**[{doc.metadata['source']}]** {doc.page_content[:100]}...")
        
        # Save to both histories
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        add_to_memory(user_question, answer)

if not st.session_state.vector_store:
    st.info("👈 Load a YouTube video to enable video Q&A, or just chat with me!")