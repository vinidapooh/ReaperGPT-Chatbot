import streamlit as st
import chromadb
import os
import sys
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings, 
    PromptTemplate
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. THE EXPERT PROMPT (Few-Shot Pattern) ---
REAPER_EXPERT_PROMPT = PromptTemplate(
    "You are the Reaper DAW Technical Support Bot. Your goal is to provide high-readability, actionable instructions.\n"
    "--- STRICT FORMATTING RULES ---\n"
    "1. Start with a 1-sentence summary.\n"
    "2. All steps MUST be in a numbered list (1., 2., 3.).\n"
    "3. Bold all UI elements, menus, and buttons (e.g., **Options > Preferences**, **FX Button**).\n"
    "4. If a step is complex, use a bullet point (-) underneath the numbered step for detail.\n"
    "\n"
    "--- PATTERN EXAMPLES ---\n"
    "User Query: How do I create a new track?\n"
    "Answer: You can insert tracks quickly to start recording.\n"
    "1. Double-click the empty space in the **Track Control Panel** (TCP).\n"
    "2. Alternatively, navigate to **Track > Insert New Track** or use the shortcut **Cmd+T**.\n"
    "\n"
    "User Query: How do I set up a Reverb Bus?\n"
    "Answer: Routing to a dedicated bus allows you to process multiple tracks with one effect.\n"
    "1. Create a new track and name it **Reverb Bus**.\n"
    "2. Click the **FX** button on the Reverb track and add your plugin (e.g., **ReaVerb**).\n"
    "3. On your source track, click the **Route** button and drag the 'send' cable to the **Reverb Bus** track.\n"
    "--- END OF EXAMPLES ---\n"
    "\n"
    "Context from manual and forums:\n"
    "{context_str}\n"
    "\n"
    "User Query: {query_str}\n"
    "Step-by-Step Instructions:"
)

# --- 2. SYSTEM INITIALIZATION ---
@st.cache_resource
def init_system():
    # 1. Serverless Cloud Embeddings: Zero download time, zero memory footprint
    # Uses the same model but runs it completely over API via Hugging Face's free tier
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    )
    
    # 2. Redirect LlamaIndex orchestration directly to Groq's high-speed free tier
    Settings.llm = OpenAILike(
        model="llama3-70b-8192", 
        api_base="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        is_chat_model=True,
        timeout=300.0
    )
    
    # Connect to ChromaDB
    db = chromadb.PersistentClient(path="./reaper_db")
    chroma_collection = db.get_collection("reaper_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load Storage Context (This pulls in the docstore.json family tree)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir="./reaper_db" 
    )
    
    # Build Index from the existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
    
    # SETUP: AutoMergingRetriever (The "Zoom-Out" Logic)
    base_retriever = index.as_retriever(similarity_top_k=8) 
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context, 
        verbose=True
    )
    
    # Create the Query Engine with our custom prompt
    return RetrieverQueryEngine.from_args(
        retriever, 
        streaming=True,
        text_qa_template=REAPER_EXPERT_PROMPT
    )

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="ReaperGPT Pro", page_icon="🎸")
st.title("🎸 ReaperGPT Pro")
st.caption("Expert knowledge from the Manual + Community Forums")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Engine
query_engine = init_system()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help with Reaper today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Query the engine
        response = query_engine.query(prompt)
        
        # Stream the response
        for token in response.response_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        
        # DISPLAY SOURCES
        if response.source_nodes:
            with st.expander("🔍 View Knowledge Sources"):
                for node in response.source_nodes:
                    meta = node.metadata
                    source_type = meta.get('source_type', 'unknown').replace('_', ' ').title()
                    title = meta.get('title', 'Reaper Technical Docs')
                    url = meta.get('url', None)
                    
                    icon = "📚" if "Manual" in source_type else "💬"
                    color = "blue" if "Manual" in source_type else "green"
                    
                    st.markdown(f":{color}[**{icon} {source_type}**] — {title}")
                    if url:
                        st.caption(f"[Open Forum Thread]({url})")
                    st.write(node.get_content()[:250] + "...")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": full_response})