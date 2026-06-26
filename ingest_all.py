import json
import os
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, Settings, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import Settings

Settings.chunk_size = 512  # Increase from 128 to 512
Settings.chunk_overlap = 50

# 1. SETUP
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. LOAD SOURCE A: The Manual (PDFs)
print("📚 Loading Manual from ./data...")
pdf_documents = SimpleDirectoryReader("./data").load_data()
for doc in pdf_documents:
    doc.metadata.update({"source_type": "official_manual"})

# 3. LOAD SOURCE B: The Forum (JSON)
print("💬 Loading Forum from forum_data.json...")
with open('forum_data.json', 'r') as f:
    forum_json = json.load(f)

forum_documents = []
for entry in forum_json:
    doc = Document(
        text=entry['content'],
        metadata={
            "title": entry['title'],
            "url": entry['url'],
            "source_type": "community_forum"
        }
    )
    forum_documents.append(doc)


print("🌐 Pulling API Docs from web...")
api_docs = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://www.reaper.fm/sdk/reascript/reascripthelp.html"]
)
for doc in api_docs:
    doc.metadata.update({"source_type": "technical_api", "title": "ReaScript Lua API"})


# Combine all documents
all_docs = pdf_documents + forum_documents + api_docs

# 4. HIERARCHICAL PARSING (The Family Tree)
print("🌳 Building Hierarchical Nodes...")
# CHANGE: Increase the smallest chunk from 128 to 256
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 256])
nodes = node_parser.get_nodes_from_documents(all_docs)
leaf_nodes = get_leaf_nodes(nodes)

# 5. PERSISTENCE SETUP
# We explicitly create the docstore and add ALL nodes to it
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

db = chromadb.PersistentClient(path="./reaper_db")
chroma_collection = db.get_or_create_collection("reaper_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore
)

# 6. INDEXING
print(f"🚀 Indexing {len(leaf_nodes)} nodes into ChromaDB...")
index = VectorStoreIndex(
    leaf_nodes, 
    storage_context=storage_context, 
    show_progress=True
)

# CRITICAL: This is the step that was missing/corrupted
# It saves the 'docstore.json' so the app can read the family tree later
storage_context.persist(persist_dir="./reaper_db")

print("✅ SUCCESS: Unified Reaper Database Created!")