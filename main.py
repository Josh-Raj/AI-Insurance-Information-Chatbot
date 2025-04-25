import pdfplumber
import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import streamlit as st

if "response" not in st.session_state:
    st.session_state.response = None

# Chunk the text
def chunk_text(full_text,chunk_size = 500, overlap = 100):
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start = start + chunk_size - overlap
    return chunks

# Extract text from the PDF file
if not os.path.exists("full_text.txt"):
    with pdfplumber.open("Insurance Policy Data.pdf") as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    with open ("full_text.txt", "wb") as f:
        f.write(full_text.encode('utf-8'))

    # Create the Chunks
    chunks = chunk_text(full_text, chunk_size=500, overlap=100)

# Save the chunks to a JSON file    
if not os.path.exists("chunks.json"):
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
with open("chunks.json", "r") as f:
    chunks = json.load(f)

# Embed the chunks using SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()
@st.cache_resource
def load_embeddings():
    return model.encode(chunks,batch_size=16, show_progress_bar=False)
embeddings = load_embeddings()

if not os.path.exists("embeddings.json"):
    with open("embeddings.json", "w") as f:
        json.dump(embeddings.tolist(), f)

# Create a ChromaDB client and collection
@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(name="insurance_docs")


collection = load_chromadb()
collection.add(
    documents=chunks,
    embeddings=[emb.tolist() for emb in embeddings],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

st.title("Insurance Policy Chatbot")

# Get the input query from the user
query = st.text_input("Enter your query: ")
button = st.button("Submit")

if query and button:
    # Embed the  using SentenceTransformer
    with st.spinner("Processing your query..."):
        embedded_query = model.encode([query])[0].tolist()  # Convert numpy array to list
        result = collection.query(
            query_embeddings=[embedded_query],
            n_results=3
        )

        # Define the context and template for the LLM
        chunks = result["documents"][0]
        context = "\n\n".join(chunks)
        prompt = f"""
        You are a strict insurance assistant.
        You will only answer questions using the provided context below.
        If the answer is not found in the context, say exactly: "Not found in knowledge base." Do not use any external knowledge.
        ---
        Context: {context}
        ---
        Question: {query}
        Answer:
        """
        response = requests.post("http://localhost:11434/api/generate", json={
            # "model": "mistral",
            "model": "phi3:3.8b",
            "prompt": prompt,
            "stream": False
        })
        answer = response.json()["response"]
        st.session_state.response = answer

if st.session_state.response:
    st.markdown("### Response")
    st.write(st.session_state.response)        

if st.session_state.response == "Not found in knowledge base." :
    st.warning("The query cannot be answered using the available information.")

    with st.expander("Escalate to a Human Agent"):
        user_email = st.text_input("Your Email (for response):")
        issue_desc = st.text_area("Add any more context (optional):")
        send = st.button("Send to Agent",disabled=not user_email)
        if not user_email:
            st.warning("Please provide your email to escalate the issue.")

        if send and user_email:
            st.success("Your query has been escalated to a human agent.")
            with open("escalated_issues.txt", "a") as f:
                f.write(f"Query: {query}\nEmail: {user_email}\nDesc: {issue_desc}\n\n")

if st.session_state.response != "Not found in knowledge base." :
    with st.expander("🔍 Retrieved Chunks"):
        for chunk in chunks:
            st.markdown(f"> {chunk}")
