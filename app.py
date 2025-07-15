import streamlit as st # type: ignore
from your_module import (
    extract_structured_query,
    retrieve_relevant_chunks,
    generate_decision,
    build_vector_store,
    extract_documents
)

# Load documents and build embeddings (run only once)
@st.cache_resource
def load_vector_store():
    docs = extract_documents("documents")  # your PDF, DOCX, EML folder
    index, chunks, metadata = build_vector_store(docs)
    return index, chunks, metadata

# UI Starts here
st.set_page_config(page_title="LLM Insurance Evaluator", layout="wide")
st.title("📄 Smart Policy Evaluator using LLMs")

st.markdown("""
Enter any insurance-style natural language query below.
Example:  
`46-year-old male, knee surgery in Pune, 3-month-old insurance policy`
""")

query = st.text_area("🔍 Enter Query:", height=100)

if st.button("🧠 Evaluate Now"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Processing..."):

            # Step 1: Load vector index
            index, chunks, metadata = load_vector_store()

            # Step 2: Extract structured fields
            structured = extract_structured_query(query)

            # Step 3: Retrieve similar clauses
            relevant_clauses = retrieve_relevant_chunks(structured, index, chunks, metadata)

            # Step 4: LLM evaluation
            decision = generate_decision(query, relevant_clauses)

        # --- Display results ---
        st.subheader("📋 Structured Query")
        st.json(structured)

        st.subheader("📑 Retrieved Clauses")
        for i, chunk in enumerate(relevant_clauses):
            st.markdown(f"**Clause #{i+1}** — from `{chunk['metadata']['filename']}`")
            st.code(chunk["text"])

        st.subheader("✅ Final Decision")
        st.json(decision)
