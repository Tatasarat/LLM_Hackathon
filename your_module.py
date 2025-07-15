
import os
import re
import pdfplumber # type: ignore
import docx # type: ignore
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore
import numpy as np # type: ignore
import spacy # type: ignore
import openai # type: ignore
import json

# ========== STEP 1: Extract Text from Documents ==========
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_eml(path):
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    body = msg.get_body(preferencelist=('html'))
    if body:
        html = body.get_content()
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    return None  # Return None if no valid HTML body is found

def extract_documents(folder="documents"):
    extracted_docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(path)
        elif file.endswith(".eml"):
            text = extract_text_from_eml(path)
        else:
            continue
        if text:  # Skip files where text extraction returns None
            extracted_docs.append((file, text))
    return extracted_docs

# ========== STEP 2: Chunking and Embedding ==========
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=1000):
    sentences = text.split('\n')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_vector_store(documents):
    all_chunks = []
    chunk_metadata = []
    for filename, text in documents:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({"filename": filename, "chunk_index": i})

    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, all_chunks, chunk_metadata

# ========== STEP 3: Structured Query Extraction ==========
nlp = spacy.load("en_core_web_sm")

def extract_structured_query(query):
    doc = nlp(query.lower())
    result = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None
    }

    age_match = re.search(r'(\d+)\s*-?\s*(year)?\s*(old)?', query)
    if age_match:
        result["age"] = int(age_match.group(1))

    if "male" in query.lower() or "\b46m\b" in query.lower():
        result["gender"] = "male"
    if "female" in query.lower() or "\b46f\b" in query.lower():
        result["gender"] = "female"

    for ent in doc.ents:
        if ent.label_ == "GPE":
            result["location"] = ent.text

    procedures = [chunk.text for chunk in doc.noun_chunks if "surgery" in chunk.text or "treatment" in chunk.text]
    if procedures:
        result["procedure"] = procedures[0]

    match = re.search(r"(\d+)\s*-?\s*(month|year)", query)
    if match:
        result["policy_duration"] = f"{match.group(1)} {match.group(2)}"

    return result

# ========== STEP 4: Semantic Retrieval ==========
def build_semantic_query(structured_query):
    query_text = (
        f"{structured_query.get('age', '')}-year-old "
        f"{structured_query.get('gender', '')}, "
        f"{structured_query.get('procedure', '')} in "
        f"{structured_query.get('location', '')}, "
        f"{structured_query.get('policy_duration', '')} insurance policy"
    )
    return query_text.strip()

def retrieve_relevant_chunks(structured_query, index, chunks, metadata, top_k=5):
    query_text = build_semantic_query(structured_query)
    query_embedding = embedder.encode([query_text])
    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        results.append({
            "text": chunks[idx],
            "metadata": metadata[idx]
        })

    return results

# ========== STEP 5: LLM Decision Making ==========
openai.api_key = "your-api-key-here"  # Replace with your actual API key

def generate_decision(query, retrieved_chunks):
    context_text = "\n".join(
        [f"{i+1}. {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
    )

    system_prompt = "You are an expert policy evaluator. Use the context to evaluate insurance eligibility."

    user_prompt = f"""
### Query:
{query}

### Context:
{context_text}

### Task:
Determine the decision based on the context. Return a JSON:
{{
  "decision": "approved or rejected or partial",
  "amount": "\u20b9 amount or covered / not covered",
  "justification": "Explain why this is the decision, quoting relevant clause numbers if possible.",
  "referenced_clauses": ["clause numbers or chunk numbers"]
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    reply = response['choices'][0]['message']['content']
    try:
        return json.loads(reply)
    except:
        return {"error": "Invalid JSON from LLM", "raw_output": reply}