# backend_rag.py
import os
import re
import time
import json
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse  # FIXED: Added import
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

# Try different Pinecone import approaches
try:
    # New Pinecone SDK (v3+)
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_NEW = True
except ImportError:
    try:
        # Old Pinecone SDK (v2)
        import pinecone
        PINECONE_NEW = False
    except ImportError:
        raise ImportError("Pinecone package not installed. Run: pip install pinecone")

# ---------- Optional: File & Web Dependencies ----------
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    BeautifulSoup = None
    requests = None

# ---------- Config ----------
INDEX_NAME = "ycotes-rag"
NAMESPACE = "default"
DIMENSION = 1536
METRIC = "cosine"
REGION = "us-east-1"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# Pricing
USD_TO_INR = 84.0
USD_PER_TOKEN_EMBED = 0.02 / 1_000_000.0
USD_PER_TOKEN_CHAT_IN = 5.0 / 1_000_000.0
USD_PER_TOKEN_CHAT_OUT = 15.0 / 1_000_000.0

TOP_K = 6
MIN_SCORE = 0.25
MAX_CONTEXT_CHARS = 7000
DEFAULT_TTL_HOURS = 24 * 7

# ---------- Init ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in .env")
if not PINECONE_API_KEY:
    raise ValueError("Set PINECONE_API_KEY in .env")

oa = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone with compatibility for both versions
if PINECONE_NEW:
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pinecone.init(api_key=PINECONE_API_KEY)

def get_pinecone_index():
    """Get Pinecone index with version compatibility"""
    if PINECONE_NEW:
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
        
        if INDEX_NAME not in index_names:
            print(f"Creating index {INDEX_NAME}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(cloud="aws", region=REGION)
            )
            # Wait for index to be ready
            while not pc.describe_index(INDEX_NAME).status.ready:
                time.sleep(1)
        
        return pc.Index(INDEX_NAME)
    else:
        # Old Pinecone version
        if INDEX_NAME not in pinecone.list_indexes():
            print(f"Creating index {INDEX_NAME}...")
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC
            )
        
        return pinecone.Index(INDEX_NAME)

# Initialize index
index = get_pinecone_index()

# ---------- Cost Helpers ----------
def cost_usd_to_inr(usd): 
    return usd * USD_TO_INR

def print_embed_cost(tokens: int):
    usd = tokens * USD_PER_TOKEN_EMBED
    inr = cost_usd_to_inr(usd)
    print(f" 🧠 Embedding tokens: {tokens} | 💵 ${usd:.8f} | ₹{inr:.6f}")
    return usd, inr

def print_chat_cost(in_t: int, out_t: int):
    usd = in_t * USD_PER_TOKEN_CHAT_IN + out_t * USD_PER_TOKEN_CHAT_OUT
    inr = cost_usd_to_inr(usd)
    print(f" 💬 Tokens in={in_t} out={out_t} | 💵 ${usd:.6f} | ₹{inr:.4f}")
    return usd, inr

# ---------- Embedding ----------
def embed_text(text: str) -> Tuple[List[float], int]:
    r = oa.embeddings.create(model=EMBED_MODEL, input=text)
    vec = r.data[0].embedding
    tokens = r.usage.prompt_tokens
    print_embed_cost(tokens)
    return vec, tokens

# ---------- Chunking by Topic ----------
def chunk_by_topic(text: str) -> List[Dict[str, str]]:
    text = re.sub(r'\r\n|\r', '\n', text)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_title = "General"
    current_content = []
    heading_pattern = re.compile(r'^(#{1,3}\s+|Chapter\s+\d+[:\-]?\s*|Section\s+\d+[:\-]?\s*|[A-Z][A-Z\s]{5,50}:?)', re.IGNORECASE)

    for para in paragraphs:
        if heading_pattern.match(para):
            if current_content:
                chunks.append({'title': current_title, 'content': '\n\n'.join(current_content)})
            current_title = re.sub(r'^[#:\-\s]+', '', para).strip() or "Untitled"
            current_content = [para]
        else:
            current_content.append(para)
    if current_content:
        chunks.append({'title': current_title, 'content': '\n\n'.join(current_content)})
    return chunks

# ---------- File & URL Extraction ----------
def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf' and PyPDF2:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return '\n\n'.join([page.extract_text() or '' for page in reader.pages])
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.docx' and docx2txt:
            return docx2txt.process(filepath)
        elif ext == '.csv':
            import csv
            rows = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for row in csv.reader(f):
                    rows.append(', '.join(row))
            return '\n'.join(rows)
        else:
            raise ValueError(f"Unsupported file: {ext}")
    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
        return ""

def scrape_url(url: str) -> str:
    if not BeautifulSoup or not requests:
        print("⚠️ Install 'beautifulsoup4' and 'requests'")
        return ""
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(resp.content, 'lxml')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        return re.sub(r'\n\s*\n+', '\n\n', text).strip()
    except Exception as e:
        print(f"⚠️ Scraping failed: {e}")
        return ""

# ---------- Upsert with TTL ----------
def upsert_chunks(chunks: List[Dict[str, str]], source: str = "unknown", ttl_hours: int = DEFAULT_TTL_HOURS):
    expiry = None
    if ttl_hours > 0:
        expiry = int((datetime.utcnow() + timedelta(hours=ttl_hours)).timestamp())
    
    for i, chunk in enumerate(chunks):
        content = chunk['content'][:10000]
        title = chunk['title'][:200]
        if not content.strip():
            continue
        
        vec, _ = embed_text(content)
        doc_id = f"{source}_{int(time.time())}_{i}"
        metadata = {
            "title": title,
            "text": content,
            "source": source,
            "created_at": int(time.time())
        }
        if expiry:
            metadata["expires_at"] = expiry
        
        # Upsert with version compatibility
        if PINECONE_NEW:
            index.upsert(
                vectors=[{"id": doc_id, "values": vec, "metadata": metadata}],
                namespace=NAMESPACE
            )
        else:
            index.upsert(
                vectors=[(doc_id, vec, metadata)],
                namespace=NAMESPACE
            )
        
        print(f"✅ Upserted: {title[:50]}...")

# ---------- Retrieval (with TTL filter) ----------
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    qvec, _ = embed_text(query)
    current_ts = int(time.time())
    
    if PINECONE_NEW:
        res = index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
            filter={"expires_at": {"$gt": current_ts}} if DEFAULT_TTL_HOURS > 0 else None
        )
        matches = res.get("matches", [])
    else:
        # Old Pinecone version
        res = index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE
        )
        matches = res.get("matches", [])
        # Manual TTL filtering for old version
        if DEFAULT_TTL_HOURS > 0:
            matches = [m for m in matches if m.get("metadata", {}).get("expires_at", float('inf')) > current_ts]
    
    return [m for m in matches if (m.get("metadata") or {}).get("text")]

def build_context(matches: List[Dict]) -> str:
    matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)
    parts, size = [], 0
    for i, m in enumerate(matches, 1):
        t = (m.get("metadata") or {}).get("text", "")
        if not t:
            continue
        chunk = f"[{i} | score={m.get('score', 0):.3f}]\n{t}\n"
        if size + len(chunk) > MAX_CONTEXT_CHARS:
            break
        parts.append(chunk)
        size += len(chunk)
    return "\n".join(parts)

# ---------- LLM Answer ----------
def ask_llm(question: str, context: str = "", style: str = "concise", lang: str = "en") -> Tuple[str, int, int]:
    lang_instr = "Answer in Hindi using Devanagari script." if lang == "hi" else "Answer in English."
    style_instr = "Provide detailed explanation with examples." if style == "detailed" else "Keep answer concise."
    sys_prompt = f"You are Ycotes, an AI tutor. {lang_instr} {style_instr} Use context if provided."
    user_prompt = f"Question:\n{question}\n\nContext:\n{context}" if context else f"Question:\n{question}"
    
    r = oa.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.4 if style == "concise" else 0.7,
        max_tokens=300 if style == "concise" else 800
    )
    ans = r.choices[0].message.content.strip()
    usage = r.usage
    in_t, out_t = usage.prompt_tokens, usage.completion_tokens
    print_chat_cost(in_t, out_t)
    return ans, in_t, out_t

def answer(question: str, style: str = "concise", lang: str = "en") -> str:
    print(f"\n🔍 Retrieving from Pinecone for: {question}")
    matches = retrieve(question)
    strong = [m for m in matches if m.get("score", 0) >= MIN_SCORE]
    
    if not strong:
        print(" No strong matches — asking LLM directly.")
        ans, _, _ = ask_llm(question, style=style, lang=lang)
        return ans
        
    print(f"✅ {len(strong)} relevant chunks found.")
    ctx = build_context(strong)
    ans, _, _ = ask_llm(question, context=ctx, style=style, lang=lang)
    return ans

# ---------- Socratic Explainer ----------
def generate_sub_questions(main_question: str, lang: str = "en") -> List[str]:
    lang_prompt = "Generate questions in Hindi." if lang == "hi" else "Generate questions in English."
    prompt = f"""
Generate exactly 3 foundational sub-questions based on this main question: "{main_question}"

Rules:
- Generate exactly 3 questions
- Make them simple and foundational
- {lang_prompt}
- Respond ONLY with the 3 questions, one per line
- No numbering, no bullets, no extra text

Example output:
What is a data structure?
What is an algorithm?
Why are data structures important?
"""
    try:
        r = oa.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        raw = r.choices[0].message.content.strip()
        
        cleaned_questions = []
        for line in raw.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-\*]\s*', '', line)
            
            if len(line) > 150 or not line.endswith('?'):
                continue
                
            cleaned_questions.append(line)

        if len(cleaned_questions) == 0:
            return [f"What is {main_question}?"]

        return cleaned_questions[:3]
    except Exception as e:
        print(f"⚠️ Sub-question gen failed: {e}")
        return [f"What is {main_question}?"]

# Export for Streamlit app
VOICE_AVAILABLE = False