```python
import os
import json
import numpy as np
import streamlit as st

from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document


# =========================================================
# SETTINGS
# =========================================================

st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Knowledge Chatbot")
st.caption("اسأل عن المعلومات الموجودة في الملفات")


# =========================================================
# GEMINI
# =========================================================

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


# =========================================================
# FILE READING
# =========================================================

def read_pdf(path):

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    return "\n\n".join(pages)


def read_docx(path):

    document = Document(path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


def read_txt(path):

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_file(path):

    filename = path.lower()

    if filename.endswith(".pdf"):
        return read_pdf(path)

    if filename.endswith(".docx"):
        return read_docx(path)

    if filename.endswith(".txt"):
        return read_txt(path)

    return ""


# =========================================================
# CHUNKING
# =========================================================

def split_text(text, chunk_size=1200, overlap=200):

    text = text.strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE KNOWLEDGE BASE
# =========================================================

@st.cache_data(show_spinner="جاري تجهيز قاعدة المعرفة...")
def build_knowledge():

    folder = "knowledge"

    all_chunks = []

    if not os.path.exists(folder):
        return []

    for filename in sorted(os.listdir(folder)):

        path = os.path.join(folder, filename)

        if not filename.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):
            continue

        text = read_file(path)

        if not text.strip():
            continue

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "filename": filename,
                "chunk": chunk_number
            })

    return all_chunks


chunks = build_knowledge()


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

@st.cache_resource(show_spinner="جاري إنشاء الـ Embeddings...")
def create_embeddings(chunks_data):

    if not chunks_data:
        return np.array([])

    embeddings = []

    batch_size = 20

    for i in range(0, len(chunks_data), batch_size):

        batch = chunks_data[i:i + batch_size]

        texts = [
            item["text"]
            for item in batch
        ]

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        for embedding in result.embeddings:
            embeddings.append(embedding.values)

    return np.array(embeddings, dtype=np.float32)


if chunks:

    embeddings = create_embeddings(chunks)

else:

    embeddings = np.array([])


# =========================================================
# NORMALIZE
# =========================================================

def normalize_vectors(vectors):

    if len(vectors) == 0:
        return vectors

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    return vectors / norms


if len(embeddings) > 0:

    embeddings = normalize_vectors(embeddings)


# =========================================================
# SEARCH
# =========================================================

def search_knowledge(question, top_k=5):

    if len(embeddings) == 0:
        return []

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    query_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return []

    query_vector = query_vector / query_norm

    scores = embeddings @ query_vector

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "filename": chunks[index]["filename"],
            "score": float(scores[index])
        })

    return results


# =========================================================
# CHAT
# =========================================================

question = st.chat_input(
    "اكتب سؤالك هنا..."
)


if question:

    with st.chat_message("user"):
        st.write(question)

    if not chunks:

        answer = (
            "لا توجد ملفات معرفة مضافة حاليًا."
        )

    else:

        retrieved = search_knowledge(
            question,
            top_k=5
        )

        # ---------------------------------------------
        # Minimum relevance check
        # ---------------------------------------------

        if not retrieved:

            answer = (
                "المعلومة دي غير موجودة "
                "في الملفات المتاحة."
            )

        else:

            best_score = retrieved[0]["score"]

            if best_score < 0.25:

                answer = (
                    "المعلومة دي غير موجودة "
                    "في الملفات المتاحة."
                )

            else:

                context_parts = []

                for item in retrieved:

                    context_parts.append(
                        f"""
SOURCE: {item["filename"]}

{item["text"]}
"""
                    )

                context = "\n\n".join(
                    context_parts
                )

                prompt = f"""
أنت مساعد ذكي يعتمد فقط على المعلومات
الموجودة في المصادر التي يتم إعطاؤها لك.

القواعد:

1. أجب فقط من المعلومات الموجودة في SOURCES.
2. لا تستخدم معلوماتك العامة للإجابة.
3. لا تخترع أي معلومة.
4. إذا كانت المصادر لا تحتوي على إجابة واضحة،
   قل:
   "المعلومة دي غير موجودة في الملفات المتاحة."
5. أجب باللغة العربية.
6. اجعل الإجابة واضحة ومباشرة.
7. لا تذكر للمستخدم تفاصيل تقنية عن الـRAG.

SOURCES:

{context}

USER QUESTION:

{question}
"""

                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt
                )

                answer = response.text

    with st.chat_message("assistant"):
        st.write(answer)
```
