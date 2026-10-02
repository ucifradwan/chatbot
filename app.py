
import os
import json
import hashlib
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
    page_title="Like It Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Like It Chatbot")
st.caption("اسأل عن المعلومات الموجودة في الملفات")


# =========================================================
# GEMINI
# =========================================================

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


# =========================================================
# PATHS
# =========================================================

KNOWLEDGE_FOLDER = "knowledge"
CACHE_FILE = "rag_cache.json"


# =========================================================
# FILE READING
# =========================================================

def read_pdf(path):

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

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

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

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
# FILE HASH
# =========================================================

def get_file_hash(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


# =========================================================
# CHUNKING
# =========================================================

def split_text(
    text,
    chunk_size=1200,
    overlap=200
):

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
# CACHE
# =========================================================

def load_cache():

    if not os.path.exists(CACHE_FILE):

        return {}

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_cache(cache):

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cache,
            file,
            ensure_ascii=False
        )


# =========================================================
# BUILD / UPDATE KNOWLEDGE
# =========================================================

def build_knowledge():

    cache = load_cache()

    current_files = {}

    all_chunks = []

    if not os.path.exists(
        KNOWLEDGE_FOLDER
    ):

        return [], np.array([])

    # -----------------------------------------
    # Read current files
    # -----------------------------------------

    for filename in sorted(
        os.listdir(KNOWLEDGE_FOLDER)
    ):

        path = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        if not filename.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):
            continue

        file_hash = get_file_hash(path)

        current_files[filename] = file_hash

        # -------------------------------------
        # File unchanged
        # -------------------------------------

        if (
            filename in cache
            and cache[filename].get("hash")
            == file_hash
        ):

            file_data = cache[filename]

            all_chunks.extend(
                file_data["chunks"]
            )

            continue

        # -------------------------------------
        # File changed / new
        # -------------------------------------

        text = read_file(path)

        if not text.strip():

            continue

        chunks = split_text(text)

        texts = [
            chunk
            for chunk in chunks
        ]

        embeddings = []

        # -------------------------------------
        # Create embeddings in batches
        # -------------------------------------

        batch_size = 20

        for i in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                i:i + batch_size
            ]

            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )

            for embedding in result.embeddings:

                embeddings.append(
                    embedding.values
                )

        file_chunks = []

        for index, chunk in enumerate(
            chunks
        ):

            file_chunks.append({
                "text": chunk,
                "filename": filename,
                "chunk": index,
                "embedding": embeddings[index]
            })

        cache[filename] = {
            "hash": file_hash,
            "chunks": file_chunks
        }

        all_chunks.extend(
            file_chunks
        )

    # -----------------------------------------
    # Remove deleted files
    # -----------------------------------------

    deleted_files = [
        filename
        for filename in cache
        if filename not in current_files
    ]

    for filename in deleted_files:

        del cache[filename]

    save_cache(cache)

    # -----------------------------------------
    # Build embedding matrix
    # -----------------------------------------

    if not all_chunks:

        return [], np.array([])

    matrix = np.array(
        [
            item["embedding"]
            for item in all_chunks
        ],
        dtype=np.float32
    )

    return all_chunks, matrix


# =========================================================
# LOAD KNOWLEDGE
# =========================================================

@st.cache_resource(
    show_spinner="جاري تجهيز قاعدة المعرفة..."
)
def load_knowledge():

    chunks, embeddings = build_knowledge()

    if len(embeddings) > 0:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        embeddings = (
            embeddings / norms
        )

    return chunks, embeddings


chunks, embeddings = load_knowledge()


# =========================================================
# SEARCH
# =========================================================

def search_knowledge(
    question,
    top_k=5
):

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

    norm = np.linalg.norm(
        query_vector
    )

    if norm == 0:

        return []

    query_vector /= norm

    scores = embeddings @ query_vector

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "filename": chunks[index]["filename"],
            "score": float(
                scores[index]
            )
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

        if not retrieved:

            answer = (
                "المعلومة دي غير موجودة "
                "في الملفات المتاحة."
            )

        else:

            best_score = retrieved[0]["score"]

            # ---------------------------------
            # Relevance threshold
            # ---------------------------------

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
أنت مساعد يعتمد فقط على المعلومات
الموجودة في المصادر التالية.

القواعد:

- أجب فقط من المصادر.
- لا تخترع معلومات.
- لا تستخدم معرفتك العامة.
- إذا لم تجد الإجابة بوضوح في المصادر،
  قل:
  "المعلومة دي غير موجودة في الملفات المتاحة."
- أجب بالعربية.
- كن واضحًا ومباشرًا.
- لا تذكر تفاصيل تقنية عن النظام.

SOURCES:

{context}

QUESTION:

{question}
"""

                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt
                )

                answer = response.text

    with st.chat_message("assistant"):

        st.write(answer)