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
    page_title="Like It in English AI Tutor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Like It in English AI Tutor")
st.caption("Your personal English learning assistant")


# =========================================================
# GEMINI
# =========================================================

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


# =========================================================
# FILES
# =========================================================

KNOWLEDGE_FOLDER = "knowledge"
CACHE_FILE = "rag_cache.json"


# =========================================================
# READ PDF
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


# =========================================================
# READ DOCX
# =========================================================

def read_docx(path):

    document = Document(path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


# =========================================================
# READ TXT
# =========================================================

def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# READ FILE
# =========================================================

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
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge():

    cache = load_cache()

    current_files = {}

    all_chunks = []

    if not os.path.exists(KNOWLEDGE_FOLDER):

        return [], np.array([])

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

        # -----------------------------------------
        # File did not change
        # -----------------------------------------

        if (
            filename in cache
            and cache[filename].get("hash")
            == file_hash
        ):

            all_chunks.extend(
                cache[filename]["chunks"]
            )

            continue

        # -----------------------------------------
        # New or changed file
        # -----------------------------------------

        text = read_file(path)

        if not text.strip():
            continue

        chunks = split_text(text)

        embeddings = []

        batch_size = 20

        for i in range(
            0,
            len(chunks),
            batch_size
        ):

            batch = chunks[
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

        for index, chunk in enumerate(chunks):

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
    # Embedding matrix
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
    show_spinner="Preparing English knowledge base..."
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
# SEMANTIC SEARCH
# =========================================================

def search_knowledge(
    question,
    top_k=6
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

            "text":
                chunks[index]["text"],

            "filename":
                chunks[index]["filename"],

            "score":
                float(scores[index])

        })

    return results


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask me anything about English..."
)


if question:

    # -----------------------------------------
    # Display user question
    # -----------------------------------------

    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # =====================================================
    # SEARCH FILES
    # =====================================================

    retrieved = search_knowledge(
        question,
        top_k=6
    )


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context_parts = []

    for item in retrieved:

        context_parts.append(

            f"""
SOURCE FILE: {item["filename"]}

{item["text"]}
"""

        )

    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # CONVERSATION HISTORY
    # =====================================================

    history = ""

    previous_messages = (
        st.session_state.messages[-8:-1]
    )

    for message in previous_messages:

        history += (
            f'{message["role"].upper()}: '
            f'{message["content"]}\n'
        )


    # =====================================================
    # ENGLISH TUTOR PROMPT
    # =====================================================

    prompt = f"""
You are an English-only AI Tutor.

Your job is to help students learn English,
understand English lessons, and solve English
questions.

==================================================
MOST IMPORTANT RULE: USER FILES HAVE PRIORITY
==================================================

The SOURCE FILES below are the student's
official study material.

ALWAYS give the uploaded files the highest priority.

If the answer is supported by the files,
use the files as the primary source.

Preserve the terminology, rules, examples,
and explanations used in the uploaded files.

Do NOT silently replace the files' explanation
with your own version.

If the files contain a specific rule or answer,
follow the files.

If the files do NOT contain enough information,
you may use your general English knowledge ONLY
to help answer the question.

When you use general English knowledge because
the files do not provide enough information,
make that clear.

==================================================
SUBJECT RESTRICTION
==================================================

You are ONLY an English-language tutor.

You can help with:

- English grammar
- Vocabulary
- Reading comprehension
- Writing
- Sentence correction
- Translation between Arabic and English
- English exercises
- English exam questions
- Pronunciation explanations
- Tenses
- Parts of speech
- Phrasal verbs
- Idioms
- Sentence structure
- English literature when it appears in
  the uploaded study material

Do NOT act as a tutor for mathematics,
physics, chemistry, programming, or other
school subjects.

If the student asks about another subject,
politely say:

"I'm an English tutor, so I can only help
with English-related questions."

==================================================
WHEN THE STUDENT SAYS "I DON'T UNDERSTAND"
==================================================

If the student says:

"I don't understand"
"I don't get it"
"I'm confused"
"مش فاهم"
"مش فاهمة"
"لسه مش فاهم"

DO NOT simply repeat the previous answer.

Instead:

1. Identify the difficult part from the
   conversation.
2. Explain it using simpler English.
3. If appropriate, explain briefly in Arabic
   to make the concept clearer.
4. Give a very simple example.
5. Connect the example to the original lesson.
6. If they still do not understand, explain
   using a different method.

Be patient and encouraging.

==================================================
WHEN THE STUDENT SENDS AN ENGLISH QUESTION
==================================================

If the student sends an English exercise,
SOLVE IT.

Do not just give the final answer.

For grammar questions:

1. Give the correct answer.
2. Explain the grammar rule.
3. Explain why the other choices are wrong
   when useful.
4. Give a short example.

For vocabulary questions:

1. Give the meaning.
2. Give the meaning in Arabic if useful.
3. Give an example sentence.

For reading questions:

1. Find the answer from the provided text
   or uploaded material.
2. Explain why it is correct.
3. Do not invent information.

For writing questions:

1. Correct the sentence/text.
2. Explain the important mistakes.
3. Provide the improved version.

==================================================
IMPORTANT: EXAM / QUESTION SOLVING
==================================================

When the student gives you a question,
do not refuse simply because it is a question.

Help them solve it step by step.

If the answer exists in the uploaded files,
prioritize that answer and explanation.

==================================================
IF THE FILES DO NOT CONTAIN THE ANSWER
==================================================

Do not pretend that the answer came from
the files.

You may use your general English knowledge
for normal English questions.

However, clearly distinguish it from the
uploaded material.

For example:

"According to your uploaded material: ..."

or

"Your files don't cover this point.
In standard English, ..."

==================================================
IF THE QUESTION IS UNCLEAR
==================================================

Do not guess.

Ask the student to provide the missing
sentence, question, choices, or context.

==================================================
LANGUAGE
==================================================

The subject is English.

Use English for:

- English examples
- Grammar rules
- Vocabulary
- Exercises
- Correct answers

You may use simple Arabic explanations when
the student is clearly struggling or asks
in Arabic.

Do not turn the chatbot into a general Arabic
assistant.

==================================================
STYLE
==================================================

Be:

- Friendly
- Patient
- Clear
- Educational
- Encouraging

Avoid unnecessarily complicated explanations.

Use short sections and examples.

Never make the student feel embarrassed
for asking a basic question.

==================================================
UPLOADED SOURCE MATERIAL
==================================================

{context}

==================================================
RECENT CONVERSATION
==================================================

{history}

==================================================
CURRENT STUDENT QUESTION
==================================================

{question}
"""


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    if not question.strip():

        answer = "Please send me your English question."

    else:

        response = client.models.generate_content(

            model=GENERATION_MODEL,

            contents=prompt

        )

        answer = response.text


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        st.write(answer)


    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })