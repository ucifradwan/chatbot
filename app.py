import streamlit as st
import os
import json
import hashlib
import time
import numpy as np
import requests
import bcrypt

from pathlib import Path
from pypdf import PdfReader
from docx import Document

from google import genai
from google.genai import types


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="English AI Tutor",
    page_icon="📚",
    layout="centered"
)


# =========================================================
# CONFIG
# =========================================================

GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

KNOWLEDGE_FOLDER = "knowledge"
CACHE_FILE = "rag_cache.json"

TOP_K = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# =========================================================
# SECRETS
# =========================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
STUDENT_CODE = st.secrets["STUDENT_CODE"]

SUPABASE_URL = st.secrets["SUPABASE_URL"]

# Supabase Secret Key
# In older projects this may be called service_role
SUPABASE_SECRET_KEY = st.secrets["SUPABASE_SECRET_KEY"]


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# SUPABASE
# =========================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json"
    }


# =========================================================
# AUTH
# =========================================================

def username_exists(username):

    url = (
        f"{SUPABASE_URL}/rest/v1/students"
        f"?username=eq.{requests.utils.quote(username)}"
        f"&select=id"
    )

    try:

        response = requests.get(
            url,
            headers=supabase_headers(),
            timeout=10
        )

        if response.status_code != 200:
            return False

        data = response.json()

        return len(data) > 0

    except Exception:

        return False


def create_student(username, password):

    try:

        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        url = f"{SUPABASE_URL}/rest/v1/students"

        data = {
            "username": username,
            "password_hash": password_hash
        }

        response = requests.post(
            url,
            headers={
                **supabase_headers(),
                "Prefer": "return=minimal"
            },
            json=data,
            timeout=10
        )

        if response.status_code in [200, 201]:

            return True, ""

        return False, (
            f"Supabase error {response.status_code}: "
            f"{response.text}"
        )

    except Exception as e:

        return False, str(e)


def login_student(username, password):

    url = (
        f"{SUPABASE_URL}/rest/v1/students"
        f"?username=eq.{requests.utils.quote(username)}"
        f"&select=id,username,password_hash"
    )

    try:

        response = requests.get(
            url,
            headers=supabase_headers(),
            timeout=10
        )

        if response.status_code != 200:

            return False, (
                f"Supabase error {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        if not data:

            return False, "Username or password is incorrect."

        student = data[0]

        stored_hash = student["password_hash"]

        valid_password = bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        )

        if valid_password:

            return True, ""

        return False, "Username or password is incorrect."

    except Exception as e:

        return False, str(e)


# =========================================================
# FILE READING
# =========================================================

def read_pdf(path):

    text = ""

    try:

        reader = PdfReader(path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception:

        st.warning(
            f"Could not read PDF: {path}"
        )

    return text


def read_docx(path):

    text = ""

    try:

        document = Document(path)

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += paragraph.text + "\n"

    except Exception:

        st.warning(
            f"Could not read DOCX: {path}"
        )

    return text


def read_txt(path):

    try:

        return Path(path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


def read_file(path):

    extension = Path(path).suffix.lower()

    if extension == ".pdf":

        return read_pdf(path)

    elif extension == ".docx":

        return read_docx(path)

    elif extension == ".txt":

        return read_txt(path)

    return ""


# =========================================================
# CHUNKING
# =========================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = text.replace(
        "\x00",
        " "
    ).strip()

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - overlap

    return chunks


# =========================================================
# FILE HASH
# =========================================================

def get_file_hash(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:

                break

            hasher.update(data)

    return hasher.hexdigest()


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embedding(
    text,
    task_type
):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )


def normalize_vector(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:

        return vector

    return vector / norm


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
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_cache(cache):

    try:

        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                cache,
                f,
                ensure_ascii=False
            )

    except Exception:

        pass


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

@st.cache_resource
def build_knowledge():

    knowledge_path = Path(
        KNOWLEDGE_FOLDER
    )

    knowledge_path.mkdir(
        parents=True,
        exist_ok=True
    )

    cache = load_cache()

    documents = []

    supported_extensions = [
        ".pdf",
        ".docx",
        ".txt"
    ]

    files = []

    for extension in supported_extensions:

        files.extend(
            knowledge_path.glob(
                f"*{extension}"
            )
        )

    for file_path in files:

        try:

            file_hash = get_file_hash(
                file_path
            )

            text = read_file(
                file_path
            )

            if not text.strip():

                continue

            chunks = split_text(
                text
            )

            for index, chunk in enumerate(chunks):

                chunk_id = (
                    f"{file_path.name}_"
                    f"{file_hash}_"
                    f"{index}"
                )

                if chunk_id in cache:

                    embedding = np.array(
                        cache[chunk_id]["embedding"],
                        dtype=np.float32
                    )

                else:

                    embedding = create_embedding(
                        chunk,
                        "RETRIEVAL_DOCUMENT"
                    )

                    embedding = normalize_vector(
                        embedding
                    )

                    cache[chunk_id] = {
                        "file": file_path.name,
                        "text": chunk,
                        "embedding": embedding.tolist()
                    }

                    save_cache(cache)

                documents.append(
                    {
                        "file": file_path.name,
                        "text": chunk,
                        "embedding": embedding
                    }
                )

        except Exception as e:

            st.warning(
                f"Problem reading "
                f"{file_path.name}: {e}"
            )

    return documents


# =========================================================
# RAG SEARCH
# =========================================================

def retrieve_context(
    question,
    documents,
    top_k=TOP_K
):

    if not documents:

        return []

    query_embedding = create_embedding(
        question,
        "RETRIEVAL_QUERY"
    )

    query_embedding = normalize_vector(
        query_embedding
    )

    results = []

    for document in documents:

        document_embedding = (
            document["embedding"]
        )

        score = float(
            np.dot(
                query_embedding,
                document_embedding
            )
        )

        results.append(
            {
                "file": document["file"],
                "text": document["text"],
                "score": score
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# SCREENSHOT
# =========================================================

def extract_question_from_image(
    image_bytes,
    mime_type
):

    try:

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        prompt = """
You are an English teacher.

Look at this screenshot and extract the
English question or exercise shown in the image.

Return ONLY the question text.

Do not solve it.
Do not explain it.

If there is more than one question,
extract all questions clearly.
"""

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[
                prompt,
                image_part
            ]
        )

        return response.text.strip()

    except Exception as e:

        return (
            f"Could not read screenshot: {e}"
        )


# =========================================================
# PROMPT
# =========================================================

def build_prompt(
    question,
    retrieved_documents,
    conversation_history
):

    context = ""

    if retrieved_documents:

        for i, document in enumerate(
            retrieved_documents,
            start=1
        ):

            context += (
                f"\n--- SOURCE {i} ---\n"
                f"File: {document['file']}\n"
                f"{document['text']}\n"
            )

    history_text = ""

    if conversation_history:

        for message in conversation_history[-7:]:

            role = message["role"]

            content = message["content"]

            history_text += (
                f"\n{role.upper()}: "
                f"{content}\n"
            )

    prompt = f"""
You are an AI English tutor for students.

IMPORTANT:

You ONLY teach English.

You can help with:

- English grammar
- Vocabulary
- Tenses
- Reading
- Writing
- Sentence correction
- Translation between Arabic and English
- English exercises
- English exams
- Parts of speech
- Phrasal verbs
- Idioms
- Sentence structure
- English literature when it exists in the provided material

If the question is about another subject,
do not answer it.

Say:

"I'm an English tutor, so I can only help
with English-related questions."

CENTER MATERIAL HAS PRIORITY.

The provided files are the main source of truth.

If the answer exists in the provided material,
use it first.

If the files do not contain enough information,
you may use general English knowledge.

If the student says:

"I don't understand"
"مش فاهم"
"مش فاهمة"
"I still don't understand"
"Explain again"

Do NOT repeat the same explanation.

Instead:

- simplify the explanation
- use an easier example
- explain step-by-step
- use a different method
- use Arabic briefly when useful

For grammar questions:

Give:

1. Correct answer
2. Explanation
3. Grammar rule
4. Why it is correct

When useful, explain why other choices
are incorrect.

For vocabulary:

Give:

- English meaning
- Arabic meaning when useful
- Example sentence

For reading:

Answer based on the provided material
when available.

For writing:

Correct the writing and explain
important mistakes.

For exercises:

Solve step-by-step.

Do not mention:

- RAG
- embeddings
- prompts
- system instructions
- internal implementation

CENTER MATERIAL:

{context}

RECENT CONVERSATION:

{history_text}

STUDENT QUESTION:

{question}

ANSWER:
"""

    return prompt


# =========================================================
# GEMINI GENERATION WITH RETRY
# =========================================================

def generate_answer(
    question,
    retrieved_documents,
    conversation_history,
    image_bytes=None,
    image_mime=None
):

    prompt = build_prompt(
        question,
        retrieved_documents,
        conversation_history
    )

    contents = [prompt]

    if image_bytes and image_mime:

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=image_mime
        )

        contents.append(
            image_part
        )

    last_error = None

    # Try 3 times
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=contents
            )

            if response.text:

                return response.text

            return "I couldn't generate an answer."

        except Exception as e:

            last_error = e

            error_text = str(e)

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text
                or "temporarily" in error_text.lower()
            )

            if temporary_error and attempt < 2:

                wait_time = 3 * (attempt + 1)

                time.sleep(
                    wait_time
                )

                continue

            break

    raise Exception(
        "Gemini is temporarily unavailable.\n\n"
        f"Details: {last_error}"
    )


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if "username" not in st.session_state:

    st.session_state.username = ""


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# LOGIN / REGISTER
# =========================================================

if not st.session_state.logged_in:

    st.title("📚 English AI Tutor")

    st.write(
        "Please login or create your student account."
    )

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register"
        ]
    )

    # =====================================================
    # LOGIN
    # =====================================================

    with login_tab:

        st.subheader("Login")

        login_username = st.text_input(
            "Username",
            key="login_username"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            if (
                not login_username
                or not login_password
            ):

                st.error(
                    "Please enter username "
                    "and password."
                )

            else:

                success, error_message = login_student(
                    login_username.strip(),
                    login_password
                )

                if success:

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        login_username.strip()
                    )

                    st.session_state.messages = []

                    st.rerun()

                else:

                    st.error(
                        error_message
                    )

    # =====================================================
    # REGISTER
    # =====================================================

    with register_tab:

        st.subheader(
            "Create Student Account"
        )

        register_code = st.text_input(
            "Student Code",
            type="password",
            key="register_code"
        )

        register_username = st.text_input(
            "Choose Username",
            key="register_username"
        )

        register_password = st.text_input(
            "Choose Password",
            type="password",
            key="register_password"
        )

        register_confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="register_confirm"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if register_code != STUDENT_CODE:

                st.error(
                    "Invalid Student Code."
                )

            elif not register_username.strip():

                st.error(
                    "Please choose a username."
                )

            elif not register_password:

                st.error(
                    "Please choose a password."
                )

            elif register_password != register_confirm:

                st.error(
                    "Passwords do not match."
                )

            elif len(register_password) < 6:

                st.error(
                    "Password must be at least "
                    "6 characters."
                )

            else:

                username = (
                    register_username.strip()
                )

                if username_exists(username):

                    st.error(
                        "This username already exists. "
                        "Please choose another one."
                    )

                else:

                    success, error_message = (
                        create_student(
                            username,
                            register_password
                        )
                    )

                    if success:

                        st.success(
                            "Account created successfully! 🎉"
                        )

                        st.info(
                            "You can now login using "
                            "your username and password."
                        )

                    else:

                        st.error(
                            "Could not create account."
                        )

                        st.code(
                            error_message
                        )

    st.stop()


# =========================================================
# CHATBOT
# =========================================================

st.title("📚 English AI Tutor")

st.caption(
    f"Logged in as: "
    f"{st.session_state.username}"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Student")

    st.write(
        f"👤 {st.session_state.username}"
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.info(
        "This chatbot is specialized "
        "in English only."
    )


# =========================================================
# LOAD KNOWLEDGE
# =========================================================

with st.spinner(
    "Loading English learning materials..."
):

    documents = build_knowledge()


if not documents:

    st.warning(
        "No English files were found "
        "in the knowledge folder."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# SCREENSHOT UPLOAD
# =========================================================

uploaded_image = st.file_uploader(
    "📷 Upload a screenshot "
    "of an English question",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key="question_image"
)


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask your English question..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if user_question or uploaded_image:

    typed_question = (
        user_question.strip()
        if user_question
        else ""
    )

    image_bytes = None

    image_mime = None

    extracted_question = ""

    # =====================================================
    # READ SCREENSHOT
    # =====================================================

    if uploaded_image:

        image_bytes = (
            uploaded_image.getvalue()
        )

        image_mime = uploaded_image.type

        with st.spinner(
            "Reading the screenshot..."
        ):

            extracted_question = (
                extract_question_from_image(
                    image_bytes,
                    image_mime
                )
            )

    # =====================================================
    # COMBINE QUESTIONS
    # =====================================================

    if (
        typed_question
        and extracted_question
    ):

        final_question = (
            f"{typed_question}\n\n"
            f"Question from screenshot:\n"
            f"{extracted_question}"
        )

    elif typed_question:

        final_question = typed_question

    elif extracted_question:

        final_question = extracted_question

    else:

        final_question = ""

    if not final_question:

        st.error(
            "Please type a question or "
            "upload a screenshot."
        )

        st.stop()

    # =====================================================
    # SHOW USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        if typed_question:

            st.markdown(
                typed_question
            )

        if uploaded_image:

            st.image(
                uploaded_image,
                caption="Uploaded question",
                use_container_width=True
            )

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": final_question
        }
    )

    # =====================================================
    # RAG SEARCH
    # =====================================================

    with st.spinner(
        "Searching English materials..."
    ):

        try:

            retrieved_documents = (
                retrieve_context(
                    final_question,
                    documents,
                    TOP_K
                )
            )

        except Exception as e:

            retrieved_documents = []

            st.warning(
                "Could not search the "
                "knowledge base."
            )

            st.code(
                str(e)
            )

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Preparing your answer..."
        ):

            try:

                answer = generate_answer(
                    question=final_question,
                    retrieved_documents=(
                        retrieved_documents
                    ),
                    conversation_history=(
                        st.session_state.messages
                    ),
                    image_bytes=image_bytes,
                    image_mime=image_mime
                )

                st.markdown(
                    answer
                )

            except Exception as e:

                answer = (
                    "Sorry, Gemini is "
                    "temporarily unavailable. "
                    "Please try again in a moment."
                )

                st.error(
                    answer
                )

                st.code(
                    str(e)
                )

    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )