import streamlit as st
import os
import json
import hashlib
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


# =========================================================
# SECRETS
# =========================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
STUDENT_CODE = st.secrets["STUDENT_CODE"]

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
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
# SUPABASE HEADERS
# =========================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }


# =========================================================
# CHECK USERNAME
# =========================================================

def username_exists(username):

    url = (
        f"{SUPABASE_URL}/rest/v1/students"
        f"?username=eq.{username}&select=id"
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


# =========================================================
# CREATE STUDENT
# =========================================================

def create_student(username, password):

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    url = f"{SUPABASE_URL}/rest/v1/students"

    data = {
        "username": username,
        "password_hash": password_hash
    }

    try:

        response = requests.post(
            url,
            headers=supabase_headers(),
            json=data,
            timeout=10
        )

        return response.status_code in [200, 201]

    except Exception:

        return False


# =========================================================
# LOGIN STUDENT
# =========================================================

def login_student(username, password):

    url = (
        f"{SUPABASE_URL}/rest/v1/students"
        f"?username=eq.{username}"
        f"&select=username,password_hash"
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

        if len(data) == 0:
            return False

        stored_hash = data[0]["password_hash"]

        return bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        )

    except Exception:

        return False


# =========================================================
# AUTHENTICATION PAGE
# =========================================================

def authentication_page():

    st.title("📚 English AI Tutor")

    st.write(
        "This AI Tutor is available only for students of the center."
    )

    tab_login, tab_register = st.tabs(
        ["🔐 Login", "📝 Create Account"]
    )

    # =====================================================
    # LOGIN
    # =====================================================

    with tab_login:

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
            use_container_width=True,
            key="login_button"
        ):

            if not login_username:

                st.error(
                    "Please enter your username."
                )

            elif not login_password:

                st.error(
                    "Please enter your password."
                )

            elif login_student(
                login_username.strip(),
                login_password
            ):

                st.session_state.logged_in = True

                st.session_state.username = (
                    login_username.strip()
                )

                st.session_state.messages = []

                st.rerun()

            else:

                st.error(
                    "Incorrect username or password."
                )

    # =====================================================
    # REGISTER
    # =====================================================

    with tab_register:

        st.subheader("Create Student Account")

        st.info(
            "You need the Student Code provided by the center."
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

        register_confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="register_confirm_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True,
            key="create_account_button"
        ):

            register_code = register_code.strip()
            register_username = register_username.strip()
            register_password = register_password.strip()
            register_confirm_password = (
                register_confirm_password.strip()
            )

            # ---------------------------------------------
            # STUDENT CODE
            # ---------------------------------------------

            if register_code != STUDENT_CODE:

                st.error(
                    "Invalid Student Code. "
                    "Please contact the center."
                )

            # ---------------------------------------------
            # USERNAME
            # ---------------------------------------------

            elif not register_username:

                st.error(
                    "Please choose a username."
                )

            elif len(register_username) < 4:

                st.error(
                    "Username must be at least 4 characters."
                )

            # ---------------------------------------------
            # PASSWORD
            # ---------------------------------------------

            elif not register_password:

                st.error(
                    "Please choose a password."
                )

            elif len(register_password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

            elif register_password != register_confirm_password:

                st.error(
                    "Passwords do not match."
                )

            # ---------------------------------------------
            # CHECK USERNAME
            # ---------------------------------------------

            elif username_exists(register_username):

                st.error(
                    "This username is already registered."
                )

            # ---------------------------------------------
            # CREATE ACCOUNT
            # ---------------------------------------------

            else:

                success = create_student(
                    register_username,
                    register_password
                )

                if success:

                    st.success(
                        "Account created successfully!"
                    )

                    st.info(
                        "You can now login using your "
                        "username and password."
                    )

                else:

                    st.error(
                        "Could not create account. "
                        "Please try again."
                    )


# =========================================================
# READ PDF
# =========================================================

def read_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =========================================================
# READ DOCX
# =========================================================

def read_docx(path):

    doc = Document(path)

    text = []

    for paragraph in doc.paragraphs:

        if paragraph.text.strip():

            text.append(
                paragraph.text
            )

    return "\n".join(text)


# =========================================================
# READ TXT
# =========================================================

def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        return file.read()


# =========================================================
# READ FILE
# =========================================================

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
# SPLIT TEXT
# =========================================================

def split_text(
    text,
    chunk_size=1200,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# FILE HASH
# =========================================================

def get_file_hash(path):

    sha = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


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
# CREATE EMBEDDING
# =========================================================

def create_embedding(
    text,
    task_type
):

    result = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=text,

        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm > 0:

        vector = vector / norm

    return vector


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

@st.cache_resource
def build_knowledge():

    os.makedirs(
        KNOWLEDGE_FOLDER,
        exist_ok=True
    )

    cache = load_cache()

    knowledge = []

    for filename in os.listdir(
        KNOWLEDGE_FOLDER
    ):

        path = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        if not os.path.isfile(path):

            continue

        extension = Path(path).suffix.lower()

        if extension not in [
            ".pdf",
            ".docx",
            ".txt"
        ]:

            continue

        file_hash = get_file_hash(path)

        # ---------------------------------------------
        # USE CACHE
        # ---------------------------------------------

        if file_hash in cache:

            for item in cache[file_hash]:

                knowledge.append({

                    "text": item["text"],

                    "embedding": np.array(
                        item["embedding"],
                        dtype=np.float32
                    ),

                    "source": item["source"]

                })

            continue

        # ---------------------------------------------
        # READ FILE
        # ---------------------------------------------

        text = read_file(path)

        if not text.strip():

            continue

        chunks = split_text(text)

        file_data = []

        for chunk in chunks:

            embedding = create_embedding(
                chunk,
                "RETRIEVAL_DOCUMENT"
            )

            item = {

                "text": chunk,

                "embedding": embedding.tolist(),

                "source": os.path.basename(path)

            }

            file_data.append(item)

            knowledge.append({

                "text": chunk,

                "embedding": embedding,

                "source": os.path.basename(path)

            })

        cache[file_hash] = file_data

    save_cache(cache)

    return knowledge


# =========================================================
# SEARCH KNOWLEDGE
# =========================================================

def search_knowledge(
    query,
    knowledge,
    top_k=TOP_K
):

    if not knowledge:

        return []

    query_embedding = create_embedding(
        query,
        "RETRIEVAL_QUERY"
    )

    scored = []

    for item in knowledge:

        score = float(
            np.dot(
                query_embedding,
                item["embedding"]
            )
        )

        scored.append(
            (
                score,
                item
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        item
        for score, item in scored[:top_k]
    ]


# =========================================================
# EXTRACT QUESTION FROM SCREENSHOT
# =========================================================

def extract_question_from_image(
    image_bytes,
    mime_type
):

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    prompt = """
You are an English exam question reader.

Look carefully at the uploaded screenshot.

Extract the English question, choices, passage,
sentences, or exercise shown in the image.

DO NOT solve the question.

Return a clean text representation of the
question that can be used for semantic search.

If there are multiple questions,
include all of them.

Do not add explanations.
"""

    response = client.models.generate_content(

        model=GENERATION_MODEL,

        contents=[
            image_part,
            prompt
        ]
    )

    return response.text.strip()


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    user_question,
    context,
    history,
    image_bytes=None,
    image_type=None
):

    # -----------------------------------------------------
    # CONVERSATION HISTORY
    # -----------------------------------------------------

    history_text = ""

    for message in history[-7:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # -----------------------------------------------------
    # MAIN PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an AI English tutor for students
of an educational center.

==================================================
SUBJECT LIMIT
==================================================

You ONLY help with English-related topics.

You can help with:

- English grammar
- Vocabulary
- Reading comprehension
- Writing
- Sentence correction
- Translation between Arabic and English
- English exercises
- English exams
- Tenses
- Parts of speech
- Sentence structure
- Phrasal verbs
- Idioms
- English literature when relevant
  to the uploaded center material

If the student asks about another subject,
politely say:

"I'm an English tutor, so I can only help
with English-related questions."

==================================================
CENTER MATERIAL
==================================================

The following information was retrieved
from the center's uploaded files.

The center material has HIGHEST PRIORITY.

CENTER MATERIAL:

{context}

==================================================
STUDENT QUESTION
==================================================

{user_question}

==================================================
PREVIOUS CONVERSATION
==================================================

{history_text}

==================================================
IMPORTANT RULES
==================================================

1. Answer the student's actual question.

2. If the answer or rule exists in the
   center material, prioritize it.

3. Do not contradict the center material.

4. For grammar questions:

   - Give the correct answer.
   - Explain the grammar rule.
   - Explain why it is correct.
   - Explain why the other choices are wrong
     when useful.

5. For vocabulary:

   - Give the meaning.
   - Give Arabic meaning when useful.
   - Give a simple example.

6. For reading:

   - Answer from the provided text.
   - Explain why the answer is correct.

7. For writing:

   - Correct mistakes.
   - Explain the mistakes.
   - Give an improved version.

8. For exercises and exams:

   - Solve step-by-step.
   - Do not give only the final answer.

9. If the student says:

   "I don't understand"
   "مش فاهم"
   "مش فاهمة"

   DO NOT simply repeat the previous explanation.

   Instead:

   - Use simpler English.
   - Give a very simple example.
   - Use Arabic briefly if helpful.
   - Explain using a different method.

10. If the uploaded screenshot contains
    an English question, understand it
    and solve it.

11. If the screenshot is unclear,
    tell the student what part is unclear.

12. Do not answer questions from subjects
    other than English.

13. If the center material does not contain
    enough information, you may use general
    English knowledge.

    Clearly distinguish general knowledge
    from the center material.

14. Keep explanations suitable for students.

Give a clear educational answer.
"""

    contents = []

    # -----------------------------------------------------
    # ADD IMAGE
    # -----------------------------------------------------

    if image_bytes is not None:

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=image_type
        )

        contents.append(image_part)

    # -----------------------------------------------------
    # ADD TEXT PROMPT
    # -----------------------------------------------------

    contents.append(prompt)

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    response = client.models.generate_content(

        model=GENERATION_MODEL,

        contents=contents
    )

    return response.text


# =========================================================
# CHATBOT PAGE
# =========================================================

def chatbot_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.title("📚 English AI Tutor")

        st.write(
            f"Logged in as: **{st.session_state.username}**"
        )

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.messages = []

            st.rerun()

        st.divider()

        st.caption(
            "English questions only."
        )

    # =====================================================
    # HEADER
    # =====================================================

    st.title("📚 English AI Tutor")

    st.write(
        "Ask your English question or upload "
        "a screenshot."
    )

    # =====================================================
    # LOAD KNOWLEDGE
    # =====================================================

    try:

        knowledge = build_knowledge()

    except Exception as e:

        st.error(
            "There was a problem loading "
            "the center files."
        )

        st.code(str(e))

        knowledge = []

    # =====================================================
    # DISPLAY CHAT HISTORY
    # =====================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # =====================================================
    # SCREENSHOT UPLOAD
    # =====================================================

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

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded question",
            use_container_width=True
        )

    # =====================================================
    # TEXT QUESTION
    # =====================================================

    user_question = st.chat_input(
        "Ask your English question..."
    )

    # =====================================================
    # PROCESS
    # =====================================================

    if user_question or uploaded_image:

        image_bytes = None
        image_type = None

        extracted_question = ""

        # =================================================
        # IMAGE
        # =================================================

        if uploaded_image:

            image_bytes = uploaded_image.getvalue()

            image_type = uploaded_image.type

            try:

                extracted_question = (
                    extract_question_from_image(
                        image_bytes,
                        image_type
                    )
                )

            except Exception as e:

                st.error(
                    "I couldn't read the screenshot."
                )

                st.code(str(e))

                return

        # =================================================
        # SEARCH QUERY
        # =================================================

        if user_question and extracted_question:

            search_query = (
                f"{user_question}\n"
                f"{extracted_question}"
            )

        elif user_question:

            search_query = user_question

        else:

            search_query = extracted_question

        # =================================================
        # RAG
        # =================================================

        retrieved = search_knowledge(
            search_query,
            knowledge
        )

        context_parts = []

        for item in retrieved:

            context_parts.append(
                f"[Source: {item['source']}]\n"
                f"{item['text']}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # =================================================
        # DISPLAY USER MESSAGE
        # =================================================

        display_question = user_question

        if not display_question:

            display_question = (
                "📷 Uploaded English question"
            )

        st.session_state.messages.append({

            "role": "user",

            "content": display_question

        })

        with st.chat_message("user"):

            st.markdown(
                display_question
            )

        # =================================================
        # GENERATE ANSWER
        # =================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking and checking "
                "the center material..."
            ):

                try:

                    answer = generate_answer(

                        user_question=search_query,

                        context=context,

                        history=st.session_state.messages,

                        image_bytes=image_bytes,

                        image_type=image_type

                    )

                    st.markdown(answer)

                except Exception as e:

                    answer = (
                        "Sorry, something went wrong "
                        "while processing your question."
                    )

                    st.error(answer)

                    st.code(str(e))

        # =================================================
        # SAVE ANSWER
        # =================================================

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })

        # =================================================
        # CLEAR IMAGE
        # =================================================

        try:

            st.session_state.question_image = None

        except Exception:

            pass


# =========================================================
# START APP
# =========================================================

if not st.session_state.logged_in:

    authentication_page()

else:

    chatbot_page()