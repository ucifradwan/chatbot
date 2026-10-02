import os
import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document


st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="🤖"
)

st.title("🤖 Knowledge Chatbot")
st.caption("اسأل عن المعلومات الموجودة في الملفات")


# =========================
# GEMINI
# =========================

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


# =========================
# READ FILES
# =========================

def read_pdf(path):
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def read_docx(path):
    document = Document(path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )


def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_file(path):

    if path.endswith(".pdf"):
        return read_pdf(path)

    if path.endswith(".docx"):
        return read_docx(path)

    if path.endswith(".txt"):
        return read_txt(path)

    return ""


# =========================
# LOAD KNOWLEDGE
# =========================

@st.cache_data
def load_knowledge():

    folder = "knowledge"

    all_text = ""

    if not os.path.exists(folder):
        return ""

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        if filename.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):

            text = read_file(path)

            all_text += (
                f"\n\n===== {filename} =====\n\n"
                + text
            )

    return all_text


knowledge = load_knowledge()


# =========================
# CHAT
# =========================

question = st.chat_input(
    "اكتب سؤالك..."
)


if question:

    with st.chat_message("user"):
        st.write(question)

    if not knowledge.strip():

        answer = "لا توجد ملفات معرفة مضافة حاليًا."

    else:

        prompt = f"""
أنت chatbot يعتمد فقط على ملفات المعرفة التالية.

قواعد مهمة جدًا:

1. أجب فقط باستخدام المعلومات الموجودة في الملفات.
2. لا تخترع أي معلومة.
3. إذا لم تجد إجابة السؤال في الملفات، قل:
"المعلومة دي غير موجودة في الملفات المتاحة."
4. أجب بالعربية بشكل واضح.
5. لا تقل إنك بحثت على الإنترنت.
6. لا تستخدم معلومات من معرفتك العامة.

===== ملفات المعرفة =====

{knowledge}

===== سؤال المستخدم =====

{question}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

    with st.chat_message("assistant"):
        st.write(answer)