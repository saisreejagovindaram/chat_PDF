import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* ==========================================
   APP BACKGROUND
   ========================================== */

.stApp {
    background:
        linear-gradient(
            135deg,
            #eaf4ff 0%,
            #f4efff 45%,
            #e8f1ff 100%
        );

    color: #111111;
}


/* ==========================================
   MAIN CONTAINER
   ========================================== */

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================
   TITLE
   ========================================== */

h1 {
    color: #000000 !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    text-align: center;
    margin-bottom: 0.3rem;
}


/* ==========================================
   SUBTITLE
   ========================================== */

.main .block-container > div > div > div > p {
    color: #333333;
}


/* ==========================================
   ALL NORMAL TEXT
   ========================================== */

p {
    color: #222222;
}


/* ==========================================
   FILE UPLOADER
   ========================================== */

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.80);
    border: 2px dashed #8b7cf6;
    border-radius: 18px;
    padding: 20px;

    box-shadow:
        0 8px 25px rgba(80, 70, 160, 0.12);

    transition: 0.3s ease;
}


[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;

    box-shadow:
        0 10px 30px rgba(99, 102, 241, 0.20);
}


/* ==========================================
   FILE UPLOADER TEXT
   ========================================== */

[data-testid="stFileUploader"] * {
    color: #222222 !important;
}


/* ==========================================
   QUESTION INPUT
   ========================================== */

.stTextInput > div > div > input {
    background-color: #ffffff !important;

    color: #000000 !important;

    border: 2px solid #9b8df5 !important;

    border-radius: 14px !important;

    padding: 14px !important;

    font-size: 16px !important;

    box-shadow:
        0 5px 15px rgba(80, 70, 160, 0.10);

    transition: 0.3s ease;
}


/* ==========================================
   INPUT FOCUS
   ========================================== */

.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}


/* ==========================================
   INPUT LABEL
   ========================================== */

.stTextInput label {
    color: #000000 !important;

    font-weight: 700 !important;

    font-size: 16px !important;
}


/* ==========================================
   SUCCESS / INFO / ERROR BOXES
   ========================================== */

[data-testid="stAlert"] {
    border-radius: 14px;

    border: none;

    box-shadow:
        0 5px 18px rgba(80, 70, 160, 0.08);
}


/* ==========================================
   EXPANDER
   ========================================== */

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.78);

    border: 1px solid #c7befa;

    border-radius: 15px;

    box-shadow:
        0 5px 18px rgba(80, 70, 160, 0.08);

    overflow: hidden;
}


/* ==========================================
   EXPANDER TEXT
   ========================================== */

[data-testid="stExpander"] p {
    color: #222222 !important;
}


/* ==========================================
   HEADINGS
   ========================================== */

h2,
h3 {
    color: #000000 !important;

    font-weight: 750 !important;
}


/* ==========================================
   DIVIDER
   ========================================== */

hr {
    border: none;

    height: 2px;

    background:
        linear-gradient(
            90deg,
            #93c5fd,
            #8b5cf6,
            #c4b5fd
        );

    margin: 30px 0;
}


/* ==========================================
   BUTTON
   ========================================== */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );

    color: white !important;

    border: none;

    border-radius: 12px;

    padding: 10px 24px;

    font-weight: 600;

    transition: 0.3s ease;
}


.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(99, 102, 241, 0.25);
}


/* ==========================================
   SPINNER
   ========================================== */

.stSpinner {
    color: #6366f1 !important;
}


/* ==========================================
   HEADER
   ========================================== */

header[data-testid="stHeader"] {
    background: transparent;
}


/* ==========================================
   MAIN CARD EFFECT
   ========================================== */

.main .block-container {
    background: rgba(255, 255, 255, 0.20);

    border-radius: 24px;
}


/* ==========================================
   ANSWER AREA
   ========================================== */

.answer-box {
    background:
        rgba(255, 255, 255, 0.85);

    border-left:
        5px solid #6366f1;

    border-radius: 15px;

    padding: 20px;

    margin-top: 10px;

    color: #111111;

    box-shadow:
        0 8px 25px rgba(80, 70, 160, 0.12);
}


/* ==========================================
   ANSWER TEXT
   ========================================== */

.answer-box p {
    color: #111111 !important;

    font-size: 16px;

    line-height: 1.7;
}


/* ==========================================
   METRIC / INFORMATION TEXT
   ========================================== */

.metric-card {
    background:
        rgba(255, 255, 255, 0.70);

    border-radius: 15px;

    padding: 15px;

    text-align: center;

    box-shadow:
        0 5px 18px rgba(80, 70, 160, 0.08);
}


/* ==========================================
   MOBILE RESPONSIVE
   ========================================== */

@media (max-width: 768px) {

    h1 {
        font-size: 2rem !important;
    }

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.title("📄 Chat with Your PDF")

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Ask questions about your PDF using RAG + Ollama"
    "</p>",
    unsafe_allow_html=True
)


# ==========================================
# TEST MESSAGE
# ==========================================

st.info("Application is running successfully!")


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📁 Upload your PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # ==========================================
    # READ PDF
    # ==========================================

    reader = PdfReader(uploaded_file)

    st.write(
        "📄 Number of pages:",
        len(reader.pages)
    )


    # ==========================================
    # EXTRACT TEXT
    # ==========================================

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"


    # ==========================================
    # CHECK TEXT
    # ==========================================

    if not text.strip():

        st.error(
            "❌ No text could be extracted from this PDF."
        )

        st.stop()


    st.write(
        "📝 Total characters:",
        len(text)
    )


    # ==========================================
    # CREATE CHUNKS
    # ==========================================

    chunk_size = 500

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(chunk)


    st.write(
        "🧩 Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # LOAD EMBEDDING MODEL
    # ==========================================

    with st.spinner(
        "Loading embedding model..."
    ):

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    st.success(
        "Embedding model loaded!"
    )


    # ==========================================
    # CREATE EMBEDDINGS
    # ==========================================

    with st.spinner(
        "Creating embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )


    st.success(
        "Embeddings created!"
    )


    # ==========================================
    # CREATE CHROMADB
    # ==========================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )


    # ==========================================
    # CREATE IDS
    # ==========================================

    ids = []

    for i in range(len(chunks)):

        ids.append(
            "chunk_" + str(i)
        )


    # ==========================================
    # REMOVE OLD DATA
    # ==========================================

    try:

        old_data = collection.get()

        if old_data["ids"]:

            collection.delete(
                ids=old_data["ids"]
            )

    except Exception:

        pass


    # ==========================================
    # STORE DOCUMENT
    # ==========================================

    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist()

    )


    st.success(
        "PDF stored successfully in ChromaDB!"
    )


    st.divider()


    # ==========================================
    # QUESTION
    # ==========================================

    question = st.text_input(
        "💬 Ask a question about the PDF",
        placeholder="Type your question here..."
    )


    # ==========================================
    # ANSWER
    # ==========================================

    if question:

        with st.spinner(
            "Searching document..."
        ):

            # ==========================================
            # CREATE QUESTION EMBEDDING
            # ==========================================

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )


            # ==========================================
            # SEARCH CHROMADB
            # ==========================================

            results = collection.query(

                query_embeddings=
                question_embedding.tolist(),

                n_results=len(chunks)

            )


            # ==========================================
            # GET DOCUMENTS
            # ==========================================

            relevant_chunks = (
                results["documents"][0]
            )


            # ==========================================
            # COMBINE CONTEXT
            # ==========================================

            context = "\n\n".join(
                relevant_chunks
            )


        # ==========================================
        # SHOW RETRIEVED DATA
        # ==========================================

        with st.expander(
            "🔎 View Retrieved Context"
        ):

            st.write(context)


        # ==========================================
        # CREATE PROMPT
        # ==========================================

        prompt = f"""

You are a helpful document assistant.

Answer the user's question using the information
available in the document context.

If the user asks for a summary, overview, or brief
description, summarize the important information
from the document context.

If the answer cannot be found in the document,
say:

"I could not find the answer in the document."

Do not make up information.

Document Context:

{context}

User Question:

{question}

Answer:

"""


        # ==========================================
        # OLLAMA
        # ==========================================

        with st.spinner(
            "🤖 Ollama is generating the answer..."
        ):

            response = ollama.chat(

                model="llama3.2",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )


        # ==========================================
        # DISPLAY ANSWER
        # ==========================================

        answer = response["message"]["content"]

        st.subheader("🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-box">
                <p>{answer}</p>
            </div>
            """,
            unsafe_allow_html=True
        )