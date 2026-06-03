import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image
import google.generativeai as genai
 
 
# ─── Environment Setup ────────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
 
 
# ─── Gemini Vision Functions ──────────────────────────────────────────
def analyze_symptoms_from_image(image):
    """Analyzes visible symptoms from a medical image using Gemini Vision."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        vision_prompt = """
        You are an expert clinical observer. Look at this medical image (e.g., skin rash, wound, eye redness)
        and accurately describe the visible symptoms in 2-3 short sentences.
        DO NOT provide a diagnosis, disease name, or medical advice.
        Just describe what you see physically (color, shape, texture, location).
        Reply in Bengali.
        """
        response = model.generate_content([vision_prompt, image])
        return response.text
    except Exception as e:
        return f"ERROR: ছবি প্রসেস করতে সমস্যা হয়েছে ({str(e)})"
 
 
def extract_text_gemini(image):
    """Extracts text from a medical prescription or report using Gemini Vision."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        ocr_prompt = """
        You are an expert pharmacist and medical document reader.
        Carefully read this medical prescription or lab report.
        Extract all the text, focusing heavily on:
        1. Medicine names
        2. Dosages (e.g., 500mg, 1+0+1)
        3. Duration (e.g., for 5 days)
        4. Any medical advice or test results.
        If it's handwritten, use your medical knowledge to accurately decipher the terms.
        Format the extracted text neatly. Do not give any medical advice here, just extract the text.
        """
        response = model.generate_content([ocr_prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini OCR ব্যর্থ হয়েছে ({str(e)})"
 
 
# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="MediAssist AI", page_icon="👩🏻‍⚕️", layout="wide")
st.title("👩🏻‍⚕️ MediAssist AI — Personal Medical Assistant")
st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা, ইংরেজি বা Banglish-এ)।")
 
 
# ─── Session State Init ───────────────────────────────────────────────
defaults = {
    "messages": [],
    "chat_history": [],
    "ocr_confirmed_text": None,
    "ocr_extracted_text": None,
    "ui_language": "বাংলা",
    "last_uploaded_file": None,
    "vision_extracted_symptoms": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
 
 
# ─── Display Chat History ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 
# ─── RAG Pipeline ─────────────────────────────────────────────────────
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
 
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
 
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question to be standalone and clear. "
         "Do NOT answer the question, just rephrase it if needed. "
         "If it's already clear, return it as is."
         ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
 
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
 
    system_prompt = (
        "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
        "Your role is to provide accurate, helpful, and easy-to-understand health information based on the given context. "
 
        "\n\n### CORE RULES ###\n"
        "- ONLY use information from the provided context to answer questions. "
        "- If the context lacks sufficient information, respond with: 'I'm sorry, I don't have enough information on this topic. Please consult a qualified doctor.' "
        "- NEVER invent, guess, or hallucinate any medical information, drug names, or dosages. "
        "- NEVER provide a specific diagnosis. You can explain symptoms and conditions, but always clarify you are an AI, not a doctor. "
 
        "\n\n### LANGUAGE RULES ###\n"
        "- Always reply in the SAME language the user used. "
        "- If the user writes in Bengali, reply fully in Bengali. If in English, reply in English. "
        "- If the user explicitly asks for a translation, switch to that language. "
        "- The user may also write in 'Banglish' (Bengali words in English letters, e.g., 'ami jore bugtesi' = 'I have fever'). "
        "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
        "- Common Banglish medical terms: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
        "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
        "'durbolta' = weakness, 'bomi' = vomiting, 'ghum nai' = insomnia, "
        "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
 
        "\n\n### FORMATTING RULES ###\n"
        "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line. "
        "- For step-by-step instructions: use numbered lists (1. 2. 3.), one step per line. "
        "- For complex topics: use clear headings with **bold** text to separate sections. "
        "- Keep answers concise but structured. Never put two ideas in one line. "
 
        "\n\n### EMERGENCY PROTOCOL ###\n"
        "- If the user describes emergency symptoms (chest pain with sweating, difficulty breathing, "
        "severe bleeding, loss of consciousness, stroke symptoms, severe allergic reaction) — "
        "IMMEDIATELY start with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
 
        "\n\n### SPECIALIST RECOMMENDATION ###\n"
        "- If asked who to consult, recommend the most appropriate specialist: "
        "chest pain → Cardiologist, skin issues → Dermatologist, headaches → Neurologist, "
        "mental health → Psychiatrist, child health → Pediatrician. "
 
        "\n\n### OCR EXTRACTED TEXT ###\n"
        "- If user input contains '[OCR থেকে পাওয়া তথ্য]', treat it as a medical prescription or report. "
        "- Identify medicine names, dosages, and instructions from the extracted text. "
 
        "\n\n### DISCLAIMER ###\n"
        "- Always end medical advice responses with: "
        "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। "
        "যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "
 
        "\n\nContext:\n{context}"
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
 
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
 
 
rag_chain = load_rag_pipeline()
 
 
# ─── Sidebar: Settings ────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.session_state.ui_language = st.sidebar.radio(
    "Language / ভাষা:",
    ["বাংলা", "English"]
)
is_bangla = st.session_state.ui_language == "বাংলা"
 
 
# ─── Sidebar: Health Dashboard ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("📊 Health Dashboard")
st.sidebar.write("আপনার প্রাথমিক স্বাস্থ্য তথ্য দিন:")
 
weight = st.sidebar.number_input("ওজন (Weight in kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)
height_cm = st.sidebar.number_input("উচ্চতা (Height in cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)
 
if st.sidebar.button("Calculate BMI & Stats"):
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    water_intake = (weight * 35) / 1000
 
    st.sidebar.markdown("---")
    st.sidebar.subheader("আপনার ফলাফল:")
    st.sidebar.write(f"**BMI:** {bmi:.2f}")
 
    if bmi < 18.5:
        st.sidebar.warning("Category: Underweight 📉")
    elif 18.5 <= bmi < 24.9:
        st.sidebar.success("Category: Normal Weight ✅")
    elif 25 <= bmi < 29.9:
        st.sidebar.warning("Category: Overweight 📈")
    else:
        st.sidebar.error("Category: Obese ⚠️")
 
    st.sidebar.write(f"**দৈনিক পানির চাহিদা:** {water_intake:.1f} লিটার 💧")
    st.sidebar.info("💡 এই ফলাফলের ভিত্তিতে মেইন চ্যাটে ডায়েট প্ল্যান চাইতে পারেন!")
 
 
# ─── Sidebar: Visual Symptom Checker ─────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Symptom Checker")
st.sidebar.caption("লক্ষণ বুঝতে ছবি আপলোড করুন (Powered by Gemini Vision)")
 
vision_upload = st.sidebar.file_uploader(
    "ছবি আপলোড করুন (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    key="vision_uploader"
)
 
if vision_upload:
    vision_image = Image.open(vision_upload)
    st.sidebar.image(vision_image, caption="আপলোড করা ছবি", use_container_width=True)
 
    if st.sidebar.button("🔍 লক্ষণ বিশ্লেষণ করুন"):
        with st.spinner("ছবি বিশ্লেষণ করা হচ্ছে..."):
            symptoms = analyze_symptoms_from_image(vision_image)
            st.session_state.vision_extracted_symptoms = symptoms
 
    if st.session_state.vision_extracted_symptoms:
        if not st.session_state.vision_extracted_symptoms.startswith("ERROR"):
            st.sidebar.success("✅ বিশ্লেষণ সম্পন্ন!")
            st.sidebar.info(f"**প্রাথমিক লক্ষণ:**\n{st.session_state.vision_extracted_symptoms}")
            st.sidebar.caption("💡 এই লক্ষণগুলোর ভিত্তিতে নিচে মূল চ্যাটে প্রশ্ন করুন।")
        else:
            st.sidebar.error(st.session_state.vision_extracted_symptoms)
 
 
# ─── Sidebar: Prescription / Report OCR ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Prescription / Report OCR")
st.sidebar.caption(
    "🔍 Gemini Vision দ্বারা পরিচালিত" if is_bangla
    else "🔍 Powered by Gemini Vision"
)
 
upload_label = "প্রেসক্রিপশন / রিপোর্টের ছবি আপলোড করুন" if is_bangla else "Upload Prescription / Report Image"
uploaded_image = st.sidebar.file_uploader(upload_label, type=["jpg", "jpeg", "png"])
 
if uploaded_image:
    # Reset state on new file
    if st.session_state.last_uploaded_file != uploaded_image.name:
        st.session_state.ocr_extracted_text = None
        st.session_state.ocr_confirmed_text = None
        st.session_state.last_uploaded_file = uploaded_image.name
 
    image = Image.open(uploaded_image)
    st.sidebar.image(
        image,
        caption="আপলোড করা ছবি" if is_bangla else "Uploaded Image",
        use_container_width=True
    )
 
    # Run OCR only once
    if st.session_state.ocr_extracted_text is None:
        with st.spinner("🔍 OCR processing হচ্ছে..." if is_bangla else "🔍 Processing OCR..."):
            extracted = extract_text_gemini(image)
            st.session_state.ocr_extracted_text = extracted
            st.session_state.ocr_confirmed_text = None
 
    # Confirmation Block
    if (
        st.session_state.ocr_extracted_text
        and not st.session_state.ocr_extracted_text.startswith("ERROR")
        and st.session_state.ocr_confirmed_text is None
    ):
        st.info("📋 ছবি থেকে নিচের text পাওয়া গেছে। এটা কি সঠিক?" if is_bangla
                else "📋 Text extracted from the image. Is this correct?")
 
        edited_text = st.text_area(
            "✏️ প্রয়োজনে সম্পাদনা করুন:" if is_bangla else "✏️ Edit if needed:",
            value=st.session_state.ocr_extracted_text,
            height=150
        )
 
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ হ্যাঁ, সঠিক" if is_bangla else "✅ Confirm"):
                st.session_state.ocr_confirmed_text = edited_text
                st.success("✅ Text confirm হয়েছে! নিচে প্রশ্ন করুন।" if is_bangla
                           else "✅ Confirmed! Ask your question below.")
        with col2:
            if st.button("🔄 আবার চেষ্টা" if is_bangla else "🔄 Retry"):
                st.session_state.ocr_extracted_text = None
                st.session_state.ocr_confirmed_text = None
                st.rerun()
 
    # Error Block
    elif (
        st.session_state.ocr_extracted_text
        and st.session_state.ocr_extracted_text.startswith("ERROR")
    ):
        st.error(f"❌ OCR ব্যর্থ হয়েছে: {st.session_state.ocr_extracted_text}")
        if st.button("🔄 আবার চেষ্টা করুন" if is_bangla else "🔄 Retry"):
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.rerun()
 
    # Confirmed Block
    elif st.session_state.ocr_confirmed_text:
        st.sidebar.success("✅ OCR Text confirm হয়েছে।" if is_bangla else "✅ OCR Text confirmed.")
        if st.sidebar.button("🗑️ OCR রিসেট করুন" if is_bangla else "🗑️ Reset OCR"):
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.session_state.last_uploaded_file = None
            st.rerun()
 
elif not uploaded_image and st.session_state.last_uploaded_file is not None:
    st.session_state.ocr_extracted_text = None
    st.session_state.ocr_confirmed_text = None
    st.session_state.last_uploaded_file = None
 
 
# ─── Main Chat Input ──────────────────────────────────────────────────
chat_placeholder = (
    "আপনার লক্ষণ বা সমস্যার কথা লিখুন (বাংলা, ইংরেজি বা Banglish)..."
    if is_bangla
    else "Type your symptoms or question (Bengali, English or Banglish)..."
)
 
user_input = st.chat_input(chat_placeholder)
 
if user_input:
    combined_input = user_input
 
    # Attach OCR text if confirmed
    if st.session_state.get("ocr_confirmed_text"):
        combined_input += (
            f"\n\n[OCR রিপোর্ট/প্রেসক্রিপশন থেকে পাওয়া তথ্য]:\n"
            f"{st.session_state.ocr_confirmed_text}"
        )
 
    # Attach vision symptoms if available
    vision_symptoms = st.session_state.get("vision_extracted_symptoms")
    if vision_symptoms and not vision_symptoms.startswith("ERROR"):
        combined_input += (
            f"\n\n[রোগীর আপলোড করা ছবির লক্ষণ]:\n"
            f"{vision_symptoms}"
        )
 
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
 
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("বই থেকে তথ্য খোঁজা হচ্ছে..." if is_bangla else "Searching knowledge base..."):
            response = rag_chain.invoke({
                "input": combined_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
 
    # Update memory
    st.session_state.chat_history.append(HumanMessage(content=combined_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.messages.append({"role": "assistant", "content": answer})
