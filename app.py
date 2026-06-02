import os
#import cv2
import re
#import numpy as np
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
#import easyocr
#import speech_recognition as sr
import google.generativeai as genai



# Gemini API কনফিগারেশন
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

def analyze_symptoms_from_image(image):
    """
    এই ফাংশনটি ছবি দেখে শুধুমাত্র বাহ্যিক লক্ষণ (Symptoms) বের করবে।
    """
    try:
        # Gemini 1.5 Flash মডেল কল করা
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # ⚠️ প্রম্পট ইঞ্জিনিয়ারিং: মডেলকে কড়া নির্দেশ দেওয়া হচ্ছে যেন সে ডাক্তারি না করে!
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
        return f"ERROR: ছবি প্রসেস করতে সমস্যা হয়েছে ({str(e)})"


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# def get_voice_input():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("🎤( Listening...)")
#         # ব্যাকগ্রাউন্ড নয়েজ কমানোর জন্য
#         r.adjust_for_ambient_noise(source) 
#         try:
#             audio = r.listen(source, timeout=5)
#             # language="bn-BD" দিলে বাংলা বুঝবে, "en-US" দিলে ইংরেজি
#             text = r.recognize_google(audio, language="bn-BD") 
#             return text
#         except sr.UnknownValueError:
#             st.error("দুঃখিত, আমি কথাটি ঠিকমতো বুঝতে পারিনি।")
#             return None
#         except sr.RequestError:
#             st.error("ইন্টারনেট কানেকশনে সমস্যা হচ্ছে!")
#             return None
#         except Exception as e:
#             return None


st.set_page_config(page_title="Personal Medical Assistant", page_icon="👩🏻‍⚕️")
st.title("👩🏻‍⚕️ Personal Medical Assistant")
st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা বা ইংরেজিতে)।")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ocr_confirmed_text" not in st.session_state:
    st.session_state.ocr_confirmed_text = None
if "ocr_extracted_text" not in st.session_state:
    st.session_state.ocr_extracted_text = None
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "বাংলা"
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# ─── Visual Symptom Checker (Vision) ──────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Symptom Checker")
st.sidebar.caption("লক্ষণ বুঝতে ছবি আপলোড করুন (Powered by Gemini)")

# ছবি আপলোডের জন্য file uploader
vision_upload = st.sidebar.file_uploader(
    "লক্ষণ চেক করতে ছবি আপলোড করুন (JPG/PNG)", 
    type=["jpg", "jpeg", "png"], 
    key="vision_uploader"
)

# যদি Session State-এ ভিশনের ডেটা না থাকে
if "vision_extracted_symptoms" not in st.session_state:
    st.session_state.vision_extracted_symptoms = None

if vision_upload:
    vision_image = Image.open(vision_upload)
    st.sidebar.image(
        vision_image, 
        caption="আপলোড করা ছবি", 
        use_container_width=True
    )
    
    # বিশ্লেষণ করার বাটন
    if st.sidebar.button("🔍 লক্ষণ বিশ্লেষণ করুন"):
        with st.spinner("ছবি বিশ্লেষণ করা হচ্ছে..."):
            # তোমার আগে লেখা ফাংশনটি কল করা হচ্ছে
            symptoms = analyze_symptoms_from_image(vision_image)
            st.session_state.vision_extracted_symptoms = symptoms
            
    # ফলাফল দেখানো
    if st.session_state.vision_extracted_symptoms:
        if not st.session_state.vision_extracted_symptoms.startswith("ERROR"):
            st.sidebar.success("✅ বিশ্লেষণ সম্পন্ন!")
            st.sidebar.info(f"**প্রাথমিক লক্ষণ:**\n{st.session_state.vision_extracted_symptoms}")
            st.sidebar.caption("💡 এই লক্ষণগুলোর ভিত্তিতে নিচে মূল চ্যাটে প্রশ্ন করতে পারেন।")
        else:
            st.sidebar.error(st.session_state.vision_extracted_symptoms)



# ─── OCR Functions ───────────────────────────────────────────────────
# @st.cache_resource
# def load_ocr():
#     return easyocr.Reader(['en', 'bn'])


# def preprocess_image(image):
#     """ছবির quality উন্নত করে OCR accuracy বাড়ায়"""
#     img = np.array(image.convert('RGB'))

#     # Step 1: Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # Step 2: Upscale করো — ছোট ছবি বড় করলে OCR ভালো কাজ করে
#     scale = 2
#     gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

#     # Step 3: Denoise — ছবির noise কমাও
#     gray = cv2.fastNlMeansDenoising(gray, h=30)

#     # Step 4: Contrast বাড়াও
#     gray = cv2.equalizeHist(gray)

#     # Step 5: Sharpen — text এর edge আরো স্পষ্ট করো
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     sharpened = cv2.filter2D(gray, -1, kernel)

#     # Step 6: Adaptive Threshold — uneven lighting handle করে
#     thresh = cv2.adaptiveThreshold(
#         sharpened, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 31, 2
#     )

#     return thresh



# import re

# def clean_ocr_text(text):
#     """OCR এর raw output কে পরিষ্কার ও সুন্দরভাবে format করে"""

#     # একাধিক space কমিয়ে একটা করো
#     text = re.sub(r' +', ' ', text)

#     # শুধু space, comma, dash, dot দিয়ে তৈরি অর্থহীন অংশ সরাও
#     text = re.sub(r'\b[,.\-\s]+\b', ' ', text)

#     # লাইনের শুরু বা শেষে থাকা comma/dot/space সরাও
#     text = re.sub(r'^[,.\-\s]+|[,.\-\s]+$', '', text, flags=re.MULTILINE)

#     # ৩+ বার repeated punctuation সরাও (যেমন ,,, বা ...)
#     text = re.sub(r'[,.\-]{2,}', '.', text)

#     # প্রতিটা detected line কে আলাদা line এ রাখো
#     lines = [line.strip() for line in text.split('.') if line.strip()]
#     text = '\n'.join(lines)

#     return text.strip()





def extract_text_gemini(image):
    """Gemini 2.5 Flash ব্যবহার করে প্রেসক্রিপশন বা রিপোর্ট থেকে টেক্সট বের করা"""
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



# ------------------Sidebar Settings -------------------------------

with st.sidebar:
    st.header("📊 Health Dashboard")
    st.write("আপনার প্রাথমিক স্বাস্থ্য তথ্য দিন:")
    
    # ইউজারের ইনপুট নেওয়া
    weight = st.number_input("ওজন (Weight in kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)
    height_cm = st.number_input("উচ্চতা (Height in cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)
    
    # ক্যালকুলেট বাটন
    if st.button("Calculate BMI & Stats"):
        # BMI হিসাব করা (ওজন / উচ্চতা^2)
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        
        # পানির চাহিদা হিসাব করা (প্রতি কেজি ওজনের জন্য আনুমানিক ৩৫ মিলি)
        water_intake = (weight * 35) / 1000
        
        st.markdown("---")
        st.subheader("আপনার ফলাফল (Results):")
        st.write(f"**BMI:** {bmi:.2f}")
        
        # BMI ক্যাটাগরি অনুযায়ী মেসেজ
        if bmi < 18.5:
            st.warning("Category: Underweight 📉 (ওজন কম)")
        elif 18.5 <= bmi < 24.9:
            st.success("Category: Normal Weight ✅ (স্বাভাবিক)")
        elif 25 <= bmi < 29.9:
            st.warning("Category: Overweight 📈 (ওজন বেশি)")
        else:
            st.error("Category: Obese ⚠️ (স্থূলতা)")
        
        st.write(f"**দৈনিক পানির চাহিদা:** {water_intake:.1f} লিটার 💧")
        
        st.info("💡 টিপস: আপনি চাইলে এই ফলাফলের ওপর ভিত্তি করে মেইন চ্যাটে এআই-এর কাছে একটি ডায়েট প্ল্যান চাইতে পারেন!")



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
        "- If the user explicitly asks for a translation (e.g., 'এটা ইংরেজিতে বলো' or 'say this in Bengali'), switch to that language. "
        "- The user may also write in 'Banglish' (Bengali words written in English letters, e.g., 'ami jore bugtesi' means 'I have fever'). "
        "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
        "- Common Banglish medical terms to understand: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
        "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
        "'durbolta' = weakness, 'bomi' = vomiting, 'pায়খানা' = diarrhea, 'ghum nai' = insomnia, "
        "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
        "- If unsure about a Banglish word's meaning, try to interpret it based on context and reply in Bengali. "

        "\n\n### FORMATTING RULES ###\n"
        "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line. Always use line breaks for readability. "
        "- For step-by-step instructions or processes: use numbered lists (1. 2. 3.), one step per line. Always use line breaks for readability. "
        "- For complex topics: use clear headings with **bold** text to separate sections. Always use line breaks for readability. "
        "- Keep answers concise. For simple questions, give short answers. For complex questions, give detailed but structured answers. "
        "- Never put two separate ideas in a single line. Always use line breaks for readability. "

        "\n\n### EMERGENCY PROTOCOL ###\n"
        "- If the user describes emergency symptoms such as: chest pain with sweating, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms, or severe allergic reaction — "
        "IMMEDIATELY start your response with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
        "- Do not wait to give other information before showing the emergency alert. "

        "\n\n### SPECIALIST RECOMMENDATION ###\n"
        "- If the user asks who to consult, recommend the most appropriate medical specialist based on their symptoms. "
        "- Examples: chest pain → Cardiologist, skin issues → Dermatologist, headaches/seizures → Neurologist, mental health → Psychiatrist, child health → Pediatrician. "
        "- Always phrase it as a recommendation, not a diagnosis. "

        "\n\n### Website Resources ###\n"
        "When extracting information from the context, strictly ignore any marketing text, website navigation menus, subscription offers, or promotional content. Focus ONLY on clinical facts, symptoms, treatments, and medical guidelines."

        "\n\n### OCR EXTRACTED TEXT ###\n"
        "- If the user's input contains text extracted from an image (marked as '[OCR থেকে পাওয়া তথ্য]'), "
        "treat it as a medical prescription or report. "
        "- Identify medicine names, dosages, and instructions from the extracted text. "
        "- If OCR text seems garbled or unclear, mention it and ask the user to clarify. "

        "\n\n### DISCLAIMER ###\n"
        "- If the user asks for medical advice or treatment suggestions, always end your response with: "
        "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "

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


# ─── Sidebar Settings ─────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.session_state.ui_language = st.sidebar.radio(
    "OCR Confirmation Language / ভাষা বেছে নিন:",
    ["বাংলা", "English"]
)

is_bangla = st.session_state.ui_language == "বাংলা"


# ─── OCR Section ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Prescription / Report OCR")
st.sidebar.caption(
    "🔍 EasyOCR দ্বারা পরিচালিত (সম্পূর্ণ local)"
    if is_bangla
    else "🔍 Powered by EasyOCR (fully local)"
)

upload_label = (
    "ছবি আপলোড করুন (JPG/PNG)"
    if is_bangla
    else "Upload Image (JPG/PNG)"
)
uploaded_image = st.sidebar.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_image:
    # নতুন ছবি upload হলে state reset করো
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

    # শুধু প্রথমবার OCR চালাও
    if st.session_state.ocr_extracted_text is None:
        with st.spinner(
            "🔍 OCR processing হচ্ছে, একটু অপেক্ষা করুন..."
            if is_bangla
            else "🔍 Processing OCR, please wait..."
        ):
            extracted = extract_text_gemini(image)
            st.session_state.ocr_extracted_text = extracted
            st.session_state.ocr_confirmed_text = None

    # ─── Confirmation Block ───────────────────────────────────────────
    if (
        st.session_state.ocr_extracted_text
        and not st.session_state.ocr_extracted_text.startswith("ERROR")
        and st.session_state.ocr_confirmed_text is None
    ):
        if is_bangla:
            st.info("📋 ছবি থেকে নিচের text পাওয়া গেছে। এটা কি সঠিক?")
        else:
            st.info("📋 The following text was extracted from the image. Is this correct?")

        # Editable text area — user চাইলে ঠিক করতে পারবে
        edited_text = st.text_area(
            "✏️ প্রয়োজনে সম্পাদনা করুন:" if is_bangla else "✏️ Edit if needed:",
            value=st.session_state.ocr_extracted_text,
            height=150
        )

        col1, col2 = st.columns(2)

        with col1:
            confirm_label = "✅ হ্যাঁ, সঠিক" if is_bangla else "✅ Confirm"
            if st.button(confirm_label):
                st.session_state.ocr_confirmed_text = edited_text
                if is_bangla:
                    st.success("✅ Text confirm হয়েছে! নিচে প্রশ্ন করুন।")
                else:
                    st.success("✅ Confirmed! Ask your question below.")

        with col2:
            retry_label = "🔄 আবার চেষ্টা" if is_bangla else "🔄 Retry"
            if st.button(retry_label):
                st.session_state.ocr_extracted_text = None
                st.session_state.ocr_confirmed_text = None
                st.rerun()

    # ─── Error Block ──────────────────────────────────────────────────
    elif (
        st.session_state.ocr_extracted_text
        and st.session_state.ocr_extracted_text.startswith("ERROR")
    ):
        if is_bangla:
            st.error(f"❌ OCR ব্যর্থ হয়েছে: {st.session_state.ocr_extracted_text}")
            st.warning("আবার চেষ্টা করতে নিচের বোতাম চাপুন।")
        else:
            st.error(f"❌ OCR failed: {st.session_state.ocr_extracted_text}")
            st.warning("Press the button below to retry.")

        if st.button("🔄 Retry" if not is_bangla else "🔄 আবার চেষ্টা করুন"):
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.rerun()

    # ─── Confirmed Block ──────────────────────────────────────────────
    elif st.session_state.ocr_confirmed_text:
        if is_bangla:
            st.sidebar.success("✅ OCR Text confirm হয়েছে।")
        else:
            st.sidebar.success("✅ OCR Text confirmed.")

        reset_label = "🗑️ OCR রিসেট করুন" if is_bangla else "🗑️ Reset OCR"
        if st.sidebar.button(reset_label):
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.session_state.last_uploaded_file = None
            st.rerun()

elif not uploaded_image:
    # ছবি না থাকলে state reset করো
    if st.session_state.last_uploaded_file is not None:
        st.session_state.ocr_extracted_text = None
        st.session_state.ocr_confirmed_text = None
        st.session_state.last_uploaded_file = None


# ─── Chat Input (Voice + Text) ────────────────────────────────────────

# chat_placeholder = (
#     "আপনার লক্ষণ বা সমস্যার কথা লিখুন..."
#     if is_bangla
#     else "Type your symptoms or question..."
# )

# # ১. টেক্সট এবং ভয়েস ইনপুট ভেরিয়েবল সেট করা
# text_input = st.chat_input(chat_placeholder)
# voice_input = None

# # ২. ভয়েস বাটন ক্লিক হলে অডিও থেকে টেক্সট নেওয়া
# if st.button("🎤 কথা বলে প্রশ্ন করুন"):
#     voice_input = get_voice_input()
#     if voice_input:
#         st.success(f"আপনি বলেছেন: {voice_input}")

# # ৩. চূড়ান্ত ইনপুট: ইউজার টাইপ করুক বা কথা বলুক, যেকোনো একটি পেলেই কাজ করবে
# user_input = text_input or voice_input

# if user_input:
#     # শুরুতে ইউজারের মূল প্রশ্নটি নিলাম
#     combined_input = user_input

#     # ১. OCR টেক্সট থাকলে সেটি যুক্ত করো
#     if st.session_state.get("ocr_confirmed_text"):
#         combined_input += (
#             f"\n\n[OCR রিপোর্ট/প্রেসক্রিপশন থেকে পাওয়া তথ্য]:\n"
#             f"{st.session_state.ocr_confirmed_text}"
#         )

#     # ২. Vision থেকে পাওয়া লক্ষণ থাকলে সেটিও যুক্ত করো (নতুন লজিক 🚀)
#     vision_symptoms = st.session_state.get("vision_extracted_symptoms")
#     if vision_symptoms and not vision_symptoms.startswith("ERROR"):
#         combined_input += (
#             f"\n\n[রোগীর আপলোড করা ছবির লক্ষণ]:\n"
#             f"{vision_symptoms}"
#         )

#     # ইউজারের মেসেজ UI-তে দেখানো (ইউজার শুধু তার টাইপ করা অংশটুকুই দেখবে)
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # এআই-এর উত্তর জেনারেট করা
#     with st.chat_message("assistant"):
#         with st.spinner("Processing..." if not is_bangla else "বই থেকে তথ্য খোঁজা হচ্ছে..."):
#             response = rag_chain.invoke({
#                 "input": combined_input, # এখন Groq ইউজারের প্রশ্ন + ছবি + OCR সবকিছু একসাথেই পাবে!
#                 "chat_history": st.session_state.chat_history
#             })
#             answer = response["answer"]
#             st.markdown(answer)

#     # মেমোরি (History) আপডেট করা
#     st.session_state.chat_history.append(HumanMessage(content=combined_input))
#     st.session_state.chat_history.append(AIMessage(content=answer))
#     st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── Chat Input (Text Only for Cloud) ────────────────────────────────────────

chat_placeholder = (
    "আপনার লক্ষণ বা সমস্যার কথা লিখুন..."
    if is_bangla
    else "Type your symptoms or question..."
)

# ক্লাউডের জন্য শুধু টেক্সট ইনপুট নেওয়া হচ্ছে (ভয়েস বাটন বাদ দেওয়া হলো)
user_input = st.chat_input(chat_placeholder)

if user_input:
    # শুরুতে ইউজারের মূল প্রশ্নটি নিলাম
    combined_input = user_input

    # ১. OCR টেক্সট থাকলে সেটি যুক্ত করো
    if st.session_state.get("ocr_confirmed_text"):
        combined_input += (
            f"\n\n[OCR রিপোর্ট/প্রেসক্রিপশন থেকে পাওয়া তথ্য]:\n"
            f"{st.session_state.ocr_confirmed_text}"
        )

    # ২. Vision থেকে পাওয়া লক্ষণ থাকলে সেটিও যুক্ত করো
    vision_symptoms = st.session_state.get("vision_extracted_symptoms")
    if vision_symptoms and not vision_symptoms.startswith("ERROR"):
        combined_input += (
            f"\n\n[রোগীর আপলোড করা ছবির লক্ষণ]:\n"
            f"{vision_symptoms}"
        )

    # ইউজারের মেসেজ UI-তে দেখানো (ইউজার শুধু তার টাইপ করা অংশটুকুই দেখবে)
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # এআই-এর উত্তর জেনারেট করা
    with st.chat_message("assistant"):
        with st.spinner("Processing..." if not is_bangla else "বই থেকে তথ্য খোঁজা হচ্ছে..."):
            response = rag_chain.invoke({
                "input": combined_input, # এখন Groq ইউজারের প্রশ্ন + ছবি + OCR সবকিছু একসাথেই পাবে!
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)

    # মেমোরি (History) আপডেট করা
    st.session_state.chat_history.append(HumanMessage(content=combined_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.messages.append({"role": "assistant", "content": answer})



# # ─── Chat Input ───────────────────────────────────────────────────────
# # ভয়েস ইনপুট বাটন
# if st.button("🎤 কথা বলে প্রশ্ন করুন"):
#     voice_text = get_voice_input()
#     if voice_text:
#         st.success(f"আপনি বলেছেন: {voice_text}")
#         # এই voice_text টি এখন তুমি তোমার মডেলের কাছে প্রম্পট হিসেবে পাঠিয়ে দিতে পারো!


# chat_placeholder = (
#     "আপনার লক্ষণ বা সমস্যার কথা লিখুন..."
#     if is_bangla
#     else "Type your symptoms or question..."
# )
# user_input = st.chat_input(chat_placeholder)

# if user_input:

#     if st.session_state.ocr_confirmed_text:
#         combined_input = (
#             f"{user_input}\n\n"
#             f"[OCR থেকে পাওয়া তথ্য / Extracted from image]:\n"
#             f"{st.session_state.ocr_confirmed_text}"
#         )
#     else:
#         combined_input = user_input

#     with st.chat_message("user"):
#         st.markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.chat_message("assistant"):
#         with st.spinner("Processing..."):
#             response = rag_chain.invoke({
#                 "input": combined_input,
#                 "chat_history": st.session_state.chat_history
#             })
#             answer = response["answer"]

#             disclaimer = "\n\n---\n*⚠️ সতর্কতা: I am a AI model, not a medical professional. Please consult a registered doctor for any medical concerns.*"
#             final_answer = answer + disclaimer

#             st.markdown(final_answer)

#     st.session_state.chat_history.append(HumanMessage(content=combined_input))
#     st.session_state.chat_history.append(AIMessage(content=final_answer))
#     st.session_state.messages.append({"role": "assistant", "content": final_answer})













# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
# from PIL import Image
# import google.generativeai as genai


# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=gemini_api_key)


# st.set_page_config(page_title="Personal Medical Assistant", page_icon="🩺")
# st.title("🩺 Personal Medical Assistant")
# st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা বা ইংরেজিতে)।")


# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "ocr_confirmed_text" not in st.session_state:
#     st.session_state.ocr_confirmed_text = None
# if "ocr_extracted_text" not in st.session_state:
#     st.session_state.ocr_extracted_text = None
# if "ui_language" not in st.session_state:
#     st.session_state.ui_language = "বাংলা"


# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# # ─── Gemini Vision OCR Function ─────────────────────────────────────
# def extract_text_gemini(image):
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content([
#             "This is a medical prescription or medical report. "
#             "Your task is to carefully extract ALL visible text from this image. "
#             "Include: patient name, doctor name, medicine names, dosages, frequency, and any instructions. "
#             "If the text is in Bengali, keep it in Bengali script. "
#             "If the text is in English, keep it in English. "
#             "If handwriting is unclear, make your best attempt to read it. "
#             "Return ONLY the extracted text, nothing else. "
#             "Do not add any explanation or commentary.",
#             image
#         ])
#         return response.text.strip()
#     except Exception as e:
#         return f"ERROR: {str(e)}"


# # ─── RAG Pipeline ────────────────────────────────────────────────────
# @st.cache_resource
# def load_rag_pipeline():

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     )
#     vector_db = Chroma(
#         persist_directory="./chroma_db", embedding_function=embeddings
#     )
#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})

#     llm = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0.3
#     )

#     contextualize_q_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "Given the chat history and the latest user question, "
#          "reformulate the question to be standalone and clear. "
#          "Do NOT answer the question, just rephrase it if needed. "
#          "If it's already clear, return it as is."
#          ),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])

#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt
#     )

#     system_prompt = (
#         "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
#         "Your role is to provide accurate, helpful, and easy-to-understand health information based on the given context. "

#         "\n\n### CORE RULES ###\n"
#         "- ONLY use information from the provided context to answer questions. "
#         "- If the context lacks sufficient information, respond with: 'I'm sorry, I don't have enough information on this topic. Please consult a qualified doctor.' "
#         "- NEVER invent, guess, or hallucinate any medical information, drug names, or dosages. "
#         "- NEVER provide a specific diagnosis. You can explain symptoms and conditions, but always clarify you are an AI, not a doctor. "

#         "\n\n### LANGUAGE RULES ###\n"
#         "- Always reply in the SAME language the user used. "
#         "- If the user writes in Bengali, reply fully in Bengali. If in English, reply in English. "
#         "- If the user explicitly asks for a translation (e.g., 'এটা ইংরেজিতে বলো' or 'say this in Bengali'), switch to that language. "
#         "- The user may also write in 'Banglish' (Bengali words written in English letters, e.g., 'ami jore bugtesi' means 'I have fever'). "
#         "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
#         "- Common Banglish medical terms to understand: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
#         "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
#         "'durbolta' = weakness, 'bomi' = vomiting, 'pায়খানা' = diarrhea, 'ghum nai' = insomnia, "
#         "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
#         "- If unsure about a Banglish word's meaning, try to interpret it based on context and reply in Bengali. "

#         "\n\n### FORMATTING RULES ###\n"
#         "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line. Always use line breaks for readability. "
#         "- For step-by-step instructions or processes: use numbered lists (1. 2. 3.), one step per line. Always use line breaks for readability. "
#         "- For complex topics: use clear headings with **bold** text to separate sections. Always use line breaks for readability. "
#         "- Keep answers concise. For simple questions, give short answers. For complex questions, give detailed but structured answers. "
#         "- Never put two separate ideas in a single line. Always use line breaks for readability. "

#         "\n\n### EMERGENCY PROTOCOL ###\n"
#         "- If the user describes emergency symptoms such as: chest pain with sweating, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms, or severe allergic reaction — "
#         "IMMEDIATELY start your response with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
#         "- Do not wait to give other information before showing the emergency alert. "

#         "\n\n### SPECIALIST RECOMMENDATION ###\n"
#         "- If the user asks who to consult, recommend the most appropriate medical specialist based on their symptoms. "
#         "- Examples: chest pain → Cardiologist, skin issues → Dermatologist, headaches/seizures → Neurologist, mental health → Psychiatrist, child health → Pediatrician. "
#         "- Always phrase it as a recommendation, not a diagnosis. "

#         "\n\n### Website Resources ###\n"
#         "When extracting information from the context, strictly ignore any marketing text, website navigation menus, subscription offers, or promotional content. Focus ONLY on clinical facts, symptoms, treatments, and medical guidelines."

#         "\n\n### DISCLAIMER ###\n"
#         "- If the user asks for medical advice or treatment suggestions, always end your response with: "
#         "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "

#         "\n\nContext:\n{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ])

#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     return rag_chain


# rag_chain = load_rag_pipeline()


# # ─── Sidebar Settings ────────────────────────────────────────────────
# st.sidebar.title("⚙️ Settings")
# st.session_state.ui_language = st.sidebar.radio(
#     "OCR Confirmation Language / ভাষা বেছে নিন:",
#     ["বাংলা", "English"]
# )

# is_bangla = st.session_state.ui_language == "বাংলা"


# # ─── OCR Section ─────────────────────────────────────────────────────
# st.sidebar.markdown("---")
# st.sidebar.subheader("📄 Prescription / Report OCR")
# st.sidebar.caption(
#     "🔍 Powered by Gemini Vision" if not is_bangla
#     else "🔍 Gemini Vision দ্বারা পরিচালিত"
# )

# upload_label = (
#     "ছবি আপলোড করুন (JPG/PNG)"
#     if is_bangla
#     else "Upload Image (JPG/PNG)"
# )
# uploaded_image = st.sidebar.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

# if uploaded_image:
#     image = Image.open(uploaded_image)
#     st.sidebar.image(
#         image,
#         caption="আপলোড করা ছবি" if is_bangla else "Uploaded Image",
#         use_container_width=True
#     )

#     # শুধু প্রথমবার OCR চালাও
#     if st.session_state.ocr_extracted_text is None:
#         with st.spinner(
#             "🔍 Gemini Vision দিয়ে OCR processing হচ্ছে..."
#             if is_bangla
#             else "🔍 Processing OCR with Gemini Vision..."
#         ):
#             extracted = extract_text_gemini(image)
#             st.session_state.ocr_extracted_text = extracted
#             st.session_state.ocr_confirmed_text = None

#     # ─── Confirmation Block ──────────────────────────────────────────
#     if (
#         st.session_state.ocr_extracted_text
#         and not st.session_state.ocr_extracted_text.startswith("ERROR")
#         and st.session_state.ocr_confirmed_text is None
#     ):
#         if is_bangla:
#             st.info("📋 ছবি থেকে নিচের text পাওয়া গেছে। এটা কি সঠিক?")
#         else:
#             st.info("📋 The following text was extracted from the image. Is this correct?")

#         st.code(st.session_state.ocr_extracted_text, language=None)

#         col1, col2 = st.columns(2)

#         with col1:
#             confirm_label = "✅ হ্যাঁ, সঠিক" if is_bangla else "✅ Yes, correct"
#             if st.button(confirm_label):
#                 st.session_state.ocr_confirmed_text = st.session_state.ocr_extracted_text
#                 if is_bangla:
#                     st.success("✅ Text confirm হয়েছে! নিচে প্রশ্ন করুন।")
#                 else:
#                     st.success("✅ Text confirmed! Ask your question below.")

#         with col2:
#             retry_label = "🔄 আবার চেষ্টা" if is_bangla else "🔄 Retry"
#             if st.button(retry_label):
#                 st.session_state.ocr_extracted_text = None
#                 st.session_state.ocr_confirmed_text = None
#                 st.rerun()

#     # ─── Error Block ─────────────────────────────────────────────────
#     elif (
#         st.session_state.ocr_extracted_text
#         and st.session_state.ocr_extracted_text.startswith("ERROR")
#     ):
#         if is_bangla:
#             st.error(f"❌ OCR ব্যর্থ হয়েছে: {st.session_state.ocr_extracted_text}")
#             st.warning("আবার চেষ্টা করতে নিচের বোতাম চাপুন।")
#         else:
#             st.error(f"❌ OCR failed: {st.session_state.ocr_extracted_text}")
#             st.warning("Press the button below to retry.")

#         if st.button("🔄 Retry" if not is_bangla else "🔄 আবার চেষ্টা করুন"):
#             st.session_state.ocr_extracted_text = None
#             st.session_state.ocr_confirmed_text = None
#             st.rerun()

#     # ─── Confirmed Block ──────────────────────────────────────────────
#     elif st.session_state.ocr_confirmed_text:
#         if is_bangla:
#             st.sidebar.success("✅ OCR Text confirm হয়েছে।")
#         else:
#             st.sidebar.success("✅ OCR Text confirmed.")

#         reset_label = "🗑️ OCR রিসেট করুন" if is_bangla else "🗑️ Reset OCR"
#         if st.sidebar.button(reset_label):
#             st.session_state.ocr_extracted_text = None
#             st.session_state.ocr_confirmed_text = None
#             st.rerun()


# # ─── Chat Input ───────────────────────────────────────────────────────
# chat_placeholder = (
#     "আপনার লক্ষণ বা সমস্যার কথা লিখুন..."
#     if is_bangla
#     else "Type your symptoms or question..."
# )
# user_input = st.chat_input(chat_placeholder)

# if user_input:

#     # OCR confirmed text থাকলে question এর সাথে যোগ করো
#     if st.session_state.ocr_confirmed_text:
#         combined_input = (
#             f"{user_input}\n\n"
#             f"[OCR থেকে পাওয়া তথ্য / Extracted from image]:\n"
#             f"{st.session_state.ocr_confirmed_text}"
#         )
#     else:
#         combined_input = user_input

#     with st.chat_message("user"):
#         st.markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.chat_message("assistant"):
#         with st.spinner("Processing..."):
#             response = rag_chain.invoke({
#                 "input": combined_input,
#                 "chat_history": st.session_state.chat_history
#             })
#             answer = response["answer"]

#             disclaimer = "\n\n---\n*⚠️ সতর্কতা: I am an AI model. This is not medical advice. Please consult a registered doctor for any serious health concerns.*"
#             final_answer = answer + disclaimer

#             st.markdown(final_answer)

#     st.session_state.chat_history.append(HumanMessage(content=combined_input))
#     st.session_state.chat_history.append(AIMessage(content=final_answer))
#     st.session_state.messages.append({"role": "assistant", "content": final_answer})














# # import os
# # import numpy as np
# # import streamlit as st
# # from dotenv import load_dotenv
# # from langchain_groq import ChatGroq
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import Chroma
# # from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain_core.messages import HumanMessage, AIMessage
# # from PIL import Image
# # import easyocr


# # load_dotenv()
# # groq_api_key = os.getenv("GROQ_API_KEY")


# # st.set_page_config(page_title="Personal Medical Assistant", page_icon="🩺")
# # st.title("🩺 Personal Medical Assistant")
# # st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা বা ইংরেজিতে)।")


# # if "messages" not in st.session_state:
# #     st.session_state.messages = []
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []
# # if "ocr_confirmed_text" not in st.session_state:
# #     st.session_state.ocr_confirmed_text = None
# # if "ocr_extracted_text" not in st.session_state:
# #     st.session_state.ocr_extracted_text = None
# # if "ui_language" not in st.session_state:
# #     st.session_state.ui_language = "বাংলা"


# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])


# # @st.cache_resource
# # def load_ocr():
# #     return easyocr.Reader(['en', 'bn'])


# # @st.cache_resource
# # def load_rag_pipeline():

# #     embeddings = HuggingFaceEmbeddings(
# #         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# #     )
# #     vector_db = Chroma(
# #         persist_directory="./chroma_db", embedding_function=embeddings
# #     )
# #     retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# #     llm = ChatGroq(
# #         groq_api_key=groq_api_key,
# #         model_name="llama-3.3-70b-versatile",
# #         temperature=0.3
# #     )

# #     contextualize_q_prompt = ChatPromptTemplate.from_messages([
# #         ("system",
# #          "Given the chat history and the latest user question, "
# #          "reformulate the question to be standalone and clear. "
# #          "Do NOT answer the question, just rephrase it if needed. "
# #          "If it's already clear, return it as is."
# #          ),
# #         MessagesPlaceholder("chat_history"),
# #         ("human", "{input}"),
# #     ])

# #     history_aware_retriever = create_history_aware_retriever(
# #         llm, retriever, contextualize_q_prompt
# #     )

# #     system_prompt = (
# #         "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
# #         "Your role is to provide accurate, helpful, and easy-to-understand health information based on the given context. "

# #         "\n\n### CORE RULES ###\n"
# #         "- ONLY use information from the provided context to answer questions. "
# #         "- If the context lacks sufficient information, respond with: 'I'm sorry, I don't have enough information on this topic. Please consult a qualified doctor.' "
# #         "- NEVER invent, guess, or hallucinate any medical information, drug names, or dosages. "
# #         "- NEVER provide a specific diagnosis. You can explain symptoms and conditions, but always clarify you are an AI, not a doctor. "

# #         "\n\n### LANGUAGE RULES ###\n"
# #         "- Always reply in the SAME language the user used. "
# #         "- If the user writes in Bengali, reply fully in Bengali. If in English, reply in English. "
# #         "- If the user explicitly asks for a translation (e.g., 'এটা ইংরেজিতে বলো' or 'say this in Bengali'), switch to that language. "
# #         "- The user may also write in 'Banglish' (Bengali words written in English letters, e.g., 'ami jore bugtesi' means 'I have fever'). "
# #         "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
# #         "- Common Banglish medical terms to understand: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
# #         "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
# #         "'durbolta' = weakness, 'bomi' = vomiting, 'pায়খানা' = diarrhea, 'ghum nai' = insomnia, "
# #         "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
# #         "- If unsure about a Banglish word's meaning, try to interpret it based on context and reply in Bengali. "

# #         "\n\n### FORMATTING RULES ###\n"
# #         "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line. Always use line breaks for readability. "
# #         "- For step-by-step instructions or processes: use numbered lists (1. 2. 3.), one step per line. Always use line breaks for readability. "
# #         "- For complex topics: use clear headings with **bold** text to separate sections. Always use line breaks for readability. "
# #         "- Keep answers concise. For simple questions, give short answers. For complex questions, give detailed but structured answers. "
# #         "- Never put two separate ideas in a single line. Always use line breaks for readability. "

# #         "\n\n### EMERGENCY PROTOCOL ###\n"
# #         "- If the user describes emergency symptoms such as: chest pain with sweating, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms, or severe allergic reaction — "
# #         "IMMEDIATELY start your response with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
# #         "- Do not wait to give other information before showing the emergency alert. "

# #         "\n\n### SPECIALIST RECOMMENDATION ###\n"
# #         "- If the user asks who to consult, recommend the most appropriate medical specialist based on their symptoms. "
# #         "- Examples: chest pain → Cardiologist, skin issues → Dermatologist, headaches/seizures → Neurologist, mental health → Psychiatrist, child health → Pediatrician. "
# #         "- Always phrase it as a recommendation, not a diagnosis. "

# #         "\n\n### Website Resources ###\n"
# #         "When extracting information from the context, strictly ignore any marketing text, website navigation menus, subscription offers, or promotional content. Focus ONLY on clinical facts, symptoms, treatments, and medical guidelines."

# #         "\n\n### DISCLAIMER ###\n"
# #         "- If the user asks for medical advice or treatment suggestions, always end your response with: "
# #         "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "

# #         "\n\nContext:\n{context}"
# #     )

# #     prompt = ChatPromptTemplate.from_messages([
# #         ("system", system_prompt),
# #         MessagesPlaceholder("chat_history"),
# #         ("human", "{input}")
# #     ])

# #     question_answer_chain = create_stuff_documents_chain(llm, prompt)
# #     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# #     return rag_chain


# # ocr_reader = load_ocr()
# # rag_chain = load_rag_pipeline()


# # # ─── UI Language Selector ───────────────────────────────────────────
# # st.sidebar.title("⚙️ Settings")
# # st.session_state.ui_language = st.sidebar.radio(
# #     "OCR Confirmation Language / ভাষা বেছে নিন:",
# #     ["বাংলা", "English"]
# # )

# # is_bangla = st.session_state.ui_language == "বাংলা"


# # # ─── OCR Section ────────────────────────────────────────────────────
# # st.sidebar.markdown("---")
# # st.sidebar.subheader("📄 Prescription / Report OCR")

# # upload_label = "ছবি আপলোড করুন (JPG/PNG)" if is_bangla else "Upload Image (JPG/PNG)"
# # uploaded_image = st.sidebar.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

# # if uploaded_image:
# #     image = Image.open(uploaded_image)
# #     st.sidebar.image(image, caption="আপলোড করা ছবি" if is_bangla else "Uploaded Image", use_container_width=True)

# #     if st.session_state.ocr_extracted_text is None:
# #         with st.spinner("🔍 OCR processing হচ্ছে..." if is_bangla else "🔍 Processing OCR..."):
# #             img_array = np.array(image)
# #             results = ocr_reader.readtext(img_array, detail=0)
# #             extracted = " ".join(results).strip()
# #             st.session_state.ocr_extracted_text = extracted
# #             st.session_state.ocr_confirmed_text = None

# #     # ─── Confirmation Block ─────────────────────────────────────────
# #     if st.session_state.ocr_extracted_text and st.session_state.ocr_confirmed_text is None:

# #         if is_bangla:
# #             st.info("📋 ছবি থেকে নিচের text পাওয়া গেছে। এটা কি সঠিক?")
# #         else:
# #             st.info("📋 The following text was extracted from the image. Is this correct?")

# #         st.code(st.session_state.ocr_extracted_text, language=None)

# #         col1, col2 = st.columns(2)

# #         with col1:
# #             confirm_label = "✅ হ্যাঁ, সঠিক আছে" if is_bangla else "✅ Yes, looks correct"
# #             if st.button(confirm_label):
# #                 st.session_state.ocr_confirmed_text = st.session_state.ocr_extracted_text
# #                 if is_bangla:
# #                     st.success("✅ Text confirm হয়েছে! নিচে প্রশ্ন করুন।")
# #                 else:
# #                     st.success("✅ Text confirmed! Ask your question below.")

# #         with col2:
# #             retry_label = "🔄 না, আবার চেষ্টা করুন" if is_bangla else "🔄 No, retry"
# #             if st.button(retry_label):
# #                 st.session_state.ocr_extracted_text = None
# #                 st.session_state.ocr_confirmed_text = None
# #                 st.rerun()

# #     elif st.session_state.ocr_confirmed_text:
# #         if is_bangla:
# #             st.sidebar.success("✅ OCR Text confirm হয়েছে।")
# #         else:
# #             st.sidebar.success("✅ OCR Text confirmed.")

# #         if st.sidebar.button("🗑️ Reset OCR" if not is_bangla else "🗑️ OCR রিসেট করুন"):
# #             st.session_state.ocr_extracted_text = None
# #             st.session_state.ocr_confirmed_text = None
# #             st.rerun()


# # # ─── Chat Input ─────────────────────────────────────────────────────
# # chat_placeholder = "আপনার লক্ষণ বা সমস্যার কথা লিখুন..." if is_bangla else "Type your symptoms or question..."
# # user_input = st.chat_input(chat_placeholder)

# # if user_input:

# #     # OCR confirmed text থাকলে সেটা question এর সাথে যোগ করো
# #     if st.session_state.ocr_confirmed_text:
# #         combined_input = (
# #             f"{user_input}\n\n"
# #             f"[OCR থেকে পাওয়া তথ্য / Extracted from image]:\n"
# #             f"{st.session_state.ocr_confirmed_text}"
# #         )
# #     else:
# #         combined_input = user_input

# #     with st.chat_message("user"):
# #         st.markdown(user_input)
# #     st.session_state.messages.append({"role": "user", "content": user_input})

# #     with st.chat_message("assistant"):
# #         with st.spinner("Processing..."):
# #             response = rag_chain.invoke({
# #                 "input": combined_input,
# #                 "chat_history": st.session_state.chat_history
# #             })
# #             answer = response["answer"]

# #             disclaimer = "\n\n---\n*⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো মেডিকেল পরামর্শ নয়, যেকোনো গুরুতর সমস্যায় অবশ্যই রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।*"
# #             final_answer = answer + disclaimer

# #             st.markdown(final_answer)

# #     st.session_state.chat_history.append(HumanMessage(content=combined_input))
# #     st.session_state.chat_history.append(AIMessage(content=final_answer))
# #     st.session_state.messages.append({"role": "assistant", "content": final_answer})













# # # import os
# # # import streamlit as st
# # # from dotenv import load_dotenv
# # # from langchain_groq import ChatGroq
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # from langchain_community.vectorstores import Chroma

# # # from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# # # from langchain.chains.combine_documents import create_stuff_documents_chain

# # # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # # from langchain_core.messages import HumanMessage, AIMessage


# # # load_dotenv()
# # # groq_api_key = os.getenv("GROQ_API_KEY")


# # # st.set_page_config(page_title="Personal Medical Assistant", page_icon="🩺")
# # # st.title("🩺 Personal Medical Assistant")
# # # st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা বা ইংরেজিতে)।")


# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []
# # # if "chat_history" not in st.session_state:
# # #     st.session_state.chat_history = []

# # # for message in st.session_state.messages:
# # #     with st.chat_message(message["role"]):
# # #         st.markdown(message["content"])


# # # @st.cache_resource
# # # def load_rag_pipeline():
    
# # #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# # #     vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# # #     retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
# # #     llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    
# # #     contextualize_q_prompt = ChatPromptTemplate.from_messages([
# # #         ("system",
# # #          "Given the chat history and the latest user question, "
# # #          "reformulate the question to be standalone and clear. "
# # #          "Do NOT answer the question, just rephrase it if needed. "
# # #          "If it's already clear, return it as is."
# # #         ),
# # #         MessagesPlaceholder("chat_history"),
# # #         ("human", "{input}"),
# # #     ])

# # #     history_aware_retriever = create_history_aware_retriever(
# # #         llm, retriever, contextualize_q_prompt
# # #     )

# # #     system_prompt = (
        
# # #     "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
# # #     "Your role is to provide accurate, helpful, and easy-to-understand health information based on the given context. "

# # #     "\n\n### CORE RULES ###\n"
# # #     "- ONLY use information from the provided context to answer questions. "
# # #     "- If the context lacks sufficient information, respond with: 'I'm sorry, I don't have enough information on this topic. Please consult a qualified doctor.' "
# # #     "- NEVER invent, guess, or hallucinate any medical information, drug names, or dosages. "
# # #     "- NEVER provide a specific diagnosis. You can explain symptoms and conditions, but always clarify you are an AI, not a doctor. "

# # #     "\n\n### LANGUAGE RULES ###\n"
# # #     "- Always reply in the SAME language the user used. "
# # #     "- If the user writes in Bengali, reply fully in Bengali. If in English, reply in English. "
# # #     "- If the user explicitly asks for a translation (e.g., 'এটা ইংরেজিতে বলো' or 'say this in Bengali'), switch to that language. "
# # #     "- The user may also write in 'Banglish' (Bengali words written in English letters, e.g., 'ami jore bugtesi' means 'I have fever'). "
# # #     "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
# # #     "- Common Banglish medical terms to understand: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
# # #     "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
# # #     "'durbolta' = weakness, 'bomi' = vomiting, 'pায়খানা' = diarrhea, 'ghum nai' = insomnia, "
# # #     "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
# # #     "- If unsure about a Banglish word's meaning, try to interpret it based on context and reply in Bengali. "

# # #     "\n\n### FORMATTING RULES ###\n"
# # #     "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line.Always use line breaks for readability. "
# # #     "- For step-by-step instructions or processes: use numbered lists (1. 2. 3.), one step per line.Always use line breaks for readability. "
# # #     "- For complex topics: use clear headings with **bold** text to separate sections.Always use line breaks for readability. "
# # #     "- Keep answers concise. For simple questions, give short answers. For complex questions, give detailed but structured answers.Always use line breaks for readability. "
# # #     "- Never put two separate ideas in a single line. Always use line breaks for readability. "

# # #     "\n\n### EMERGENCY PROTOCOL ###\n"
# # #     "- If the user describes emergency symptoms such as: chest pain with sweating, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms, or severe allergic reaction — "
# # #     "IMMEDIATELY start your response with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
# # #     "- Do not wait to give other information before showing the emergency alert. "

# # #     "\n\n### SPECIALIST RECOMMENDATION ###\n"
# # #     "- If the user asks who to consult, recommend the most appropriate medical specialist based on their symptoms. "
# # #     "- Examples: chest pain → Cardiologist, skin issues → Dermatologist, headaches/seizures → Neurologist, mental health → Psychiatrist, child health → Pediatrician. "
# # #     "- Always phrase it as a recommendation, not a diagnosis. "

# # #     "\n\n### Website Resources ###\n"
# # #     "When extracting information from the context, strictly ignore any marketing text, website navigation menus, subscription offers, or promotional content. Focus ONLY on clinical facts, symptoms, treatments, and medical guidelines."

# # #     "\n\n### DISCLAIMER ###\n"
# # #     "- If the user asks for medical advice or treatment suggestions, always end your response with: "
# # #     "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "

# # #     "\n\nContext:\n{context}"
# # #     )
    
    
# # #     prompt = ChatPromptTemplate.from_messages([
# # #         ("system", system_prompt),
# # #         MessagesPlaceholder("chat_history"),
# # #         ("human", "{input}")
# # #     ])
    
# # #     question_answer_chain = create_stuff_documents_chain(llm, prompt)
# # #     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
# # #     return rag_chain

# # # rag_chain = load_rag_pipeline()


# # # user_input = st.chat_input("আপনার লক্ষণ বা সমস্যার কথা লিখুন...")

# # # if user_input:
    
# # #     with st.chat_message("user"):
# # #         st.markdown(user_input)
# # #     st.session_state.messages.append({"role": "user", "content": user_input})
    
# # #     with st.chat_message("assistant"):
# # #         with st.spinner("Processing..."):
            
# # #             response = rag_chain.invoke({
# # #                 "input": user_input,
# # #                 "chat_history": st.session_state.chat_history
# # #             })
# # #             answer = response["answer"]
            
# # #             disclaimer = "\n\n---\n*⚠️ Disclaimer:I am an AI model. This is for informational purposes only. Please consult a registered doctor for any serious health issues.*"
# # #             final_answer = answer + disclaimer
            
# # #             st.markdown(final_answer)

    
# # #     st.session_state.chat_history.append(HumanMessage(content=user_input))
# # #     st.session_state.chat_history.append(AIMessage(content=final_answer))
# # #     st.session_state.messages.append({"role": "assistant", "content": final_answer})















# # # #     import os
# # # # import streamlit as st
# # # # from dotenv import load_dotenv
# # # # from langchain_groq import ChatGroq
# # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # from langchain_community.vectorstores import Chroma

# # # # from langchain.chains import create_retrieval_chain
# # # # from langchain.chains.combine_documents import create_stuff_documents_chain

# # # # from langchain_core.prompts import ChatPromptTemplate


# # # # load_dotenv()
# # # # groq_api_key = os.getenv("GROQ_API_KEY")


# # # # st.set_page_config(page_title="Personal Medical Assistant", page_icon="🩺")
# # # # st.title("🩺 Personal Medical Assistant")
# # # # st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা বা ইংরেজিতে)।")


# # # # if "messages" not in st.session_state:
# # # #     st.session_state.messages = []

# # # # for message in st.session_state.messages:
# # # #     with st.chat_message(message["role"]):
# # # #         st.markdown(message["content"])


# # # # @st.cache_resource
# # # # def load_rag_pipeline():
    
# # # #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# # # #     vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# # # #     retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # find top 3 relevant chunks
    
    
# # # #     llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    
# # # #     system_prompt = (
        
# # # #     "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
# # # #     "Your role is to provide accurate, helpful, and easy-to-understand health information based on the given context. "

# # # #     "\n\n### CORE RULES ###\n"
# # # #     "- ONLY use information from the provided context to answer questions. "
# # # #     "- If the context lacks sufficient information, respond with: 'I'm sorry, I don't have enough information on this topic. Please consult a qualified doctor.' "
# # # #     "- NEVER invent, guess, or hallucinate any medical information, drug names, or dosages. "
# # # #     "- NEVER provide a specific diagnosis. You can explain symptoms and conditions, but always clarify you are an AI, not a doctor. "

# # # #     "\n\n### LANGUAGE RULES ###\n"
# # # #     "- Always reply in the SAME language the user used. "
# # # #     "- If the user writes in Bengali, reply fully in Bengali. If in English, reply in English. "
# # # #     "- If the user explicitly asks for a translation (e.g., 'এটা ইংরেজিতে বলো' or 'say this in Bengali'), switch to that language. "
# # # #     "- The user may also write in 'Banglish' (Bengali words written in English letters, e.g., 'ami jore bugtesi' means 'I have fever'). "
# # # #     "- If the user writes in Banglish, understand it as Bengali and reply in proper Bengali script. "
# # # #     "- Common Banglish medical terms to understand: 'jore bugti/bugtesi' = fever, 'mathay betha' = headache, "
# # # #     "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
# # # #     "'durbolta' = weakness, 'bomi' = vomiting, 'pায়খানা' = diarrhea, 'ghum nai' = insomnia, "
# # # #     "'khide nai' = loss of appetite, 'shordi' = cold/cough, 'khansi' = cough. "
# # # #     "- If unsure about a Banglish word's meaning, try to interpret it based on context and reply in Bengali. "

# # # #     "\n\n### FORMATTING RULES ###\n"
# # # #     "- For lists of symptoms, diseases, or medicines: use bullet points with ➡️ emoji, one point per line.Always use line breaks for readability. "
# # # #     "- For step-by-step instructions or processes: use numbered lists (1. 2. 3.), one step per line.Always use line breaks for readability. "
# # # #     "- For complex topics: use clear headings with **bold** text to separate sections.Always use line breaks for readability. "
# # # #     "- Keep answers concise. For simple questions, give short answers. For complex questions, give detailed but structured answers.Always use line breaks for readability. "
# # # #     "- Never put two separate ideas in a single line. Always use line breaks for readability. "

# # # #     "\n\n### EMERGENCY PROTOCOL ###\n"
# # # #     "- If the user describes emergency symptoms such as: chest pain with sweating, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms, or severe allergic reaction — "
# # # #     "IMMEDIATELY start your response with: '🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!' "
# # # #     "- Do not wait to give other information before showing the emergency alert. "

# # # #     "\n\n### SPECIALIST RECOMMENDATION ###\n"
# # # #     "- If the user asks who to consult, recommend the most appropriate medical specialist based on their symptoms. "
# # # #     "- Examples: chest pain → Cardiologist, skin issues → Dermatologist, headaches/seizures → Neurologist, mental health → Psychiatrist, child health → Pediatrician. "
# # # #     "- Always phrase it as a recommendation, not a diagnosis. "

# # # #     "\n\n### Website Resources ###\n"
# # # #     "When extracting information from the context, strictly ignore any marketing text, website navigation menus, subscription offers, or promotional content. Focus ONLY on clinical facts, symptoms, treatments, and medical guidelines."

# # # #     "\n\n### DISCLAIMER ###\n"
# # # #     "- If the user asks for medical advice or treatment suggestions, always end your response with: "
# # # #     "'⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো চূড়ান্ত মেডিকেল পরামর্শ নয়। যেকোনো স্বাস্থ্য সমস্যায় অবশ্যই একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।' "

# # # #     "\n\nContext:\n{context}"







# # # #         # "You are an expert AI Medical Information Assistant. Use the given context to answer the user's health-related question. "
# # # #         # "If you don't know the answer or the context doesn't have the information, just say 'I am sorry, I don't have enough information on this.' "
# # # #         # "DO NOT invent or hallucinate any medical information or medicines. "
# # # #         # "Always reply in the same language the user asked the question (If Bengali, reply in Bengali. If English, reply in English). "
# # # #         # "If the user asks for bangla translation in English, reply in Bengali. If the user asks for English translation in Bengali, reply in English."
# # # #         # "If the user asks for a list of symptoms, diseases, or medicines, provide the answer in a bullet-point format.use dark mode formatting for the answer and arrow emojis (➡️) for each point.provide the answer in a list format so that two sentence doesn't come in a single line."
# # # #         # "and if the user asks for a step-by-step process, provide the answer in a numbered list format. "
# # # #         # "Always try to keep your answer concise and to the point, but if the question is complex, provide a detailed answer."
# # # #         # "Based on the user's symptoms, if the user ask then recommend the exact medical specialist they should consult (e.g., Cardiologist, Dermatologist, Neurologist, etc.). "
# # # #         # "If the symptoms indicate a severe emergency (like chest pain, sweating, severe bleeding, or breathing difficulty), urge them to visit the nearest hospital emergency immediately. "

# # # #         "\n\nContext:\n{context}"
# # # #     )
    
# # # #     prompt = ChatPromptTemplate.from_messages([
# # # #         ("system", system_prompt),
# # # #         ("human", "{input}")
# # # #     ])
    
# # # #     question_answer_chain = create_stuff_documents_chain(llm, prompt)
# # # #     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
# # # #     return rag_chain

# # # # rag_chain = load_rag_pipeline()


# # # # user_input = st.chat_input("আপনার লক্ষণ বা সমস্যার কথা লিখুন...")

# # # # if user_input:
    
# # # #     with st.chat_message("user"):
# # # #         st.markdown(user_input)
# # # #     st.session_state.messages.append({"role": "user", "content": user_input})
    
   
# # # #     with st.chat_message("assistant"):
# # # #         with st.spinner("Processing..."):
# # # #             response = rag_chain.invoke({"input": user_input})
# # # #             answer = response["answer"]
            
            
# # # #             disclaimer = "\n\n---\n*⚠️ সতর্কতা: আমি একটি এআই মডেল। এটি কোনো মেডিকেল পরামর্শ নয়, যেকোনো গুরুতর সমস্যায় অবশ্যই রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।*"
# # # #             final_answer = answer + disclaimer
            
# # # #             st.markdown(final_answer)
    
# # # #     st.session_state.messages.append({"role": "assistant", "content": final_answer})