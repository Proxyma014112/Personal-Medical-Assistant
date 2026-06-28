import os
from groq import Groq
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from PIL import Image
import google.generativeai as genai


# ─── Environment Setup ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

groq_api_key   = os.environ.get("GROQ_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Groq client for Whisper voice transcription
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


# ─── Gemini Vision Functions ──────────────────────────────────────────
def analyze_symptoms_from_image(image):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = """
        You are an expert clinical observer. Look at this medical image and accurately
        describe the visible symptoms in 2-3 short sentences.
        DO NOT provide a diagnosis or medical advice.
        Just describe what you see physically. Reply in Bengali.
        """
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"ERROR: ছবি প্রসেস করতে সমস্যা হয়েছে ({str(e)})"


def extract_text_gemini(image):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = """
        You are an expert pharmacist. Read this medical prescription or lab report carefully.
        Extract: medicine names, dosages, duration, and any medical advice.
        Format the extracted text neatly. Do not give medical advice.
        """
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini OCR ব্যর্থ হয়েছে ({str(e)})"


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="MediAssist AI", page_icon="👩🏻‍⚕️", layout="wide")
st.title("👩🏻‍⚕️ MediAssist AI — Personal Medical Assistant")
st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন (বাংলা, ইংরেজি বা Banglish-এ)।")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY পাওয়া যায়নি। HF Space Settings → Secrets এ add করুন।")
    st.stop()


# ─── Session State Init ───────────────────────────────────────────────
defaults = {
    "messages": [],
    "chat_history": [],
    "ocr_confirmed_text": None,
    "ocr_extracted_text": None,
    "ui_language": "বাংলা",
    "last_uploaded_file": None,
    "vision_extracted_symptoms": None,
    "rag_chain": None,
    "triage_active": False,
    "triage_questions": [],
    "triage_original_input": "",
    "triage_answers": {},
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─── Display Chat History ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── RAG Pipeline (Modern LangChain) ─────────────────────────────────
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    chroma_path = "./chroma_db"
    if not os.path.exists(chroma_path):
        chroma_path = "/app/chroma_db"

    vector_db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # ── Contextualize question using chat history ──
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question to be standalone and clear. "
         "Do NOT answer it, just rephrase if needed. Return as is if already clear."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(x):
        if x["chat_history"]:
            reformulate_chain = contextualize_prompt | llm | StrOutputParser()
            question = reformulate_chain.invoke({
                "input": x["input"],
                "chat_history": x["chat_history"]
            })
        else:
            question = x["input"]
        docs = retriever.invoke(question)
        return format_docs(docs)

    # ── Main QA Prompt ──
    system_prompt = (
        "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant. "
        "Your role is to provide accurate, helpful health information based on the given context.\n\n"

        "### CORE RULES ###\n"
        "- ONLY use information from the provided context.\n"
        "- If context lacks info, say: 'I don't have enough information. Please consult a doctor.'\n"
        "- NEVER invent medical information, drug names, or dosages.\n"
        "- NEVER provide a specific diagnosis.\n\n"

        "### LANGUAGE RULES ###\n"
        "- Reply in the SAME language the user used.\n"
        "- Bengali input → Bengali reply. English input → English reply.\n"
        "- Banglish input (Bengali in English letters) → reply in proper Bengali script.\n"
        "- Common Banglish terms: 'jore bugti' = fever, 'mathay betha' = headache, "
        "'buk betha' = chest pain, 'pet betha' = stomach pain, 'shash kosto' = breathing difficulty, "
        "'durbolta' = weakness, 'bomi' = vomiting, 'shordi' = cold, 'khansi' = cough.\n\n"

        "### FORMATTING ###\n"
        "- Symptoms/medicines list: use ➡️ bullet points, one per line.\n"
        "- Step-by-step: use numbered lists.\n"
        "- Complex topics: use **bold** headings.\n\n"

        "### EMERGENCY PROTOCOL ###\n"
        "- For emergency symptoms (chest pain + sweating, breathing difficulty, severe bleeding, "
        "loss of consciousness, stroke) → IMMEDIATELY say: "
        "'🚨 EMERGENCY ALERT: Please call emergency services or go to the nearest hospital RIGHT NOW!'\n\n"

        "### SPECIALIST ###\n"
        "- Chest pain → Cardiologist, Skin → Dermatologist, Headache → Neurologist, "
        "Mental health → Psychiatrist, Child → Pediatrician.\n\n"

        "### DISCLAIMER ###\n"
        "- Always end with: '⚠️ সতর্কতা: আমি একটি এআই মডেল। যেকোনো স্বাস্থ্য সমস্যায় "
        "একজন রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।'\n\n"

        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(get_context))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ─── Sidebar: Settings ────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.session_state.ui_language = st.sidebar.radio("Language / ভাষা:", ["বাংলা", "English"])
is_bangla = st.session_state.ui_language == "বাংলা"


# ─── Sidebar: Health Dashboard ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("📊 Health Dashboard")
st.sidebar.write("আপনার প্রাথমিক স্বাস্থ্য তথ্য দিন:")

weight    = st.sidebar.number_input("ওজন (kg)",  min_value=10.0,  max_value=200.0, value=70.0,  step=0.5)
height_cm = st.sidebar.number_input("উচ্চতা (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)

if st.sidebar.button("Calculate BMI & Stats"):
    bmi          = weight / ((height_cm / 100) ** 2)
    water_intake = (weight * 35) / 1000
    st.sidebar.markdown("---")
    st.sidebar.subheader("আপনার ফলাফল:")
    st.sidebar.write(f"**BMI:** {bmi:.2f}")
    if bmi < 18.5:      st.sidebar.warning("Underweight 📉")
    elif bmi < 24.9:    st.sidebar.success("Normal Weight ✅")
    elif bmi < 29.9:    st.sidebar.warning("Overweight 📈")
    else:               st.sidebar.error("Obese ⚠️")
    st.sidebar.write(f"**দৈনিক পানির চাহিদা:** {water_intake:.1f} লিটার 💧")
    st.sidebar.info("💡 মেইন চ্যাটে ডায়েট প্ল্যান চাইতে পারেন!")


# ─── Sidebar: Visual Symptom Checker ─────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Symptom Checker")
st.sidebar.caption("লক্ষণ বুঝতে ছবি আপলোড করুন (Powered by Gemini Vision)")

vision_upload = st.sidebar.file_uploader(
    "ছবি আপলোড করুন (JPG/PNG)", type=["jpg", "jpeg", "png"], key="vision_uploader"
)
if vision_upload:
    vision_image = Image.open(vision_upload)
    st.sidebar.image(vision_image, caption="আপলোড করা ছবি", use_container_width=True)
    if st.sidebar.button("🔍 লক্ষণ বিশ্লেষণ করুন"):
        with st.spinner("ছবি বিশ্লেষণ করা হচ্ছে..."):
            st.session_state.vision_extracted_symptoms = analyze_symptoms_from_image(vision_image)

    if st.session_state.vision_extracted_symptoms:
        if not st.session_state.vision_extracted_symptoms.startswith("ERROR"):
            st.sidebar.success("✅ বিশ্লেষণ সম্পন্ন!")
            st.sidebar.info(f"**প্রাথমিক লক্ষণ:**\n{st.session_state.vision_extracted_symptoms}")
        else:
            st.sidebar.error(st.session_state.vision_extracted_symptoms)


# ─── Sidebar: Prescription OCR ───────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Prescription / Report OCR")
st.sidebar.caption("🔍 Gemini Vision দ্বারা পরিচালিত" if is_bangla else "🔍 Powered by Gemini Vision")

upload_label  = "প্রেসক্রিপশন / রিপোর্টের ছবি আপলোড করুন" if is_bangla else "Upload Prescription / Report Image"
uploaded_image = st.sidebar.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_image:
    if st.session_state.last_uploaded_file != uploaded_image.name:
        st.session_state.ocr_extracted_text  = None
        st.session_state.ocr_confirmed_text  = None
        st.session_state.last_uploaded_file  = uploaded_image.name

    image = Image.open(uploaded_image)
    st.sidebar.image(image, caption="আপলোড করা ছবি" if is_bangla else "Uploaded Image", use_container_width=True)

    if st.session_state.ocr_extracted_text is None:
        with st.spinner("🔍 OCR processing হচ্ছে..." if is_bangla else "🔍 Processing OCR..."):
            st.session_state.ocr_extracted_text = extract_text_gemini(image)
            st.session_state.ocr_confirmed_text = None

    if (st.session_state.ocr_extracted_text
            and not st.session_state.ocr_extracted_text.startswith("ERROR")
            and st.session_state.ocr_confirmed_text is None):
        st.info("📋 ছবি থেকে text পাওয়া গেছে। এটা কি সঠিক?" if is_bangla else "📋 Text extracted. Is this correct?")
        edited_text = st.text_area("✏️ প্রয়োজনে সম্পাদনা করুন:" if is_bangla else "✏️ Edit if needed:",
        value=st.session_state.ocr_extracted_text, height=150)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ হ্যাঁ, সঠিক" if is_bangla else "✅ Confirm"):
                st.session_state.ocr_confirmed_text = edited_text
                st.success("✅ Confirmed!")
        with col2:
            if st.button("🔄 আবার চেষ্টা" if is_bangla else "🔄 Retry"):
                st.session_state.ocr_extracted_text = None
                st.session_state.ocr_confirmed_text = None
                st.rerun()

    elif (st.session_state.ocr_extracted_text
            and st.session_state.ocr_extracted_text.startswith("ERROR")):
        st.error(f"❌ OCR ব্যর্থ: {st.session_state.ocr_extracted_text}")

    elif st.session_state.ocr_confirmed_text:
        st.sidebar.success("✅ OCR confirmed.")
        if st.sidebar.button("🗑️ Reset OCR"):
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.session_state.last_uploaded_file = None
            st.rerun()

elif st.session_state.last_uploaded_file is not None:
    st.session_state.ocr_extracted_text = None
    st.session_state.ocr_confirmed_text = None
    st.session_state.last_uploaded_file = None


# ─── Sidebar: Hospital & Specialist Finder ────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 Nearby Hospital Finder")
st.sidebar.caption("আপনার কাছের হাসপাতাল বা বিশেষজ্ঞ খুঁজুন")

bangladesh_districts = [
    "Dhaka", "Chittagong (Chattogram)", "Sylhet", "Rajshahi", "Khulna",
    "Barishal", "Rangpur", "Mymensingh", "Comilla", "Narayanganj",
    "Gazipur", "Tangail", "Bogura", "Jessore (Jashore)", "Cox's Bazar",
    "Dinajpur", "Pabna", "Brahmanbaria", "Sirajganj", "Noakhali",
    "Faridpur", "Narsingdi", "Kishoreganj", "Habiganj", "Moulvibazar",
    "Sunamganj", "Netrokona", "Sherpur", "Jamalpur", "Magura",
    "Kushtia", "Meherpur", "Chuadanga", "Jhenaidah", "Narail",
    "Satkhira", "Bagerhat", "Pirojpur", "Jhalokathi", "Patuakhali",
    "Bhola", "Barguna", "Lakshmipur", "Chandpur", "Feni",
    "Khagrachhari", "Rangamati", "Bandarban", "Natore", "Chapainawabganj",
    "Naogaon", "Joypurhat", "Thakurgaon", "Panchagarh", "Nilphamari",
    "Lalmonirhat", "Kurigram", "Gaibandha", "Jashore", "Shariatpur",
    "Madaripur", "Gopalganj", "Munshiganj", "Manikganj", "Rajbari"
]

specialist_types = [
    "হাসপাতাল (General Hospital)",
    "হৃদরোগ বিশেষজ্ঞ (Cardiologist)",
    "চর্মরোগ বিশেষজ্ঞ (Dermatologist)",
    "স্নায়ু বিশেষজ্ঞ (Neurologist)",
    "শিশু বিশেষজ্ঞ (Pediatrician)",
    "মানসিক স্বাস্থ্য (Psychiatrist)",
    "হাড় বিশেষজ্ঞ (Orthopedic)",
    "চক্ষু বিশেষজ্ঞ (Ophthalmologist)",
    "দন্ত চিকিৎসক (Dentist)",
    "ডায়াবেটিস বিশেষজ্ঞ (Diabetologist)",
    "কিডনি বিশেষজ্ঞ (Nephrologist)",
    "ক্যান্সার বিশেষজ্ঞ (Oncologist)",
    "গাইনি বিশেষজ্ঞ (Gynecologist)",
    "ইএনটি বিশেষজ্ঞ (ENT Specialist)",
]

selected_district   = st.sidebar.selectbox("আপনার জেলা বেছে নিন:", bangladesh_districts)
selected_specialist = st.sidebar.selectbox("কী ধরনের সাহায্য দরকার?", specialist_types)

if st.sidebar.button("🔍 হাসপাতাল খুঁজুন"):
    specialist_en = selected_specialist.split("(")[-1].replace(")", "").strip()
    query    = f"{specialist_en} hospital near {selected_district} Bangladesh"
    maps_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"

    st.sidebar.success(f"✅ **{selected_district}** এ **{selected_specialist}** খোঁজা হচ্ছে!")
    st.sidebar.markdown(f"### 🗺️ [Google Maps এ দেখুন →]({maps_url})")
    st.sidebar.info(
        "💡 **টিপস:**\n"
        "- লিংকে click করলে Google Maps খুলবে\n"
        "- Rating ও Reviews দেখে হাসপাতাল বেছে নিন\n"
        "- যাওয়ার আগে phone করে appointment নিন\n"
        "- জরুরি অবস্থায় সরাসরি চলে যান"
    )
    st.sidebar.warning(
        "⚠️ MediAssist AI কোনো নির্দিষ্ট হাসপাতালকে recommend করে না। "
        "Google Maps এর তথ্য যাচাই করে নিন।"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("🆘 **জরুরি: 999 call করুন**")


# ─── Emergency Numbers Section (Main Area) ───────────────────────────
with st.expander("🚨 জরুরি হেল্পলাইন নম্বর — Emergency Helpline Numbers (ক্লিক করুন)"):
    st.markdown("### 🇧🇩 বাংলাদেশ জরুরি হেল্পলাইন")
    st.markdown("---")

    st.markdown("#### 🚑 প্রধান জরুরি সেবা")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**999**")
        st.markdown("পুলিশ / ফায়ার / অ্যাম্বুলেন্স\n*(24/7 available)*")
    with col2:
        st.error("**16263**")
        st.markdown("DGHS স্বাস্থ্য হেল্পলাইন\n*(Health Emergency)*")
    with col3:
        st.error("**333**")
        st.markdown("জাতীয় স্বাস্থ্য সেবা\n*(Health Helpline)*")

    st.markdown("---")

    st.markdown("#### 🏥 স্বাস্থ্য সেবা")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.warning("**16000**")
        st.markdown("জাতীয় হেল্পলাইন\n*(National Helpline)*")
    with col5:
        st.warning("**09611667777**")
        st.markdown("বিষ নিয়ন্ত্রণ কেন্দ্র\n*(Poison Control)*")
    with col6:
        st.warning("**10655**")
        st.markdown("মানসিক স্বাস্থ্য\n*(Mental Health)*")

    st.markdown("---")

    st.markdown("#### 👩‍👧 নারী ও শিশু সেবা")
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.info("**109**")
        st.markdown("নারী হেল্পলাইন\n*(Women Helpline)*")
    with col8:
        st.info("**10921**")
        st.markdown("নারী সহায়তা\n*(Women Support)*")
    with col9:
        st.info("**1098**")
        st.markdown("শিশু হেল্পলাইন\n*(Child Helpline)*")
    with col10:
        st.info("**16430**")
        st.markdown("আইনি সহায়তা\n*(Legal Aid)*")

    st.markdown("---")
    st.caption(
        "⚠️ জরুরি অবস্থায় সরাসরি **999** dial করুন। "
        "উপরের নম্বরগুলো Bangladesh-এর সরকারি হেল্পলাইন।"
    )


# ─── Smart Symptom Detection + MCQ Triage ────────────────────────────
def is_symptom_query(text: str) -> bool:
    symptom_keywords = [
        "ব্যথা", "বেথা", "জ্বর", "কাশি", "শ্বাস", "বমি", "মাথা", "বুক",
        "পেট", "গলা", "চোখ", "কান", "নাক", "হাত", "পা", "দুর্বল", "ক্লান্ত",
        "ঘুম", "খিদে", "র্যাশ", "চুলকানি", "ফোলা", "রক্ত", "ডায়রিয়া",
        "betha", "jor", "jore", "bugti", "bugtesi", "kashi", "shash", "bomi",
        "matha", "buk", "pet", "gola", "chokh", "kan", "nak", "durbolta",
        "ghum", "khide", "rash", "cholkani", "fola", "shordi", "khansi",
        "mathay", "pate", "buke",
        "pain", "fever", "cough", "headache", "nausea", "vomit", "dizzy",
        "tired", "weak", "rash", "swelling", "bleeding", "diarrhea",
        "chest", "stomach", "throat", "eye", "ear", "breathing", "itching",
        "burning", "ache", "sore", "hurt", "suffering", "feeling",
    ]
    return any(kw in text.lower() for kw in symptom_keywords)


def generate_triage_questions(symptom_text: str) -> list:
    try:
        import json
        client = Groq(api_key=groq_api_key)
        prompt = f"""A patient described: "{symptom_text}"
Generate 3-5 MCQ questions to understand this condition better.
Rules:
- Match the patient language (Bengali input = Bengali questions)
- Each question has exactly 4 options
- Return ONLY valid JSON, no extra text

JSON format:
{{
"questions": [
    {{"id": 1, "question": "question text", "options": ["A", "B", "C", "D"]}}
]
}}"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip()).get("questions", [])
    except Exception:
        return []


# ─── Voice Language Config ────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "বাংলা": {
        "voice_prompt":  "এটি একটি চিকিৎসা সংক্রান্ত কথোপকথন।",
        "spinner_msg":   "ভয়েস প্রসেস করা হচ্ছে...",
        "error_msg":     "ভয়েস প্রসেসিং ব্যর্থ হয়েছে",
        "whisper_lang":  "bn",
    },
    "English": {
        "voice_prompt":  "This is a medical conversation.",
        "spinner_msg":   "Processing voice...",
        "error_msg":     "Voice processing failed",
        "whisper_lang":  "en",
    },
}


# ─── Shared helper: build enriched input & call RAG ──────────────────
def _run_rag(user_input: str, enriched_input: str) -> str:
    """Load RAG chain if needed, invoke it, and return the answer string."""
    if st.session_state.rag_chain is None:
        with st.spinner("লোড হচ্ছে... ⏳"):
            st.session_state.rag_chain = load_rag_pipeline()
    try:
        answer = st.session_state.rag_chain.invoke({
            "input": enriched_input,
            "chat_history": st.session_state.chat_history,
        })
    except Exception as e:
        answer = f"❌ সমস্যা: {str(e)}"
    return answer


def _append_context(base: str) -> str:
    """Append OCR and vision context to the base input string."""
    if st.session_state.get("ocr_confirmed_text"):
        base += f"\n\n[OCR]:\n{st.session_state.ocr_confirmed_text}"
    vision = st.session_state.get("vision_extracted_symptoms")
    if vision and not vision.startswith("ERROR"):
        base += f"\n\n[ছবির লক্ষণ]:\n{vision}"
    return base


def _save_to_history(user_input: str, enriched_input: str, answer: str):
    """Persist message to session state history."""
    st.session_state.chat_history.extend([
        HumanMessage(content=enriched_input),
        AIMessage(content=answer),
    ])
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── Main Chat ────────────────────────────────────────────────────────
placeholder = (
    "আপনার লক্ষণ বা সমস্যার কথা লিখুন অথবা মাইক্রোফোন ব্যবহার করুন..."
    if is_bangla else
    "Type your symptoms or question, or use the microphone..."
)
@st.dialog("🩺 লক্ষণ মূল্যায়ন", width="large")
def triage_dialog():
    st.caption("সব প্রশ্নের উত্তর দিন তারপর submit করুন")
    questions = st.session_state.triage_questions

    # ── Collect answers INTO session_state directly (not a local dict) ──
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            with st.container(border=True):
                st.caption(f"প্রশ্ন {i+1}")
                st.markdown(f"**{q['question']}**")
                val = st.radio(
    q["question"],                    # ← non-empty label
    q["options"],
    key=f"dq_{q['id']}",
    index=None,
    label_visibility="hidden"         # ← hides it visually
)
                if val:
                    st.session_state.triage_answers[str(q["id"])] = val

    answered = len(st.session_state.triage_answers)
    total    = len(questions)
    st.progress(answered / total if total else 0)
    st.caption(f"✅ {answered} / {total} answered")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("⏭️ Skip", use_container_width=True):
            # Mark skip — dialog will close, main flow handles it
            st.session_state.triage_skip     = True
            st.session_state.triage_active   = False
            st.session_state.triage_questions = []
            st.rerun()
    with col2:
        if st.button("✅ Submit", type="primary", use_container_width=True,
                     disabled=answered < total):
            st.session_state.triage_submit   = True
            st.session_state.triage_active   = False
            st.session_state.triage_questions = []
            st.rerun()


# ── Initialize flags ──────────────────────────────────────────────────
for k, v in {"triage_submit": False, "triage_skip": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Triage MCQ UI ─────────────────────────────────────────────────────
if st.session_state.triage_active and st.session_state.triage_questions:
    with st.chat_message("assistant"):
        st.markdown("🩺 **আপনার সমস্যা আরো ভালোভাবে বুঝতে কিছু প্রশ্ন করছি:**"
                    if is_bangla else "🩺 **To better understand your condition, please answer:**")
    triage_dialog()   # ← opens the modal


# ── After dialog closes — run RAG with enriched input ─────────────────
if st.session_state.get("triage_submit") or st.session_state.get("triage_skip"):
    submitted = st.session_state.pop("triage_submit", False)
    st.session_state.pop("triage_skip", False)

    enriched = st.session_state.triage_original_input

    if submitted and st.session_state.triage_answers:
        enriched += "\n\n[রোগীর অতিরিক্ত তথ্য]:\n"
        for q in (st.session_state.get("triage_questions_backup") or []):
            ans = st.session_state.triage_answers.get(str(q["id"]), "N/A")
            enriched += f"- {q['question']}: {ans}\n"

    enriched = _append_context(enriched)

    original_input = st.session_state.triage_original_input  # save before reset

    # Reset triage state
    st.session_state.triage_answers          = {}
    st.session_state.triage_original_input   = ""
    st.session_state.triage_questions_backup = []

    # Run RAG and show answer
    with st.chat_message("assistant"):
        label = "বিশ্লেষণ করা হচ্ছে... 🔍" if is_bangla else "Analyzing... 🔍"
        with st.spinner(label):
            answer = _run_rag(enriched, enriched)
        st.markdown(answer)

    _save_to_history(original_input, enriched, answer)  # ← handles both appends
    # ← REMOVED: the extra st.session_state.messages.append({"role": "assistant", ...})
    st.rerun()

else:
    # ── Normal chat input (text + audio) ─────────────────────────────

    prompt = st.chat_input(
        placeholder,
        accept_audio=True,

    )


    if prompt:
        user_input = ""
        lang_cfg   = LANGUAGE_CONFIG[st.session_state.ui_language]

        # ── Voice branch — transcribe only, write to input bar ────────
        if prompt.audio:
            with st.spinner(lang_cfg["spinner_msg"]):
                try:
                    prompt.audio.name = "recording.wav"
                    transcription = groq_client.audio.transcriptions.create(
                        file=prompt.audio,
                        model="whisper-large-v3-turbo",
                        language=lang_cfg["whisper_lang"],
                        prompt=lang_cfg["voice_prompt"],
                    )
                    user_input = transcription.text
                except Exception as e:
                    st.error(f"{lang_cfg['error_msg']}: {e}")


        # ── Text branch ───────────────────────────────────────────────
        elif prompt.text:
            user_input = prompt.text

        if user_input:
            # Show user bubble (raw input only — no internal context)
            with st.chat_message("user"):
                st.markdown(user_input)

            combined = _append_context(user_input)

            if is_symptom_query(user_input):
                # Try to generate triage MCQ first
                with st.spinner("🩺 লক্ষণ বিশ্লেষণ করা হচ্ছে..." if is_bangla
                                else "🩺 Analyzing symptoms..."):
                    questions = generate_triage_questions(user_input)

                if questions:
                    # Store for MCQ UI on next render
                    st.session_state.triage_active         = True
                    st.session_state.triage_questions      = questions
                    st.session_state.triage_original_input = user_input
                    st.session_state.triage_answers        = {}
                    # Keep user message in display history before rerun
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.rerun()
                else:
                    # No MCQ generated — answer directly
                    with st.chat_message("assistant"):
                        with st.spinner("তথ্য খোঁজা হচ্ছে..." if is_bangla else "Searching..."):
                            answer = _run_rag(user_input, combined)
                        st.markdown(answer)
                    _save_to_history(user_input, combined, answer)

            else:
                # Non-symptom query — answer directly
                with st.chat_message("assistant"):
                    with st.spinner("তথ্য খোঁজা হচ্ছে..." if is_bangla else "Searching..."):
                        answer = _run_rag(user_input, combined)
                    st.markdown(answer)
                _save_to_history(user_input, combined, answer)