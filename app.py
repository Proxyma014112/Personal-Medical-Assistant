import os
import json
import logging
from io import BytesIO
import streamlit as st
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mediassist")

# ─── Environment Setup ────────────────────────────────────────────────
try:
    load_dotenv()
except Exception:
    pass

groq_api_key   = os.environ.get("GROQ_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("gemini-2.5-flash")


# ─── Gemini Vision Functions ──────────────────────────────────────────
def analyze_symptoms_from_image(image, bangla: bool):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = get_gemini_model()
        prompt = f"""You are a clinical observer describing ONLY what is physically visible in this image — no interpretation, no diagnosis, no medical advice.

STEP 1 — Verify: Does this image show a visible physical symptom, condition, or body part relevant to a health concern (e.g. rash, swelling, wound, discoloration)?
- If NO (e.g. unrelated photo, object, scenery, no visible symptom): respond with exactly "NOT_A_SYMPTOM_IMAGE" and nothing else.

STEP 2 — If YES, describe in 2-3 short sentences:
- Location on the body (if identifiable)
- Visual characteristics only: color, size, texture, shape, swelling, discharge, etc.
- Do NOT name a condition, disease, or likely cause. Do NOT say things like "this looks like X."

Reply entirely in {'Bengali' if bangla else 'English'}. Do not mix languages."""
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        if text == "NOT_A_SYMPTOM_IMAGE":
            return "ERROR: এই ছবিতে কোনো লক্ষণ শনাক্ত করা যায়নি।" if bangla else "ERROR: No visible symptom detected in this image."
        return text
    except Exception:
        logger.exception("Gemini vision analysis failed")
        return "ERROR: ছবি প্রসেস করতে সমস্যা হয়েছে। আবার চেষ্টা করুন।"


def extract_text_gemini(image, bangla: bool):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = get_gemini_model()
        prompt = f"""You are transcribing a medical prescription or lab report image. Your ONLY job is to transcribe what is written, not to interpret or advise.

STEP 1 — Verify: Is this image actually a medical prescription, lab report, or medicine packaging?
- If NO (e.g. unrelated photo, blank image, illegible scan): respond with exactly "NOT_A_PRESCRIPTION" and nothing else.

STEP 2 — If YES, transcribe using this exact structure, leaving a field blank if not present:
Medicines:
- [name] — [dosage] — [frequency] — [duration]

Other instructions on document: [transcribe verbatim any advice/instructions written on the document — do not add your own]

Doctor/Clinic name: [if visible]
Date: [if visible]

RULES:
- Transcribe text exactly as written. Do not correct, complete, or guess illegible medicine names — write "[illegible]" instead.
- Do not add any medical advice, warnings, or interpretation of your own.
- Output language: write field labels and any transcribed advice in {'Bengali' if bangla else 'English'}, but keep medicine names in their original script as written on the prescription."""
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        if text == "NOT_A_PRESCRIPTION":
            return "ERROR: এই ছবিতে কোনো প্রেসক্রিপশন বা রিপোর্ট খুঁজে পাওয়া যায়নি।" if bangla else "ERROR: No prescription or lab report detected in this image."
        return text
    except Exception:
        logger.exception("Gemini OCR extraction failed")
        return "ERROR: Gemini OCR ব্যর্থ হয়েছে। আবার চেষ্টা করুন।"


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="MediAssist AI", page_icon="👩🏻‍⚕️", layout="wide")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY পাওয়া যায়নি।")
    st.stop()


# ─── Session State Init ───────────────────────────────────────────────

defaults = {
    "messages":                  [],
    "chat_history":              [],
    "ocr_confirmed_text":        None,
    "ocr_extracted_text":        None,
    "ui_language":                "বাংলা",
    "vision_extracted_symptoms": None,
    "vision_confirmed_symptoms": None,
    "vision_image_bytes":        None,
    "vision_image_name":         None,
    "rag_chain":                 None,
    "triage_active":             False,
    "triage_questions":          [],
    "triage_questions_backup":   [],
    "triage_original_input":     "",
    "triage_answers":            {},
    "triage_submit":             False,
    "triage_skip":               False,
    "ocr_image_bytes":           None,
    "ocr_image_name":            None,
    "audio_widget_warmed":       False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

is_bangla = st.session_state.ui_language == "বাংলা"


# ─── Static Data ──────────────────────────────────────────────────────
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
    "Madaripur", "Gopalganj", "Munshiganj", "Manikganj", "Rajbari",
]
specialist_types = [
    "হাসপাতাল (General Hospital)", "হৃদরোগ বিশেষজ্ঞ (Cardiologist)",
    "চর্মরোগ বিশেষজ্ঞ (Dermatologist)", "স্নায়ু বিশেষজ্ঞ (Neurologist)",
    "শিশু বিশেষজ্ঞ (Pediatrician)", "মানসিক স্বাস্থ্য (Psychiatrist)",
    "হাড় বিশেষজ্ঞ (Orthopedic)", "চক্ষু বিশেষজ্ঞ (Ophthalmologist)",
    "দন্ত চিকিৎসক (Dentist)", "ডায়াবেটিস বিশেষজ্ঞ (Diabetologist)",
    "কিডনি বিশেষজ্ঞ (Nephrologist)", "ক্যান্সার বিশেষজ্ঞ (Oncologist)",
    "গাইনি বিশেষজ্ঞ (Gynecologist)", "ইএনটি বিশেষজ্ঞ (ENT Specialist)",
]


# ─── SIDEBAR: collapsible sections instead of dialogs ─────────────────
st.sidebar.markdown("## 👩🏻‍⚕️ MediAssist AI")
st.sidebar.caption("আপনার ব্যক্তিগত স্বাস্থ্য সহকারী" if is_bangla else "Your personal health assistant")
st.sidebar.markdown("---")

# ── Settings ────────────────────────────────────────────────────────
with st.sidebar.expander("⚙️ সেটিংস" if is_bangla else "⚙️ Settings"):
    choice = st.radio(
        "Language / ভাষা:", ["বাংলা", "English"],
        index=0 if st.session_state.ui_language == "বাংলা" else 1,
        key="lang_radio",
    )
    # Reacts immediately on next natural rerun (radio change already reruns) —
    # no extra st.rerun() needed.
    if choice != st.session_state.ui_language:
        st.session_state.ui_language = choice
        is_bangla = choice == "বাংলা"

# ── BMI & Health Stats ──────────────────────────────────────────────
with st.sidebar.expander("📊 বিএমআই ও স্বাস্থ্য তথ্য" if is_bangla else "📊 BMI & Health Stats"):
    weight = st.number_input(
        "ওজন (kg)" if is_bangla else "Weight (kg)",
        min_value=10.0, max_value=200.0, value=70.0, step=0.5, key="bmi_weight",
    )
    height_cm = st.number_input(
        "উচ্চতা (cm)" if is_bangla else "Height (cm)",
        min_value=50.0, max_value=250.0, value=170.0, step=1.0, key="bmi_height",
    )
    if st.button("📊 হিসাব করুন" if is_bangla else "📊 Calculate", use_container_width=True, key="bmi_calc"):
        bmi = weight / ((height_cm / 100) ** 2)
        water_intake = (weight * 35) / 1000
        st.session_state["_bmi_result"] = (bmi, water_intake)

    result = st.session_state.get("_bmi_result")
    if result:
        bmi, water_intake = result
        c1, c2 = st.columns(2)
        c1.metric("BMI", f"{bmi:.1f}")
        c2.metric("💧 পানি/দিন" if is_bangla else "💧 Water/day", f"{water_intake:.1f} L")
        if bmi < 18.5:
            st.warning("📉 কম ওজন" if is_bangla else "📉 Underweight")
        elif bmi < 25:
            st.success("✅ স্বাভাবিক ওজন" if is_bangla else "✅ Normal Weight")
        elif bmi < 30:
            st.warning("📈 অতিরিক্ত ওজন" if is_bangla else "📈 Overweight")
        else:
            st.error("⚠️ স্থূলতা" if is_bangla else "⚠️ Obese")
        st.caption("💡 মেইন চ্যাটে ডায়েট প্ল্যান চাইতে পারেন!" if is_bangla
                else "💡 Ask for a diet plan in the main chat!")

# ── Visual Symptom Checker ──────────────────────────────────────────
with st.sidebar.expander("👁️ ভিজ্যুয়াল লক্ষণ পরীক্ষা" if is_bangla else "👁️ Visual Symptom Checker"):
    st.caption("Powered by Gemini Vision")
    v_uploaded = st.file_uploader(
        "ছবি আপলোড করুন (JPG/PNG)" if is_bangla else "Upload image (JPG/PNG)",
        type=["jpg", "jpeg", "png"], key="vision_uploader",
    )
    if v_uploaded is not None:
        raw = v_uploaded.getvalue()
        if raw != st.session_state.vision_image_bytes:
            st.session_state.vision_image_bytes = raw
            st.session_state.vision_image_name = v_uploaded.name
            st.session_state.vision_extracted_symptoms = None
            st.session_state.vision_confirmed_symptoms = None

    if st.session_state.vision_image_bytes:
        img = Image.open(BytesIO(st.session_state.vision_image_bytes))
        st.image(img, use_container_width=True)

        if st.session_state.vision_extracted_symptoms is None:
            if st.button("🔍 বিশ্লেষণ করুন" if is_bangla else "🔍 Analyze",
                        use_container_width=True, key="vision_analyze"):
                with st.spinner("বিশ্লেষণ করা হচ্ছে..."):
                    st.session_state.vision_extracted_symptoms = analyze_symptoms_from_image(img, is_bangla)

        col_vr, col_vx = st.columns(2)
        with col_vr:
            if st.session_state.vision_extracted_symptoms and st.button(
                "🔄 আবার" if is_bangla else "🔄 Re-analyze", use_container_width=True, key="vision_reanalyze"
            ):
                with st.spinner("বিশ্লেষণ করা হচ্ছে..."):
                    st.session_state.vision_extracted_symptoms = analyze_symptoms_from_image(img, is_bangla)
                st.session_state.vision_confirmed_symptoms = None
        with col_vx:
            if st.button("🗑️ মুছুন" if is_bangla else "🗑️ Clear", use_container_width=True, key="vision_clear"):
                st.session_state.vision_image_bytes = None
                st.session_state.vision_image_name = None
                st.session_state.vision_extracted_symptoms = None
                st.session_state.vision_confirmed_symptoms = None

        if st.session_state.vision_confirmed_symptoms:
            st.success("✅ চ্যাটে এই তথ্য ব্যবহার হবে" if is_bangla else "✅ Confirmed — used as chat context")
            st.markdown(st.session_state.vision_confirmed_symptoms)
            if st.button("✏️ আবার সম্পাদনা" if is_bangla else "✏️ Edit again",
                        use_container_width=True, key="vision_edit_again"):
                st.session_state.vision_confirmed_symptoms = None

        elif st.session_state.vision_extracted_symptoms:
            syms = st.session_state.vision_extracted_symptoms
            if syms.startswith("ERROR"):
                st.error(syms)
            else:
                edited_syms = st.text_area(
                    "✏️ প্রয়োজনে সম্পাদনা করুন:" if is_bangla else "✏️ Edit if needed:",
                    value=syms, height=150, key="vision_text_area",
                )
                if st.button("✅ নিশ্চিত করুন" if is_bangla else "✅ Confirm & use in chat",
                            use_container_width=True, key="vision_confirm"):
                    st.session_state.vision_confirmed_symptoms = edited_syms

# ── Prescription / Report OCR ───────────────────────────────────────
with st.sidebar.expander("📄 প্রেসক্রিপশন OCR" if is_bangla else "📄 Prescription OCR"):
    st.caption("Powered by Gemini Vision")
    o_uploaded = st.file_uploader(
        "প্রেসক্রিপশন / রিপোর্টের ছবি" if is_bangla else "Upload Prescription / Report",
        type=["jpg", "jpeg", "png"], key="ocr_uploader",
    )
    if o_uploaded is not None:
        raw = o_uploaded.getvalue()
        if raw != st.session_state.ocr_image_bytes:
            st.session_state.ocr_image_bytes = raw
            st.session_state.ocr_image_name = o_uploaded.name
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None

    if st.session_state.ocr_image_bytes:
        oimg = Image.open(BytesIO(st.session_state.ocr_image_bytes))
        st.image(oimg, caption=st.session_state.ocr_image_name, use_container_width=True)

        if st.session_state.ocr_extracted_text is None:
            if st.button("🔍 টেক্সট বের করুন" if is_bangla else "🔍 Extract Text",
                        use_container_width=True, key="ocr_extract"):
                with st.spinner("🔍 OCR হচ্ছে..." if is_bangla else "🔍 Extracting..."):
                    result = extract_text_gemini(oimg, is_bangla)
                st.session_state.ocr_extracted_text = result

        col_r, col_x = st.columns(2)
        with col_r:
            if st.session_state.ocr_extracted_text and st.button(
                "🔄 আবার" if is_bangla else "🔄 Re-extract", use_container_width=True, key="ocr_reextract"
            ):
                with st.spinner("🔍 Re-extracting..."):
                    st.session_state.ocr_extracted_text = extract_text_gemini(oimg, is_bangla)
                st.session_state.ocr_confirmed_text = None
        with col_x:
            if st.button("🗑️ মুছুন" if is_bangla else "🗑️ Clear", use_container_width=True, key="ocr_clear"):
                st.session_state.ocr_image_bytes = None
                st.session_state.ocr_image_name = None
                st.session_state.ocr_extracted_text = None
                st.session_state.ocr_confirmed_text = None

        if st.session_state.ocr_confirmed_text:
            st.success("✅ চ্যাটে এই তথ্য ব্যবহার হবে" if is_bangla else "✅ Confirmed — used as chat context")
            st.markdown(st.session_state.ocr_confirmed_text)
            if st.button("✏️ আবার সম্পাদনা" if is_bangla else "✏️ Edit again",
                        use_container_width=True, key="ocr_edit_again"):
                st.session_state.ocr_confirmed_text = None

        elif st.session_state.ocr_extracted_text:
            if st.session_state.ocr_extracted_text.startswith("ERROR"):
                st.error(st.session_state.ocr_extracted_text)
            else:
                edited = st.text_area(
                    "✏️ প্রয়োজনে সম্পাদনা করুন:" if is_bangla else "✏️ Edit if needed:",
                    value=st.session_state.ocr_extracted_text, height=200, key="ocr_text_area",
                )
                if st.button("✅ নিশ্চিত করুন" if is_bangla else "✅ Confirm & use in chat",
                            use_container_width=True, key="ocr_confirm"):
                    st.session_state.ocr_confirmed_text = edited

# ── Hospital Finder ─────────────────────────────────────────────────
with st.sidebar.expander("🏥 হাসপাতাল ফাইন্ডার" if is_bangla else "🏥 Hospital Finder"):
    district = st.selectbox(
        "জেলা বেছে নিন:" if is_bangla else "Select district:", bangladesh_districts, key="hosp_district",
    )
    specialist = st.selectbox(
        "কী ধরনের সাহায্য?" if is_bangla else "Type of help?", specialist_types, key="hosp_specialist",
    )
    if st.button("🔍 খুঁজুন" if is_bangla else "🔍 Search", use_container_width=True, key="hosp_search"):
        spec_en = specialist.split("(")[-1].replace(")", "").strip()
        maps_url = f"https://www.google.com/maps/search/{spec_en}+hospital+near+{district}+Bangladesh".replace(" ", "+")
        st.success(f"**{district}** — {specialist}")
        st.markdown(f"[🗺️ Google Maps এ দেখুন →]({maps_url})")
        st.caption("💡 Rating ও Reviews দেখে বেছে নিন।" if is_bangla
                else "💡 Check ratings/reviews before choosing.")

st.sidebar.markdown("---")

# ── Emergency Helplines ─────────────────────────────────────────────
with st.sidebar.expander("🚨 জরুরি হেল্পলাইন" if is_bangla else "🚨 Emergency Helplines"):
    st.markdown("**`999`** — " + ("পুলিশ/ফায়ার/অ্যাম্বুলেন্স · 24/7" if is_bangla else "Police/Fire/Ambulance · 24/7"))
    st.markdown("**`16263`** — " + ("DGHS স্বাস্থ্য হেল্পলাইন" if is_bangla else "DGHS Health Helpline"))
    st.markdown("**`333`** — " + ("জাতীয় স্বাস্থ্য সেবা" if is_bangla else "National Health Service"))
    st.markdown("**`16000`** — " + ("জাতীয় হেল্পলাইন" if is_bangla else "National Helpline"))
    st.markdown("**`09611667777`** — " + ("বিষ নিয়ন্ত্রণ কেন্দ্র" if is_bangla else "Poison Control"))
    st.markdown("**`10655`** — " + ("মানসিক স্বাস্থ্য" if is_bangla else "Mental Health"))
    st.markdown("**`109`** / **`10921`** — " + ("নারী হেল্পলাইন" if is_bangla else "Women's Helpline"))
    st.markdown("**`1098`** — " + ("শিশু হেল্পলাইন" if is_bangla else "Child Helpline"))
    st.markdown("**`16430`** — " + ("আইনি সহায়তা" if is_bangla else "Legal Aid"))
    st.caption("⚠️ জরুরি অবস্থায় সরাসরি **999** dial করুন।" if is_bangla
               else "⚠️ In an emergency, dial **999** directly.")

st.sidebar.markdown("🆘 **জরুরি অবস্থায় `999` ডায়াল করুন**" if is_bangla
                    else "🆘 **In an emergency, dial `999`**")


# ─── Main Area ──────────────────────────────────────────────────────
st.title("👩🏻‍⚕️ MediAssist AI")
st.caption(
    "আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন... " if is_bangla
    else "Ask any question related to your health..."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── RAG Pipeline ────────────────────────────────────────────────────
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    chroma_path = "./chroma_db"
    if not os.path.exists(chroma_path):
        chroma_path = "/app/chroma_db"
    vector_db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.3)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Given the chat history and the latest user question, reformulate the question "
        "to be standalone and clear. Reply in the SAME language as the user's input. "
        "Do NOT answer it, just rephrase if needed. Return as it is if already clear."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(x):
        if x["chat_history"]:
            q = (contextualize_prompt | llm | StrOutputParser()).invoke(
                {"input": x["input"], "chat_history": x["chat_history"]})
        else:
            q = x["input"]
        return format_docs(retriever.invoke(q))

    
    system_prompt = (
    "### LANGUAGE RULE — ###\n"
    "Detect the language from the user message and follow STRICTLY:\n"
    "- Bengali script → reply ENTIRELY in Bengali. Only medicine names may stay in English.\n"
    "- Banglish (Bengali in English letters) → reply ENTIRELY in Bengali script.\n"
    "- English → reply ENTIRELY in English.\n"
    "NEVER mix languages. This overrides all other rules.\n\n"
    "### IDENTITY ###\n"
    "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant.\n\n"
    "### INPUT STRUCTURE ###\n"
    "The user's message may include, in addition to their question, these optional blocks:\n"
    "- [Patient answers]: triage answers about onset, severity, duration, associated symptoms\n"
    "- [OCR Prescription]: text transcribed from a prescription or lab report\n"
    "- [Visual Symptoms]: a description of a symptom visible in an uploaded photo\n"
    "Treat these as patient-reported facts. Combine them with the retrieved Context below "
    "to give a more specific, grounded answer — do not ignore them, and do not ask the user "
    "to repeat information already present in these blocks.\n\n"
    "### REASONING ORDER ###\n"
    "1. Check if [Patient answers] / [OCR Prescription] / [Visual Symptoms] are present — use them as the patient's specific situation.\n"
    "2. Check the retrieved Context for relevant general medical information.\n"
    "3. If both are present, connect them explicitly (e.g. relate the patient's reported symptom/duration to what the context says about it).\n"
    "4. Only ask the user a clarifying question if the missing information is something NOT already covered by triage answers, OCR, or visual symptoms, AND is necessary to give a safe, non-generic answer.\n"
    "5. If context lacks relevant info and you are unsure, tell the user to seek help from a doctor or provide more information — do not guess.\n\n"
    "### CORE RULES ###\n"
    "- ONLY use medical facts from the provided context. Never invent medicine names, dosages, or facts.\n"
    "- NEVER provide a specific diagnosis — describe possibilities and general guidance, not a verdict.\n"
    "- Avoid generic follow-up questions like 'what is the reason' or 'can you tell me more' — "
    "if you need more information, ask ONE specific, clinically relevant question "
    "(e.g. duration, severity 1-10, associated symptoms, what makes it better/worse).\n\n"
    "### FORMATTING ###\n"
    "- Use Markdown for the output. But the font size should not be different.\n\n"
    "### EMERGENCY ###\n"
    "- Chest pain + sweating / breathing difficulty / severe bleeding / unconsciousness / stroke "
    "→ '🚨 EMERGENCY: Call 999 or go to the nearest hospital immediately!'\n\n"
    "### DISCLAIMER ###\n"
    "End every reply with disclaimer in user's language:\n"
    "Bengali: '⚠️ সতর্কতা: আমি একটি এআই মডেল। যেকোনো স্বাস্থ্য সমস্যায় রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।'\n"
    "English: '⚠️ Disclaimer: I am an AI. Please consult a registered doctor for any health concern.'\n\n"
    "note that, here you can give specific kind of doctor, for example if it is a skin problem specify darmatologist "
    "Context:\n{context}"
)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    return (
        RunnablePassthrough.assign(context=RunnableLambda(get_context))
        | qa_prompt | llm | StrOutputParser()
    )


# ─── Voice Config ─────────────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "বাংলা":   {"voice_prompt": "এটি একটি চিকিৎসা সংক্রান্ত কথোপকথন।",
                "spinner_msg": "ভয়েস প্রসেস করা হচ্ছে...",
                "error_msg":   "ভয়েস প্রসেসিং ব্যর্থ হয়েছে", "whisper_lang": "bn"},
    "English": {"voice_prompt": "This is a medical conversation.",
                "spinner_msg":  "Processing voice...",
                "error_msg":    "Voice processing failed",      "whisper_lang": "en"},
}


# ─── Helpers ─────────────────────────────────────────────────────────
def _run_rag(enriched_input: str) -> str:
    if st.session_state.rag_chain is None:
        with st.spinner("লোড হচ্ছে... ⏳"):
            st.session_state.rag_chain = load_rag_pipeline()
    try:
        return st.session_state.rag_chain.invoke({
            "input": enriched_input,
            "chat_history": st.session_state.chat_history,
        })
    except Exception as e:
        return f"❌ সমস্যা: {str(e)}"


def _append_context(base: str) -> str:
    if st.session_state.get("ocr_confirmed_text"):
        base += f"\n\n[OCR Prescription]:\n{st.session_state.ocr_confirmed_text}"
    v = st.session_state.get("vision_confirmed_symptoms")
    if v and not v.startswith("ERROR"):
        base += f"\n\n[Visual Symptoms]:\n{v}"
    return base

def _save_to_history(raw: str, enriched: str, answer: str):
    st.session_state.chat_history.extend([
        HumanMessage(content=enriched),
        AIMessage(content=answer),
    ])
    st.session_state.messages.append({"role": "assistant", "content": answer})


def is_symptom_query(text: str) -> bool:
    kws = [
        "ব্যথা","বেথা","জ্বর","কাশি","শ্বাস","বমি","মাথা","বুক","পেট","গলা","চোখ","কান","নাক",
        "হাত","পা","দুর্বল","ক্লান্ত","ঘুম","খিদে","র্যাশ","চুলকানি","ফোলা","রক্ত","ডায়রিয়া",
        "betha","jor","jore","bugti","kashi","shash","bomi","matha","buk","pet","gola",
        "chokh","kan","nak","durbolta","ghum","khide","rash","cholkani","fola","shordi","khansi",
        "mathay","pate","buke",
        "pain","fever","cough","headache","nausea","vomit","dizzy","tired","weak","swelling",
        "bleeding","diarrhea","chest","stomach","throat","eye","ear","breathing","itching",
        "burning","ache","sore","hurt","suffering","feeling",
    ]
    return any(k in text.lower() for k in kws)


def generate_triage_questions(symptom_text: str) -> list:
    try:
        prompt = f"""Patient said: "{symptom_text}"

You are a clinical triage assistant generating structured follow-up questions — NOT generic open-ended ones like "what is the reason" or "tell me more."

Generate exactly 3-5 MCQ questions, each with 4 options, prioritizing the most clinically useful categories that fit this symptom:
1. Duration/onset (e.g. "How long have you had this?")
2. Severity (e.g. mild/moderate/severe or a 1-10 style scale as options)
3. Associated symptoms (what else is present alongside the main complaint)
4. Aggravating/relieving factors (what makes it better or worse)
5. Red flag check (something that would indicate urgency, if relevant to this symptom)

Each question must be answerable by someone with no medical background, and each option must be a concrete, specific choice — not vague ranges like "a little" or "a lot."

STRICT RULE:
- Bengali script or Banglish input → questions and options in Bengali script ONLY.
- English input → English ONLY.
- Never mix languages.

Return ONLY valid JSON, no markdown, no preamble.
Format: {{"questions":[{{"id":1,"question":"...","options":["A","B","C","D"]}}]}}"""

        raw = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1000,
        ).choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip()).get("questions", [])
    except Exception:
        return []


# ─── Triage: dialog-based (modal) ───────────

@st.dialog("🩺 লক্ষণ মূল্যায়ন" if is_bangla else "🩺 Symptoms Evaluation", width="medium")
def triage_dialog():
    st.caption("সব প্রশ্নের উত্তর দিন তারপর submit করুন" if is_bangla
            else "Please answer all questions and then submit")
    questions = st.session_state.triage_questions
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            with st.container(border=True):
                st.caption(f"প্রশ্ন {i+1}" if is_bangla else f"Question {i+1}")
                st.markdown(f"**{q['question']}**")
                val = st.radio(q["question"], q["options"],
                            key=f"dq_{q['id']}", index=None, label_visibility="hidden")
                if val:
                    st.session_state.triage_answers[str(q["id"])] = val

    answered = len(st.session_state.triage_answers)
    total = len(questions)
    st.progress(answered / total if total else 0)
    st.caption(f"✅ {answered} / {total} answered")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.triage_skip = True
            st.session_state.triage_active = False
            st.session_state.triage_questions = []
            st.rerun()
    with c2:
        if st.button("✅ Submit", type="primary", use_container_width=True, disabled=answered < total):
            st.session_state.triage_submit = True
            st.session_state.triage_active = False
            st.session_state.triage_questions = []
            st.rerun()


if st.session_state.triage_active and st.session_state.triage_questions:
    with st.chat_message("assistant"):
        st.markdown(
            "🩺 **আপনার সমস্যা আরো ভালোভাবে বুঝতে কিছু প্রশ্ন করছি:**" if is_bangla
            else "🩺 **To better understand your condition, please answer:**"
        )
    triage_dialog()

if st.session_state.get("triage_submit") or st.session_state.get("triage_skip"):
    submitted = st.session_state.pop("triage_submit", False)
    st.session_state.pop("triage_skip", False)

    enriched = st.session_state.triage_original_input
    if submitted and st.session_state.triage_answers:
        enriched += "\n\n[Patient answers]:\n"
        for q in (st.session_state.get("triage_questions_backup") or []):
            ans = st.session_state.triage_answers.get(str(q["id"]), "N/A")
            enriched += f"- {q['question']}: {ans}\n"

    enriched = _append_context(enriched)
    original = st.session_state.triage_original_input

    st.session_state.triage_answers = {}
    st.session_state.triage_original_input = ""
    st.session_state.triage_questions_backup = []

    with st.chat_message("assistant"):
        with st.spinner("বিশ্লেষণ করা হচ্ছে... 🔍" if is_bangla else "Analyzing... 🔍"):
            answer = _run_rag(enriched)
        st.markdown(answer)
    _save_to_history(original, enriched, answer)
    st.rerun()

else:
    # ─── Mobile mic warm-up ─────────────────────────────────────────

    if not st.session_state.audio_widget_warmed:
        _warmup = st.empty()
        with _warmup.container():
            st.chat_input("warmup", accept_audio=True, key="__audio_warmup__")
        _warmup.empty()
        st.session_state.audio_widget_warmed = True
        st.rerun()

    placeholder = (
        "আপনার লক্ষণ বা সমস্যার কথা লিখুন অথবা মাইক্রোফোন ব্যবহার করুন..."
        if is_bangla else
        "Type your symptoms or question, or use the microphone..."
    )
    prompt = st.chat_input(placeholder, accept_audio=True)

    if prompt:
        user_input = ""
        lang_cfg = LANGUAGE_CONFIG[st.session_state.ui_language]

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
        elif prompt.text:
            user_input = prompt.text

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            combined = _append_context(user_input)

            if is_symptom_query(user_input):
                with st.spinner("🩺 লক্ষণ বিশ্লেষণ করা হচ্ছে..." if is_bangla else "🩺 Analyzing symptoms..."):
                    questions = generate_triage_questions(user_input)
                if questions:
                    st.session_state.triage_active = True
                    st.session_state.triage_questions = questions
                    st.session_state.triage_questions_backup = questions
                    st.session_state.triage_original_input = user_input
                    st.session_state.triage_answers = {}
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.rerun()
                else:
                    with st.chat_message("assistant"):
                        with st.spinner("তথ্য খোঁজা হচ্ছে..." if is_bangla else "Searching..."):
                            answer = _run_rag(combined)
                        st.markdown(answer)
                    _save_to_history(user_input, combined, answer)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("তথ্য খোঁজা হচ্ছে..." if is_bangla else "Searching..."):
                        answer = _run_rag(combined)
                    st.markdown(answer)
                _save_to_history(user_input, combined, answer)