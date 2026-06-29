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
from dotenv import load_dotenv
import logging


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

# FIX 2: Single cached client — no re-creation on every rerun
@st.cache_resource
def get_groq_client():
    return Groq(api_key=groq_api_key) if groq_api_key else None

groq_client = get_groq_client()


# ─── Gemini Vision Functions ──────────────────────────────────────────
def analyze_symptoms_from_image(image):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""You are an expert clinical observer. Look at this medical image and accurately
        describe the visible symptoms in 2-3 short sentences.
        DO NOT provide a diagnosis or medical advice.
        Just describe what you see physically. Reply in {'Bengali' if is_bangla else 'English'}."""
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        logger.exception("Gemini vision analysis failed")
        return "ERROR: ছবি প্রসেস করতে সমস্যা হয়েছে। আবার চেষ্টা করুন।"


def extract_text_gemini(image):
    if not gemini_api_key:
        return "ERROR: Gemini API key পাওয়া যায়নি।"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = """You are an expert pharmacist. Read this medical prescription or lab report carefully.
        Extract: medicine names, dosages, duration, and any medical advice.
        Format the extracted text neatly. Do not give medical advice."""
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        logger.exception("Gemini OCR extraction failed")
        return "ERROR: Gemini OCR ব্যর্থ হয়েছে। আবার চেষ্টা করুন।"


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="MediAssist AI", page_icon="👩🏻‍⚕️", layout="wide")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY পাওয়া যায়নি।")
    st.stop()


# ─── Session State Init ───────────────────────────────────────────────
defaults = {
    "messages":                 [],
    "chat_history":             [],
    "ocr_confirmed_text":       None,
    "ocr_extracted_text":       None,
    "ocr_edited_text":          None,
    "ui_language":              "বাংলা",
    "last_uploaded_file":       None,
    "vision_extracted_symptoms": None,
    "rag_chain":                None,
    "triage_active":            False,
    "triage_questions":         [],
    "triage_questions_backup":  [],   # FIX 1: added to defaults
    "triage_original_input":    "",
    "triage_answers":           {},
    "triage_submit":            False,
    "triage_skip":              False,
    # dialog open flags
    "open_settings":            False,
    "open_bmi":                 False,
    "open_vision":              False,
    "open_ocr":                 False,
    "open_hospital":            False,
    "open_emergency":           False,
    # OCR image bytes — persists across reruns unlike file_uploader
    "ocr_image_bytes":          None,
    "ocr_image_name":           None,
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


# ─── DIALOGS ──────────────────────────────────────────────────────────

@st.dialog("⚙️ Settings", width="small")
def dialog_settings():
    choice = st.radio(
        "Language / ভাষা:", ["বাংলা", "English"],
        index=0 if st.session_state.ui_language == "বাংলা" else 1
    )
    if st.button("✅ Save & Close", type="primary", use_container_width=True):
        st.session_state.ui_language = choice
        st.session_state.open_settings = False
        st.rerun()


@st.dialog("📊 বিএমআই ও স্বাস্থ্য তথ্য" if is_bangla else "📊 BMI & Health Stats", width="small")
def dialog_bmi():
    st.caption("আপনার স্বাস্থ্য তথ্য দিন" if is_bangla else "Enter your health info")
    weight    = st.number_input("ওজন (kg)" if is_bangla else "Weight (kg)",
    min_value=10.0, max_value=200.0, value=70.0,  step=0.5)
    height_cm = st.number_input("উচ্চতা (cm)" if is_bangla else "Height (cm)",
    min_value=50.0, max_value=250.0, value=170.0, step=1.0)
    if st.button("📊 হিসাব করুন" if is_bangla else "📊 Calculate", type="primary", use_container_width=True):
        bmi          = weight / ((height_cm / 100) ** 2)
        water_intake = (weight * 35) / 1000
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("BMI", f"{bmi:.1f}")
        c2.metric("💧 পানি/দিন" if is_bangla else "💧 Water/day", f"{water_intake:.1f} L")
        if bmi < 18.5:   st.warning("📉 কম ওজন" if is_bangla else "📉 Underweight")
        elif bmi < 25:   st.success("✅ স্বাভাবিক ওজন" if is_bangla else "✅ Normal Weight")
        elif bmi < 30:   st.warning("📈 অতিরিক্ত ওজন" if is_bangla else "📈 Overweight")
        else:            st.error("⚠️ স্থূলতা" if is_bangla else "⚠️ Obese")
        st.info("💡 মেইন চ্যাটে ডায়েট প্ল্যান চাইতে পারেন!" if is_bangla
                else "💡 Ask for a diet plan in the main chat!")


@st.dialog("👁️ ভিজ্যুয়াল লক্ষণ পরীক্ষা" if is_bangla else "👁️ Visual Symptom Checker", width="medium")
def dialog_vision():
    st.caption("লক্ষণ বুঝতে ছবি আপলোড করুন — Powered by Gemini Vision")
    uploaded = st.file_uploader(
        "ছবি আপলোড করুন (JPG/PNG)" if is_bangla else "Upload image (JPG/PNG)",
        type=["jpg", "jpeg", "png"], key="vision_uploader_dialog"
    )
    if not uploaded:
        return

    img = Image.open(uploaded)
    c_img, c_res = st.columns([1, 1], gap="medium")

    with c_img:
        st.image(img, use_container_width=True)
        analyze = st.button("🔍 বিশ্লেষণ করুন" if is_bangla else "🔍 Analyze",
                            type="primary", use_container_width=True)
        if analyze:
            with st.spinner("বিশ্লেষণ করা হচ্ছে..."):
                result = analyze_symptoms_from_image(img)
            st.session_state.vision_extracted_symptoms = result

    with c_res:
        syms = st.session_state.vision_extracted_symptoms
        if syms:
            if syms.startswith("ERROR"):
                st.error(syms)
            else:
                st.success("✅ বিশ্লেষণ সম্পন্ন!" if is_bangla else "✅ Analysis complete!")
                heading = "**প্রাথমিক লক্ষণ:**" if is_bangla else "**Preliminary symptoms:**"
                st.markdown(f"{heading}\n\n{syms}")
        else:
            st.markdown("*ফলাফল এখানে দেখাবে*" if is_bangla else "*Results will appear here*")


@st.dialog("📋 Prescription / Report OCR", width="medium")
def dialog_ocr():
    st.caption("🔍 Gemini Vision দ্বারা পরিচালিত" if is_bangla else "🔍 Powered by Gemini Vision")

    if st.session_state.ocr_image_bytes is None:
        uploaded = st.file_uploader(
            "প্রেসক্রিপশন / রিপোর্টের ছবি" if is_bangla else "Upload Prescription / Report",
            type=["jpg", "jpeg", "png"], key="ocr_uploader_dialog"
        )
        if uploaded:
            st.session_state.ocr_image_bytes = uploaded.getvalue()
            st.session_state.ocr_image_name  = uploaded.name
            st.session_state.ocr_extracted_text = None
            st.session_state.ocr_confirmed_text = None
            st.session_state.ocr_edited_text    = None
            st.rerun()
        else:
            return

    from io import BytesIO
    img = Image.open(BytesIO(st.session_state.ocr_image_bytes))

    c_img, c_text = st.columns([1, 1], gap="medium")

    with c_img:
        st.image(img, caption=st.session_state.ocr_image_name, use_container_width=True)

        if st.session_state.ocr_extracted_text is None:
            if st.button("🔍 টেক্সট বের করুন" if is_bangla else "🔍 Extract Text",
                type="primary", use_container_width=True):
                with st.spinner("🔍 OCR হচ্ছে..." if is_bangla else "🔍 Extracting..."):
                    result = extract_text_gemini(img)
                st.session_state.ocr_extracted_text = result
                st.session_state.ocr_edited_text    = result

        col_r, col_x = st.columns(2)
        with col_r:
            reextract_label = "🔄 পুনরায় বের করুন" if is_bangla else "🔄 Re-extract"
            if st.session_state.ocr_extracted_text and st.button(reextract_label, use_container_width=True):
                with st.spinner("🔍 Re-extracting..."):
                    result = extract_text_gemini(img)
                st.session_state.ocr_extracted_text = result
                st.session_state.ocr_edited_text    = result
                st.session_state.ocr_confirmed_text = None
        with col_x:
            if st.button("🗑️ ছবি মুছুন" if is_bangla else "🗑️ Clear Image", use_container_width=True):
                st.session_state.ocr_image_bytes    = None
                st.session_state.ocr_image_name     = None
                st.session_state.ocr_extracted_text = None
                st.session_state.ocr_confirmed_text = None
                st.session_state.ocr_edited_text    = None
                st.session_state.last_uploaded_file = None
                st.rerun()

    with c_text:
        if st.session_state.ocr_confirmed_text:
            st.success("✅ নিশ্চিত করা হয়েছে — চ্যাটে এই তথ্য ব্যবহার হবে" if is_bangla
                    else "✅ Confirmed — this will be used as context in chat")
            st.markdown(st.session_state.ocr_confirmed_text)
            if st.button("✏️ আবার সম্পাদনা করুন" if is_bangla else "✏️ Edit Again", use_container_width=True):
                st.session_state.ocr_confirmed_text = None

        elif st.session_state.ocr_extracted_text:
            if st.session_state.ocr_extracted_text.startswith("ERROR"):
                st.error(st.session_state.ocr_extracted_text)
            else:
                st.markdown("**✏️ প্রয়োজনে সম্পাদনা করুন:**" if is_bangla else "**✏️ Edit if needed:**")
                edited = st.text_area(
                    "ocr_edit_area",
                    value=st.session_state.ocr_extracted_text,
                    height=250,
                    label_visibility="collapsed",
                    key="ocr_text_area_live"
                )
                confirm_label = "✅ নিশ্চিত করুন ও চ্যাটে ব্যবহার করুন" if is_bangla else "✅ Confirm & Use in Chat"
                if st.button(confirm_label, type="primary", use_container_width=True):
                    st.session_state.ocr_confirmed_text = edited
                    st.session_state.ocr_edited_text    = edited
        else:
            st.markdown(
                "*ছবি থেকে text extract হলে এখানে দেখাবে*" if is_bangla
                else "*Extracted text will appear here after clicking Extract*"
            )


@st.dialog("🏥 হাসপাতাল খুঁজুন" if is_bangla else "🏥 Hospital Finder", width="medium")
def dialog_hospital():
    st.caption("আপনার কাছের হাসপাতাল বা বিশেষজ্ঞ খুঁজুন" if is_bangla
            else "Find hospitals or specialists near you")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        district = st.selectbox(
            "জেলা বেছে নিন:" if is_bangla else "Select district:", bangladesh_districts)
    with c2:
        specialist = st.selectbox(
            "কী ধরনের সাহায্য?" if is_bangla else "Type of help?", specialist_types)

    if st.button("🔍 খুঁজুন" if is_bangla else "🔍 Search", type="primary", use_container_width=True):
        spec_en  = specialist.split("(")[-1].replace(")", "").strip()
        maps_url = f"https://www.google.com/maps/search/{spec_en}+hospital+near+{district}+Bangladesh".replace(" ", "+")
        st.success(f"**{district}** — {specialist}")
        st.markdown(f"### 🗺️ [Google Maps এ দেখুন →]({maps_url})")
        st.divider()
        st.info("💡 Rating ও Reviews দেখে বেছে নিন। যাওয়ার আগে appointment নিন।" if is_bangla
                else "💡 Check ratings and reviews before choosing, and book an appointment ahead.")
        st.warning("⚠️ MediAssist AI কোনো নির্দিষ্ট হাসপাতালকে recommend করে না।" if is_bangla
                else "⚠️ MediAssist AI does not recommend any specific hospital.")


@st.dialog("🚨 জরুরি হেল্পলাইন" if is_bangla else "🚨 Emergency Helplines", width="medium")
def dialog_emergency():
    st.markdown("### 🇧🇩 বাংলাদেশ জরুরি হেল্পলাইন" if is_bangla else "### 🇧🇩 Bangladesh Emergency Helplines")

    st.markdown("##### 🚑 প্রধান জরুরি সেবা" if is_bangla else "##### 🚑 Primary Emergency Services")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown("**`999`**")
        st.caption("পুলিশ / ফায়ার / অ্যাম্বুলেন্স · 24/7" if is_bangla else "Police / Fire / Ambulance · 24/7")
    with c2:
        st.markdown("**`16263`**")
        st.caption("DGHS স্বাস্থ্য হেল্পলাইন" if is_bangla else "DGHS Health Helpline")
    with c3:
        st.markdown("**`333`**")
        st.caption("জাতীয় স্বাস্থ্য সেবা" if is_bangla else "National Health Service")

    st.divider()

    st.markdown("##### 🏥 স্বাস্থ্য সেবা" if is_bangla else "##### 🏥 Health Services")
    c4, c5, c6 = st.columns(3, gap="small")
    with c4:
        st.markdown("**`16000`**")
        st.caption("জাতীয় হেল্পলাইন" if is_bangla else "National Helpline")
    with c5:
        st.markdown("**`09611667777`**")
        st.caption("বিষ নিয়ন্ত্রণ কেন্দ্র" if is_bangla else "Poison Control Center")
    with c6:
        st.markdown("**`10655`**")
        st.caption("মানসিক স্বাস্থ্য" if is_bangla else "Mental Health")

    st.divider()

    st.markdown("##### 👩‍👧 নারী ও শিশু সেবা" if is_bangla else "##### 👩‍👧 Women & Child Services")
    c7, c8, c9, c10 = st.columns(4, gap="small")
    with c7:
        st.markdown("**`109`**")
        st.caption("নারী হেল্পলাইন" if is_bangla else "Women's Helpline")
    with c8:
        st.markdown("**`10921`**")
        st.caption("নারী সহায়তা" if is_bangla else "Women's Support")
    with c9:
        st.markdown("**`1098`**")
        st.caption("শিশু হেল্পলাইন" if is_bangla else "Child Helpline")
    with c10:
        st.markdown("**`16430`**")
        st.caption("আইনি সহায়তা" if is_bangla else "Legal Aid")

    st.divider()
    st.caption("⚠️ জরুরি অবস্থায় সরাসরি **999** dial করুন।" if is_bangla
               else "⚠️ In an emergency, dial **999** directly.")


# ─── SIDEBAR (wrapped in fragment for instant dialog opening) ──────────
# FIX 3: @st.fragment means clicking a sidebar button only reruns this
# block, not the entire page (chat history, RAG pipeline, etc.)
# ─── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👩🏻‍⚕️ MediAssist AI")
    st.caption("আপনার ব্যক্তিগত স্বাস্থ্য সহকারী" if is_bangla else "Your personal health assistant")
    st.markdown("---")

    st.markdown("**🛠️ টুলস**" if is_bangla else "**🛠️ Tools**")
    
    # Launching dialogs immediately on click prevents fragment delays
    if st.button("⚙️ সেটিংস" if is_bangla else "⚙️ Settings", use_container_width=True):
        dialog_settings()
        
    if st.button("📊 বিএমআই ও স্বাস্থ্য ড্যাশবোর্ড" if is_bangla else "📊 BMI & Health Dashboard", use_container_width=True):
        dialog_bmi()
        
    if st.button("👁️ ভিজ্যুয়াল লক্ষণ পরীক্ষা" if is_bangla else "👁️ Visual Symptom Checker", use_container_width=True):
        dialog_vision()
        
    if st.button("📄 প্রেসক্রিপশন OCR" if is_bangla else "📄 Prescription OCR", use_container_width=True):
        dialog_ocr()
        
    if st.button("🏥 হাসপাতাল ফাইন্ডার" if is_bangla else "🏥 Hospital Finder", use_container_width=True):
        dialog_hospital()

    st.markdown("---")
    if st.button("🚨 জরুরি হেল্পলাইন" if is_bangla else "🚨 Emergency Helplines", use_container_width=True, type="primary"):
        dialog_emergency()

    st.markdown("🆘 **জরুরি অবস্থায় `999` ডায়াল করুন**" if is_bangla else "🆘 **In an emergency, dial `999`**")

    # ─── Open dialogs ─────────────────────────────────────────────────
    if st.session_state.open_settings:
        st.session_state.open_settings = False
        dialog_settings()

    if st.session_state.open_bmi:
        st.session_state.open_bmi = False
        dialog_bmi()

    if st.session_state.open_vision:
        st.session_state.open_vision = False
        dialog_vision()

    if st.session_state.open_ocr:
        st.session_state.open_ocr = False
        dialog_ocr()

    if st.session_state.open_hospital:
        st.session_state.open_hospital = False
        dialog_hospital()

    if st.session_state.open_emergency:
        st.session_state.open_emergency = False
        dialog_emergency()




# ─── Main Area ────────────────────────────────────────────────────────
st.title("👩🏻‍⚕️ MediAssist AI")
st.caption("আপনার স্বাস্থ্য বিষয়ক যেকোনো প্রশ্ন করুন" if is_bangla
        else "Ask any question related to your health, for better try adding prescription, photo of symptom from sidebar")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── RAG Pipeline ─────────────────────────────────────────────────────
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
        "### LANGUAGE RULE — HIGHEST PRIORITY ###\n"
        "Detect the language from the user message and follow STRICTLY:\n"
        "- Bengali script → reply ENTIRELY in Bengali. Only medicine names may stay in English.\n"
        "- Banglish (Bengali in English letters) → reply ENTIRELY in Bengali script.\n"
        "- English → reply ENTIRELY in English.\n"
        "NEVER mix languages. This overrides all other rules.\n\n"

        "### IDENTITY ###\n"
        "You are MediAssist AI, a knowledgeable and compassionate Medical Information Assistant.\n\n"

        "### CORE RULES ###\n"
        "- ONLY use information from the provided context.\n"
        "- If context lacks info and you are unsure, then tell him to seek help from doctor.\n"
        "- NEVER invent medical information, drug names, or dosages.\n"
        "- NEVER provide a specific diagnosis.\n\n"
        "- If you do not know anything for the context then no need to invent any info from llm "

        "### FORMATTING ###\n"
        "-Use Markdown for the output. But the fontsize should not be different.\n\n"

        "### EMERGENCY ###\n"
        "- Chest pain + sweating / breathing difficulty / severe bleeding / unconsciousness / stroke "
        "→ '🚨 EMERGENCY: Call 999 or go to the nearest hospital immediately!'\n\n"

        "### DISCLAIMER ###\n"
        "End every reply with disclaimer in user's language:\n"
        "Bengali: '⚠️ সতর্কতা: আমি একটি এআই মডেল। যেকোনো স্বাস্থ্য সমস্যায় রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।'\n"
        "English: '⚠️ Disclaimer: I am an AI. Please consult a registered doctor for any health concern.'\n\n"
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


# ─── Voice Config ──────────────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "বাংলা":   {"voice_prompt": "এটি একটি চিকিৎসা সংক্রান্ত কথোপকথন।",
                 "spinner_msg": "ভয়েস প্রসেস করা হচ্ছে...",
                 "error_msg":   "ভয়েস প্রসেসিং ব্যর্থ হয়েছে", "whisper_lang": "bn"},
    "English": {"voice_prompt": "This is a medical conversation.",
                "spinner_msg":  "Processing voice...",
                "error_msg":    "Voice processing failed",      "whisper_lang": "en"},
}


# ─── Helpers ──────────────────────────────────────────────────────────
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
    v = st.session_state.get("vision_extracted_symptoms")
    if v and not v.startswith("ERROR"):
        base += f"\n\n[Visual Symptoms]:\n{v}"
    return base


def _save_to_history(raw: str, enriched: str, answer: str):
    st.session_state.chat_history.extend([
        HumanMessage(content=enriched),
        AIMessage(content=answer),
    ])
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── Symptom Detection ────────────────────────────────────────────────
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


# FIX 2: Removed duplicate `client = Groq(...)` and reuses cached groq_client
def generate_triage_questions(symptom_text: str) -> list:
    try:
        import json
        prompt = f"""Patient said: "{symptom_text}"

STRICT RULE:
- Bengali script or Banglish input → questions and options in Bengali script ONLY.
- English input → English ONLY.
- Never mix languages.

Generate 3-5 MCQ questions (exactly 4 options each) to understand the condition.
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


# ─── Triage Dialog ────────────────────────────────────────────────────
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
    total    = len(questions)
    st.progress(answered / total if total else 0)
    st.caption(f"✅ {answered} / {total} answered")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.triage_skip      = True
            st.session_state.triage_active    = False
            st.session_state.triage_questions = []
            st.rerun()
    with c2:
        if st.button("✅ Submit", type="primary", use_container_width=True, disabled=answered < total):
            st.session_state.triage_submit    = True
            st.session_state.triage_active    = False
            st.session_state.triage_questions = []
            st.rerun()


# ─── Triage UI ────────────────────────────────────────────────────────
if st.session_state.triage_active and st.session_state.triage_questions:
    with st.chat_message("assistant"):
        st.markdown("🩺 **আপনার সমস্যা আরো ভালোভাবে বুঝতে কিছু প্রশ্ন করছি:**"
                    if is_bangla else "🩺 **To better understand your condition, please answer:**")
    triage_dialog()

if st.session_state.get("triage_submit") or st.session_state.get("triage_skip"):
    submitted = st.session_state.pop("triage_submit", False)
    st.session_state.pop("triage_skip", False)

    enriched = st.session_state.triage_original_input
    if submitted and st.session_state.triage_answers:
        enriched += "\n\n[Patient answers]:\n"
        # FIX 1: triage_questions_backup is now properly set, so this loop actually runs
        for q in (st.session_state.get("triage_questions_backup") or []):
            ans = st.session_state.triage_answers.get(str(q["id"]), "N/A")
            enriched += f"- {q['question']}: {ans}\n"

    enriched = _append_context(enriched)
    original = st.session_state.triage_original_input

    st.session_state.triage_answers          = {}
    st.session_state.triage_original_input   = ""
    st.session_state.triage_questions_backup = []

    with st.chat_message("assistant"):
        with st.spinner("বিশ্লেষণ করা হচ্ছে... 🔍" if is_bangla else "Analyzing... 🔍"):
            answer = _run_rag(enriched)
        st.markdown(answer)
    _save_to_history(original, enriched, answer)
    st.rerun()

else:
    # ─── Normal Chat Input ────────────────────────────────────────────
    placeholder = (
        "আপনার লক্ষণ বা সমস্যার কথা লিখুন অথবা মাইক্রোফোন ব্যবহার করুন..."
        if is_bangla else
        "Type your symptoms or question, or use the microphone..."
    )
    prompt = st.chat_input(placeholder, accept_audio=True, key="main_chat_input")

    if prompt:
        user_input = ""
        lang_cfg   = LANGUAGE_CONFIG[st.session_state.ui_language]

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
                with st.spinner("🩺 লক্ষণ বিশ্লেষণ করা হচ্ছে..." if is_bangla
                                else "🩺 Analyzing symptoms..."):
                    questions = generate_triage_questions(user_input)
                if questions:
                    st.session_state.triage_active           = True
                    st.session_state.triage_questions        = questions
                    st.session_state.triage_questions_backup = questions   # FIX 1: save backup
                    st.session_state.triage_original_input   = user_input
                    st.session_state.triage_answers          = {}
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