# 👩🏻‍⚕️ MediAssist AI

**MediAssist AI** is a bilingual (Bengali/English) medical information assistant built with Streamlit, combining Retrieval-Augmented Generation (RAG), vision-based symptom analysis, prescription OCR, and structured clinical triage to help users in Bangladesh get safe, grounded, and context-aware health guidance.

> ⚠️ **Disclaimer:** MediAssist AI is an informational tool, not a substitute for professional medical advice. It does not diagnose conditions. Users are always directed to consult a registered doctor and, in emergencies, to dial **999**.

---

## ✨ Features

- **Bilingual Chat Interface** — Full support for Bengali (বাংলা) and English, with automatic language detection and Banglish (Bengali typed in English letters) handling.
- **RAG-Powered Medical Q&A** — Answers are grounded in a curated knowledge base via a Chroma vector store, retrieved with multilingual sentence embeddings and synthesized using a Groq-hosted LLaMA 3.3 70B model.
- **Structured Symptom Triage** — Automatically detects symptom-related queries and generates targeted multiple-choice follow-up questions (onset, severity, associated symptoms, aggravating factors, red flags) instead of generic prompts.
- **Visual Symptom Checker** — Upload a photo of a visible symptom (rash, swelling, wound, etc.) for a purely descriptive, non-diagnostic visual analysis powered by Gemini Vision.
- **Prescription & Lab Report OCR** — Transcribes medicines, dosages, instructions, and metadata from photographed prescriptions or lab reports, with built-in safeguards against hallucinated or "completed" text.
- **Voice Input** — Record symptoms via microphone; transcribed using Groq Whisper (large-v3-turbo) in the selected language.
- **BMI & Hydration Calculator** — Quick health stats with category feedback (underweight, normal, overweight, obese) and daily water intake estimate.
- **Hospital & Specialist Finder** — Locate hospitals and specialists by district across all 64 districts of Bangladesh, with direct Google Maps search links.
- **Emergency Helpline Directory** — One-click access to national emergency, health, poison control, mental health, women's, and child helplines.
- **Built-in Safety Guardrails** — Automatic emergency detection (chest pain, breathing difficulty, severe bleeding, stroke symptoms) with an immediate "call 999" warning, plus a mandatory disclaimer on every response.

---

## 🏗️ Architecture

```
User Input (text / voice / image)
        │
        ├── Symptom Image ──► Gemini Vision ──► Visual Symptom Description
        ├── Prescription Image ──► Gemini Vision ──► Structured OCR Transcript
        └── Text / Voice Query
                │
                ├── Symptom Detected? ──► Groq LLaMA 3.3 ──► Triage MCQ Questions
                │                                                   │
                │                                          User Answers (modal)
                │                                                   │
                └────────────────► Context Enrichment ◄─────────────┘
                                         │
                            Contextualize Question (chat history aware)
                                         │
                              Chroma Vector Store Retrieval
                                         │
                              Groq LLaMA 3.3 70B (RAG Chain)
                                         │
                              Grounded, Disclaimer-Appended Answer
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI / App Framework | [Streamlit](https://streamlit.io/) |
| LLM (chat / triage) | Groq — `llama-3.3-70b-versatile` |
| Vision (image analysis & OCR) | Google Gemini — `gemini-2.5-flash` |
| Speech-to-Text | Groq Whisper — `whisper-large-v3-turbo` |
| Orchestration | LangChain (`langchain-groq`, `langchain-community`, `langchain-core`) |
| Embeddings | HuggingFace `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Store | Chroma |
| Image Handling | Pillow (PIL) |

---

## 📋 Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)
- A [Google Gemini API key](https://aistudio.google.com/)
- A pre-built Chroma vector store (`./chroma_db` or `/app/chroma_db`) populated with your medical knowledge base

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mediassist-ai
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Set up the vector database**

   Ensure a populated Chroma persistence directory exists at `./chroma_db` (or `/app/chroma_db` for containerized deployments) containing your embedded medical reference documents.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
pillow
groq
python-dotenv
google-generativeai
langchain
langchain-groq
langchain-huggingface
langchain-community
chromadb
sentence-transformers
```

*(Pin exact versions in `requirements.txt` for reproducible deployments.)*

---

## 🗂️ Project Structure

```
mediassist-ai/
├── app.py              # Main Streamlit application
├── chroma_db/          # Persisted vector store (medical knowledge base)
├── .env                # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 🔐 Safety & Design Principles

- **No diagnosis, only guidance** — The system prompt explicitly forbids definitive diagnoses; it offers possibilities and general, context-grounded information.
- **Grounded responses** — All medical facts are constrained to the retrieved context; the model is instructed never to invent medicines, dosages, or claims.
- **Verification before extraction** — Both the vision symptom checker and prescription OCR first verify the image is relevant before processing, rejecting unrelated images gracefully.
- **Emergency-first detection** — Critical symptom combinations trigger an immediate emergency call-to-action ahead of any other response.
- **Consistent disclaimers** — Every AI-generated answer ends with a clear disclaimer in the user's selected language, directing them to a registered doctor.
- **Specific, non-generic triage** — Follow-up questions are structured MCQs targeting duration, severity, associated symptoms, and red flags rather than vague prompts.

---

## 🌍 Localization

The interface and AI responses support full bilingual operation:

- **বাংলা (Bengali)** — native script, including detection of Banglish (Bengali typed using English letters).
- **English** — full feature parity.

Language is switched live from the sidebar **Settings** panel and propagates to the chat model, triage questions, voice transcription, and all UI labels.

---

## 🚑 Emergency Resources (Bangladesh)

| Number | Service |
|---|---|
| 999 | Police / Fire / Ambulance (24/7) |
| 16263 | DGHS Health Helpline |
| 333 | National Health Service |
| 16000 | National Helpline |
| 09611667777 | Poison Control |
| 10655 | Mental Health |
| 109 / 10921 | Women's Helpline |
| 1098 | Child Helpline |
| 16430 | Legal Aid |

---

## 📄 License

Specify your project's license here (e.g., MIT, Apache 2.0).

---

## 🙏 Acknowledgements

- [Groq](https://groq.com/) for high-speed LLM and Whisper inference
- [Google Gemini](https://ai.google.dev/) for vision-language capabilities
- [LangChain](https://www.langchain.com/) for RAG orchestration
- [Streamlit](https://streamlit.io/) for the application framework