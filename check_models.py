import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env থেকে API Key লোড করা
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

print("তোমার API Key-এর জন্য যে মডেলগুলো সাপোর্ট করে:\n" + "="*50)

# যেসব মডেল ছবি/টেক্সট জেনারেট করতে পারে, শুধু তাদের লিস্ট করা
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")