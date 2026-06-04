# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma


# pdf_path = "First_Aid.pdf"
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )
# chunks = text_splitter.split_documents(documents)



# print("মাল্টিলিঙ্গুয়াল এমবেডিং মডেল ডাউনলোড হচ্ছে (প্রথমবার একটু সময় লাগবে)...")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )


# persist_directory = "./chroma_db"
# print("ভেক্টর ডেটাবেস তৈরি হচ্ছে...")
# vector_db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory=persist_directory
# )

# print("✅ দারুণ! ডেটাবেস সফলভাবে তৈরি হয়েছে এবং 'chroma_db' ফোল্ডারে সেভ হয়েছে!")





# import os
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# print("ডেটা লোড করা হচ্ছে... একটু অপেক্ষা করো...")

# # ১. পিডিএফ ডেটা লোড করা
# pdf_loader = PyPDFLoader("First_Aid.pdf")
# pdf_docs = pdf_loader.load()

# # ২. ওয়েবসাইটের লিংক থেকে সরাসরি ডেটা পড়া (এখানে তুমি যেকোনো মেডিকেল আর্টিকেলের লিংক দিতে পারো)
# medical_url = "https://www.who.int/news-room/fact-sheets/detail/asthma" # এটি WHO-এর অ্যাজমা বিষয়ক একটি পেজ
# print(f"ওয়েবসাইট থেকে তথ্য সংগ্রহ করা হচ্ছে: {medical_url}")
# web_loader = WebBaseLoader(medical_url)
# web_docs = web_loader.load()

# # ৩. পিডিএফ এবং ওয়েবসাইটের ডেটা একসাথে যুক্ত করা
# all_documents = pdf_docs + web_docs

# # ৪. টেক্সটকে ছোট অংশে ভাগ করা
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )
# chunks = text_splitter.split_documents(all_documents)
# print(f"সব মিলিয়ে মোট {len(chunks)} টি খণ্ড তৈরি হয়েছে!")

# # ৫. এমবেডিং মডেল
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# # ৬. ডেটাবেস আপডেট করা
# persist_directory = "./chroma_db"
# print("ডেটাবেসে আপডেট হচ্ছে...")
# vector_db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory=persist_directory
# )

# print("✅ দারুণ! ওয়েবসাইটের রিয়েল-টাইম তথ্য ডেটাবেসে যুক্ত হয়েছে!")



import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("ডেটা লোড করা হচ্ছে... একটু অপেক্ষা করো...")

# ১. ফোল্ডার থেকে একসাথে সব পিডিএফ লোড করা (Magic of DirectoryLoader!)
pdf_folder_path = "./medical_books" # তোমার প্রজেক্টের ভেতরের ফোল্ডারের নাম
print(f"'{pdf_folder_path}' ফোল্ডার থেকে সব পিডিএফ চেক করা হচ্ছে...")

# PyPDFDirectoryLoader ব্যবহার করে ফোল্ডারের সব পিডিএফ পড়া
pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
all_documents = pdf_loader.load()

print(f"✅ ফোল্ডার থেকে সব পিডিএফ লোড হয়েছে! মোট পেজ: {len(all_documents)}")

# ২. টেক্সটকে ছোট অংশে ভাগ করা (Text Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(all_documents)
print(f"সব মিলিয়ে মোট {len(chunks)} টি খণ্ড তৈরি হয়েছে!")

# ৩. এমবেডিং ও ভেক্টর স্টোর (Vector Store) তৈরি
print("ডেটাবেস আপডেট হচ্ছে...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

persist_directory = "./chroma_db"
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

print("🎉 দারুণ! ফোল্ডারের সব পিডিএফ সফলভাবে ডেটাবেসে যুক্ত হয়েছে!")