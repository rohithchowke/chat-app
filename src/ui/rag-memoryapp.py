# import streamlit as st
# import os
# import PyPDF2
# import chromadb
# from pathlib import Path
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from sklearn.metrics.pairwise import cosine_similarity
# import uuid

# # Load ENV
# ENV_DIR = Path().absolute() / ".env"
# DEV_ENV_FILE_PATH = ENV_DIR / "dev.env"
# load_dotenv(DEV_ENV_FILE_PATH, override=True)

# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_SLN"],
#     api_key=os.environ["AZURE_OPENAI_API_KEY_SLN"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
# )

# # Function to read PDF
# def read_pdf(uploadedfile):
#     pdf_text = ""
#     pdfReader = PyPDF2.PdfReader(uploadedfile)
#     for page_num in range(len(pdfReader.pages)):
#         page = pdfReader.pages[page_num]
#         pdf_text += page.extract_text()
#     return pdf_text

# # Cache the embeddings generation
# @st.cache_resource
# def generate_embeddings(texts):
#     response = client.embeddings.create(
#         model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME_ADA_SLN"],
#         input=[texts]
#     )
#     return response.data[0].embedding

# # Initialize ChromaDB client and collection
# chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection(name="all-my-document")

# # Split text into smaller chunks for embeddings
# def split_text_into_chunks(text, max_chunk_size=512):
#     sentences = text.split('. ')
#     chunks = []
#     current_chunk = ""
#     current_length = 0

#     for sentence in sentences:
#         sentence_length = len(sentence.split())
#         if current_length + sentence_length <= max_chunk_size:
#             current_chunk += (sentence + '. ')
#             current_length += sentence_length
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + '. '
#             current_length = sentence_length

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks

# # Cache all document data
# @st.cache_resource
# def cache_documents(uploaded_file, department):
#     collection_data = st.session_state.get("collection_data", {})
#     document_content = read_pdf(uploaded_file)
#     if document_content:
#         clean_text = document_content.replace('\n', ' ').strip()
#         texts = split_text_into_chunks(clean_text)
#         metadatas = [{"chunk_index": i, "department": department} for i in range(len(texts))]
#         document_uuid = str(uuid.uuid4())
#         ids = [f"doc_{i}_{document_uuid}" for i in range(len(texts))]
#         embeds = [generate_embeddings(text) for text in texts]

#         collection_data.update({
#             document_uuid: {
#                 "texts": texts,
#                 "metadatas": metadatas,
#                 "ids": ids,
#                 "embeddings": embeds,
#             }
#         })

#         st.session_state["collection_data"] = collection_data
#         st.sidebar.write("Documents cached successfully.")

# def add_cached_data_to_chromadb():
#     collection_data = st.session_state.get("collection_data", {})
#     for cached_document in collection_data.values():
#         collection.upsert(
#             documents=cached_document["texts"],
#             metadatas=cached_document["metadatas"],
#             ids=cached_document["ids"],
#             embeddings=cached_document["embeddings"],
#         )

# @st.cache_data
# def fetch_documents_by_department(department):
#     all_docs = collection.get(include=["documents", "metadatas"])
#     filtered_docs = [" ".join(doc) for i, doc in enumerate(all_docs['documents']) if all_docs['metadatas'][i]['department'] == department]
#     return " ".join(filtered_docs) if filtered_docs else ""

# def is_summarization(query):
#     summarize_keywords = ['summarize', 'summary', 'summarise', 'overview', 'about the document', 'details of the document']
#     return any(keyword in query.lower() for keyword in summarize_keywords)

# def chat_with_ai(query, department):
#     context = fetch_documents_by_department(department)
#     user_embedding = generate_embeddings(query)
#     results = collection.query(
#         query_embeddings=[user_embedding],
#         n_results=3,  # Fetch top 3 results
#         include=["documents", "distances"]
#     )

#     if results and 'documents' in results and 'distances' in results:
#         # Flatten results and calculate cosine similarity
#         context_documents = [doc for sublist in results['documents'] for doc in sublist]
#         distances = [dist for sublist in results['distances'] for dist in sublist]
#         context = context_documents[0]  # Use the top result initially

#         # If additional context from other documents is beneficial, include them based on distance
#         for i in range(1, len(context_documents)):
#             similarity_score = cosine_similarity([user_embedding], [generate_embeddings(context_documents[i])])[0][0]
#             if similarity_score > 0.7:  # Adjusting the threshold
#                 context += " " + context_documents[i]

#         messages = [
#             {"role": "system", "content": "You are an AI assistant. Answer strictly based on the content provided. Understand the query properly and answer accordingly."},
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": f"Relevant context from document:\n\n{context}"},
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": "You are an AI assistant. Answer strictly based on the known content. If you do not know the answer for sure, indicate your uncertainty politely rather than stating 'I don't know.'."},
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": "No relevant context was found directly from the documents. However, I'm here to help. Please clarify your question or provide more details."},
#         ]

#     try:
#         response = client.chat.completions.create(
#             model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_SLN"],
#             messages=messages,
#             temperature=0.7,
#             max_tokens=1000,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#         )
#         response_content = response.choices[0].message.content
#         return response_content
#     except Exception as e:
#         return f"An error occurred: {e}"

# def main():
#     st.set_page_config(page_title="Azure OpenAI Chat", layout="wide")
#     st.title("Azure OpenAI Chat with Document Context")

#     st.sidebar.header("Upload PDF Document")
#     department = st.sidebar.text_input("Enter the department")

#     if st.sidebar.button("Confirm Department"):
#         st.session_state["department"] = department
#         st.sidebar.success(f"Department set to: {department}")

#     uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

#     if uploaded_file is not None:
#         if "department" in st.session_state and st.session_state["department"]:
#             cache_documents(uploaded_file, st.session_state["department"])
#             add_cached_data_to_chromadb()
#         else:
#             st.sidebar.error("Please confirm the department before uploading the file.")

#     st.header("Chat with AI")
#     query = st.text_input("Enter your query:")

#     if st.button("Submit"):
#         if query:
#             if "department" in st.session_state and st.session_state["department"]:
#                 response = chat_with_ai(query, st.session_state["department"])
#                 st.write("Response from AI:")
#                 st.write(response)
#             else:
#                 st.error("Please confirm the department before submitting a query.")
#         else:
#             st.error("Please enter a query.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
import PyPDF2
import chromadb
from dotenv import load_dotenv
from pathlib import Path
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Load ENV
ENV_DIR = Path().absolute() / ".env"
DEV_ENV_FILE_PATH = ENV_DIR / "dev.env"
load_dotenv(DEV_ENV_FILE_PATH, override=True)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_SLN"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_SLN"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

def read_pdf(uploadedfile):
    pdf_text = ""
    pdfReader = PyPDF2.PdfReader(uploadedfile)
    for page_num in range(len(pdfReader.pages)):
        page = pdfReader.pages[page_num]
        pdf_text += page.extract_text()
    return pdf_text

# Cache the embeddings generation
@st.cache_resource
def generate_embeddings(texts):
    response = client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME_ADA_SLN"],
        input=[texts]
    )
    return response.data[0].embedding

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="all-my-document")

def split_text_into_chunks(text, max_chunk_size=512):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_size:
            current_chunk += (sentence + '. ')
            current_length += sentence_length
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def cache_documents(uploaded_file, department):
    document_content = read_pdf(uploaded_file)
    clean_text = document_content.replace('\n', ' ').strip()
    texts = split_text_into_chunks(clean_text)
    metadatas = [{"chunk_index": i, "department": department} for i in range(len(texts))]
    document_uuid = str(uuid.uuid4())
    ids = [f"doc_{i}_{document_uuid}" for i in range(len(texts))]
    embeds = [generate_embeddings(text) for text in texts]

    collection_data = {
        document_uuid: {
            "texts": texts,
            "metadatas": metadatas,
            "ids": ids,
            "embeddings": embeds,
        }
    }
    return collection_data

def add_cached_data_to_chromadb(cached_data):
    for cached_document in cached_data.values():
        collection.upsert(
            documents=cached_document["texts"],
            metadatas=cached_document["metadatas"],
            ids=cached_document["ids"],
            embeddings=cached_document["embeddings"],
        )
    st.sidebar.write("Documents cached successfully.")

@st.cache_data
def fetch_documents_by_department(department):
    all_docs = collection.get(include=["documents", "metadatas"])
    filtered_docs = [" ".join(doc) for i, doc in enumerate(all_docs['documents']) if all_docs['metadatas'][i]['department'] == department]
    return " ".join(filtered_docs) if filtered_docs else ""

def chat_with_ai(query, department):
    context = fetch_documents_by_department(department)
    user_embedding = generate_embeddings(query)
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    if results and 'documents' in results and 'distances' in results:
        context_documents = [doc for sublist in results['documents'] for doc in sublist]
        distances = [dist for sublist in results['distances'] for dist in sublist]
        context = context_documents[0]

        for i in range(1, len(context_documents)):
            similarity_score = cosine_similarity([user_embedding], [generate_embeddings(context_documents[i])])[0][0]
            if similarity_score > 0.7:
                context += " " + context_documents[i]

        messages = [
            {"role": "system", "content": "You are an AI assistant. Answer strictly based on the content provided. Understand the query properly and answer accordingly."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Relevant context from document:\n\n{context}"},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are an AI assistant. Answer strictly based on the known content. If you do not know the answer for sure, indicate your uncertainty politely rather than stating 'I don't know.'."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "No relevant context was found directly from the documents. However, I'm here to help. Please clarify your question or provide more details."},
        ]

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_SLN"],
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_content = response.choices[0].message.content
    return response_content

def parseDepartmentFromFilename(file):
    try:
        return file.name.split(".")[0]
    except:
        return None

def main():
    st.set_page_config(page_title="Azure OpenAI Chat", layout="wide")
    st.title("Azure OpenAI Chat with Document Context")

    st.sidebar.header("Upload PDF Documents")

    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    for file in uploaded_files:
        department = parseDepartmentFromFilename(file)
        if department:
            cached_data = cache_documents(file, department)
            add_cached_data_to_chromadb(cached_data)
        else:
            st.sidebar.error('Unable to fetch department from filename. Please follow the file naming convention.')

    st.header("Chat with AI")
    query = st.text_input("Enter your query:")
    department = st.text_input("Enter the department for query:")

    if st.button("Submit"):
        if query and department:
            response = chat_with_ai(query, department)
            st.write("Response from AI:")
            st.write(response)
        else:
            st.error("Please enter a query and department.")

if __name__ == "__main__":
    main()