import streamlit as st
import os
import PyPDF2
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

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

# Function to read PDF
def read_pdf(uploadedfile):
    pdf_text = ""
    pdfReader = PyPDF2.PdfReader(uploadedfile)
    for page_num in range(len(pdfReader.pages)):
        page = pdfReader.pages[page_num]
        pdf_text += page.extract_text()
    return pdf_text

# Cache the embeddings generation
@st.cache_data
def generate_embeddings(texts):
    response = client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME_ADA_SLN"],
        input=texts
    )
    return response.data[0].embedding

# Create or get ChromaDB collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="all-my-documents")

def split_text_into_chunks(text, max_chunk_size=512):
    # Split text into sentences
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

def add_pdf_to_chromadb(uploadedfile):
    # Read PDF contents
    pdf_text = read_pdf(uploadedfile)

    # Remove newline characters and extra spaces
    clean_text = pdf_text.replace('\n', ' ').strip()

    if clean_text:  # Ensure the text is not empty
        # Split PDF text into chunks
        texts = split_text_into_chunks(clean_text)
        metadatas = [{"chunk_index": i} for i in range(len(texts))]
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeds = [generate_embeddings(text) for text in texts]

        # Add documents with embeddings to ChromaDB
        collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeds,
        )
    else:
        st.sidebar.error("Uploaded PDF is empty or could not be read.")

@st.cache_data
def fetch_all_documents():
    all_docs = collection.get(include=["documents"])
    documents = [doc for sublist in all_docs['documents'] for doc in sublist]
    return " ".join(documents)

def is_summarization(query):
    # Define keywords that suggest the user is asking for a summary
    summarize_keywords = ['summarize', 'summary', 'summarise', 'overview', 'about the document', 'details of the document']
    return any(keyword in query.lower() for keyword in summarize_keywords)

def chat_with_ai(query):
    # Fetch all documents context
    context = fetch_all_documents()

    # Add document context to message if summarization is detected or if a general inquiry about the document is identified
    if is_summarization(query):
        messages = [
            {"role": "system", "content": "You are an AI assistant. Provide information based on the content provided from the document."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Document content:\n\n{context}"},
        ]
    else:
        user_embedding = generate_embeddings(query)

        # Search for the most similar document in ChromaDB
        results = collection.query(
            query_embeddings=[user_embedding],
            n_results=1,
            include=["documents", "distances"]
        )

        if results and 'documents' in results and 'distances' in results and \
                len(results['documents']) > 0 and results['documents'][0] and \
                len(results['distances']) > 0:

            context = results['documents'][0][0]  # Extract the first document
            messages = [
                {"role": "system", "content": "You are an AI assistant. Answer strictly based on the content provided. Understand the query properly and if the query is out of the context of the provided document, respond with 'I don't know.'"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"Relevant context from document:\n\n{context}"},
            ]
        else:
            return "I don't know."

    try:
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
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.set_page_config(page_title="Azure OpenAI Chat", layout="wide")
    st.title("Azure OpenAI Chat with Document Context")

    st.sidebar.header("Upload PDF Document")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        add_pdf_to_chromadb(uploaded_file)
        st.sidebar.write("Document added to database.")

    st.header("Chat with AI")
    query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if query:
            response = chat_with_ai(query)
            st.write("Response from AI:")
            st.write(response)
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()