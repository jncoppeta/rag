import os
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
import json
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

os.environ["OPENAI_API_KEY"] = ""
openai_client = OpenAI()

milvus_client = MilvusClient(uri="./milvus_test.db")

collection_name = "test_collection"

create_new = True

if create_new:
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=1536,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )

def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

def query(context, question):
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

def add_data_to_collection(lines):
    try:
        data = []

        for i, line in enumerate(tqdm(lines, desc="Creating embeddings")):
            data.append({"id": i, "vector": emb_text(line), "text": line})

        milvus_client.insert(collection_name=collection_name, data=data)
        return "Success"
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
def search_collection(question):
    try:
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[
                emb_text(question)
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=3,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        # Parse top 3 responses into an array
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]

        # Convert top responses into a string
        context = "\n".join(
            [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
        )

        # Return the string result
        return context
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def add_data():
    # Get data in desirable format
    data = []
    def get_text(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        sentences = sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 5]

    # Loop through the 'pdfs' folder
    pdf_folder = "pdfs"  # Specify the folder containing PDFs
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):  # Process only PDF files
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {filename}...")
            pdf_text = get_text(pdf_path)
            data.extend(pdf_text)  # Add extracted sentences to the data list
    add_data_to_collection(data)

# Parse PDFs and add to collection
add_data()

# Prepare question
question = "What are the primary findings of the study regarding sex differences in recovery after resistance training?"
context = search_collection(question)

# Query model for response
response = query(context, question)

print(response)
