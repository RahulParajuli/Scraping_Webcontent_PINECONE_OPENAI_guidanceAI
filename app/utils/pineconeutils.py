import os
from tqdm.auto import tqdm
from uuid import uuid4
import openai
import pinecone
import tiktoken
tiktoken.encoding_for_model("gpt-3.5-turbo")
import pandas as pd
from openai.embeddings_utils import get_embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from settings import PINECONE_APIKEY, PINECONE_ENV, OPENAI_APIKEY

pinecone.init(api_key= PINECONE_APIKEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_APIKEY


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def upsert(data, index):
    """
    Upsert data to Pinecone Index
    input: data, index
    output: None
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=10,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    data = data.to_dict('records')


    model_name = 'text-embeddings-ada-002'
    embed = OpenAIEmbeddings(model_name=model_name, openai_api_key=OPENAI_APIKEY)

    index = pinecone.Index(index)
    batch_size = 1500
    texts = []
    metadatas = []
    for i, record in enumerate(tqdm(data)):
        metadata = {
            "id": str(record["id"]),
            "Course": record["Course"],
            "HTML": record["HTML"],
        }
        record_texts = text_splitter.split_text(record["text_data"])
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)
        ]
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        if len(texts) >= batch_size:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

def dataloader():
    """
    input: None
    output: dataframe
    """
    data = pd.read_excel("D:/projects/upwork/active/James L langchain model/application/app/utils/data/data.xlsx")
    data["text_data"] = "Course: " + data["Course"]  + "; HTML:" + data["HTML"]
    data["id"] = [int(i) for i in range (len(data))]
    return data

def check_pinecone_index():
    """
    input: None
    output: list of indexes
    """
    indexes = None
    try:
        indexes = pinecone.list_indexes()
        print(indexes if indexes else 'No Indexes Found')
    except Exception as e:
        print('Error: {}'.format(e))
    return indexes

def create_index(index_name=None, dimension=None, metric=None):
    """
    input: index_name, dimension, metric
    output: Pinecone Index
    """
    try:
        index = None
        index = pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
        print('Index Created: {}'.format(index))
        index = pinecone.Index(index_name)
    except Exception as e:
        print('Error: {}'.format(e))
    return index


def delete_index(index_name=None):
    """
    Delete Pinecone Index
    input: index_name
    output: None
    """
    try:
        pinecone.delete_index(name=index_name)
        print('Index Deleted: {}'.format(index_name))
    except Exception as e:
        print('Error: {}'.format(e))

if __name__ == "__main__":
    while True:
        choice = input("""enter a number to select an option:
        1. Create Index and upload data
        2. Delete Index
        3. Check Index
        4. Upsert Data
        5. Exit\n
        Type your choice: 
        """
                       )
        if choice == "1":
            index_name = input("Enter Index Name: ")
            if index_name in check_pinecone_index():
                print("Index Already Exists")
                continue
            data = dataloader()
            print("---Data Loaded---")
            print("---Creating Index---")
            create_index(index_name=index_name, dimension=1536, metric="cosine")
            print("---Uploading Data---")
            upsert(data=data, index=index_name)
            print("---Data Uploaded Successfully---")
        elif choice == "2":
            check_pinecone_index()
            index_name = input("Enter Index Name: ")
            delete_index(index_name=index_name)
        elif choice == "3":
            check_pinecone_index()
        elif choice == "4":
            data = dataloader()
            upsert(data=data, index="udemyrecords")
        elif choice == "5":
            break
        else:
            print("Invalid Choice")
