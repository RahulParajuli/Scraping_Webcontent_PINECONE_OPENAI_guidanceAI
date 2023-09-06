from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_APIKEY = os.getenv("OPENAI_API_KEY")
PINECONE_APIKEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

DATA_PATH = "./data/"