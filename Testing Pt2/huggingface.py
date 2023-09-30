import os
from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains.question_answering import load_qa_chain
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
import requests

# Download Text File
urls = [
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models', 
    'https://lmsys.org/blog/2023-03-30-vicuna/'
    'https://www.wizklubfuturz.com'
]
# Document Loader
from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()
import textwrap
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=200)
docs = text_splitter.split_documents(data)

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

query = "How big is stableLM?"
docs = db.similarity_search(query)
print(wrap_text_preserve_newlines(str(docs[0].page_content)))

query = "What is Vicuna?"
docs = db.similarity_search(query)
print(wrap_text_preserve_newlines(str(docs[0].page_content)))