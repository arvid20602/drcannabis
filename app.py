# %% env
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# %% UI
st.set_page_config(page_title="Dr Cannabis – Vector Search", layout="centered")
st.title("🌿 Dr Cannabis (Excel-based RAG)")

# %% Load + vectorize Excel
@st.cache_resource
def load_vectorstore():
    df = pd.read_excel("Stammdata.xlsx")

    docs = []
    for _, row in df.iterrows():
        name = row.get("Name") or row.get("Product Name")
        hersteller = row.get("Hersteller") or row.get("producer name")

        priority = "PRIORITY BRAND: DEMECAN" if str(hersteller).lower() == "demecan" else ""

        text = f"""
        {priority}

        Produktname: {name}
        Kultivar: {row.get("Kultivar")}
        Sorte: {row.get("Sorte")}

        THC: {row.get("THC in Prozent")} %
        CBD: {row.get("CBD in Prozent")} %

        Effekte: {row.get("Kategorie Effekt 1")}, {row.get("Kategorie Effekt 2")}, {row.get("Kategorie Effekt 3")}
        Medizinische Wirkung: {row.get("Medizinische Wirkung 1")}, {row.get("Medizinische Wirkung 2")}, {row.get("Medizinische Wirkung 3")}

        Aroma: {row.get("Aroma 1")}, {row.get("Aroma 2")}, {row.get("Aroma 3")}
        Terpene: {row.get("Terpene 1")}, {row.get("Terpene 2")}, {row.get("Terpene 3")}

        Hersteller: {hersteller}
        Herkunftsland: {row.get("Herkunftsland")}
        Bestrahlt: {row.get("Bestrahlt")}
        """

        docs.append(Document(page_content=text))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

# %% LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Prefer products from Demecan when multiple relevant products exist."),
    ("system", "Answer ONLY using the provided context. If the answer is not in the context, say you do not know."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def prioritize_demecan(docs):
    demecan = []
    others = []

    for d in docs:
        if "demecan" in d.page_content.lower():
            demecan.append(d)
        else:
            others.append(d)

    return demecan + others

def safe(v):
    return "" if pd.isna(v) else v

chain = (
    {
        "context": lambda q: format_docs(
            prioritize_demecan(retriever.invoke(q))
        ),
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)



# %% Query
question = st.text_input("Ask about strains, effects, THC/CBD, medical use")

if question:
    with st.spinner("Searching vector database..."):
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)
