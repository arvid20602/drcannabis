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
st.set_page_config(page_title="Dr Cannabis", layout="centered")
st.title("🍃 Dr Cannabis")


# %% Helpers
def safe(v):
    return "" if pd.isna(v) else str(v)


# %% Load + vectorize Excel (TWO STORES)
@st.cache_resource
def load_vectorstores():
    df = pd.read_excel("Stammdata.xlsx")

    demecan_docs = []
    all_docs = []

    for _, row in df.iterrows():
        name = safe(row.get("Name")) or safe(row.get("Product Name"))

        raw_hersteller = safe(row.get("Hersteller")) or safe(row.get("producer name"))
        hersteller_norm = raw_hersteller.lower().strip()

        text = f"""
        Produktname: {name}
        Kultivar: {safe(row.get("Kultivar"))}
        Sorte: {safe(row.get("Sorte"))}

        THC: {safe(row.get("THC in Prozent"))} %
        CBD: {safe(row.get("CBD in Prozent"))} %

        Effekte: {safe(row.get("Kategorie Effekt 1"))}, {safe(row.get("Kategorie Effekt 2"))}, {safe(row.get("Kategorie Effekt 3"))}
        Medizinische Wirkung: {safe(row.get("Medizinische Wirkung 1"))}, {safe(row.get("Medizinische Wirkung 2"))}, {safe(row.get("Medizinische Wirkung 3"))}

        Aroma: {safe(row.get("Aroma 1"))}, {safe(row.get("Aroma 2"))}, {safe(row.get("Aroma 3"))}
        Terpene: {safe(row.get("Terpene 1"))}, {safe(row.get("Terpene 2"))}, {safe(row.get("Terpene 3"))}

        Hersteller: {raw_hersteller}
        Herkunftsland: {safe(row.get("Herkunftsland"))}
        Bestrahlt: {safe(row.get("Bestrahlt"))}
        """

        doc = Document(page_content=text)
        all_docs.append(doc)

        if "demecan" in hersteller_norm:
            demecan_docs.append(doc)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    demecan_store = FAISS.from_documents(demecan_docs, embeddings)
    general_store = FAISS.from_documents(all_docs, embeddings)

    return demecan_store, general_store


demecan_store, general_store = load_vectorstores()


# %% LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# %% Retrieval logic (DETERMINISTIC)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_context(question: str) -> str:
    demecan_hits = demecan_store.similarity_search(question, k=2)

    if demecan_hits:
        return format_docs(demecan_hits)

    return format_docs(general_store.similarity_search(question, k=4))


# %% Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Prefer Demecan products when they are relevant."),
    ("system", "Answer ONLY using the provided context. If the answer is not in the context, say you do not know."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# %% Chain
chain = (
    {
        "context": lambda q: get_context(q),
        "question": lambda q: q
    }
    | prompt
    | llm
    | StrOutputParser()
)


# %% UI Query
question = st.text_input("Ask about strains, effects, THC/CBD, medical use")

if question:
    with st.spinner("Searching product knowledge base..."):
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)
#%%
