import os
import streamlit as st
from dotenv import load_dotenv
from agentmold import Agent, tool
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import uuid

# --- CONFIGURAÇÃO ---
load_dotenv()
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

# --- 1. CRIAR O BANCO DE VETORES (RAG) ---
@st.cache_resource
def criar_banco_vetorial(arquivo_pdf):
    # Lê o PDF
    reader = PdfReader(arquivo_pdf)
    texto_completo = ""
    for page in reader.pages:
        texto_completo += page.extract_text() + "\n"
    
    # Divide o texto em pedaços (chunks) de 1000 caracteres
    chunks = [texto_completo[i:i+1000] for i in range(0, len(texto_completo), 1000)]
    
    # Cria o banco de dados local
    client = chromadb.Client()
    
    # 🚀 AQUI: NÃO passamos embedding_function, então o ChromaDB usa o modelo padrão dele
    collection = client.create_collection(name="documento_pdf")
    
    # Adiciona os pedaços ao banco
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, ids=ids)
    
    return collection

# --- 2. FERRAMENTA DE BUSCA ---
@tool
def buscar_no_pdf(pergunta: str) -> str:
    """Busca informações no documento PDF carregado e responde à pergunta."""
    # Busca os 3 pedaços de texto mais relevantes para a pergunta
    resultados = st.session_state.colecao.query(query_texts=[pergunta], n_results=3)
    trechos = resultados['documents'][0]
    
    contexto = "\n\n".join(trechos)
    return f"Aqui estão os trechos relevantes do documento:\n\n{contexto}"

# --- 3. CONFIGURAÇÃO DO AGENTE ---
llm_config = {"provider": "openai", "model": "openai/gpt-oss-120b"}

if "colecao" not in st.session_state:
    st.session_state.colecao = None

# --- 4. INTERFACE DO STREAMLIT ---
st.set_page_config(page_title="Analista de PDFs", page_icon="📄")
st.title("📄 Analista de PDFs Financeiros")
st.markdown("Faça upload de um PDF e pergunte qualquer coisa sobre ele.")

# Upload do arquivo
arquivo = st.file_uploader("Escolha um arquivo PDF", type="pdf")

if arquivo is not None:
    # Salva o arquivo temporariamente para o pypdf ler
    with open("temp.pdf", "wb") as f:
        f.write(arquivo.getbuffer())
    
    # Cria o banco de vetores (só roda uma vez por arquivo)
    if st.session_state.colecao is None:
        with st.spinner("Lendo e processando o PDF..."):
            st.session_state.colecao = criar_banco_vetorial("temp.pdf")
        st.success("PDF processado com sucesso!")
    
    # Cria o agente
    agente_pdf = Agent(
        name="Analista",
        instructions="Você é um analista que responde perguntas baseadas APENAS no contexto fornecido. Se a resposta não estiver no contexto, diga 'Não encontrei essa informação no documento'. Não use markdown (**).",
        tools=[buscar_no_pdf],
        llm=llm_config,
    )
    
    # Histórico de chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
    
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Pergunte algo sobre o documento..."):
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analisando o documento..."):
                resposta = agente_pdf.run(prompt)
                st.markdown(resposta.replace("**", ""))
        
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})