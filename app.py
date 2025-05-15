import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

# Configurar a API do Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Inicializar o modelo para embeddings
embedding_model = genai.GenerativeModel(model_name="embed-text-embedding-3-large")

# Stopwords estáticas em português (adicione suas stopwords aqui)
stop_words = [...]

# Função para gerar embedding de texto com Gemini
def gerar_embedding(texto, max_retries=3, wait_time=1):
    for i in range(max_retries):
        try:
            response = embedding_model.generate_content([{"text": texto}])
            return response.parts[0].embedding.values
        except Exception as e:
            st.error(f"Erro ao gerar embedding (tentativa {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(wait_time)
            else:
                return None
    return None

# Função para obter sugestões usando TF-IDF para filtragem inicial e, opcionalmente, embeddings para refinamento
def obter_sugestoes_filtradas(descricao_usuario, base_job_codes, top_n_tfidf=5, top_k_embeddings=3):
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=1, ngram_range=(1, 2))
    matriz_tfidf = tfidf.fit_transform(base_job_codes['Descricao em 2024'])
    similaridades_tfidf = cosine_similarity(tfidf.transform([descricao_usuario]), matriz_tfidf)[0]
    indices_similares_tfidf = similaridades_tfidf.argsort()[-top_n_tfidf:][::-1]
    resultados_tfidf = base_job_codes.iloc[indices_similares_tfidf]

    embeddings_subconjunto = []
    indices_validos_tfidf = []
    for index, row in resultados_tfidf.iterrows():
        embedding = gerar_embedding(row['Descricao em 2024'])
        if embedding is not None:
            embeddings_subconjunto.append(embedding)
            indices_validos_tfidf.append(index)

    if not embeddings_subconjunto:
        return resultados_tfidf[['Job Code', 'Descricao em 2024', 'Titulo em 2024']].to_dict('records')

    embedding_usuario = gerar_embedding(descricao_usuario)
    if embedding_usuario is None:
        return resultados_tfidf[['Job Code', 'Descricao em 2024', 'Titulo em 2024']].to_dict('records')

    similaridades_embeddings = cosine_similarity([embedding_usuario], embeddings_subconjunto)[0]
    indices_similares_embeddings = similaridades_embeddings.argsort()[-top_k_embeddings:][::-1]

    resultados_finais = []
    df_validos_tfidf = resultados_tfidf.loc[indices_validos_tfidf].reset_index(drop=True)
    for idx in indices_similares_embeddings:
        if idx < len(df_validos_tfidf):
            resultados_finais.append({
                "Job Code": df_validos_tfidf.iloc[idx]['Job Code'],
                "Titulo": df_validos_tfidf.iloc[idx]['Titulo em 2024'],
                "Descricao": df_validos_tfidf.iloc[idx]['Descricao em 2024']
            })

    return resultados_finais

def gerar_descricao_gemini(descricao_base):
    try:
        response = embedding_model.generate_content([{"text": descricao_base}])
        return response.parts[0].text
    except Exception as e:
        st.error(f"Erro ao gerar descrição com Gemini: {e}")
        return None

# Funções para carregar as bases
@st.cache_data
def carregar_base_job_codes():
    try:
        return pd.read_excel("base_job_codes.xlsx")
    except Exception as e:
        st.error(f"Erro ao carregar base de códigos: {e}")
        return None

@st.cache_data
def carregar_base_substituicao():
    try:
        return pd.read_excel("base_substituicao.xlsx")
    except Exception as e:
        st.error(f"Erro ao carregar base de substituição: {e}")
        return None

# Função para registrar feedback
def registrar_feedback(entrada, codigo):
    with open("feedback.csv", "a", encoding="utf-8") as f:
        f.write(f'"{entrada}","{codigo}"\n')

# Níveis de carreira disponíveis
NIVEIS_CARREIRA = {
    "Estágio": "EST",
    "Trainee": "TRN",
    "Júnior": "JR",
    "Pleno": "PL",
    "Sênior": "SR",
    "Coordenador": "COORD",
    "Especialista": "ESP",
    "Gerente": "GR",
    "Diretor": "DIR",
    "Superintendente": "SUP"
}

# Interface do Streamlit
st.title("Sistema de Sugestão de Job Code")

modo_busca = st.radio("Escolha o modo de busca:", [
    "Descrição da Atividade",
    "Colaborador (Ativo ou Desligado)",
    "Gestor e Cargo"
])

base_job_codes = carregar_base_job_codes()
base_substituicao = carregar_base_substituicao()

# Modo 1: Descrição da Atividade com filtragem TF-IDF e embeddings
if modo_busca == "Descrição da Atividade":
    descricao_usuario = st.text_area("Digite a descrição do cargo:")

    if "opcoes_descricao" not in st.session_state:
        st.session_state.opcoes_descricao = []

    if "selecao_descricao" not in st.session_state:
        st.session_state.selecao_descricao = None

    if st.button("Buscar Código"):
        if descricao_usuario.strip():
            if base_job_codes is not None:
                with st.spinner("Buscando sugestões..."):
                    resultados = obter_sugestoes_filtradas(descricao_usuario, base_job_codes)
                st.session_state.opcoes_descricao = [
                    (r['Job Code'], r['Descricao'], r['Titulo']) for r in resultados
                ]
                if not st.session_state.opcoes_descricao:
                    st.warning("Nenhuma opção encontrada.")
            else:
                st.error("Erro ao carregar os dados.")
        else:
            st.warning("Por favor, insira uma descrição válida.")

    if st.session_state.opcoes_descricao:
        for i, (codigo, descricao, titulo) in enumerate(st.session_state.opcoes_descricao, 1):
            st.markdown(f"### Opção {i}")
            st.write(f"**Título:** {titulo}")
            st.write(f"**Código:** {codigo}")
            with st.spinner("Gerando descrição detalhada..."):
                descricao_detalhada = gerar_descricao_gemini(descricao)
            if descricao_detalhada:
                st.write(f"**Descrição:** {descricao_detalhada}")
            else:
                st.write(f"**Descrição (Base):** {descricao} (Erro ao gerar descrição detalhada)")

        opcao_selecionada = st.selectbox(
            "Selecione a opção:",
            [f"Opção {i}" for i in range(1, len(st.session_state.opcoes_descricao) + 1)],
            key="selecao_descricao"
        )
        nivel_carreira = st.selectbox("Selecione o nível de carreira:", list(NIVEIS_CARREIRA.keys()))

        if st.button("Confirmar Seleção"):
            index_selecionado = int(opcao_selecionada.split()
