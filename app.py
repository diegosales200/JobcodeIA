import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configurar a API do Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Função para gerar embedding de texto com Gemini
def gerar_embedding(texto):
    response = genai.embeddings.create(
        model="embed-text-embedding-3-large",
        input=[texto]
    )
    return response.data[0].embedding

# Função para calcular similaridade e obter top 3 sugestões da base usando embeddings
def obter_sugestoes_embeddings(descricao, base_job_codes):
    # Gerar embedding da descrição do usuário
    embedding_usuario = gerar_embedding(descricao)
    
    # Gerar embeddings para a base toda (cache para evitar custo em chamadas repetidas)
    if "embeddings_base" not in st.session_state:
        st.session_state.embeddings_base = []
        for desc in base_job_codes['Descricao em 2024']:
            st.session_state.embeddings_base.append(gerar_embedding(desc))
        st.session_state.embeddings_base = np.array(st.session_state.embeddings_base)
    
    # Calcular similaridades (cosine similarity)
    similaridades = cosine_similarity([embedding_usuario], st.session_state.embeddings_base)[0]
    
    # Pegar índices dos top 3 mais similares
    top_indices = similaridades.argsort()[-3:][::-1]
    
    resultados = []
    for idx in top_indices:
        row = base_job_codes.iloc[idx]
        resultados.append({
            "Job Code": row['Job Code'],
            "Titulo": row['Titulo em 2024'],
            "Descricao": row['Descricao em 2024']
        })
    return resultados

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

# Modo 1: Descrição da Atividade com embeddings + similaridade
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
                    resultados = obter_sugestoes_embeddings(descricao_usuario, base_job_codes)
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
            st.write(f"**Descrição:** {descricao}")

        opcao_selecionada = st.selectbox(
            "Selecione a opção:",
            [f"Opção {i}" for i in range(1, len(st.session_state.opcoes_descricao) + 1)],
            key="selecao_descricao"
        )
        nivel_carreira = st.selectbox("Selecione o nível de carreira:", list(NIVEIS_CARREIRA.keys()))

        if st.button("Confirmar Seleção"):
            index_selecionado = int(opcao_selecionada.split()[1]) - 1
            codigo, _, _ = st.session_state.opcoes_descricao[index_selecionado]
            complemento = NIVEIS_CARREIRA[nivel_carreira]
            codigo_completo = f"{codigo}-{complemento}"
            registrar_feedback(descricao_usuario, codigo_completo)
            st.success(f"Código Completo Selecionado: {codigo_completo}")

# Modo 2: Busca por Colaborador
elif modo_busca == "Colaborador (Ativo ou Desligado)":
    if base_substituicao is not None:
        substituido = st.selectbox("Selecione o nome do colaborador:", sorted(base_substituicao['Substituido'].dropna().unique()))
        if substituido:
            ultimo_registro = base_substituicao[base_substituicao['Substituido'] == substituido].sort_values(by='Data Referencia', ascending=False).iloc[0]
            st.markdown("### Último Registro Encontrado")
            st.write(f"**Job Code:** {ultimo_registro['Job Code']}")
            st.write(f"**Título:** {ultimo_registro['Titulo Job Code']}")
            st.write(f"**Cargo:** {ultimo_registro['Cargo']}")
            st.write(f"**Gestor:** {ultimo_registro['Gestor']}")
            st.write(f"**Descrição:** {ultimo_registro['Descricao em 2024']}")
    else:
        st.error("Base de substituição não carregada.")

# Modo 3: Busca por Gestor e Cargo
elif modo_busca == "Gestor e Cargo":
    if base_substituicao is not None:
        gestor = st.selectbox("Passo 1 - Selecione o gestor:", sorted(base_substituicao['Gestor'].dropna().unique()))

        if gestor:
            cargos_filtrados = base_substituicao[base_substituicao['Gestor'] == gestor]['Cargo'].dropna().unique()
            cargo = st.selectbox("Passo 2 - Selecione o cargo:", sorted(cargos_filtrados))
        else:
            cargo = None

        if cargo:
            resultado = base_substituicao[
                (base_substituicao['Gestor'] == gestor) & (base_substituicao['Cargo'] == cargo)
            ].sort_values(by='Data Referencia', ascending=False)

            if not resultado.empty:
                st.markdown("### Resultados Encontrados")
                job_codes_exibidos = set()
                for _, linha in resultado.iterrows():
                    job_code = linha['Job Code']
                    if job_code not in job_codes_exibidos:
                        job_codes_exibidos.add(job_code)
                        st.write(f"**Job Code:** {job_code}")
                        st.write(f"**Título:** {linha['Titulo Job Code']}")
                        st.write(f"**Descrição:** {linha['Descricao em 2024']}")
            else:
                st.warning("Nenhum resultado encontrado para a combinação selecionada.")
        else:
            st.warning("Por favor, selecione um cargo válido.")
    else:
        st.error("Base de substituição não carregada.")
