import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _get_vector_store():
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def search_prompt(question=None):
    """Retorna uma função que responde perguntas com base no conteúdo do PDF.

    Se `question` for informada, responde diretamente e retorna a string.
    """
    for var in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(var):
            print(f"Variável de ambiente {var} não definida. Confira seu arquivo .env")
            return None

    try:
        store = _get_vector_store()
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao inicializar a busca: {exc}")
        return None

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )
    chain = prompt | llm | StrOutputParser()

    def answer(user_question: str) -> str:
        # 1. Vetoriza a pergunta e busca os 10 chunks mais relevantes
        results = store.similarity_search_with_score(user_question, k=10)

        # 2. Concatena os resultados para montar o contexto
        contexto = "\n\n".join(doc.page_content.strip() for doc, _score in results)

        # 3. Monta o prompt, chama a LLM e retorna a resposta
        return chain.invoke({"contexto": contexto, "pergunta": user_question}).strip()

    if question:
        return answer(question)
    return answer
