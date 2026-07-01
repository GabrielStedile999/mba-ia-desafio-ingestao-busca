# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica com LangChain e PostgreSQL (pgVector)

Software capaz de ingerir um arquivo PDF em um banco vetorial (PostgreSQL + pgVector) e responder perguntas via CLI **com base exclusivamente no conteúdo do PDF**.

## Como funciona

1. **Ingestão** (`src/ingest.py`): o PDF é carregado com `PyPDFLoader`, dividido em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`), convertido em embeddings (`text-embedding-3-small`) e armazenado no PostgreSQL com pgVector via `PGVector`.
2. **Busca** (`src/search.py`): a pergunta do usuário é vetorizada, os **10 chunks mais relevantes** são recuperados com `similarity_search_with_score(query, k=10)`, o contexto é montado no prompt e enviado à LLM (`gpt-5-nano`).
3. **Chat** (`src/chat.py`): CLI interativa no terminal.

## Tecnologias

- Python + LangChain
- PostgreSQL + pgVector (via Docker Compose)
- OpenAI (`text-embedding-3-small` e `gpt-5-nano`)

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca semântica + prompt
│   └── chat.py           # CLI de perguntas e respostas
├── document.pdf          # PDF para ingestão
└── README.md
```

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Uma API Key da OpenAI

## Configuração

1. Clone o repositório:

```bash
git clone https://github.com/GabrielStedile999/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

2. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

Edite o `.env` e preencha sua `OPENAI_API_KEY`. Os demais valores já vêm preenchidos:

```
OPENAI_API_KEY=sua-chave-aqui
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
OPENAI_MODEL='gpt-5-nano'
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='pdf_documents'
PDF_PATH='document.pdf'
```

## Ordem de execução

1. Suba o banco de dados:

```bash
docker compose up -d
```

Aguarde alguns segundos até o healthcheck do Postgres passar (a extensão `vector` é criada automaticamente pelo serviço `bootstrap_vector_ext`).

2. Execute a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rode o chat:

```bash
python src/chat.py
```

## Exemplo de uso

```
Faça sua pergunta: (digite 'sair' para encerrar)

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para sair, digite `sair` (ou `Ctrl+C`).
