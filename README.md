

# Enterprise ITSM AI Agent

A production-grade multi-tenant AI agent built on real ServiceNow data.
Routes employee queries to the right knowledge source, retrieves relevant 
context using hybrid search, and generates grounded answers — scoped by department.

## Architecture

User Query → Router Agent → KB Worker / Incident Worker → Synthesizer → Answer
                                    ↓
                            Security Cross-Check
                            (department_id enforcement)


## Stack
- **Orchestration:** LangGraph
- **Vector Store:** ChromaDB (persistent, local)
- **Embeddings:** OpenAI text-embedding-3-small (1536 dimensions)
- **Reranking:** Cross-Encoder ms-marco-MiniLM-L-6-v2
- **LLM:** GPT-4o (synthesizer), GPT-3.5-turbo (router)
- **Data:** ServiceNow PDI — 5,076 KB articles + 3,698 incidents

## Multi-Tenant Architecture
Every document is tagged with `department_id` at ingestion.
Every query is scoped to the authenticated user's department.
The security cross-check in each worker node verifies the router's 
department recommendation matches the user's verified tenant before 
executing any database query.

## Agent Graph
- **Router Node** — classifies query as `kb_article`, `incident`, or `out_of_scope`
- **KB Worker** — searches KB articles collection with department scoping + reranking
- **Incident Worker** — searches incident corpus with 25-candidate recall + reranking
- **Synthesizer** — combines KB and incident context, generates grounded answer
- **Security Alert Node** — handles cross-department access violations
- **Refusal Node** — handles out-of-scope queries

## Data Pipeline
- 5,076 KB articles exported from ServiceNow PDI
- HTML cleaning with BeautifulSoup
- Fallback to short_description for articles with no body text
- RecursiveCharacterTextSplitter — 512 char chunks, 50 char overlap
- 3,698 incidents with close_notes — boilerplate stripped, indexed as single docs
- Total: 24,747 KB chunks + 3,698 incident docs in ChromaDB

## Retrieval Pipeline
1. Query embedded with OpenAI text-embedding-3-small
2. ChromaDB pre-filtered by department_id (pre-filter, not post-filter)
3. Top 25 candidates retrieved
4. Cross-encoder reranking → top 3 passed to synthesizer

## Cost Per Query
- Embedding: ~$0.000002 per query (1 API call, ~500 tokens)
- LLM (router): ~$0.0001 (GPT-3.5-turbo, ~200 tokens)
- LLM (synthesizer): ~$0.003 (GPT-4o, ~1500 tokens)
- **Total: ~$0.003 per query**

## Example Queries
| Query | Route | Department |
|-------|-------|-----------|
| "I can't access Okta" | incident | DT-GPS |
| "How do I submit a leave request?" | kb_article | Global People |
| "VPN not connecting" | incident | DT-Network |
| "What is the matching gifts policy?" | kb_article | Perks and Programs |

## How to Run
```bash
git clone https://github.com/YOUR_USERNAME/enterprise-ai-agent
cd enterprise-ai-agent
pip install -r requirements.txt
# Add your OpenAI API key to .env
# Run notebooks/ITSM_RAG_Agent.ipynb
```

## What's Next (Layer 2)
- FastAPI wrapper with `/query`, `/health`, `/eval` endpoints
- Langfuse observability — latency, token usage, cost per query
- RAGAS evaluation suite — faithfulness, context recall metrics
- Semantic caching with Redis
