# Enterprise ITSM AI Agent

A LangGraph agent that routes ServiceNow employee queries to the right knowledge source, retrieves context via BM25 + dense hybrid search with cross-encoder reranking, and generates grounded answers scoped by department.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  validate_input  │  ← prompt injection check (pattern regex, 6 categories)
└────────┬────────┘
         │ blocked → ┌──────────┐
         │           │ refusal  │ → END
         ▼           └──────────┘
┌─────────────────┐
│     router      │  ← GPT-3.5-turbo structured output → RouteAction
└────────┬────────┘    {source_type: kb_article | incident | out_of_scope,
         │              department_id, refined_query}
         │
    ┌────┴─────┐
    ▼          ▼
┌──────────┐ ┌───────────────┐
│ kb_worker│ │incident_worker│  ← security check: router dept == verified tenant?
└────┬─────┘ └───────┬───────┘    if not → security_alert → END
     │               │
     └──────┬─────────┘
            ▼
     BM25 (top 20) + Dense (top 20)
            │
         RRF fusion → top 10 candidates
            │
     Confidence gate (ConfidenceEvaluator)
     ABSTAIN/ESCALATE → escalation message → END
            │ ANSWER/CLARIFY
     Cross-encoder rerank → top 3
            │
            ▼
    ┌───────────────┐
    │  synthesizer  │  ← GPT-4o, context from KB + incident results
    └───────────────┘    latency profiler prints breakdown on exit
            │
           END → final_answer
```

**Stack:**
- Orchestration: LangGraph 0.6
- Vector store: ChromaDB (persistent, local)
- Embeddings: OpenAI `text-embedding-3-small` (1536-dim)
- Keyword retrieval: BM25 (Robertson-Sparck Jones IDF, implemented from scratch)
- Fusion: Reciprocal Rank Fusion (RRF, k=60)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Router LLM: GPT-3.5-turbo (structured output via function calling)
- Synthesizer LLM: GPT-4o
- Data: 5,076 KB articles (24,922 chunks) + 3,698 incidents from ServiceNow PDI

---

## Retrieval Eval

13-query golden set, evaluated against real ChromaDB chunk IDs. q013 is intentionally out-of-scope and correctly misses for both strategies.

```
Strategy                  Hit@1    Hit@3    Hit@5    MRR
────────────────────────────────────────────────────────
BM25                      23.1%    61.5%    76.9%   0.454
Hybrid (RRF)              46.2%    92.3%    92.3%   0.692
Hybrid + Reranker (k=10)  58.3%   100.0%   100.0%   0.764
```

Each stage adds measurably. BM25 alone misses paraphrased queries entirely. Adding dense retrieval via RRF gets Hit@3 to 92.3%. Adding the cross-encoder reranker pushes Hit@3 to 100% and MRR from 0.692 to 0.764 — it's promoting the correct doc from rank 2–3 to rank 1 on the harder queries.

The reranker candidate pool was validated at k=10 vs k=25: identical Hit@3/MRR at both sizes, with k=10 saving ~88ms/query. k=25 adds noise for the reranker to process without contributing any new correct candidates.

Run evals yourself:
```bash
python src/itsm_agent/eval/run_eval.py \
  --eval-set src/itsm_agent/eval/eval_set.json \
  --verbose
```

---

## Key Technical Decisions

**Why hybrid retrieval instead of dense-only**

BM25 catches exact terminology — incident numbers, error codes, product names — that semantic embeddings compress away. A query for `ORA-12541` should return the Oracle listener error doc, not the most semantically similar text in the corpus. Hybrid gets both. The eval numbers confirm it: +20 points MRR over BM25 alone, and dense fills the exact-match gaps BM25 misses.

**Why RRF instead of score normalization**

BM25 scores (~0–20) and cosine similarity scores (~0.0–1.0) are on incompatible scales with different distributions. Normalizing them requires tuning a weight coefficient that is corpus-specific and unstable. RRF uses only rank position — `1 / (60 + rank)` — which is scale-invariant and requires no tuning. k=60 is the empirically validated default; it dampens the impact of very high ranks without collapsing the contribution curve.

**Why pre-filter by department_id, not post-filter**

Post-filtering (retrieve globally, then discard unauthorized results) has two problems: it leaks document existence across tenants via timing side-channels, and it wastes embedding compute on documents that will be thrown away. Pre-filtering passes `where={"department_id": {"$eq": dept}}` directly to ChromaDB and filters BM25 candidates before scoring. The security check in each worker node also validates that the router's recommended department matches the user's verified tenant — the router can be wrong, the ACL cannot.

**Why GPT-3.5-turbo for routing and GPT-4o for synthesis**

Routing is a classification task with structured output. GPT-3.5-turbo with function calling is reliable, fast (~200ms), and costs ~$0.0001/call. Synthesis requires reasoning over multi-document context and producing coherent prose — that's where GPT-4o's quality advantage justifies the cost (~$0.003/call). Total per-query cost is dominated by synthesis.

**Why BM25 from scratch instead of a library**

BM25 libraries (rank-bm25, elasticsearch) typically apply stemming and stop-word removal. ITSM queries contain high-signal identifiers — `INC0012345`, `KB0082877`, `ORA-12541` — that stemming would corrupt. The custom implementation uses a simple alphanumeric tokenizer that preserves these tokens. It also exposes per-chunk metadata for department pre-filtering in the same pass as scoring, which library wrappers don't support cleanly.

**Why cross-encoder reranking as a third stage**

Bi-encoder retrieval (BM25 + dense) optimizes recall — get the right documents into the top-20 candidate set. Cross-encoders optimize precision — given a small candidate set, score each document jointly against the query with full attention. Running cross-encoder on all 24,922 chunks would be unusable (~45s on CPU). Running it on the top-10 RRF candidates takes ~80ms and moves the correct document to rank 1 in most cases. A/B eval confirmed that expanding the candidate pool to 25 adds 88ms/query with zero improvement in Hit@3 or MRR — the correct document is always in the top-10 RRF results when it's retrievable at all.

---

## Multi-Tenancy

Every document in ChromaDB and the BM25 index is tagged with `department_id` at ingestion. Every query runs scoped to `verified_tenant_id` from the calling context. The security check in `kb_worker_node` and `incident_worker_node` compares the router's `department_id` decision against the caller's verified tenant — mismatches route to `security_alert_node` before any retrieval executes.

Four departments are registered in `DEPARTMENT_REGISTRY` with description and keywords used by the router:

| Department | Keywords |
|---|---|
| IT Software | Git, IntelliJ, VS Code, Python, Slack, Office 365 |
| DT-GPS | Okta, VPN, SSO, Zscaler, Password Reset, Access Denied |
| Global People Live Chat Agents | Benefits, Gym Reimbursement, Payroll, HR, Matching Gifts |
| WPS - Badging | Badge access, Desk booking, Physical Maintenance |

---

## How to Run

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
git clone https://github.com/gurukeertha-chintala/enterprise-ai-agent
cd enterprise-ai-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:
```
OPENAI_API_KEY=sk-...
CHROMA_PATH=data/processed/chromadb
```

**Step 1 — Build the index** (requires `data/raw/kb_knowledge.csv` and `data/raw/incident.csv`)
```bash
python scripts/build_index.py \
  --kb-csv data/raw/kb_knowledge.csv \
  --incident-csv data/raw/incident.csv \
  --chroma-path data/processed/chromadb
```
~30–45 min on first run (24k+ OpenAI embedding calls). Subsequent runs are incremental via ChromaDB `upsert`.

**Step 2 — Run a query**
```bash
python scripts/run_agent.py \
  --query "I can't connect to VPN after my password reset" \
  --tenant "DT-GPS"
```

**Step 3 — Run evals**
```bash
python src/itsm_agent/eval/run_eval.py \
  --eval-set src/itsm_agent/eval/eval_set.json \
  --verbose
```

---

## Design Tradeoffs

**Local ChromaDB over Pinecone/Weaviate**

Portability. A portfolio project should run with `git clone` and an API key, not require cloud credentials and a running managed service. ChromaDB persistent client writes to a local directory. The retrieval interface (`collection.query`) is identical to what Pinecone/Weaviate expose — swapping the backend is a one-file change in `chroma_retriever.py`.

**Static BM25 index over Elasticsearch**

The BM25 index is built at module import time from the ChromaDB corpus. This is a startup cost (~2s for 24k chunks), not a per-query cost. Elasticsearch would add operational complexity — a running service, index management, connection pooling — for no recall benefit at this corpus size. At 500k+ documents, that calculus changes.

**Retrieval eval over RAGAS**

RAGAS measures faithfulness and answer relevance using an LLM-as-judge — it requires an LLM call per query per metric. Retrieval eval (Hit@k, MRR) measures whether the right documents are in the retrieved set. That's the question that matters when comparing retrieval strategies, and it runs in milliseconds with no additional LLM cost. RAGAS is useful for measuring synthesis quality; that's a separate concern.

**Confidence gate thresholds are calibrated for RRF score range, not cosine similarity**

RRF scores are bounded by `1/(k+rank)` and max out at ~0.033 (rank 1 in both retrievers). The default `ConfidenceEvaluator` thresholds (0.65, 0.45) assume cosine similarity scores in [0, 1] — applying them to RRF scores would escalate every query. The evaluator is instantiated in `nodes.py` with RRF-calibrated thresholds: `answer_threshold=0.020`, `escalate_threshold=0.014`. This is a threshold calibration decision, not a bug in the evaluator.
