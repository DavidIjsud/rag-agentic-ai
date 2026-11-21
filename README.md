# rag-agentic-ai

My path to RAG and agentic AI engineering: weekly builds toward an AI assistant for an online shoe store.

## Learning log

### Week 0 · Setup (10–16 Nov 2026)

**What I set up**
- Python 3.12 project managed with uv (`pyproject.toml` + `uv.lock`)
- Gemini 3.5 Flash on Google AI Studio's free tier, called through LangChain's `init_chat_model`, so switching models is one line in `.env`
- LangSmith tracing (agent `rag-course`, environment `local`)
- Supabase Postgres with the pgvector extension
- `check_setup.py`, which checks everything above in one run

**Results** (`week00/first_call.py`, 9 calls)

```text
(rag-agentic-ai) davidgarcia@Davids-MacBook-Pro rag-agentic-ai % uv run week00/first_call.py
setting                  time    in   out think  cost       answer
A thinking=minimal      44.8s    33    70     0  $0.000679  Para andar cómodo todo el día por las subidas y bajadas de l
A thinking=low          15.3s    33   368   293  $0.003361  Para aguantar todo el día caminando por las subidas y bajada
A thinking=medium       40.9s    33   640   581  $0.005810  Le recomiendo unas zapatillas con buena amortiguación y suel
A thinking=high         47.9s    33   717   653  $0.006502  Para aguantar todo el día en las subidas y adoquines de La P
B temperature=0 #1      20.0s    33   735   658  $0.006664  Para las subidas y adoquines de nuestra querida La Paz, te r
B temperature=0 #2      52.7s    33   726   667  $0.006583  Para aguantar las subidas y bajadas de nuestra hermosa La Pa
B default #1            59.0s    33   744   683  $0.006745  Para aguantar el trajín por las empinadas calles de nuestra 
B default #2            54.6s    33   537   477  $0.004883  Para recorrer sin problemas las subidas y bajadas de nuestra
C lite temperature=0     1.2s    33    51     0  $0.000000  Para las empinadas calles de La Paz, te recomiendo unas zapa
                       ⚠️  Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.

Total cost on a paid plan: $0.041229
```

**What surprised me**
One of the thing surprised me is that the thinking minimal can work effectively with low cost but the answer is direct and without much adornment.