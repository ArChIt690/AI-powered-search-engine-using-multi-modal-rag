# Part 3 LLM Architecture: plan

> Approved plan (2026-09-30). The user also asked for **both LLM providers (Groq and Gemini) to be used together**
> across the project, so neither free rate limit is reached quickly.

## Flow (v1 diagram)
```
LLM ─► MCP
    ─► TOOLS
    ─► GUARDRAIL ──UNSAFE──► USER ("sends the reason alongside")
             └──SAFE──► EVAL ─► RESULT ─┬─► FAISS SEMANTIC CACHE ("goes for caching")
                          │             └─► FINAL RESULT to USER
                          └─► "all the eval results are stored separately"
```
Part 2's `LLMAgent.answer` (one direct call) is replaced from the inside; the Retrieval pipeline around it stays.

## Two providers (`infra/llm_client.py`)
`.env`: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_FAST_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`. Each role prefers one
provider and falls back to the other (LangChain `with_fallbacks`) on an error, rate limit or unusable output:

| Role | Used by | First choice | Falls back to |
|---|---|---|---|
| fast | Query Enhancement, Guardrail | Groq `gpt-oss-20b` | Gemini |
| answer | the LLM (agent) | Groq `gpt-oss-120b` | Gemini |
| judge | Eval | Gemini `gemini-2.5-flash` | Groq `gpt-oss-20b` |

Gemini's free limit is 5 requests/minute, so it gets one call per new question (Eval), which also means an answer
is not judged by the model that wrote it.

## Steps
1. `schemas/eval.py`: `GuardrailVerdict` (safe, reason, category), `EvalResult` (faithfulness, relevance,
   citation correctness 1–5, passed), `EvalRecord` (what is stored).
2. **TOOLS**, `llm/tools/`, one file per tool, LangChain `@tool`:
   `search_documents` (hybrid search + rerank again with a new query), `read_surrounding` (the chunks before and
   after a chunk), `list_files` (what is in the corpus), `calculator` (safe arithmetic). Chunks a tool returns
   join the numbered context so they can be cited.
3. **MCP**, `llm/mcp.py`: `langchain-mcp-adapters` `MultiServerMCPClient`; servers from `MCP_SERVERS` in `.env`
   (none by default: the user chose to add real servers later). A tiny MCP server in the tests proves the path.
4. **LLM**, `llm/agent.py`: LangChain `create_agent` (LangGraph) with the answer model, the local tools and the
   MCP tools; same rules (answer only from the context, cite [n]); a cap on tool calls; falls back to the direct
   answer if the agent fails.
5. **GUARDRAIL**, `llm/guardrail.py`, on the LLM output: rule checks (leaked secrets, empty) + a safety model on
   Groq. UNSAFE → USER with the reason, nothing cached, EVAL skipped. SAFE → EVAL.
6. **EVAL**, `llm/eval.py`: LLM-as-judge (judge role, JSON-schema output): faithfulness, relevance, citation
   correctness. The answer is always returned with its scores; only answers that pass are cached.
7. **"all the eval results are stored separately"**, `llm/eval_store.py`: one JSON line per eval in
   `data/eval_results/<date>.jsonl`; never ingested.
8. **RESULT**, `llm/pipeline.py`: `LLMPipeline.run()` = agent → guardrail → eval → result. The Retrieval pipeline
   calls it; the FAISS and Redis cache writes move behind GUARDRAIL + EVAL ("goes for caching").
9. Tests (unit per box, e2e "Done when"), then a real run.

## Cost
A new question: Query Enhancement + agent (1 + tool calls) + guardrail + eval = 4+ LLM calls (2 today), split
over the two providers. A cache hit: 1.

## New dependencies
`langgraph`, `langchain-mcp-adapters`, `mcp`

## What changed while building (2026-09-30)
- **Guardrail model:** Groq `openai/gpt-oss-safeguard-20b` (a model made to judge content against a written
  policy), role `guard`, Gemini as fallback. Groq's `llama-prompt-guard-2` models only classify prompt injection
  in a single message, so they are not used. Real check: 7 of 7 answers judged correctly, 0.2–0.3 s each.
- **Eval sees tool results.** A correct file list from `list_files` first scored 2/2/1, because the judge saw
  only the passages. The results of `list_files`, `calculator` and MCP tools are now given to the judge as
  evidence (passages from `search_documents` / `read_surrounding` are in the numbered context already).
- **Guardrail unreachable:** the answer passes on the rules only, marked unchecked, and is never cached.
- **`【calculator】`-style marks** (gpt-oss naming a tool as a source) are removed from answers.
- **Ingestion skips `eval_results_dir`**, so eval results can't be ingested even when the whole `data/` folder is.
- **`.env` names:** two values with the same name overwrite each other, so each provider has its own
  (`GROQ_*`, `GEMINI_*`); the old `LLM_*` names are gone.
- Real run (9 files, Groq + Gemini): calculator question → 436 and 32.80 %, eval 5/5/5, cached; repeat → FAISS
  cache in 4.6 s; two-part question cited both files; file list via `list_files`, eval 5/5/5 after the fix.
