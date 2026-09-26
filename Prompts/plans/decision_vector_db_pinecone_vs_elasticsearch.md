# Decision: Pinecone vs Elasticsearch

> Decided: **keep Elasticsearch** (chosen by the user). Kept for reference.

## Context
The question was whether to replace Elasticsearch with Pinecone as the Vector Database. The ingestion code was
still empty at the time, so switching would have cost little. The real question was which one fits the architecture.

## Recommendation: keep Elasticsearch

1. **The diagram names ELASTICSEARCH** (Elasticsearch Hybrid Search, BM25 keyword + semantic). Changing the
   database means changing the diagram first.
2. **Hybrid search needs real BM25.** ES runs BM25 on the `text` field with an English analyzer, plus kNN on
   `text_embedding` and `image_embedding`, all in one index and one query. Pinecone has no built-in keyword
   engine: sparse vectors (BM25Encoder or SPLADE) would have to be built, fitted on the corpus, and refitted as
   documents stream in.
3. **Metadata Filtering.** Filters on typed `metadata.*` fields (dates, modality, file type, page) need rich
   filters and range queries. Pinecone's filters are simpler, and each record's metadata has a size limit.
4. **Multimodal in one place.** One ES index holds both the 384-dim text vectors and the 512-dim CLIP vectors.
   A Pinecone index has a single dimension, so this would need two indexes and a manual join.
5. **Local, free, testable.** ES runs in `docker-compose.yml` and the integration tests use it. Pinecone is
   cloud-only (paid past the free tier), needs an API key and network access, and can't run in CI without them.

When Pinecone would make sense: pure vector search at very large scale, with no infrastructure to run.
That isn't this project.
