# ADR-006: Semantic Chunking and Lost in the Middle Mitigation

## Status

Accepted

## Context

Fixed-token chunking splits at arbitrary boundaries, producing semantically incomplete
chunks. LLM performance degrades for information in the middle of the context window
(Liu et al., 2023 — "Lost in the Middle").

## Decision

1. Chunk on semantic boundaries: paragraph breaks preferred, sentence breaks as fallback,
   hard token cap as last resort.
2. Assembled context uses Lost in the Middle ordering: highest-relevance chunk at
   position 0, second-highest at position -1, remaining chunks fill middle in
   descending relevance order.
3. Cache key is content hash, not file path. Same chunk appearing in two documents
   is stored once.
4. Memory write uses actor_id="document_agent" to enforce write isolation
   (pre-empting Day 8 write-isolation ADR).

## Tradeoff

Semantic chunking increases average chunk size ~15% vs fixed-token splitting.
This reduces retrieval cardinality (fewer distinct chunks per document).
Accepted: correctness of chunk boundaries outweighs retrieval breadth.

## Consequences

Day 18 knowledge layer can assume DocumentChunk metadata (section_path, page_number,
content_hash) is always present. Day 25 eval harness can build document grounding
evals against this schema.
