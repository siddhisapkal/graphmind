# GraphMind Two-Layer Memory Implementation

## 1. What "two-layer memory" means in this codebase

In GraphMind, the runtime retrieval path is effectively **two-layered**:

1. **Layer 1: Graph memory**
   - Durable, structured memory in **Neo4j**
   - Stores user-to-entity facts such as `STRUGGLES_WITH`, `STUDIES`, `TARGETS`, `STRENGTH_IN`
   - Also stores entity-to-entity structural links like `PART_OF`, `USED_IN`, `DEPENDS_ON`, `RELATED_TO`

2. **Layer 2: Vector memory**
   - Semantic recall over prior chat messages using **FAISS**
   - Persisted in **SQLite** (`backend/vector_store.sqlite3`)
   - Used to retrieve semantically similar past utterances when exact graph facts are missing or incomplete

There is also a very important **staging layer during ingestion**:

- **Ephemeral memory**
  - Temporary aggregate store backed by **Redis** if available, else **SQLite** (`backend/ephemeral_memory.sqlite3`)
  - Holds candidate user signals before they are promoted into durable graph memory
  - Prevents noisy one-off mentions from immediately polluting the graph

So the best mental model is:

`User message -> vector write immediately`

`User message -> extracted signals -> ephemeral aggregation -> graph promotion when strong enough`

`New user query -> retrieve from graph + vector in parallel -> optionally add web only if enabled`

## 2. Technologies used

### Core backend

- **FastAPI**: API server and request handling
- **Uvicorn**: ASGI server
- **Pydantic**: request/response models
- **Python 3.11+**

### Memory and storage

- **Neo4j**: durable graph memory
- **SQLite**:
  - vector persistence (`backend/vector_store.sqlite3`)
  - ephemeral fallback store (`backend/ephemeral_memory.sqlite3`)
  - relation semantic cache (`backend/relations.sqlite3`)
  - event log (`backend/events.sqlite3`)
  - chat history (`backend/chat_history.sqlite3`)
- **Redis**: optional ephemeral and cache backend
- **FAISS**: in-memory approximate nearest-neighbor search for vector recall
- **NumPy**: vector normalization and FAISS matrix handling

### LLM and semantic processing

- **Google Gemini**
  - triple extraction / memory signal extraction
  - embeddings via `gemini-embedding-001`
  - optional chat generation depending on configuration
- **Groq**
  - default response generation provider in current config path
- **Heuristic fallback logic**
  - used when LLM extraction is unavailable or disabled

## 3. Main files involved

- `backend/main.py`
  - `/chat` orchestration
  - parallel retrieval
  - background memory pipeline scheduling
- `backend/graph/service.py`
  - graph schema management
  - signal aggregation and promotion
  - graph evidence and graph context retrieval
- `backend/graph/ephemeral.py`
  - temporary signal store
  - TTL-backed aggregation
- `backend/vector_store.py`
  - message embedding persistence
  - FAISS bucket loading
  - semantic retrieval
- `backend/gemini_chat.py`
  - extraction of triple candidates
  - embedding resolution
  - final reply construction
- `backend/topic_router.py`
  - semantic topic detection from graph topics
- `backend/section_resolver.py`
  - converts intent into retrieval views like weaknesses, skills, goals, topics
- `backend/relation_semantics.py`
  - classifies relations into family, polarity, tags, strength
- `backend/web_research.py`
  - optional web fallback planning and web-derived signal inference

## 4. High-level architecture

### Durable memory

The durable memory is split across two stores with different responsibilities:

- **Graph memory**
  - best for typed, explainable, relationship-heavy knowledge
  - example:
    - `User -[STRUGGLES_WITH]-> Dynamic Programming`
    - `Fourier Transform -[PART_OF]-> Signals and Systems`

- **Vector memory**
  - best for fuzzy semantic recall of prior messages
  - example:
    - retrieve an older message like "I keep getting stuck in DP state transitions" when the user now asks "what are my weak areas?"

### Staging memory

- **Ephemeral aggregates**
  - keyed by `user_id + relation + entity_type + entity`
  - track:
    - max confidence
    - mention count
    - sources
    - last raw text
    - whether already promoted

This staging step is what makes the graph cleaner than raw extraction output.

## 5. Ingestion logic in detail

### 5.1 Entry point: `/chat`

When a user sends a message to `/chat`, the backend immediately does four things:

1. Saves the user message as an event record
2. Saves the message into Neo4j as a `Message` node / conversation-linked artifact
3. Writes the message into the vector store via `add_message(...)`
4. Schedules background memory extraction via `_process_memory_pipeline(...)`

Important behavior:

- **Vector memory is updated immediately**
- **Graph memory is updated asynchronously in the background**

This keeps the chat response path fast while still learning continuously.

### 5.2 Vector ingestion path

Implemented in `backend/vector_store.py`.

For every message written through `add_message(...)`:

1. The system resolves an embedding via `_resolve_embeddings(...)`
   - preferred path: Gemini embedding model
   - fallback path: local hashed embedding (`local-hash-v1`)
2. A lightweight `topic_key` is generated from normalized topic terms in the text
3. The message record is inserted into SQLite table `message_vectors`
4. The in-memory FAISS bucket cache is updated
   - user-level bucket
   - optional topic bucket

Stored fields include:

- `id`
- `user_id`
- `conversation_id`
- `role`
- `created_at`
- `text`
- `embedding_json`
- `embedding_kind`
- `topic_key`

Why this matters:

- vector recall is available as soon as the message is saved
- even before graph promotion happens, the system can semantically recall that message later

### 5.3 Background extraction path

Implemented in `_process_memory_pipeline(...)` in `backend/main.py`.

The pipeline receives:

- `user_id`
- `conversation_id`
- raw `message`
- `source`
- `source_event_id`
- `created_at`
- optional `inferred_raw_signals` from search planning

It then:

1. Calls `extract_triple_candidates(...)`
2. Converts user-subject triples into raw signals
3. Merges those with `inferred_raw_signals`
4. Passes triples and signals into graph memory service
5. Logs promotion outcomes
6. Optionally enriches relation semantics with LLM classification

### 5.4 Triple extraction

Implemented in `backend/gemini_chat.py`.

`extract_triple_candidates(...)` works in two modes:

- **Heuristic mode**
- **LLM-assisted mode**

The extraction contract separates:

- `user_facts`
  - facts from the user to an object
  - example: `User -> STRUGGLES_WITH -> Dynamic Programming`
- `concept_relations`
  - structural knowledge between concepts
  - example: `Fourier Transform -> PART_OF -> Signals and Systems`

Each triple candidate carries:

- subject type/name
- relation
- object type/name
- confidence
- source
- raw text
- action linkage flag
- promotion hint

### 5.5 Signal aggregation into ephemeral memory

Implemented by `EphemeralMemoryStore.upsert_signal(...)`.

Each signal is stored under a stable key:

- `user_id | normalized_relation | entity_type | entity`

If the same signal appears again:

- `mention_count` increases
- `max_confidence` is updated
- `last_seen` is refreshed
- `last_raw_text` is replaced
- `linked_to_action` is retained if ever true
- `sources` are merged

This means GraphMind does not treat every message as a brand-new fact. It first aggregates repeated evidence.

### 5.6 Promotion rules from ephemeral to graph

Implemented in `GraphMemoryService._should_promote(...)`.

A user memory signal is promoted if **any** of these are true:

- `max_confidence >= 0.8`
- `mention_count >= 2`
- `linked_to_action == True`

This is the key graph-quality filter.

Interpretation:

- high-confidence single mentions can be promoted
- medium-confidence repeated mentions can be promoted
- action-linked facts can be promoted early because they usually matter operationally

### 5.7 Graph promotion

Implemented in `GraphMemoryService._promote_to_graph(...)`.

When a signal is promoted:

1. The relation is canonicalized
2. The entity label is resolved (`Company`, `Topic`, `Skill`, `Goal`, `Document`, `Entity`)
3. The entity name is canonicalized
4. A canonical entity key is computed
5. Relation semantics are classified
6. Neo4j `MERGE` writes:
   - `User` node
   - typed entity node
   - relation edge from user to entity

The edge stores rich metadata:

- `confidence`
- `reinforcement_count`
- `mention_count`
- `sources`
- `linked_to_action`
- `last_signal_text`
- `relation_family`
- `relation_polarity`
- `section_tags`
- `relation_strength`
- timestamps like `first_promoted_at` and `last_reinforced`

This is what makes graph retrieval explainable and filterable.

### 5.8 Structural concept promotion

For triples that are **not** user-subject triples:

- they go through entity resolution in `backend/entity_resolution.py`
- subject and object are matched or canonicalized against graph nodes
- promotion happens through `_promote_resolved_triple_to_graph(...)`

These become concept graph edges like:

- `PART_OF`
- `USED_IN`
- `DEPENDS_ON`
- `RELATED_TO`

### 5.9 Co-mentioned and structural linking

If multiple aggregates are promoted in the same pass, the graph service also creates:

- **co-mentioned entity links** via `RELATED_TO`
- **structural edges** inferred from text patterns between entities

This increases traversal quality for later graph evidence retrieval.

## 6. Retrieval logic in detail

### 6.1 Retrieval starts in `/chat`

The chat flow in `backend/main.py` is memory-first.

For every incoming message:

1. detect semantic topic
2. classify route intent
3. build optional web search plan
4. resolve section view
5. launch graph retrieval
6. launch vector retrieval
7. optionally launch web retrieval only if allowed

Graph and vector retrieval run in parallel through a shared executor.

### 6.2 Topic detection

Implemented in `backend/topic_router.py`.

The topic router:

- builds an in-memory FAISS index from graph topic/entity names and aliases
- uses hashed semantic embeddings
- ignores generic words like `study`, `practice`, `topic`
- returns the best topic if score is above `0.48`

This helps map fuzzy phrasing onto existing graph topics.

Example:

- query: "what should I revise in signals?"
- detected topic: `Signals and Systems`

### 6.3 Section-aware retrieval

Implemented in `backend/section_resolver.py`.

The section resolver converts user intent into logical retrieval lenses such as:

- `weaknesses`
- `skills`
- `topics`
- `companies`
- `goals`

It produces:

- `section_tags`
- `section_families`
- `focus_entity`

These are used to bias graph context retrieval toward relevant memory instead of dumping the full graph.

### 6.4 Graph retrieval

Implemented mainly in:

- `GraphMemoryService.fetch_graph_evidence(...)`
- `GraphMemoryService.fetch_section_context(...)`
- `GraphMemoryService.fetch_graph_context(...)`

#### Step A: build a graph view

`fetch_graph_view(...)` materializes:

- user memory edges
- entity-to-entity edges
- typed nodes

#### Step B: score candidate nodes

For each node, the system computes:

- **query term overlap**
- **memory edge weight**
- **adjacency bonus** from related entity edges

Conceptually:

- overlap is weighted strongly
- direct user memory edges matter a lot
- nearby entity relations contribute a smaller bonus

The implementation uses:

- `score = overlap * 3.0 + memory_weight + related_bonus`

#### Step C: seed selection

The top-scoring nodes become seed nodes.

If no query overlap exists, the system falls back to nodes with strongest outgoing user memory.

#### Step D: best-first subgraph expansion

`_best_first_subgraph(...)` expands around those seed nodes for up to 3 hops, preferring:

- strong memory edges
- semantically overlapping nodes
- helpful neighboring graph structure

#### Step E: build evidence output

The result is returned as:

- `facts`
- `paths`
- `citations`

Example path:

- `user-123 -[STRUGGLES_WITH]-> Dynamic Programming`
- `Dynamic Programming -[PART_OF]-> Algorithms`

#### Step F: graph confidence gating

Back in `/chat`, the graph result is considered relevant only if:

- `graph_max_score >= 0.45`

If graph confidence is below that threshold, graph context is discarded for that request.

### 6.5 Vector retrieval

Implemented in `backend/vector_store.py::search(...)`.

#### Step A: embed the query

The query is embedded with the same embedding family used for stored messages.

#### Step B: choose search bucket

The vector store first tries a **topic bucket**:

- query text is converted to `topic_keys`
- if a topic-specific FAISS bucket has enough records, it is used

If no suitable topic bucket exists, it falls back to:

- full user-level bucket

#### Step C: FAISS similarity search

The store runs inner-product search over normalized vectors.

Search fanout:

- `search_k = min(max(k * 4, 20), len(bucket.records))`

This over-fetches candidates and then filters them.

#### Step D: result filtering

Results are filtered by:

- valid index bounds
- optional matching `conversation_id`
- duplicate suppression

The chat path later keeps only vector hits with:

- `score >= 0.5`

Then it compresses those texts into snippets for reply generation.

### 6.6 Memory hit decision

In `/chat`, memory hit is determined using both retrieval layers:

- graph hit if graph score passes threshold
- vector hit if at least one vector snippet survives filtering

So:

- `memory_hit = relevant_graph_hit OR relevant_vector_hit`

This means the assistant can answer from:

- structured graph memory
- semantic message memory
- or both together

### 6.7 Optional web retrieval

Web search is not part of the two core memory layers.

It is only used when:

- `allow_web_search == true` or the message explicitly requests web/current info
- the search planner thinks web search is useful
- queries were successfully built

If memory is missing and web is not allowed:

- the system returns a transparent miss like:
  - `I don't have any memory of that yet.`

If web is enabled and valid:

- response mode becomes `memory_plus_web`

## 7. End-to-end flow

### 7.1 Ingestion flow

```text
User message
  -> log raw event
  -> save message in Neo4j
  -> add message to vector store (SQLite + FAISS)
  -> schedule background memory pipeline

Background memory pipeline
  -> extract triple candidates
  -> convert user triples to raw signals
  -> merge with inferred signals
  -> upsert into ephemeral aggregate store
  -> if promotion threshold met:
       -> promote to Neo4j graph
       -> mark aggregate as promoted
       -> refresh topic router
  -> optionally enrich relation semantics with LLM
  -> log promotion summary
```

### 7.2 Retrieval flow

```text
New user query
  -> detect semantic topic
  -> resolve route + section plan
  -> launch graph retrieval
  -> launch vector retrieval
  -> optionally launch web retrieval
  -> graph score threshold check
  -> vector score threshold check
  -> if memory hit:
       -> build reply from graph facts + vector snippets (+ optional web)
     else:
       -> return transparent memory miss
```

## 8. Why this design works

### Strengths of graph layer

- explainable memory
- typed relations
- supports section filtering like weaknesses/goals/skills
- supports multi-hop evidence paths

### Strengths of vector layer

- catches paraphrases and fuzzy recall
- works immediately after a message is sent
- helpful before graph promotion has happened

### Strengths of ephemeral staging

- reduces noisy graph writes
- lets repeated evidence accumulate
- supports promotion based on confidence or repeated mentions

## 9. Important thresholds and heuristics

- Topic router acceptance: `0.48`
- Graph relevance threshold in chat: `0.45`
- Vector snippet acceptance in chat: `0.5`
- Graph promotion confidence threshold: `0.8`
- Graph promotion repeated-mentions threshold: `2`
- Max graph traversal hops during evidence expansion: `3`

These values directly control memory precision vs recall.

## 10. Practical interpretation of the layers

If a user says:

- "I struggle with dynamic programming"

Then:

1. the exact message is immediately searchable in vector memory
2. extracted signal enters ephemeral memory
3. if confidence is strong enough, or repeated later, it becomes:
   - `User -[STRUGGLES_WITH]-> Dynamic Programming`

If the user later asks:

- "what are my weak topics?"

The system may answer from:

- graph memory if promotion already happened
- vector memory if the earlier message exists semantically but was not yet graph-promoted
- both if both layers contain useful evidence

## 11. Current implementation summary

The current GraphMind implementation is best described as:

- **retrieval**: two-layer memory system
  - Neo4j graph memory
  - FAISS vector memory
- **ingestion**: staged promotion pipeline
  - extraction
  - ephemeral aggregation
  - graph promotion
  - vector persistence in parallel

That separation is intentional:

- vector memory provides immediate semantic recall
- graph memory provides durable structured recall
- ephemeral staging protects graph quality
