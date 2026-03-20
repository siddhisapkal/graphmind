# GraphMind Memory-First Design

## Overview

GraphMind now operates under a **memory-first retrieval paradigm**:

1. **Always query memory first**: Graph (Neo4j) + vector (FAISS) stores are searched for relevant user history.
2. **Transparent memory miss**: If no memory is found and web search is disabled, the user is informed and offered an opt-in option.
3. **Optional web search**: Web search is only performed when explicitly enabled via the `allow_web_search` flag in the chat request.
4. **Continuous memory building**: User messages are **always** processed for memory extraction in the background, independent of retrieval needs.

## Behavior

### Case 1: Memory Hit (Graph or Vector)
- User asks: "What should I practice for Google?"
- System finds stored weaknesses, topics, or goals related to Google.
- Answer: personalized recommendation based on stored memory.
- Response: `"retrieval_mode": "memory_hit"`

### Case 2: Memory Hit + Web Search Enabled
- User asks: "What should I practice for Google?" with `allow_web_search=true`
- System finds memory AND retrieves current web results.
- Answer: combines stored context with live findings.
- Response: `"retrieval_mode": "memory_plus_web"`

### Case 3: No Memory, No Web Search
- User asks: "Tell me my weak topics"
- System finds no relevant stored memory.
- **No LLM call made** (cost savings).
- Answer: "I couldn't find a relevant memory record for that question yet. If you want, you can re-send this query with `allow_web_search=true` to query the internet."
- Response: `"retrieval_mode": "memory_miss"`, `"memory_found": false`, `"suggest_web_search": true`

### Case 4: No Memory, Web Search Enabled
- User asks: "What's the best strategy for Google interviews?" with `allow_web_search=true`
- System finds no memory but retrieves web results.
- Answer: web-driven guidance without user context.
- Response: `"retrieval_mode": "memory_plus_web"`, `"web_search_used": true`

## API Changes

### `/chat` Endpoint

**Request Body:**
```json
{
  "user_id": "user-123",
  "message": "what should I practice for Google",
  "conversation_id": "optional-convo-id",
  "source": "chat",
  "allow_web_search": false
}
```

**New Field:**
- `allow_web_search` (boolean, default `false`): Enable optional internet search if memory is insufficient.

**Response Fields:**
```json
{
  "answer": "...",
  "memory_found": true|false,
  "web_search_used": true|false,
  "suggest_web_search": true|false,
  "retrieval_mode": "memory_hit|memory_plus_web|memory_miss",
  "memory_update_mode": "background",
  ...other fields...
}
```

**New Response Fields:**
- `memory_found`: Boolean indicating if graph/vector memory was matched.
- `web_search_used`: Boolean indicating if web results were included.
- `suggest_web_search`: Boolean hint for UI to offer web search button if no memory was found.
- `retrieval_mode`: One of `"memory_hit"`, `"memory_plus_web"`, or `"memory_miss"`.

## Background Memory Processing

Every user message **always** triggers background memory extraction, regardless of retrieval:

1. Extract entity/relation triples from message.
2. Classify semantics (family, polarity, strength).
3. Promote high-confidence signals to Neo4j graph.
4. Accumulate stats in ephemeral store.

This ensures the knowledge graph grows even when chat doesn't need retrieval. Over time, memory becomes richer and retrieval hits increase.

## Benefits

| Benefit | Why |
|---------|-----|
| **Cost savings** | No LLM generation when no memory found; no random web search. |
| **Better UX** | User knows when memory is missing; can explicitly opt-in to web. |
| **Focused retrieval** | Web search only used when relevant (after memory exhaustion). |
| **Memory growth** | Background extraction builds knowledge graph continuously. |
| **Transparency** | Response flags make it clear whether data is from memory or web. |

## Implementation Details

### Memory Hit Detection
```python
memory_hit = bool(graph_context or retrieved)
```
- `graph_context`: Neo4j graph evidence (users, weaknesses, goals, topics).
- `retrieved`: FAISS vector matches (past messages, session history).
- Hit = if either source has relevant matches.

### Web Search Gate
```python
if req.allow_web_search and search_plan.should_search and search_plan.queries:
    web_future = RETRIEVAL_EXECUTOR.submit(...)
```
- Web search ONLY triggered if:
  1. User explicitly set `allow_web_search=true`.
  2. Search planner determined web is relevant (prep-guidance, current-info, etc.).
  3. Valid search queries were generated.

### Response Routing
```python
if not memory_hit and not web_search_used:
    # Memory miss → inform user
    answer = "I couldn't find a relevant memory record..."
else:
    # Memory hit or web hit → generate reply
    reply_bundle = generate_reply_bundle(...)
```

## Example Flows

### Scenario A: Google Interview Prep (Memory Hit)
1. User: "what should I study for my next interview Google"
2. System: Detects memory extraction needed; routes to retrieval.
3. Graph finds: `User -[TARGETS]-> Google`, `User -[STRUGGLES_WITH]-> Algorithms`
4. Vector finds: past messages on "Google interview tips"
5. Answer: "Based on your goal to target Google and your weakness in Algorithms, focus on: Data Structures (fundamental), System Design (critical), Behavior (practice)."
6. Response: `memory_found=true`, `retrieval_mode="memory_hit"`

### Scenario B: Interview Prep No Memory
1. User: "what should I study for my next interview Google"
2. System: No stored memory about Google or interviews.
3. Answer: "I couldn't find a relevant memory record for that question yet. If you want, you can re-send this query with `allow_web_search=true` to query the internet."
4. Response: `memory_found=false`, `suggest_web_search=true`, `retrieval_mode="memory_miss"`
5. User re-sends with `allow_web_search=true`.
6. System: Fetches live web results on Google interview prep.
7. Answer: "Google interviews typically focus on: System Design, Algorithms, Behavioral..."

### Scenario C: Tell Me My Weak Topics (Continuous Memory)
1. First call: User says "I struggle with Dynamic Programming"
   - Memory: Empty (no prior DP struggles recorded).
   - Answer: "I don't have memory of that yet."
   - **Background**: Extracts `User -[STRUGGLES_WITH]-> Dynamic Programming` and promotes to graph.

2. Second call (later): User asks "tell me my weak topics"
   - Memory: Graph now has DP weakness recorded.
   - Answer: "Your recorded weaknesses are: Dynamic Programming, System Design, ..."
   - Response: `memory_found=true`, `retrieval_mode="memory_hit"`

## Testing

### Test Case 1: New User Query (No Memory)
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "message": "what should I practice",
    "allow_web_search": false
  }'
```
Expected: `memory_found=false`, `suggest_web_search=true`

### Test Case 2: New User Query with Web Search
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "message": "what should I practice",
    "allow_web_search": true
  }'
```
Expected: `web_search_used=true` (if search is relevant), answer includes web facts.

### Test Case 3: Memory Growth Over Time
1. First message: "I'm struggling with dynamic programming" (store as memory)
2. Second message: "tell me my weaknesses"
   - Expected: System recalls DP weakness from message 1.

---

## Future Enhancements

1. **Caching**: Add memory-hit telemetry dashboard to track retrieval success rates.
2. **Adaptive web search**: Learn user preferences for when to auto-enable web search.
3. **Memory quality scoring**: Rank memory hits by freshness and relevance; deprioritize stale facts.
4. **Multi-turn context**: Track conversation context to improve memory queries across turns.
5. **Fallback strategies**: Offer clarifying follow-ups ("Did you mean topic X or Y?") when memory is ambiguous.

---

**Version**: 1.0  
**Date**: March 20, 2026  
**Status**: Live
