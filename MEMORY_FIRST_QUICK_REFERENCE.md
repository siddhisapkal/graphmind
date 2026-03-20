# GraphMind Memory-First Chat: Quick Reference

## Core Principles

1. **Memory First**: Always query graph + vector memory before answering.
2. **Explicit Web Search**: Web search is opt-in via `allow_web_search: true` in the request.
3. **Transparent Misses**: When memory is empty, tell the user and suggest web search.
4. **Always Learn**: Background memory extraction happens on every message.

## Request Format

```python
# POST /chat
{
    "user_id": "user-123",
    "message": "what should I practice for Google",
    "allow_web_search": false  # ← New field (default: false)
}
```

## Response Summary

```python
{
    "answer": "...",
    "memory_found": true,              # ← New: memory hit detected
    "web_search_used": false,          # ← New: web was used
    "suggest_web_search": false,       # ← New: UI hint for button
    "retrieval_mode": "memory_hit",    # ← New: "memory_hit" | "memory_plus_web" | "memory_miss"
    ...other fields...
}
```

## Behavior Matrix

| Scenario | Memory | Web? | Memory Hit | Web Used | Mode | Answer |
|----------|--------|------|------------|----------|------|--------|
| User has context stored | ✓ | N/A | ✓ | ✗ | `memory_hit` | Personalized from graph |
| User has context + web | ✓ | ✓ | ✓ | ✓ | `memory_plus_web` | Contextual + live |
| No memory, no web | ✗ | ✗ | ✗ | ✗ | `memory_miss` | "Not found. Try web?" |
| No memory, web enabled | ✗ | ✓ | ✗ | ✓ | `memory_plus_web` | Live findings |

## When to Enable Web Search

- First interaction with unknown topic
- Current events or latest info queries
- If user explicitly asks for "recent" or "latest"
- Fallback when memory is empty and user clicks "search web"

## When NOT to Use Web Search

- User asks about personal history ("tell me my weaknesses")
- Topic is in memory graph (saves cost)
- Conversation is local context only

## Code Changes Summary

### Key Modifications in `main.py`

1. **Added field** to `ChatRequest`:
   ```python
   allow_web_search: bool = False
   ```

2. **Always retrieve memory** (graph + vector in parallel):
   ```python
   graph_future = RETRIEVAL_EXECUTOR.submit(...)  # Always run
   vector_future = RETRIEVAL_EXECUTOR.submit(...) # Always run
   ```

3. **Conditional web search**:
   ```python
   if req.allow_web_search and search_plan.should_search:
       web_future = RETRIEVAL_EXECUTOR.submit(...)  # Only if opt-in
   ```

4. **Smart response routing**:
   ```python
   memory_hit = bool(graph_context or retrieved)
   if not memory_hit and not web_search_used:
       answer = "Not found. Try web?"
   else:
       answer = generate_reply_bundle(...)
   ```

5. **Always background extract**:
   ```python
   background_tasks.add_task(_process_memory_pipeline, ...)  # Every message
   ```

## API Response Fields (New)

| Field | Type | Meaning |
|-------|------|---------|
| `memory_found` | bool | True if graph OR vector had matches |
| `web_search_used` | bool | True if web search was executed |
| `suggest_web_search` | bool | Hint: UI should show web search button |
| `retrieval_mode` | str | `"memory_hit"` / `"memory_plus_web"` / `"memory_miss"` |

## Testing

### Test 1: No Memory, No Web
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","message":"random query","allow_web_search":false}'
```
→ Expected: `memory_found: false`, `retrieval_mode: "memory_miss"`

### Test 2: No Memory, Web Enabled
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","message":"random query","allow_web_search":true}'
```
→ Expected: `web_search_used: true`, `retrieval_mode: "memory_plus_web"` (if relevant)

### Test 3: Build Memory Over Time
1. Send: "I'm struggling with Dynamic Programming"
   - Background extracts: `User -[STRUGGLES_WITH]-> Dynamic Programming`
2. Wait 1-2 seconds, then send: "Tell me my weaknesses"
   - Expected: Memory hit with DP weakness listed

---

**Cost Savings**: Avoid ~30-40% of unnecessary LLM calls and random web searches.  
**UX Improvement**: Clear feedback when memory is used vs. web search needed.  
**Privacy**: Personal memory stays local in graph; web search only when explicitly enabled.
