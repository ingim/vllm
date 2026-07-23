# PLEX serving policies

PLEX loads an operator-provided WebAssembly policy package into an asynchronous
worker beside the vLLM scheduler. The scheduler never waits for Wasm. It
consumes cached standing request and retention plans, while lifecycle snapshots
and coalesced feedback are evaluated off the decode hot path.

## Installation and startup

Install the optional `pie-plex` runtime, then pass a `.plexpkg` file:

```bash
pip install "vllm[plex]"
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --plex-policy /path/to/policy.plexpkg
```

For a source checkout where `pie-plex` is not published on the configured
package index, install the PLEX Python binding directly before starting vLLM.

The offline API accepts the same configuration:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    plex_policy="/path/to/policy.plexpkg",
)
```

## Request metadata

The OpenAI Chat, Completions, and Responses APIs accept a structured `plex`
request field:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "plex": {
            "logical_request_id": "workflow-42",
            "generation_id": 0,
            "tenant": "acme",
        }
    },
)
```

The offline equivalent uses `SamplingParams.extra_args`:

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "plex": {
            "logical_request_id": "workflow-42",
            "generation_id": 0,
        }
    },
)
```

Metadata keys other than the reserved lifecycle keys become
`fields.metadata`. The adapter exposes `prompt_token_ids`, `max_tokens`, and
`priority` in `fields.body`.

## Logical request continuations

Ordinary requests are terminal and remove their PLEX state when generation
finishes. To preserve policy state for another model call, mark the generation
nonterminal:

```json
{
  "logical_request_id": "workflow-42",
  "generation_id": 0,
  "terminal": false,
  "completion_event": "tool-boundary"
}
```

The next request uses the same `logical_request_id` and increments
`generation_id` to `1`. PLEX receives a `continue` lifecycle event and retains
shared request fields and scratch from generation zero.

## Attachment semantics

| PLEX hook | vLLM attachment |
|---|---|
| `admit` | Not attached in-engine; use the deployment router or ingress |
| `schedule` | Cached standing plan produced after membership changes |
| `evict` | Cached request-level retention decision |
| `feedback` | Coalesced completion, abort, and preemption feedback |
| `route` | Not attached; routing belongs to the deployment router |

The scheduler never waits for PLEX. A missing, stale, unavailable, or failed
decision uses vLLM's native policy.
Invalid engine events and backend errors are surfaced instead of being silently
converted to native scheduling.

PLEX actions and per-token feedback are not advertised by this integration.
Each engine process owns
an in-memory PLEX runtime; cross-process or router-to-engine policy state
requires a distributed PLEX state backend.
