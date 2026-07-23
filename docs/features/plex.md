# PLEX serving policies

PLEX loads an operator-provided v0.6 WebAssembly policy package into a bounded
worker beside the vLLM V1 scheduler. The scheduler never waits for Wasm, JSON,
state access, or Python callbacks. It consumes an immutable cached schedule or
cache plan and immediately uses native scheduling when a plan is missing,
stale, unavailable, or failed.

## Installation and startup

Install the optional `pie-plex` 0.6 runtime, then pass a package. vLLM rejects
runtime releases outside `>=0.6,<0.7` or an incomplete `AsyncRuntime` API.

```bash
pip install "vllm[plex]"
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --plex-policy /path/to/policy.plexpkg
```

The offline API accepts the same configuration:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    plex_policy="/path/to/policy.plexpkg",
)
```

## Request metadata

OpenAI Chat, Completions, and Responses requests accept a `plex` object:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "plex": {
            "request_id": "call-42",
            "principal_id": "acme",
            "group_id": "workflow-42",
            "generation_id": 0,
            "terminal": False,
            "metadata": {"service_class": "interactive"},
        }
    },
)
```

The offline equivalent uses `SamplingParams.extra_args`.

`request_id` is stable across continuations. A later generation increments
`generation_id` by exactly one. `group_id` is optional and identifies a trusted
coordination scope in PLEX state; deployments must derive `principal_id` and
group authorization from authenticated ingress rather than blindly trusting
public request bodies.

`terminal=False` preserves request policy state for a continuation. Set
`close_group=True` on a terminal generation only when that generation closes
the whole group. `group_limits` may specify positive `max_members` and
`max_scratch_bytes`.

The legacy `logical_request_id` key is accepted as an alias for `request_id`
when only one is supplied.

## vLLM attachment semantics

| PLEX operation | vLLM attachment |
|---|---|
| `admit` | Not attached in-engine; use ingress or a router |
| `route` | Not attached; routing belongs to the deployment router |
| `schedule` | One-shot cached singleton selections and token budgets |
| `cache` | One-shot ordered request-level preemption decisions |
| `feedback` | Coalesced per-step progress, enactment, completion, abort, and preemption feedback |

The cache adapter presents one virtual reclaim unit per active request. It is a
request-preemption seam, not object/page-level cache admission. Prospective
cache admission and exact shared-block accounting are therefore unsupported by
this adapter.

vLLM does not advertise PLEX actions or
`schedule.atomic-enqueue@1`. Packages requiring those mechanics fail
attachment. Optional actions remain unavailable.

vLLM refuses to enact outcomes containing actions or mutations to the reserved
request `fields.body` mirror. Other request fields, shared state, group scratch,
and request scratch remain policy-private state and do not directly mutate a
vLLM `Request`. A rejected body mutation is replaced with the canonical host
value before the next policy invocation. Multi-request selection units are not
enacted because vLLM cannot guarantee their all-or-none allocation semantics.

## Staleness and fallback

Snapshots are published outside `Scheduler.schedule()`. A bounded Rust queue
serializes PLEX transactions and atomically publishes the latest result per
operation. The hot path only reads cached dictionaries.

Each plan is tied to one submitted opportunity and request-membership epoch. It
is consumed at most once, revalidated against current native feasibility, and
discarded after 250 ms. Progress and schedule snapshots are coalesced to at most
one publication window every 25 ms; cache snapshots are published only under KV
pressure. Request membership changes, preemption, and completion invalidate
older submissions. Queue-full, stale, unavailable, invalid, unsupported, and
failed outcomes all use native vLLM behavior.

After model output, vLLM reports committed input/output/service token deltas and
observed service time. It also reports whether each schedule selection and
request-level cache reclaim decision was enacted, partially enacted, or rejected
by current native constraints. Feedback is submitted before the next policy
snapshot whenever no lifecycle bootstrap is pending.

Each engine process owns an in-memory PLEX backend. Cross-process state requires
a distributed `PolicyStateBackend`.
