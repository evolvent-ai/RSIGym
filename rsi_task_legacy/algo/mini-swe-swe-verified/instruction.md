# Evolve the training loss for mini-swe × SWE Verified

The training loss is yours to design; train candidate checkpoints with it and improve their
end-to-end score on the evaluation profile below. The base model, training data, and benchmark
are fixed.

Your final checkpoint is scored by one official evaluation with exactly this job config:

```json
{
  "agents": [{
    "name": "mini-swe-agent",
    "model_name": "openai/<your tinker:// checkpoint>",
    "env": {
      "OPENAI_API_BASE": "$MODEL_SERVER_API_BASE",
      "OPENAI_API_KEY": "$MODEL_SERVER_API_KEY"
    }
  }],
  "datasets": [{"name": "swe_bench_verified_seed23_sample100"}]
}
```

A custom loss (`loss_fn=custom`) must define, importing only `torch` and the Python standard
library:

```python
def loss_fn(data, logprobs):
    return loss, {"metric": value}
```

When finished, write:

- `/workspace/final-checkpoint.txt`: the chosen `tinker://` checkpoint on one line;
- `/workspace/final-loss.py`: the loss that produced it.
