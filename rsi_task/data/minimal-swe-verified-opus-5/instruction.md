# Evolve the training data for the minimal agent on SWE-bench Verified

Your goal is to make a fine-tune of `Qwen/Qwen3.5-35B-A3B-Base` solve as many SWE-bench Verified
tasks as possible by designing its training data. Train checkpoint candidates from that
base model on your data; everything else is fixed: the base model, the training loss, the
stock minimal agent harness that drives the checkpoint, and the benchmark —
`swe_bench_verified_seed23_sample100`, a fixed 100-task sample of SWE-bench Verified.

On each benchmark task the harness receives a real GitHub issue and works inside a checkout
of that issue's repository, calling your checkpoint for every model turn; the task scores 1
if the repository passes the held-out tests afterwards, 0 otherwise. Every task is run three
times and the final score is the mean over all 300 runs.

Before you design, execute, or revise a plan, use systematic deep research to fully
understand the goal, the task environment, the evaluation criteria, and the approaches that
already exist — gathering evidence from any credible, real source: the internet, code
repositories, public datasets, data factories, technical documentation, papers, and public
discussion.

Your final checkpoint is scored by one official evaluation with exactly this job config:

```json
{
  "n_attempts": 3,
  "agents": [{
    "name": "minimal",
    "env": {
      "OPENAI_API_BASE": "$MODEL_SERVER_API_BASE",
      "OPENAI_API_KEY": "$MODEL_SERVER_API_KEY",
      "OPENAI_MODEL": "<your tinker:// checkpoint>"
    }
  }],
  "datasets": [{"name": "swe_bench_verified_seed23_sample100"}]
}
```

Training data (`dataset=custom`) is messages JSONL: one object per line with a non-empty
`messages` list. If you want rollouts from a stronger model, an OpenAI-compatible gateway is
available at `$ROLLOUT_SERVER_URL`.

Every training example you submit must be produced by you; run someone else's data pipeline if
you like, but do not train on existing agent trajectories or SFT data.

The checkpoint you submit must not be trained on evaluation data or anything derived from it —
the 100 tasks you are scored on, the rest of SWE-bench Verified, any other benchmark's held-out
set. Its training data must be free of contamination in any form.

Every training run uses exactly this training config:

```json
{
  "base_model": "Qwen/Qwen3.5-35B-A3B-Base",
  "dataset": "custom",
  "loss_fn": "cross_entropy",
  "num_epochs": 1,
  "batch_size": 16,
  "max_length": 65536,
  "train_on_what": "all_assistant_messages",
  "lr_schedule": "linear",
  "lora_config": {"rank": 16, "seed": 0, "train_unembed": true, "train_mlp": true, "train_attn": true},
  "adam_params": {"learning_rate": 3e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
                  "weight_decay": 0.0, "grad_clip_norm": 0.0}
}
```

When finished, write:

- `/workspace/final-checkpoint.txt`: the chosen `tinker://` checkpoint on one line;
- `/workspace/final-data.jsonl`: the data that produced it.
