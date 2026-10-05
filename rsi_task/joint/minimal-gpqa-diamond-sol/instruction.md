# Improve the agent system on GPQA Diamond

Your goal is to make an agent system answer as many GPQA Diamond questions as possible by
improving the system as a whole. What you may change is the complete, mutable minimal agent
repository at `/workspace/agent` together with the model it drives, a fine-tune of
`Qwen/Qwen3.5-35B-A3B-Base` trained on data you design; what is fixed is the base model you
start from and the benchmark — `gpqa_diamond_seed23_sample100`, a fixed 100-question sample
of GPQA Diamond.

On each benchmark task the harness receives the question as its instruction on stdin and
works inside a terminal prepared for that task, calling your checkpoint for every model turn;
the task scores 1 if the answer is correct, 0 otherwise. Every task is run three times and
the final score is the mean over all 300 runs.

Before you design, execute, or revise a plan, use systematic deep research to fully
understand the goal, the task environment, the evaluation criteria, and the approaches that
already exist — gathering evidence from any credible, real source: the internet, code
repositories, public datasets, data factories, technical documentation, papers, and public
discussion.

Your final system is scored by one official evaluation with exactly this job config:

```json
{
  "n_attempts": 3,
  "agents": [{
    "name": "custom",
    "kwargs": {"install_cmd": "bash /agent/install.sh", "run_cmd": "bash /agent/run.sh"},
    "env": {
      "OPENAI_API_BASE": "$MODEL_SERVER_API_BASE",
      "OPENAI_API_KEY": "$MODEL_SERVER_API_KEY",
      "OPENAI_MODEL": "<your tinker:// checkpoint>"
    }
  }],
  "datasets": [{"name": "gpqa_diamond_seed23_sample100"}]
}
```

Training data (`dataset=custom`) is messages JSONL: one object per line with a non-empty
`messages` list. If you want rollouts from a stronger model, an OpenAI-compatible gateway is
available at `$ROLLOUT_SERVER_URL`.

Every training example you submit must be produced by you; run someone else's data pipeline if
you like, but do not train on existing agent trajectories or SFT data.

The checkpoint you submit must not be trained on evaluation data or anything derived from it —
the 100 questions you are scored on, the rest of GPQA, any other benchmark's held-out set.
Its training data must be free of contamination in any form.

Every training run fixes exactly these three fields; the rest of the training config is yours:

```json
{
  "base_model": "Qwen/Qwen3.5-35B-A3B-Base",
  "dataset": "custom",
  "renderer": "qwen3_5_preserve_thinking"
}
```

When finished, write:

- `/workspace/final-agent.tar.gz`: the agent repository as a tar.gz. Its archive root must
  contain readable `install.sh` and `run.sh` files; `run.sh` must read the task instruction
  from stdin;
- `/workspace/final-checkpoint.txt`: the chosen `tinker://` checkpoint on one line;
- `/workspace/final-data.jsonl`: the data that produced it.
