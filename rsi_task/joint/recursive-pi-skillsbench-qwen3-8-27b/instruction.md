# Improve the agent system on SkillsBench

Your goal is to make an agent system solve as many SkillsBench tasks as possible by improving
the system as a whole. That system is you: the harness you are running as, installed at
`/agent` from the repository at `/workspace/agent`, and the checkpoint it drives — a fine-tune
of `Qwen/Qwen3.8-27B`, or `Qwen/Qwen3.8-27B` itself in the first round. What you submit becomes
the next round's agent. What you may change is the complete, mutable repository together with
the model, trained further on data you design; what is fixed is the weights you continue from
and the benchmark — `skillsbench_science_office_finance`, a fixed 37-task subset of SkillsBench.

This round lasts {hours} hours, after which the final files below are taken as they stand. Its
budget is ${budget}, which pays for everything on your key — training, every model_server
sample including this session's own turns, and benchmark jobs. `/workspace/memory` (keep the
directory) reaches the next round; its `rounds.jsonl` lists each earlier round's submitted
checkpoint, training state, and official reward.

On each benchmark task the harness receives the task's request as its instruction on stdin and
works inside a terminal prepared for that task, calling your checkpoint for every model turn;
the task scores 1 if the held-out tests pass afterwards, 0 otherwise. Each task's terminal
comes with reference skills for that task under `/root/.agents/skills`. Every task is run three
times and the final score is the mean over all 111 runs.

Before you design, execute, or revise a plan, use systematic deep research to fully understand
the goal, the task environment, the evaluation criteria, and the approaches that already exist
— gathering evidence from any credible, real source: the internet, code repositories, public
datasets, data factories, technical documentation, papers, and public discussion.

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
  "datasets": [{"name": "skillsbench_science_office_finance"}]
}
```

The next round runs your submission the way this round runs you: in a fresh copy of this
environment it uploads the archive, extracts it to `/agent`, runs `bash /agent/install.sh`, then
pipes this instruction into `bash /agent/run.sh` for the whole round. The official evaluation
uploads and installs the same archive in each benchmark task's container. Each install has
network access and a 30-minute limit, and nothing from your container but `/workspace/memory`
carries over. This session's own log is under `/logs/agent/pi/sessions`; leave it intact.

Training data (`dataset=custom`) is messages JSONL: one object per line with a non-empty
`messages` list. Any rollouts you need come from you as well: sample them from the checkpoint
you are running on (`$OPENAI_MODEL`) or from checkpoints you train from it, through
`$MODEL_SERVER_API_BASE` with `$MODEL_SERVER_API_KEY` — no other model.

Every training example you submit must be produced by you; run someone else's data pipeline if
you like, but do not train on existing agent trajectories or SFT data.

The checkpoint you submit must not be trained on evaluation data or anything derived from it —
the 37 tasks you are scored on, the rest of SkillsBench, any other benchmark's held-out set.
Its training data must be free of contamination in any form.

Every training run fixes these fields; the rest of the training config is yours:

```json
{
  "base_model": "Qwen/Qwen3.8-27B",
  "dataset": "custom",
  "renderer": "qwen3_8_xhigh_reasoning",
  "save_state": true
}
```

In the first round every run starts from `Qwen/Qwen3.8-27B`. From the second round on,
`init_checkpoint` is fixed too, to the training state of the checkpoint you are running on, and
`lora_config` (rank, `train_mlp`, `train_attn`, `train_unembed`) must match it; a run that does
not fails before its first step, at no cost, naming the expected values.

pi in print mode (`-p`) exits on a model reply with no tool call the model server can parse. If
this session exits while a final file is missing, it is restarted, at most three times; once
all three exist, an exit ends the round. A round that ends without all three ends the chain.

The final files:

- `/workspace/final-agent.tar.gz`: the agent repository as a tar.gz. Its archive root must
  contain readable `install.sh` and `run.sh` files; `run.sh` must read the task instruction
  from stdin and keep its agent's session log under `/logs/agent/pi/sessions`, the record of
  the run that the audit reads;
- `/workspace/final-checkpoint.txt`: the chosen `tinker://` checkpoint on one line — the
  `checkpoint` of a training run on your key, not its `state_checkpoint`;
- `/workspace/final-data.jsonl`: the data that produced it.
