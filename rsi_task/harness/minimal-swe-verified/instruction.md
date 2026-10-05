# Improve the minimal agent on SWE-bench Verified

Your goal is to make `Qwen/Qwen3.6-35B-A3B` solve as many SWE-bench Verified tasks as
possible by improving the agent harness that drives it. The model is fixed — it is served
behind the OpenAI-compatible endpoint in the config below — and so is the benchmark:
`swe_bench_verified_seed23_sample100`, a fixed 100-task sample of SWE-bench Verified. What
you may change is the harness: the complete, mutable minimal agent repository at
`/workspace/agent`.

On each benchmark task the harness receives a real GitHub issue as its instruction on stdin
and works inside a checkout of that issue's repository; the task scores 1 if the repository
passes the held-out tests afterwards, 0 otherwise. Every task is run three times and the
final score is the mean over all 300 runs.

You may improve any part of the harness, including but not limited to prompt design, tool use,
context and memory management, control flow, planning and execution strategies, feedback and
verification mechanisms, session management, and interactions between the model and the task
environment. These examples are intended only to clarify the scope of the task and do not limit
the approaches you may take; any other harness-level improvement is equally within the scope of
this task.

Before you design, execute, or revise a plan, use systematic deep research to fully
understand the goal, the task environment, the evaluation criteria, and the approaches that
already exist — gathering evidence from any credible, real source: the internet, code
repositories, public datasets, technical documentation, papers, and public discussion.

Your final archive is scored by one official evaluation with exactly this job config:

```json
{
  "n_attempts": 3,
  "agents": [{
    "name": "custom",
    "kwargs": {"install_cmd": "bash /agent/install.sh", "run_cmd": "bash /agent/run.sh"},
    "env": {
      "OPENAI_API_BASE": "$MODEL_SERVER_API_BASE",
      "OPENAI_API_KEY": "$MODEL_SERVER_API_KEY",
      "OPENAI_MODEL": "Qwen/Qwen3.6-35B-A3B"
    }
  }],
  "datasets": [{"name": "swe_bench_verified_seed23_sample100"}]
}
```

The final submission is the agent repository itself. Its archive root must contain readable
`install.sh` and `run.sh` files; `run.sh` must read the task instruction from stdin.

When finished, write the final repository as a tar.gz to `/workspace/final-agent.tar.gz`.
