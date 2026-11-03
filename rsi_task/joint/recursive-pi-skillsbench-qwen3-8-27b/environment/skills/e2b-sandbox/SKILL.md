---
name: e2b-sandbox
description: Run code in disposable cloud sandboxes (e2b) — spin up an isolated Linux VM in seconds, execute commands, read and write files, run background services, and build reusable custom images. Use this skill whenever you need a scratch environment, want to execute experimental code safely outside your own container, or need many parallel isolated workers.
---

# E2B Sandbox

You can create disposable cloud sandboxes: isolated Linux VMs that boot in
seconds, run your commands, hold files, reach the internet, and are reclaimed
when their timeout expires. You drive them with the e2b Python SDK.

## Setup

`pip install e2b==2.39.1` (newer releases might not be supported by this endpoint). Two environment variables:

| Variable | Value |
|----------|-------|
| `E2B_API_URL` | Base URL of the sandbox endpoint |
| `E2B_API_KEY` | Your API key |

If either is not set, stop and ask the operator — do not guess.

Build the connection arguments once and pass them to every `Sandbox` /
`Template` classmethod:

```python
import os
from e2b import Sandbox

config = {
    "api_url": os.environ["E2B_API_URL"],
    "api_key": os.environ["E2B_API_KEY"],
    "validate_api_key": False,
}
```

## Demo

Creating a sandbox, running commands, working with files:

```python
sbx = Sandbox.create(timeout=300, **config)      # default "base" image
print(sbx.commands.run("echo hi").stdout)
sbx.files.write("/work/app.py", "print('hi')")
print(sbx.commands.run("python /work/app.py").stdout)
Sandbox.kill(sbx.sandbox_id, **config)
```

Building a custom image and starting sandboxes from it:

```python
from e2b import Template

tpl = Template().from_image("python:3.12-slim").run_cmd("pip install pytest")
Template.build(tpl, "dev-env", cpu_count=4, memory_mb=8192, **config)

sbx = Sandbox.create("dev-env", timeout=600, **config)
```


## Notes

- Refer to the e2b SDK documentation for the full surface.
- Sandbox forking, snapshots, and volumes are not available (403).
