"""Deploy (or update) the server as a Docker Space on Hugging Face and verify it.

Environment:
    HF_TOKEN        Hugging Face access token with *write* permission
    HF_SPACE        target Space id, e.g. "your-hf-username/trading-mcp"
    MCP_AUTH_TOKEN  connector token (>= 24 chars); stored as a Space *secret*, never printed

The Space is public (Claude.ai must reach it); /mcp stays protected by MCP_AUTH_TOKEN.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = """---
title: Trading MCP
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: MCP server for market data, backtesting and ML research
---

Remote MCP server (Streamable HTTP at `/mcp`, health at `/health`).
Source: https://github.com/{github_repo}. Research and paper trading only - not investment advice.
"""


def main() -> None:
    hf_token, space, mcp_token = (os.environ.get(k, "") for k in ("HF_TOKEN", "HF_SPACE", "MCP_AUTH_TOKEN"))
    if not (hf_token and space and "/" in space):
        sys.exit("HF_TOKEN and HF_SPACE (owner/name) are required")
    if len(mcp_token) < 24:
        sys.exit("MCP_AUTH_TOKEN must be set (>= 24 characters)")
    api = HfApi(token=hf_token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.add_space_secret(space, "MCP_AUTH_TOKEN", mcp_token)
    api.add_space_variable(space, "MARKET_DATA_PROVIDER", "yahoo")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in ("Dockerfile", ".dockerignore", "pyproject.toml", "src"):
            src = ROOT / item
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, stage / item)
        (stage / "README.md").write_text(SPACE_README.format(github_repo=os.environ.get("GITHUB_REPOSITORY", "")))
        commit = api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space",
                                   commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
                                   delete_patterns=["*"])
        print("uploaded:", commit.commit_url if hasattr(commit, "commit_url") else commit)

    host = f"https://{space.replace('/', '-').replace('_', '-').replace('.', '-').lower()}.hf.space"
    time.sleep(45)  # let the Space leave the previous RUNNING state and start building the new commit
    deadline = time.time() + 20 * 60
    stage_name = "?"
    while time.time() < deadline:
        stage_name = api.get_space_runtime(space).stage
        print("space stage:", stage_name)
        if stage_name == "RUNNING":
            break
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            sys.exit(f"Space failed: {stage_name}. See build logs on huggingface.co/spaces/{space}")
        time.sleep(20)
    else:
        sys.exit(f"Space not running after 20 min (last stage {stage_name})")

    print("MCP URL:", f"{host}/mcp", "(append ?key=<MCP_AUTH_TOKEN>)")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"host={host}\n")
    env = {**os.environ, "MCP_VERIFY_TOKEN": mcp_token}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_endpoint.py"), host, "--retries", "12"],
                   check=True, env=env)


if __name__ == "__main__":
    main()
