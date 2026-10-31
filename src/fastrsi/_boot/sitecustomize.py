"""Starts the agent in every Python process that inherits FASTRSI_AGENT_DIR."""

import os

if os.environ.get("FASTRSI_AGENT_DIR"):
    from fastrsi import agent

    agent.boot(os.environ["FASTRSI_AGENT_DIR"])
