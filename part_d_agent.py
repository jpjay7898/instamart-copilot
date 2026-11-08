
import json
import time
from pathlib import Path

from mock_llm import chat
from part_d_tools import TOOLS_SPEC, TOOL_MAP


STATE_DIR = Path("state")
STATE_DIR.mkdir(exist_ok=True)


def save_checkpoint(state):
    """Save the complete agent state after every tool execution."""
    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)


def get_status(final_answer):
    """Extract the status from the beginning of the final answer."""
    if ":" in final_answer:
        return final_answer.split(":", 1)[0].strip()

    return "UNKNOWN"


def make_fallback(status, steps):
    """Create a short handoff message."""
    summary = ", ".join(step["tool"] for step in steps)

    if not summary:
        summary = "No tools executed"

    return (
        f"{status}: Agent stopped safely. "
        f"Tools completed: {summary}. "
        "Please continue with human support."
    )


def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    """Continue a ReAct agent from its current state."""

    start_time = time.monotonic()
    initial_tool_count = len(state["steps"])

    while state["llm_calls"] < max_iterations:

        # Time limit check before every model call
        if time.monotonic() - start_time >= time_limit_seconds:
            answer = make_fallback(
                "FALLBACK_TIMEOUT",
                state["steps"]
            )

            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"]
            }

        # Call the mock LLM
        response = chat(state["messages"], tools=TOOLS_SPEC)

        usage = response.usage.total_tokens
        state["tokens_used"] += usage
        state["llm_calls"] += 1

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Stop immediately if the token budget is exceeded
        if state["tokens_used"] > token_limit:
            answer = make_fallback(
                "FALLBACK_TOKEN_LIMIT",
                state["steps"]
            )

            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"]
            }

        # If no tool calls, the agent has finished
        if not tool_calls:
            final_answer = message.content or ""

            return {
                "status": get_status(final_answer),
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"]
            }

        # Save assistant tool-call message
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": []
        }

        for tool_call in tool_calls:
            assistant_message["tool_calls"].append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments)
                }
            })

        state["messages"].append(assistant_message)

        # Execute each requested tool
        for tool_call in tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments

            if tool_name not in TOOL_MAP:
                observation = {"error": "Unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            # Add tool observation to conversation
            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tool_call.id,
                "content": json.dumps(observation)
            })

            step = {
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)
            state["tools_executed"].append(tool_name)

            # Save checkpoint after every tool observation
            save_checkpoint(state)

            # Simulate crash after the requested number of new tools
            if (
                crash_after_tools is not None
                and len(state["steps"]) - initial_tool_count
                >= crash_after_tools
            ):
                raise RuntimeError("SIMULATED_CRASH")

    # Maximum iterations reached
    answer = make_fallback(
        "FALLBACK_MAX_ITERATIONS",
        state["steps"]
    )

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"]
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    """Start a new ReAct agent run."""

    state = {
        "run_id": scenario_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AGENT. Use the supplied tools to "
                    "process refund requests safely."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "tools_executed": []
    }

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Load a checkpoint and continue the interrupted agent."""

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run: {run_id}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        state = json.load(file)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )
