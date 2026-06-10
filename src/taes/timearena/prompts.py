"""LLM prompts for TimeArena TAES candidate generation."""


TAES_ACTION_PROMPT = """You are planning the next action for a time-management task. Given the current state, suggest {K} different actions.

=== TASKS ===
{task_descriptions}

=== CURRENT STATE ===
{state_summary}

=== AVAILABLE ACTIONS ===
{available_actions}

=== INSTRUCTIONS ===
Choose {K} different actions from the available actions list. Each should be a valid action string exactly matching one from the list.

Strategy tips:
- Prioritize starting non-blocking actions first (they run in background while you do other things)
- Start long-duration non-blocking tasks early
- Only do blocking actions when no non-blocking action is available or when it's the only remaining work
- "wait" should be used only when the agent must wait for a background task to complete
- Consider dependencies: some actions require others to finish first

Return a JSON array of exactly {K} action strings. Example:
["activate kettle", "wash cup", "wait"]

Return ONLY the JSON array, no other text.
"""


TAES_TRAJECTORY_PROMPT = """You are planning short action sequences for a time-management task. Given the current state, suggest {K} different candidate plans.

=== TASKS ===
{task_descriptions}

=== CURRENT STATE ===
{state_summary}

=== AVAILABLE FIRST ACTIONS ===
{available_actions}

=== INSTRUCTIONS ===
Return exactly {K} different candidate plans. Each candidate plan should be a JSON array of up to {H} action strings.

Rules:
- The first action in each candidate plan must exactly match one action from AVAILABLE FIRST ACTIONS.
- Later actions should be valid likely follow-ups based on dependencies, resources, and task goals.
- Prefer starting long non-blocking actions early, then use their running time to complete blocking prerequisites.
- Use "wait" only when no productive action is likely possible while background work finishes.
- Do not include explanations or markdown.

Return ONLY a JSON array of arrays. Example:
[
  ["wash dish", "pick rice", "cook rice in pot"],
  ["pick beef", "chop beef", "cook beef in pot"]
]
"""


def build_action_prompt(state, K: int = 3) -> str:
    available = state.get_available_actions()
    action_details = []
    for action in available:
        if action == 'wait':
            action_details.append("wait (skip this timestep)")
        else:
            parts = action.split()
            is_blocking = state.action_occupy.get(parts[0], True)
            blocking_str = "blocking - you must wait" if is_blocking else "non-blocking - runs in background"
            action_details.append(f"{action} ({blocking_str})")

    return TAES_ACTION_PROMPT.format(
        K=K,
        task_descriptions='\n'.join(f"- {desc}" for desc in state.task_descriptions),
        state_summary=state.summary(),
        available_actions='\n'.join(f"- {a}" for a in action_details),
    )


def build_trajectory_prompt(state, K: int = 3, H: int = 4) -> str:
    available = state.get_available_actions()
    action_details = []
    for action in available:
        if action == 'wait':
            action_details.append("wait (skip this timestep)")
        else:
            parts = action.split()
            if len(parts) >= 3:
                occupy_key = f"{parts[0]} {parts[2]}"
            else:
                occupy_key = parts[0]
            is_blocking = state.action_occupy.get(occupy_key, state.action_occupy.get(parts[0], True))
            blocking_str = "blocking - you must keep doing it until complete" if is_blocking else "non-blocking - runs in background"
            action_details.append(f"{action} ({blocking_str})")

    return TAES_TRAJECTORY_PROMPT.format(
        K=K,
        H=H,
        task_descriptions='\n'.join(f"- {desc}" for desc in state.task_descriptions),
        state_summary=state.summary(),
        available_actions='\n'.join(f"- {a}" for a in action_details),
    )
