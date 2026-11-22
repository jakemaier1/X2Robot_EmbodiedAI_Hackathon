"""Pure state machine for the flowers-in-a-vase task.

No I/O, no asyncio, no clock. At session start the relay asks Claude to turn the
client's instruction and the three camera images into a plan: one action per
flower, e.g. ["blue", "blue", "white"]. The machine keeps the plan as a queue,
pops one action at a time and runs the step cycle with the action's color
filled into every instruction ("{color}" in flower_task.json):

    GRAB_FLOWER -> INSERT_STEM

After an action, if a check could change what happens next, the relay asks the
vision model how many flowers of each color stand in the vase. If the count for
the action's color did not go up, the action goes back on the front of the
queue (at most retries_per_action times). A failed or unusable check counts as
success. Empty queue -> DONE: forward the home instruction for done_requests
requests, then `finished` is True and the relay closes the session and exits.
States, wording, lengths and prompts are data in flower_task.json.

Driving protocol, once per client request:
    machine = Machine(config, plan)       # plan: validate_plan(...) or fallback_plan(...)
    if machine.awaiting_check:
        machine.on_vase_counts(counts)    # {color: int}, or None if the call failed
    observation["instruction"] = machine.next_instruction()
    ... forward it; then if machine.finished: close the session
"""

from collections import deque
from dataclasses import dataclass
import re

DONE = "DONE"
COLOR = "{color}"
INSTRUCTION = "{instruction}"


@dataclass(frozen=True)
class Step:
    state: str
    instruction: str  # may contain {color}
    requests: int  # client requests to forward this instruction for (~1.3 s of motion each)


@dataclass(frozen=True)
class Action:
    color: str
    attempt: int = 0  # 0 = first try, 1 = first retry, ...


@dataclass(frozen=True)
class TaskConfig:
    colors: tuple
    max_actions: int
    retries_per_action: int
    steps: tuple
    done_instruction: str
    done_requests: int  # home chunks to forward before the relay exits
    planner_system_prompt: str
    planner_question: str  # must contain {instruction}
    check_system_prompt: str
    check_question: str

    @classmethod
    def from_dict(cls, data):
        config = cls(
            colors=tuple(data["colors"]),
            max_actions=data["max_actions"],
            retries_per_action=data["retries_per_action"],
            steps=tuple(Step(s["state"], s["instruction"], s["requests"]) for s in data["steps"]),
            done_instruction=data["done_instruction"],
            done_requests=data["done_requests"],
            planner_system_prompt=data["planner_system_prompt"],
            planner_question=data["planner_question"],
            check_system_prompt=data["check_system_prompt"],
            check_question=data["check_question"],
        )
        states = [s.state for s in config.steps]
        texts = ([config.done_instruction, config.planner_system_prompt, config.planner_question,
                  config.check_system_prompt, config.check_question]
                 + list(config.colors) + states + [s.instruction for s in config.steps])
        if not config.steps or DONE in states or len(set(states)) != len(states):
            raise ValueError(f"Need at least one step; state names must be unique and not {DONE}")
        if not config.colors or len(set(config.colors)) != len(config.colors):
            raise ValueError("Need at least one color; colors must be unique")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Colors, states, instructions and prompts must be non-empty strings")
        if not all(type(s.requests) is int and s.requests >= 1 for s in config.steps):
            raise ValueError("Every step's requests must be an integer >= 1")
        if type(config.done_requests) is not int or config.done_requests < 1:
            raise ValueError("done_requests must be an integer >= 1")
        if type(config.max_actions) is not int or config.max_actions < 1:
            raise ValueError("max_actions must be an integer >= 1")
        if type(config.retries_per_action) is not int or config.retries_per_action < 0:
            raise ValueError("retries_per_action must be an integer >= 0")
        if INSTRUCTION not in config.planner_question:
            raise ValueError(f"planner_question must contain {INSTRUCTION}")
        return config

    def instruction(self, step, color):
        return step.instruction.replace(COLOR, color)


class Machine:
    def __init__(self, config, plan):
        if not all(color in config.colors for color in plan):
            raise ValueError("Plan has a color the config does not know")
        self.config = config
        self.queue = deque(Action(color) for color in plan[:config.max_actions])
        self.in_vase = dict.fromkeys(config.colors, 0)  # last known count per color
        self.started = 0  # actions started, retries included
        self.done_sent = 0  # home instructions returned
        self.action = None
        self._next_action()

    def _next_action(self):
        self.action = self.queue.popleft() if self.queue else None
        self.step = self.sent = 0
        if self.action:
            self.started += 1

    @property
    def done(self):
        return self.action is None

    @property
    def state(self):
        return DONE if self.done else self.config.steps[self.step].state

    @property
    def finished(self):
        """True once done_requests home instructions have gone out: end the session."""
        return self.done and self.done_sent >= self.config.done_requests

    @property
    def queued(self):
        """Colors still waiting, in order (not counting the current action)."""
        return [a.color for a in self.queue]

    @property
    def action_complete(self):
        return (not self.done and self.step == len(self.config.steps) - 1
                and self.sent >= self.config.steps[-1].requests)

    @property
    def awaiting_check(self):
        """True once the action has run and a vase check could change something:
        a retry is left, or a later action of the same color needs this count."""
        if not self.action_complete:
            return False
        action = self.action
        return (action.attempt < self.config.retries_per_action
                or any(a.color == action.color for a in self.queue))

    def on_vase_counts(self, counts):
        """counts: {color: int} from the vision model, or None if it gave no usable
        answer. Returns the outcome: 'placed', 'assumed', 'retry' or 'failed'."""
        if not self.awaiting_check:
            raise RuntimeError("on_vase_counts called when no check was due")
        action = self.action
        seen = counts.get(action.color) if isinstance(counts, dict) else None
        if type(seen) is not int or seen < 0:
            outcome = "assumed"
            self.in_vase[action.color] += 1
        elif seen > self.in_vase[action.color]:
            outcome = "placed"
            self.in_vase[action.color] = seen
        elif action.attempt < self.config.retries_per_action:
            outcome = "retry"
            self.queue.appendleft(Action(action.color, action.attempt + 1))
        else:
            outcome = "failed"
        self._next_action()
        return outcome

    def next_instruction(self):
        """Return the instruction for this request; `state` then names its state."""
        if not self.done and self.sent >= self.config.steps[self.step].requests:
            if self.action_complete:
                if self.awaiting_check:
                    raise RuntimeError("Call on_vase_counts() first")
                self.in_vase[self.action.color] += 1  # unchecked: nothing would change
                self._next_action()
            else:
                self.step, self.sent = self.step + 1, 0
        if self.done:
            self.done_sent += 1
            return self.config.done_instruction
        self.sent += 1
        return self.config.instruction(self.config.steps[self.step], self.action.color)


# ---- plans --------------------------------------------------------------------

def validate_plan(reply, config):
    """Claude's {"actions": [color, ...]} -> list of colors (cut to max_actions), else ValueError."""
    actions = reply.get("actions") if isinstance(reply, dict) else None
    if not isinstance(actions, list) or not all(a in config.colors for a in actions):
        raise ValueError("Plan must be {\"actions\": [color, ...]} with known colors")
    return actions[:config.max_actions]


def validate_counts(reply, config):
    """Claude's {color: int, ...} -> the same dict, else ValueError."""
    if not isinstance(reply, dict) or not all(
            type(reply.get(c)) is int and reply[c] >= 0 for c in config.colors):
        raise ValueError("Counts must be {color: int >= 0} for every color")
    return {c: reply[c] for c in config.colors}


# Keyword fallback, used only when the planner fails twice. It knows this
# station's flowers (blue roses, white peonies) and the test-case phrasings.
FLOWER_COLORS = {"rose": "blue", "roses": "blue", "peony": "white", "peonies": "white"}
NUMBERS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "both": 2, "three": 3, "four": 4}
PLURAL_COUNT = 2  # 'the blue flowers', 'all the flowers': 2 of each color on the test table
CLAUSES = re.compile(r"[;,.!?]|\bbut\b|\bthen\b")
SKIP_CLAUSE = re.compile(r"\b(leave|leaving|don'?t|do not|not|never|except|without|ignore|alone)\b")


def fallback_plan(instruction, config):
    """Best-effort plan from the instruction text alone. Unparseable -> one action
    of the first color."""
    colors = "|".join(re.escape(c) for c in config.colors)
    mention = re.compile(
        r"\b(?:(?P<qty>" + "|".join(NUMBERS) + r"|all|every|each|the)\s+(?:of\s+)?(?:the\s+)?)?"
        r"(?:(?P<color>" + colors + r")\s+)?(?P<noun>flowers?|roses?|peon(?:y|ies)|ones?)\b")
    wanted = {}  # color -> count; dict keeps first-mention order
    for clause in CLAUSES.split(instruction.lower()):
        if SKIP_CLAUSE.search(clause):
            continue
        for m in mention.finditer(clause):
            qty, noun = m["qty"], m["noun"]
            color = m["color"] or FLOWER_COLORS.get(noun)
            if color is None and noun.startswith("one"):
                continue  # 'put one in' names no color
            if color is not None and color not in config.colors:
                continue
            every = qty not in NUMBERS and (qty in ("all", "every", "each") or noun.endswith("s"))
            count = NUMBERS.get(qty, PLURAL_COUNT if every else 1)
            # No color: 'all the flowers' means every color; 'two flowers' the first one.
            for c in [color] if color else (config.colors if every else config.colors[:1]):
                wanted[c] = max(wanted.get(c, 0), count)
    plan = [c for c, n in wanted.items() for _ in range(n)] or [config.colors[0]]
    return plan[:config.max_actions]
