"""Pure state machine for the flowers-in-a-bottle task (MVP).

No I/O, no asyncio, no clock. Each state forwards its instruction for a fixed
number of client requests, then moves on. No verification and no retries.
One cycle per blue flower:

    GRAB_BOTTLE -> GRAB_FLOWER -> INSERT_STEM -> PLACE_BOTTLE

After each cycle the relay asks the vision model whether any blue flower is
still on the table: yes, or no answer -> another cycle; no -> DONE (forward the
home instruction until the WebUI ends the session). The states, their wording
and their lengths are data in flower_task.json.

Driving protocol, once per client request:
    if machine.cycle_complete:
        machine.on_flowers_left(answer)   # True, False, or None if the call failed
    observation["instruction"] = machine.next_instruction()
"""

from dataclasses import dataclass

DONE = "DONE"


@dataclass(frozen=True)
class Step:
    state: str
    instruction: str
    requests: int  # client requests to forward this instruction for (~1.3 s of motion each)


@dataclass(frozen=True)
class TaskConfig:
    steps: tuple
    done_instruction: str
    flowers_left_question: str
    system_prompt: str

    @classmethod
    def from_dict(cls, data):
        config = cls(
            steps=tuple(Step(s["state"], s["instruction"], s["requests"]) for s in data["steps"]),
            done_instruction=data["done_instruction"],
            flowers_left_question=data["flowers_left_question"],
            system_prompt=data["system_prompt"],
        )
        states = [s.state for s in config.steps]
        texts = ([config.done_instruction, config.flowers_left_question, config.system_prompt]
                 + states + [s.instruction for s in config.steps])
        if not config.steps or DONE in states or len(set(states)) != len(states):
            raise ValueError(f"Need at least one step; state names must be unique and not {DONE}")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("States, instructions and prompts must be non-empty strings")
        if not all(type(s.requests) is int and s.requests >= 1 for s in config.steps):
            raise ValueError("Every step's requests must be an integer >= 1")
        return config


class Machine:
    def __init__(self, config):
        self.config = config
        self.flowers_done = 0  # completed cycles
        self.step = 0  # index into config.steps
        self.sent = 0  # requests forwarded so far in this step
        self.done = False

    @property
    def state(self):
        return DONE if self.done else self.config.steps[self.step].state

    @property
    def cycle_complete(self):
        """True once the last step has run: ask whether blue flowers are left."""
        return (not self.done and self.step == len(self.config.steps) - 1
                and self.sent >= self.config.steps[-1].requests)

    def on_flowers_left(self, left):
        """left: the vision model's True/False, or None if it gave no usable answer.
        Only an explicit False ends the task; None runs another cycle."""
        if not self.cycle_complete:
            raise RuntimeError("on_flowers_left called before the cycle was complete")
        self.flowers_done += 1
        self.step = self.sent = 0
        self.done = left is False

    def next_instruction(self):
        """Return the instruction for this request; `state` then names its state."""
        if self.done:
            return self.config.done_instruction
        if self.sent >= self.config.steps[self.step].requests:
            if self.cycle_complete:
                raise RuntimeError("Call on_flowers_left() first")
            self.step, self.sent = self.step + 1, 0
        self.sent += 1
        return self.config.steps[self.step].instruction
