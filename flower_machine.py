"""Pure state machine for the flowers-in-a-bottle task.

No I/O, no asyncio, no clock. The states, their wording, their lengths and the
scene classes are data in flower_task.json.

Each step forwards its instruction for `requests` client requests. Then the
relay asks the vision model to classify the scene into one of the configured
scenes, and the machine goes to that scene's `next` state:

    - a different step  -> enter it (fresh `requests` budget)
    - the current step  -> keep going, classify again after `reclassify_every`
    - DONE              -> forward done_instruction until the WebUI ends the session
    - no match / failed -> the `unclear` step (open the grippers) for its
                           `requests`, then back to the running step's `back`
                           step (default: the one before it in the list; the
                           first step goes back to itself); no classification
                           during or right after it

So the scene, not a fixed order, decides what runs next: a flower that drops
sends the robot back to the step that fixes it.

Driving protocol, once per client request:
    if machine.classify_due:
        machine.on_scene(scene_id)        # a configured scene id, or None
    observation["instruction"] = machine.next_instruction()
"""

from dataclasses import dataclass

DONE = "DONE"
NONE = "NONE"  # the answer for "no scene matches"


@dataclass(frozen=True)
class Step:
    state: str
    instruction: str
    requests: int  # requests to forward before classifying (~1.3 s of motion each)
    back: str = None  # step to return to after an unclear scene; None = the previous step


@dataclass(frozen=True)
class Scene:
    id: str
    description: str
    next: str  # a step's state, or DONE


@dataclass(frozen=True)
class TaskConfig:
    steps: tuple
    scenes: tuple       # in priority order: the first matching scene wins
    unclear: Step       # runs when no scene matches or the call fails, then steps back
    done_instruction: str
    system_prompt: str
    reclassify_every: int = 3

    @classmethod
    def from_dict(cls, data):
        config = cls(
            steps=tuple(Step(s["state"], s["instruction"], s["requests"], s.get("back"))
                        for s in data["steps"]),
            scenes=tuple(Scene(s["id"], s["description"], s["next"]) for s in data["scenes"]),
            unclear=Step(data["unclear"]["state"], data["unclear"]["instruction"],
                         data["unclear"]["requests"]),
            done_instruction=data["done_instruction"],
            system_prompt=data["system_prompt"],
            **({"reclassify_every": data["reclassify_every"]} if "reclassify_every" in data else {}),
        )
        states = [s.state for s in config.steps]
        all_states = states + [config.unclear.state]
        ids = [s.id for s in config.scenes]
        texts = ([config.done_instruction, config.system_prompt, config.unclear.instruction]
                 + all_states + ids
                 + [s.instruction for s in config.steps] + [s.description for s in config.scenes])
        if not config.steps or DONE in all_states or len(set(all_states)) != len(all_states):
            raise ValueError(f"Need at least one step; state names (including unclear) must be "
                             f"unique and not {DONE}")
        if not config.scenes or NONE in ids or len(set(ids)) != len(ids):
            raise ValueError(f"Need at least one scene; scene ids must be unique and not {NONE}")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("States, scene ids, instructions and descriptions must be non-empty strings")
        targets = set(states) | {DONE}
        bad = [s.next for s in config.scenes if s.next not in targets]
        if bad:
            raise ValueError(f"Scene next must be a step state or {DONE}: {bad}")
        bad = [s.back for s in config.steps if s.back is not None and s.back not in states]
        if bad:
            raise ValueError(f"Step back must be a step state: {bad}")
        if not all(type(n) is int and n >= 1 for n in [config.reclassify_every, config.unclear.requests]
                   + [s.requests for s in config.steps]):
            raise ValueError("reclassify_every and every step's requests must be integers >= 1")
        return config

    @property
    def classify_question(self):
        lines = [f"- {s.id}: {s.description}" for s in self.scenes]
        return ("Classify the current scene. Choose the FIRST scene in this list that matches "
                "what you see:\n" + "\n".join(lines)
                + f"\n- {NONE}: none of the scenes above clearly matches.\n"
                'Answer with JSON: {"reason": "<one short sentence>", "scene": "<scene id>"}')


class Machine:
    def __init__(self, config):
        self.config = config
        self._steps = {s.state: s for s in config.steps}
        self._steps[config.unclear.state] = config.unclear
        self._scenes = {s.id: s for s in config.scenes}
        self.resume = None  # step to go back to after the unclear step
        self._enter(config.steps[0].state)

    @property
    def classify_due(self):
        """True when the scene must be classified before the next request is forwarded."""
        return self.state not in (DONE, self.config.unclear.state) and self.wait <= 0

    def on_scene(self, scene_id):
        """scene_id: a configured scene id, or anything else (None, NONE, unknown) for
        no match. Returns the state the machine is in afterwards."""
        if not self.classify_due:
            raise RuntimeError("on_scene called when no classification was due")
        self.classifications += 1
        scene = self._scenes.get(scene_id)
        if scene is None:  # open the grippers, then go back
            states = [s.state for s in self.config.steps]
            self.resume = (self._steps[self.state].back
                           or states[max(states.index(self.state) - 1, 0)])
            self._enter(self.config.unclear.state)
        elif scene.next == self.state:
            self.wait = self.config.reclassify_every  # still needed: keep going, check again soon
        else:
            self._enter(scene.next)
        return self.state

    def next_instruction(self):
        """Return the instruction to forward for this request."""
        if self.state == DONE:
            return self.config.done_instruction
        if self.classify_due:
            raise RuntimeError("Classify the scene before forwarding")
        if self.state == self.config.unclear.state and self.wait <= 0:
            self._enter(self.resume)  # grippers opened: back to the earlier step, no classify
        self.wait -= 1
        return self._steps[self.state].instruction

    def _enter(self, state):
        self.state, self.classifications = state, 0
        self.wait = self._steps[state].requests if state != DONE else 0
