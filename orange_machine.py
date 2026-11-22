"""Pure state machine for the orange-item task.

No network, no asyncio, no clock: the caller passes `now` (seconds, any
monotonic origin) into the methods that need it. See orange_harness.py for the
relay that drives this machine and for the list of "open" decisions.

Per item: GRAB -> VERIFY_GRAB -> DROP -> VERIFY_DROP -> next item; then DONE.

Driving protocol, once per client request:
    question = machine.tick(now)          # non-None => a verify is due now
    if question:
        machine.on_verdict(verdict_or_None, now)
    forward machine.instruction to WALLX
    machine.forwarded()
    log machine.drain_events()
"""

from dataclasses import dataclass

GRAB, VERIFY_GRAB, DROP, VERIFY_DROP, DONE = "GRAB", "VERIFY_GRAB", "DROP", "VERIFY_DROP", "DONE"

_TEMPLATE_KEYS = ("grab_instruction", "drop_instruction", "grab_question", "drop_question")


@dataclass(frozen=True)
class Item:
    name: str
    grab_instruction: str
    drop_instruction: str
    grab_question: str
    drop_question: str


@dataclass(frozen=True)
class TaskConfig:
    items: tuple
    system_prompt: str
    done_instruction: str
    grab_pass: tuple
    drop_pass: tuple
    max_attempts: int = 3
    retry_max_attempts: int = 1
    time_budget_s: float = 420.0
    verify_after: int = 8
    reverify_every: int = 4

    @classmethod
    def from_dict(cls, data):
        """Build from the task JSON. Items are names, or dicts that override templates."""
        templates = data.get("templates", {})
        items = []
        for entry in data["items"]:
            spec = {"name": entry} if isinstance(entry, str) else dict(entry)
            name = spec.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each item needs a non-empty name")
            fields = {}
            for key in _TEMPLATE_KEYS:
                text = spec.get(key, templates.get(key))
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing {key} for item {name!r}")
                # replace(), not format(): prompts contain literal JSON braces.
                fields[key] = text.replace("{item}", name)
            items.append(Item(name=name, **fields))
        config = cls(
            items=tuple(items),
            system_prompt=data["system_prompt"],
            done_instruction=data["done_instruction"],
            grab_pass=tuple(data["grab_pass"]),
            drop_pass=tuple(data["drop_pass"]),
            **{k: data[k] for k in ("max_attempts", "retry_max_attempts", "time_budget_s",
                                    "verify_after", "reverify_every") if k in data},
        )
        if not config.items:
            raise ValueError("Task needs at least one item")
        if not config.grab_pass or not config.drop_pass:
            raise ValueError("grab_pass and drop_pass need at least one key")
        if config.max_attempts < 1 or config.retry_max_attempts < 0:
            raise ValueError("max_attempts must be >= 1 and retry_max_attempts >= 0")
        if config.verify_after < 0 or config.reverify_every < 0 or config.time_budget_s <= 0:
            raise ValueError("Cadence values must be >= 0 and time_budget_s > 0")
        return config


def judge(verdict, keys):
    """True if every key is exactly True, False if all are bools but one is not True,
    None (no verdict) if the verdict is not a dict or any key is missing/non-bool."""
    if not isinstance(verdict, dict):
        return None
    values = [verdict.get(k) for k in keys]
    if not all(isinstance(v, bool) for v in values):
        return None
    return all(values)


class Machine:
    def __init__(self, config, now):
        self.config = config
        self.started = now
        self.queue = list(config.items)  # queue[0] is the current item
        self.placed, self.skipped, self.failed = [], [], []
        self.round = 1  # 1 = main queue, 2 = retry of skipped items
        self.attempts = 0
        self.state = GRAB
        self.done_reason = None
        self._wait = config.verify_after  # forwarded requests left before the next verify
        self._events = []
        self._emit("start", items=[i.name for i in config.items])

    # ---- read-only views -------------------------------------------------

    @property
    def item(self):
        return self.queue[0] if self.state != DONE and self.queue else None

    @property
    def instruction(self):
        if self.state == DONE:
            return self.config.done_instruction
        if self.state in (GRAB, VERIFY_GRAB):
            return self.item.grab_instruction
        return self.item.drop_instruction

    @property
    def verify_question(self):
        if self.state == VERIFY_GRAB:
            return self.item.grab_question
        if self.state == VERIFY_DROP:
            return self.item.drop_question
        return None

    def elapsed(self, now):
        return now - self.started

    def snapshot(self, now=None):
        snap = {
            "state": self.state,
            "item": self.item.name if self.item else None,
            "attempts": self.attempts,
            "round": self.round,
            "placed": list(self.placed),
            "skipped": list(self.skipped),
            "failed": list(self.failed),
            "done_reason": self.done_reason,
        }
        if now is not None:
            snap["elapsed_s"] = round(self.elapsed(now), 2)
        return snap

    def drain_events(self):
        events, self._events = self._events, []
        return events

    # ---- driving ---------------------------------------------------------

    def tick(self, now):
        """Call once per client request. Returns the verify question if one is due."""
        self._check_time(now)
        if self.state in (GRAB, DROP) and self._wait <= 0:
            self._go(VERIFY_GRAB if self.state == GRAB else VERIFY_DROP, "verify_due")
            return self.verify_question
        return None

    def forwarded(self):
        """Call after a request was forwarded to WALLX with self.instruction."""
        if self.state in (GRAB, DROP):
            self._wait -= 1

    def on_verdict(self, verdict, now):
        """Feed the vision LLM's verdict dict, or None if the call failed."""
        if self.state not in (VERIFY_GRAB, VERIFY_DROP):
            raise RuntimeError(f"on_verdict called in state {self.state}")
        grabbing = self.state == VERIFY_GRAB
        result = judge(verdict, self.config.grab_pass if grabbing else self.config.drop_pass)
        if result is None:
            # No verdict: not an attempt. Keep the same instruction, ask again soon.
            self._wait = self.config.reverify_every
            self._go(GRAB if grabbing else DROP, "no_verdict")
        elif result and grabbing:
            self._wait = self.config.verify_after
            self._go(DROP, "grab_verified")
        elif result:
            self.placed.append(self.item.name)
            self._advance("drop_verified")
        else:
            self.attempts += 1
            self._note_recovery(grabbing, verdict)
            if self.attempts >= self._max_attempts():
                self._skip()
            elif grabbing:
                # Same instruction keeps running; WALLX may just need more time.
                self._wait = self.config.reverify_every
                self._go(GRAB, "grab_failed")
            else:
                self._wait = self.config.verify_after
                self._go(GRAB, "drop_failed")
        self._check_time(now)

    # ---- internals -------------------------------------------------------

    def _max_attempts(self):
        return self.config.max_attempts if self.round == 1 else self.config.retry_max_attempts

    def _skip(self):
        name = self.item.name
        (self.skipped if self.round == 1 else self.failed).append(name)
        self._advance("skipped" if self.round == 1 else "failed_after_retry")

    def _advance(self, reason):
        self.queue.pop(0)
        self.attempts = 0
        self._wait = self.config.verify_after
        if not self.queue and self.round == 1 and self.skipped and self.config.retry_max_attempts > 0:
            self.round = 2
            self.queue, self.skipped = list(self._items(self.skipped)), []
            self._emit("retry_skipped", items=[i.name for i in self.queue])
        if self.queue:
            self._go(GRAB, reason)
        else:
            self._finish(reason, "queue_complete")

    def _items(self, names):
        by_name = {i.name: i for i in self.config.items}
        return [by_name[n] for n in names]

    def _check_time(self, now):
        if self.state != DONE and self.elapsed(now) >= self.config.time_budget_s:
            self._finish("time_budget", "time_budget")

    def _finish(self, reason, done_reason):
        # Items not reached are left in queue for the log; DONE ignores them.
        self.done_reason = done_reason
        self._go(DONE, reason, unfinished=[i.name for i in self.queue])

    def _note_recovery(self, grabbing, verdict):
        # v1 has no recovery states; only log situations that would need one.
        if grabbing and verdict.get("holding") is True and verdict.get("is_target") is False:
            self._emit("recovery_not_built", situation="holding a non-target object")
        elif not grabbing:
            self._emit("recovery_not_built",
                       situation="drop failed; item may be outside the basket or still held")

    def _go(self, to, reason, **extra):
        before, self.state = self.state, to
        self._emit("transition", **{"from": before, "to": to, "reason": reason}, **extra)

    def _emit(self, event, **fields):
        self._events.append({
            "event": event,
            "item": self.item.name if self.item else None,
            "attempts": self.attempts,
            "round": self.round,
            "skipped": list(self.skipped),
            **fields,
        })
