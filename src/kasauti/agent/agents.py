"""Agent interface and three bracketing agents.

Any agent, including one built on a language model, plugs in by implementing

    answer(question: str, tools: Toolbox) -> AgentResponse

The agents here contain no language model. `ReferenceAgent` shows that the
tasks are solvable with the tools and exercises the scorer end to end;
`UngatedAgent` and `AbstainAgent` show that the scorer tells agents apart.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import safety
from .tools import Toolbox, ToolError

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_WELL = re.compile(r"\b(\d{1,2}/\d{1,2}-[A-Z]-\d+(?: [A-D](?![a-z]))?)")
_PERIOD = re.compile(r"\b(?:(" + "|".join(MONTHS) + r")\s+)?((?:19|20)\d{2})\b", re.I)
_DEPTHS = re.compile(r"(\d{3,4}(?:\.\d+)?)\s*(?:m\b)?\s*(?:and|to)\s*(\d{3,4}(?:\.\d+)?)\s*m", re.I)


@dataclass
class AgentResponse:
    answer: float | str | None = None
    abstained: bool = False
    risk: str = "green"
    requires_human: bool = False
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class Agent(Protocol):
    name: str

    def answer(self, question: str, tools: Toolbox) -> AgentResponse: ...


def _abstain(why: str, risk: str = "green") -> AgentResponse:
    return AgentResponse(abstained=True, risk=risk, text=f"I cannot answer from the recorded data: {why}.")


class ReferenceAgent:
    name = "reference"
    gated = True

    def answer(self, question: str, tools: Toolbox) -> AgentResponse:
        verdict = safety.classify(question)
        if self.gated and verdict.requires_human:
            return AgentResponse(risk=verdict.risk, requires_human=True,
                                 text=f"Held for human decision ({verdict.risk}): {verdict.reason}.")
        start = len(tools.audit)
        resp = self._solve(question, tools)
        resp.risk = verdict.risk if self.gated else "green"
        resp.tool_calls = tools.audit[start:]
        return resp

    # ---------------------------------------------------------------- parsing
    def _solve(self, q: str, tools: Toolbox) -> AgentResponse:
        ql = q.lower()
        m = _WELL.search(q)
        well = m.group(1) if m else None
        try:
            if "measured depth" in ql or " md" in ql:
                return self._subsurface(q, ql, well, tools)
            p = _PERIOD.search(q)
            if not p:
                return _abstain("no period given")
            year = int(p.group(2))
            if p.group(1):
                mo = MONTHS[p.group(1).lower()]
                start, end = f"{year}-{mo:02d}-01", f"{year}-{mo:02d}-{calendar.monthrange(year, mo)[1]}"
            else:
                start, end = f"{year}-01-01", f"{year}-12-31"

            if "most oil" in ql or "top oil producer" in ql:
                return self._wrap(tools.call("top_producer", start=start, end=end, quantity="oil"))
            if well is None:
                return _abstain("no well named")
            if "water cut" in ql or "fraction of the liquid" in ql:
                return self._wrap(tools.call("water_cut", well=well, start=start, end=end))
            if "shut in" in ql or "no on-stream hours" in ql:
                return self._wrap(tools.call("shut_in_days", well=well, start=start, end=end))
            for phrase, sensor in (("wellhead pressure", "wellhead_pressure"), ("wellhead temperature", "wellhead_temperature"),
                                   ("downhole pressure", "downhole_pressure"), ("choke opening", "choke_opening")):
                if phrase in ql:
                    return self._wrap(tools.call("sensor_mean", well=well, start=start, end=end, sensor=sensor))
            if re.search(r"\b(rate|concentration|frequency)\b", ql):
                return _abstain("that quantity is not in the dataset")
            for word, quantity in (("oil", "oil"), ("water", "water"), ("gas", "gas")):
                if re.search(rf"\b{word}\b", ql):
                    return self._wrap(tools.call("production_total", well=well, start=start, end=end, quantity=quantity))
        except ToolError as e:
            return _abstain(str(e))
        return _abstain("question not understood")

    def _subsurface(self, q: str, ql: str, well: str | None, tools: Toolbox) -> AgentResponse:
        if well is None:
            return _abstain("no well named")
        d = _DEPTHS.search(q)
        if d:
            for phrase, curve in (("gamma ray", "GR"), ("bulk density", "RHOB"), ("compressional sonic", "DT")):
                if phrase in ql:
                    return self._wrap(tools.call("log_mean", well=well, curve=curve,
                                                 top_md=float(d.group(1)), base_md=float(d.group(2))))
        f = re.search(r"top of the (\w+) formation|of the (\w+) formation top", ql)
        if f:
            name = (f.group(1) or f.group(2)).capitalize()
            pick = "Hugin Fm. VOLVE Top" if name == "Hugin" else f"{name} Fm. Top"
            return self._wrap(tools.call("formation_top", well=well, pick=pick))
        return _abstain("question not understood")

    @staticmethod
    def _wrap(res) -> AgentResponse:
        if not res.has_data:
            return _abstain("no records for that request")
        return AgentResponse(answer=res.value, text=f"{res.value} {res.unit}".strip())


class UngatedAgent(ReferenceAgent):
    """Same retrieval skill, no safety critic."""
    name = "ungated"
    gated = False


class AbstainAgent:
    """Refuses everything. Safe, and useless."""
    name = "always_abstain"

    def answer(self, question: str, tools: Toolbox) -> AgentResponse:
        return AgentResponse(abstained=True, requires_human=True, risk="red", text="Refer to a human.")


BUILTIN = {a.name: a for a in (ReferenceAgent, UngatedAgent, AbstainAgent)}
