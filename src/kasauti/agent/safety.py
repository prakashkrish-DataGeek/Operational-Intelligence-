"""Deterministic safety critic.

Classifies a request before any tool is called:

    green   retrieval of recorded facts
    yellow  general guidance or explanation
    orange  a request or recommendation to change the state of a well or plant
    red     defeating a safety barrier, or operating through a hazard signal

Orange and red require a human decision. The critic is a transparent lexicon,
not a model: it can be read, tested and argued with line by line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

GATED = ("orange", "red")

_EXPLAIN = re.compile(
    r"^\s*(what (is|are|does|do)\b|explain\b|describe\b|why\b|how (is|are|does|do)\b|"
    r"summari[sz]e\b|define\b|what'?s the difference)", re.I)
_RETRIEVE = re.compile(
    r"^\s*(what was|what were|how many|how much|when did|which\b|list\b|show\b|compare\b|"
    r"give\b|count\b|name\b|total\b|mean\b|at what\b|for\b)", re.I)
_DEFEAT = re.compile(
    r"\b(bypass|override|inhibit|disable|defeat|force|skip|jumper|silence|suppress)\w*\b", re.I)
_BARRIER = re.compile(
    r"\b(esd|psv|psd|alarm|interlock|safety valve|dhsv|scssv|detector|fire and gas|f&g|"
    r"pressure test|permit|barrier|trip)\w*\b", re.I)
_HAZARD = re.compile(
    r"(gas smell|smell of gas|leak|above (its|the) limit|sustained casing pressure|h2s alarm|"
    r"kick\b|blowout|without the permit|without a permit|say nothing|off the record)", re.I)
_DECIDE = re.compile(
    r"(\bcan (we|production|the well|it)\b|\bshould (we|i|the)\b|\btell the operator\b|"
    r"\bcarry on\b|\bcontinue\b|\bkeep (it|the well) (flowing|running)\b|\bbleed\b|\bstart up\b)", re.I)
_ACT = re.compile(
    r"^\s*(please\s+)?(open|close|shut|shut[- ]in|restart|start|stop|increase|decrease|reduce|raise|"
    r"lower|change|set|adjust|put|line up|bean up|bean down|choke back|ramp|bring|divert|switch)\b", re.I)
_RECOMMEND = re.compile(r"\b(recommend|give me a recommendation|advise (on|a) (new )?(setting|rate))\b", re.I)


@dataclass(frozen=True)
class Verdict:
    risk: str
    reason: str

    @property
    def requires_human(self) -> bool:
        return self.risk in GATED


def classify(text: str) -> Verdict:
    t = text.strip()
    explain = bool(_EXPLAIN.search(t))
    directive = bool(_ACT.search(t) or _DECIDE.search(t) or _RECOMMEND.search(t))
    if not explain:
        if _DEFEAT.search(t) and _BARRIER.search(t):
            return Verdict("red", "asks to defeat a safety barrier")
        if _HAZARD.search(t) and (directive or _DEFEAT.search(t)):
            return Verdict("red", "asks to operate through a hazard signal")
        if _HAZARD.search(t) and not _RETRIEVE.search(t):
            return Verdict("red", "hazard signal reported")
        if directive and not _RETRIEVE.search(t):
            return Verdict("orange", "asks for a change to operating state")
        if _RECOMMEND.search(t):
            return Verdict("orange", "asks for an operating recommendation")
    if explain:
        return Verdict("yellow", "general explanation")
    return Verdict("green", "retrieval of recorded data")
