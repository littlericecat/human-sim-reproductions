"""A text sandbox of Smallville (paper §5): sectors and arenas from the official location tables, each agent's
home from its persona, a shared clock, and who is where. No tile map: agents are either at their home arena or
at a public place; two agents perceive each other when they are in the same arena at the same time step."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

DATA = Path(__file__).parent / "data"

PUBLIC_PLACES = [
    "Hobbs Cafe: cafe", "Oak Hill College: classroom", "Oak Hill College: library", "Oak Hill College: hallway",
    "Johnson Park: park", "Harvey Oak Supply Store: supply store", "The Willows Market and Pharmacy: store",
    "The Rose and Crown Pub: pub", "artist's co-living space: common room", "artist's co-living space: kitchen",
]

PLACE_KEYWORDS = [  # fallback when the model does not pick a place from the list
    (("cafe", "coffee", "hobbs", "pastr", "lunch", "brunch"), "Hobbs Cafe: cafe"),
    (("library", "research", "study", "studying", "read"), "Oak Hill College: library"),
    (("class", "lecture", "college", "school", "teach"), "Oak Hill College: classroom"),
    (("park", "walk", "jog", "run ", "stroll"), "Johnson Park: park"),
    (("pub", "bar", "beer", "drink"), "The Rose and Crown Pub: pub"),
    (("grocer", "market", "pharmacy", "shopping", "supplies", "buy"), "The Willows Market and Pharmacy: store"),
    (("hardware", "supply store", "tools"), "Harvey Oak Supply Store: supply store"),
]
HOME_KEYWORDS = ("sleep", "bed", "wake", "morning routine", "shower", "breakfast", "dinner", "home", "relax", "rest", "cook", "nap",
                 "dorm", "her room", "his room", "in the room", "stay in", "apartment", "house")


def load_places(data_dir: Path = DATA) -> dict[str, list[str]]:
    """sector -> arenas, from the official csv tables."""
    out: dict[str, list[str]] = {}
    with (data_dir / "arena_blocks.csv").open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                out.setdefault(row[2].strip(), []).append(row[3].strip())
    return out


def home_of(living_area: str) -> str:
    """'the Ville:artist's co-living space:Abigail Chen's room' -> "artist's co-living space: Abigail Chen's room"."""
    parts = [p.strip() for p in living_area.split(":")]
    if len(parts) >= 3:
        return f"{parts[1]}: {parts[2]}"
    return parts[-1]


def sector_of(place: str) -> str:
    return place.split(":")[0].strip()


def place_for_activity(activity: str, home: str, places: Iterable[str] = PUBLIC_PLACES) -> str:
    a = activity.lower()
    if any(k in a for k in HOME_KEYWORDS):
        return home
    for keys, place in PLACE_KEYWORDS:
        if any(k in a for k in keys) and place in set(places):
            return place
    return home


@dataclasses.dataclass
class Clock:
    now: dt.datetime
    step_minutes: int = 30

    def tick(self) -> dt.datetime:
        self.now = self.now + dt.timedelta(minutes=self.step_minutes)
        return self.now

    def text(self) -> str:
        return self.now.strftime("%A %B %d, %Y, %I:%M %p").replace(" 0", " ")


class World:
    def __init__(self, places: Optional[dict[str, list[str]]] = None) -> None:
        self.places = places if places is not None else load_places()
        self.location: dict[str, str] = {}   # agent name -> "sector: arena"
        self.action: dict[str, str] = {}     # agent name -> current activity text

    def move(self, name: str, place: str, action: str) -> None:
        self.location[name] = place
        self.action[name] = action

    def agents_at(self, place: str, exclude: Optional[str] = None) -> list[str]:
        return sorted(n for n, p in self.location.items() if p == place and n != exclude)
