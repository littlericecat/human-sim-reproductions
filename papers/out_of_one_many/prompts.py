"""Conditioning texts. Study 3's interview template is the paper's Figure 10 / Table 11 wording verbatim; Study 2's
first-person backstory fragments follow Appendix C.1 ("phrasing selected to closely match the ANES, translated into
first-person declarations"; the paper does not print the fragments, so these are ours and are listed in REPORT.md)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from papers.out_of_one_many import data as D

QUESTIONS = {
    "gender": 'What is your gender? Are you "male" or "female"?',
    "race": 'I am going to read you a list of four race categories. What race do you consider yourself to be? "White", "Black", "Asian", or "Hispanic"?',
    "age": "What is your age in years?",
    "education": 'What is the highest level of school you have completed, or the highest degree you have received? Is it "high school", "some college", "a four-year college degree", or "an advanced degree"?',
    "patriotism": 'When you see the American flag flying, how does it make you feel? Does it make you feel "extremely good", "moderately good", "a little good", "neither good nor bad", "a little bad", "moderately bad", or "extremely bad"?',
    "discuss": 'Do you ever discuss politics with your family and friends? Please respond with "yes" or "no".',
    "interest": 'How interested would you say you are in politics? Are you "very interested", "somewhat interested", "not very interested", or "not at all interested"?',
    "ideology": 'In general, how would you describe your political views? Are you "extremely liberal", "liberal", "slightly liberal", "moderate", "slightly conservative", "conservative", or "extremely conservative"?',
    "pid7": 'Which would you say best describes your partisan identification. Would you say you are a "strong democrat", "not very strong democrat", "independent, but closer to the Democratic party", "independent", "independent, but closer to the Republican party", "not very strong Republican", or "strong Republican"?',
    "voted": 'Did you vote in the 2016 general election? Please answer with "yes" or "no".',
    "vote": 'Which presidential candidate did you vote for in the 2016 presidential election, "Hillary Clinton", "Donald Trump", or "someone else"?',
    "church": 'Lots of things come up that keep people from attending religious services even if they want to. Thinking about your life these days, do you ever attend religious services? Please respond with "yes" or "no".',
}
# Figure 10 order: gender, race, age, education, patriotism, discuss, interest, pid7, voted, vote, church (ideology is item 9 in Table 11's list;
# the figure omits it). We keep the figure's order and insert ideology after interest, as Table 11 lists it before partisanship.
ORDER = ["gender", "race", "age", "education", "patriotism", "discuss", "interest", "ideology", "pid7", "voted", "vote", "church"]
OPTIONS = {"gender": D.GENDER, "race": D.RACE, "education": {v: k for k, v in D.EDU.items()}, "church": D.CHURCH, "patriotism": D.PATRIOTISM,
           "discuss": D.DISCUSS, "interest": D.INTEREST, "ideology": D.IDEOLOGY, "pid7": D.PID7, "voted": {1: "yes", 0: "no"}, "vote": {1: "Hillary Clinton", 2: "Donald Trump", 3: "someone else"}}


def answer_text(item: str, code: Any) -> Optional[str]:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    if item == "age":
        return str(int(code))
    return OPTIONS[item].get(int(code))


def interview(row: dict[str, Any], target: str) -> str:
    """The Study 3 conditioning text: eleven answered items in order, the target question last, unanswered."""
    lines = []
    for item in ORDER:
        if item == target:
            continue
        if item == "vote" and row.get("voted") == 0:
            continue  # non-voters were not asked who they voted for
        a = answer_text(item, row.get(item))
        if a is None:
            continue  # missing ANES responses drop the fragment (Appendix D.1 footnote 2)
        lines.append(f"Interviewer: {QUESTIONS[item]}\nMe: {a}")
    lines.append(f"Interviewer: {QUESTIONS[target]}\nMe:")
    return "\n".join(lines)


def backstory(row: dict[str, Any]) -> str:
    """Study 2 first-person backstory (our fragments; the paper's are not printed) followed by the vote prompt."""
    parts = []
    r = answer_text("race", row.get("race"))
    if r:
        parts.append(f"Racially, I am {r}.")
    g = answer_text("gender", row.get("gender"))
    if g:
        parts.append(f"I am {g}.")
    a = answer_text("age", row.get("age"))
    if a:
        parts.append(f"I am {a} years old.")
    i = answer_text("ideology", row.get("ideology"))
    if i:
        parts.append(f"Ideologically, I am {i}.")
    p = answer_text("pid7", row.get("pid7"))
    if p:
        parts.append(f"Politically, I am a {p}." if not p.startswith("independent") else f"Politically, I am {p}.")
    it = answer_text("interest", row.get("interest"))
    if it:
        parts.append(f"I am {it} in politics.")
    c = answer_text("church", row.get("church"))
    if c:
        parts.append("I attend religious services." if c == "yes" else "I do not attend religious services.")
    d = answer_text("discuss", row.get("discuss"))
    if d:
        parts.append("I discuss politics with my family and friends." if d == "yes" else "I do not discuss politics with my family and friends.")
    f = answer_text("patriotism", row.get("patriotism"))
    if f:
        parts.append(f"When I see the American flag flying, it makes me feel {f}.")
    return " ".join(parts) + "\nIn 2016, I voted for"


# Appendix C.1 token sets for 2016 (lexical variants are matched case-insensitively after stripping spaces)
TRUMP_TOKENS = {"trump", "donald", "republican", "conservative"}
CLINTON_TOKENS = {"clinton", "hillary", "rodham", "senator", "democrat", "democratic", "liberal"}
