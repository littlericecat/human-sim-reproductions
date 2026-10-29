"""Load and recode the CC0 replication file (2016 ANES items + GPT-3 answers).

Codes follow the ANES 2016 Time Series codebook; negative codes are missing. Education is collapsed to the four
bins of the paper's template (high school / some college / a four-year college degree / an advanced degree); the
`*_gpt3` columns use the same bins coded 9 / 12 / 13 / 16. Vote turnout and choice are combined into one variable
as in the paper's analysis (Clinton / Trump / someone else / did not vote).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA = Path(__file__).parent / "data" / "anesgpt3_task3.csv"

GENDER = {1: "male", 2: "female"}
RACE = {1: "white", 2: "black", 3: "asian", 5: "hispanic"}      # the paper's four categories (ANES 4 = native american, 6 = other: dropped)
EDU = {"high school": 1, "some college": 2, "a four-year college degree": 3, "an advanced degree": 4}
EDU_GPT3 = {9: 1, 12: 2, 13: 3, 16: 4}
CHURCH = {1: "yes", 2: "no"}
PATRIOTISM = {1: "extremely good", 2: "moderately good", 3: "a little good", 4: "neither good nor bad", 5: "a little bad", 6: "moderately bad", 7: "extremely bad"}
DISCUSS = {1: "yes", 2: "no"}
INTEREST = {1: "very interested", 2: "somewhat interested", 3: "not very interested", 4: "not at all interested"}
IDEOLOGY = {1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate", 5: "slightly conservative", 6: "conservative", 7: "extremely conservative"}
PID7 = {1: "strong democrat", 2: "not very strong democrat", 3: "independent, but closer to the Democratic party", 4: "independent",
        5: "independent, but closer to the Republican party", 6: "not very strong Republican", 7: "strong Republican"}
VOTE = {1: "Hillary Clinton", 2: "Donald Trump", 3: "someone else", 4: "someone else", 5: "someone else"}

ITEMS = ["gender", "race", "age", "education", "church", "patriotism", "discuss", "interest", "ideology", "pid7", "voted", "vote"]
ANES_COL = {"gender": "V161342", "race": "V161310x", "age": "V161267", "education": "V161270", "church": "V161244", "patriotism": "V162125x",
            "discuss": "V162174", "interest": "V162256", "ideology": "V161126", "pid7": "V161158x", "voted": "V162031x", "vote": "V162062x"}
GPT3_COL = {"gender": "gender_gpt3", "race": "race_gpt3", "age": "age_gpt3", "education": "education_gpt3", "church": "church_goer_gpt3",
            "patriotism": "patriotism_gpt3", "discuss": "discuss_politics_gpt3", "interest": "political_interest_gpt3", "ideology": "ideology_gpt3",
            "pid7": "pid7_gpt3", "voted": "voted_2016_gpt3", "vote": "votechoice_2016_gpt3"}


def edu_bin(code: int) -> Optional[int]:
    if code < 1:
        return None
    if code <= 9:
        return 1          # up to high-school graduate / GED
    if code <= 12:
        return 2          # some college, associate degrees
    if code == 13:
        return 3          # bachelor's
    return 4              # master's, professional, doctorate


def recode_human(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-coded items with NaN for missing; codes are the template option indices (see the dicts above)."""
    out = pd.DataFrame({"id": df["V160001_orig"]})
    def keep(col, valid):
        return df[col].where(df[col].isin(list(valid)))
    out["gender"] = keep("V161342", GENDER)
    out["race"] = keep("V161310x", RACE)
    out["age"] = df["V161267"].where(df["V161267"] >= 18)
    out["education"] = df["V161270"].map(lambda c: edu_bin(int(c)) if c >= 1 else None)
    out["church"] = keep("V161244", CHURCH)
    out["patriotism"] = keep("V162125x", PATRIOTISM)
    out["discuss"] = keep("V162174", DISCUSS)
    out["interest"] = keep("V162256", INTEREST)
    out["ideology"] = keep("V161126", IDEOLOGY)
    out["pid7"] = keep("V161158x", PID7)
    out["voted"] = df["V162031x"].where(df["V162031x"].isin([0, 1]))
    out["vote"] = keep("V162062x", VOTE).map(lambda v: {1: 1, 2: 2, 3: 3, 4: 3, 5: 3}.get(v, v) if pd.notna(v) else v)
    return out


def recode_gpt3(df: pd.DataFrame) -> pd.DataFrame:
    """GPT-3's answers in the same coding (-1 = no parsable answer -> NaN)."""
    out = pd.DataFrame({"id": df["V160001_orig"]})
    for item in ITEMS:
        col = df[GPT3_COL[item]]
        if item == "education":
            out[item] = col.map(lambda c: EDU_GPT3.get(int(c)) if c > 0 else None)
        elif item == "age":
            out[item] = col.where(col >= 18)
        elif item == "vote":
            out[item] = col.where(col.isin([1, 2, 42])).map(lambda v: 3 if v == 42 else v)   # 42 is the package's code for a non-matching / other answer
        elif item == "voted":
            out[item] = col.where(col.isin([0, 1]))
        else:
            out[item] = col.where(col > 0)
    return out


def vote_combined(voted: pd.Series, vote: pd.Series) -> pd.Series:
    """The paper's single vote.2016 variable: 0 = did not vote, 1 = Clinton, 2 = Trump, 3 = someone else."""
    out = vote.copy()
    out[(voted == 0)] = 0
    return out


def age_band(age: float) -> Optional[str]:
    if pd.isna(age):
        return None
    return "18 to 30 years old" if age <= 30 else "31 to 45 years old" if age <= 45 else "46 to 60 years old" if age <= 60 else "Over 60"


def load(path: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    return raw, recode_human(raw), recode_gpt3(raw)
