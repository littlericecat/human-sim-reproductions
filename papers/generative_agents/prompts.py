"""Prompt texts. Where the paper prints a prompt (§4.1 importance, §4.2 reflection, §4.3 daily plan, §4.3.1
reaction) the wording is kept; the rest follows the paper's description. Every prompt is a plain string so the
cassettes are stable."""

from __future__ import annotations

IMPORTANCE = ("On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely "
              "poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\n"
              "Memory: {memory}\nRating: ")

REFLECTION_QUESTIONS = ("{statements}\n\nGiven only the information above, what are 3 most salient high-level questions we can "
                        "answer about the subjects in the statements? Write one question per line, numbered 1-3.")

REFLECTION_INSIGHTS = ("Statements about {name}\n{statements}\n\nWhat 5 high-level insights can you infer from the above statements? "
                       "(example format: insight (because of 1, 5, 3))\nWrite one insight per line, numbered 1-5.")

DAILY_PLAN = ("Name: {name} (age: {age})\nInnate traits: {innate}\n{learned} {currently} {lifestyle}\n"
              "{yesterday}Today is {today}. Here is {first}'s plan today in broad strokes, as 5 to 8 numbered items each with a "
              "start time (e.g. '1) wake up and complete the morning routine at 7:00 am'):\n1)")

HOURLY_PLAN = ("{summary}\nToday is {today}. {first}'s plan for the day in broad strokes:\n{broad}\n\n"
               "Break the plan into hourly activities from {start} to {end}. Write one line per hour in the exact form "
               "'HH:MM - activity' (24-hour clock), nothing else.")

CHOOSE_LOCATION = ("{name} is going to: {activity}.\n{name} lives at {home}. Known places in town: {places}.\n"
                   "Which single place from the list is the most appropriate? Answer with the place name exactly as written in the list.")

# §4.3.1, verbatim structure; the answer is free text and is read for a decision to converse.
REACT = ("{summary}\nIt is {now}.\n{name}'s status: {status}\nObservation: {observation}\n"
         "Summary of relevant context from {name}'s memory: {context}\n"
         "Should {name} react to the observation, and if so, what would be an appropriate reaction? Answer in one or two sentences.")

CONTEXT_SUMMARY = ("{name}'s relevant memories:\n{memories}\n\nIn two sentences, summarize what {name} knows about {other} and about: "
                   "{observation}")

DIALOGUE_TURN = ("{summary}\nIt is {now}. {name} is at {place}.\n{name}'s memories relevant to this conversation:\n{memories}\n"
                 "{name} and {other} are talking.\n{history}\nWhat does {name} say next? Reply with one or two sentences of "
                 "dialogue only, in the voice of {name}, no name prefix. If the conversation has naturally ended, reply exactly: [END]")

INTERVIEW = ("{summary}\nIt is {now}.\n{name}'s relevant memories:\n{memories}\n\n"
             "Interviewer: {question}\n{name}: ")

SUMMARY_DESCRIPTION = "Name: {name} (age: {age})\nInnate traits: {innate}\n{learned}\n{currently}\n{lifestyle}"
