"""Synthetic chats for the demo, tests and benchmarks.

Every name, number and message here is invented. No real conversation data is
used anywhere in this repository.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

SAMPLE_FILENAME = "synthetic-sample-chat.txt"

# A small, hand-written Android-style group chat (DD/MM/YYYY, 24-hour).
SAMPLE_CHAT = """\
02/03/2024, 09:00 - Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more.
02/03/2024, 09:00 - Asha created group "Weekend Hike 🥾"
02/03/2024, 09:01 - Asha added Bruno and Chen Wei
02/03/2024, 09:02 - Asha: Morning all! Planning the hike for Saturday 🌄
02/03/2024, 09:05 - Bruno: Sounds great 👍🏽 What time?
02/03/2024, 09:06 - Asha: Let's meet at 7:30 at the north gate.
Bring water and snacks.
02/03/2024, 09:20 - Chen Wei: I can drive two people 🚗
02/03/2024, 09:21 - Chen Wei: <Media omitted>
02/03/2024, 09:40 - Bruno: Decision: we take Chen's car + the bus. Can someone check the bus times?
02/03/2024, 10:02 - Asha: I'll check https://example.org/bus-times and report back
02/03/2024, 18:30 - Asha: Bus leaves 07:05 🚌 ✅
03/03/2024, 08:15 - Bruno: This message was deleted
03/03/2024, 08:16 - Bruno: Do we need permits? 🤔
03/03/2024, 12:45 - Chen Wei: No permit needed for day hikes 🇳🇿
05/03/2024, 20:10 - Asha: Weather looks good! ☀️☀️
05/03/2024, 20:11 - Bruno: Great, see you Saturday 🙌
05/03/2024, 20:12 - Chen Wei: 👨‍👩‍👧 family might join too
09/03/2024, 07:25 - Asha: At the gate now
09/03/2024, 07:31 - Bruno: Running 2 min late, sorry!
09/03/2024, 07:33 - Chen Wei: Parking now
09/03/2024, 19:02 - Asha: <Media omitted>
09/03/2024, 19:03 - Asha: Thanks everyone, amazing day 🥾🌄
09/03/2024, 19:30 - Bruno: Thank you! Action item: I'll share photos by Monday
14/03/2024, 21:00 - Chen Wei: Photos received, thanks Bruno 😊
14/03/2024, 21:05 - Bruno left
"""

_NAMES = ["Asha", "Bruno", "Chen Wei", "Dana", "+1 555 0100", "Élodie", "Ravi Kumar", "Søren"]
_WORDS = [
    "hike",
    "plan",
    "meeting",
    "lunch",
    "train",
    "bus",
    "coffee",
    "tomorrow",
    "today",
    "weekend",
    "photos",
    "park",
    "lake",
    "trail",
    "running",
    "late",
    "okay",
    "sure",
    "thanks",
    "please",
    "great",
    "idea",
    "check",
    "tickets",
    "weather",
    "sunny",
    "rain",
]
_EMOJI = ["😀", "👍🏽", "🎉", "🇮🇳", "👨‍👩‍👧", "❤️", "😂", "🙏🏾"]


def generate_chat(
    n_messages: int,
    *,
    seed: int = 7,
    style: str = "android",
    participants: int = 4,
    start: datetime = datetime(2022, 1, 1, 8, 0),
) -> str:
    """Generate a deterministic synthetic export with ``n_messages`` records."""
    rng = random.Random(seed)  # noqa: S311 - deterministic synthetic data, not security
    names = _NAMES[: max(1, min(participants, len(_NAMES)))]
    ts = start
    lines: list[str] = []
    for i in range(n_messages):
        ts += timedelta(seconds=rng.choice([20, 60, 300, 900, 3600, 4 * 3600, 20 * 3600]))
        name = rng.choice(names)
        r = rng.random()
        if r < 0.05:
            body = "<Media omitted>" if style == "android" else "‎image omitted"
        elif r < 0.07:
            body = "This message was deleted"
        else:
            words = rng.sample(_WORDS, k=rng.randint(2, 9))
            if rng.random() < 0.3:
                words.append(rng.choice(_EMOJI))
            if rng.random() < 0.05:
                words.append("https://example.org/x" + str(i))
            body = " ".join(words)
            if rng.random() < 0.05:
                body += "\nsecond line: " + rng.choice(_WORDS)
        if style == "ios":
            hour12 = ts.hour % 12 or 12
            ampm = "AM" if ts.hour < 12 else "PM"
            header = f"[{ts.month}/{ts.day}/{ts:%y}, {hour12}:{ts:%M:%S} {ampm}]"
            lines.append(f"{header} {name}: {body}")
        else:
            lines.append(f"{ts:%d/%m/%Y}, {ts:%H:%M} - {name}: {body}")
    return "\n".join(lines) + "\n"
