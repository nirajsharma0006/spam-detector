"""
features.py — Helpers to extract suspicious signals from a message.
Not part of the model; only used by the app to show the "Why?" section.
"""

import re

# ------------------- Regex Patterns -------------------
_LINK_RE = re.compile(
    r"https?://|www\.|bit\.ly|tinyurl|goo\.gl|t\.co|\.xyz\b|\.online\b|\.info\b|\.top\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?:\+91[\s\-]?)?\b[6-9]\d{9}\b")
_AMOUNT_RE = re.compile(
    r"₹|\brs\.?\s*\d|\brupees\b|\blakhs?\b|\bcrores?\b",
    re.IGNORECASE,
)

# ------------------- Word lists -------------------
_URGENCY_WORDS = [
    "urgent", "immediately", "today only", "abhi", "turant",
    "last chance", "expire", "expires", "expired",
    "final notice", "within 24",
]
_SENSITIVE_WORDS = [
    "kyc", "otp", "otps", "pin", "password", "cvv",
    "account block", "blocked", "block ho",
    "verify", "verified", "verification",
    "pan card", "pan number", "aadhaar", "aadhar",
    "debit card", "credit card", "net banking", "netbanking",
]
_REWARD_WORDS = [
    "winner", "winners", "lottery", "prize", "prizes",
    "gift", "gifts", "free", "cashback",
    "lucky", "won", "claim", "reward", "rewards",
]


def _has_word(text, words):
    """Check whether text contains any of the given words/phrases (word boundary safe)."""
    for w in words:
        pattern = r"\b" + re.escape(w.strip()).replace(" ", r"\s+") + r"\b"
        if re.search(pattern, text):
            return True
    return False


def extract_signals(text):
    """Extract suspicious signals from a text and return a dict of booleans."""
    if not isinstance(text, str):
        text = str(text)
    lower = text.lower()

    # Uppercase ratio (counted over letters only)
    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0

    return {
        "link": bool(_LINK_RE.search(text)),
        "phone": bool(_PHONE_RE.search(text)),
        "amount": bool(_AMOUNT_RE.search(text)),
        "urgency": _has_word(lower, _URGENCY_WORDS),
        "sensitive": _has_word(lower, _SENSITIVE_WORDS),
        "reward": _has_word(lower, _REWARD_WORDS),
        "uppercase": upper_ratio > 0.4,
    }


def signal_messages(signals):
    """Convert the signals dict into a list of human-readable strings (only True ones)."""
    messages = []
    if signals.get("link"):
        messages.append("Message contains a link")
    if signals.get("phone"):
        messages.append("Message contains a phone number")
    if signals.get("amount"):
        messages.append("Message mentions money / an amount")
    if signals.get("urgency"):
        messages.append("Message creates urgency (asking you to act fast)")
    if signals.get("sensitive"):
        messages.append("Message contains sensitive words (KYC / OTP / PIN / bank etc.)")
    if signals.get("reward"):
        messages.append("Message mentions a reward / prize / free offer")
    if signals.get("uppercase"):
        messages.append("Message has too many CAPITAL letters")

    if not messages:
        messages.append("No suspicious signals found — the message looks clean")
    return messages