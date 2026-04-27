import re

CATEGORY_KEYWORDS = {
    "Electricity": ["electricity", "power", "bijli", "light", "transformer", "meter"],
    "Water Supply": ["water", "pani", "pipeline", "supply", "tap"],
    "Sanitation & Garbage": ["garbage", "kooda", "trash", "waste", "sewage", "gutter", "toilet"],
    "Roads & Infrastructure": ["road", "pothole", "gaddha", "street", "bridge", "footpath"],
    "Public Transport": ["bus", "train", "metro", "transport", "rickshaw"],
    "Healthcare": ["hospital", "doctor", "clinic", "ambulance", "health"],
    "Government Services": ["ration", "pension", "certificate", "office", "service", "sarkari"]
}

SEVERITY_KEYWORDS = [
    "urgent", "very urgent", "extremely urgent", "immediately", "asap",
    "right now", "without delay", "priority", "top priority",
    "severe", "very severe", "serious", "very serious", "critical",
    "extreme", "intense", "major", "worst", "out of control",
    "dangerous", "very dangerous", "hazard", "hazardous", "risk",
    "high risk", "unsafe", "not safe", "life threatening",
    "threat", "accident risk", "fire risk", "electrocution risk",
    "emergency", "emergency situation", "critical situation",
    "needs immediate attention", "urgent attention required",
    "affecting many people", "affecting everyone",
    "whole area affected", "entire area", "massive issue",
    "large scale problem", "big problem",
    "jaldi", "turant", "abhi", "abhi ke abhi",
    "jaldi se theek karo", "turant theek karo",
    "bahut zyada", "bohot zyada", "bahut jyada",
    "bahut serious", "bohot serious",
    "bahut buri halat", "bohot buri halat",
    "bahut kharab", "bohot kharab",
    "khatarnaak", "bahut khatarnaak",
    "jaan ko khatra", "risk hai", "danger hai",
    "unsafe hai", "safe nahi hai",
    "emergency hai", "urgent hai", "serious problem hai",
    "issue bahut bada hai", "problem bahut bada hai",
    "control ke bahar", "situation kharab hai",
    "haalat bahut kharab hai",
    "bahut bahut", "bohot bohot", "very very",
    "extremely", "too much", "too severe"
]

PRIORITY_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

REVERSE_PRIORITY_MAP = {
    1: "Low",
    2: "Medium",
    3: "High"
}


def detect_multiple_issues(text):
    text_lower = text.lower()
    matched_categories = 0

    for keywords in CATEGORY_KEYWORDS.values():
        if any(keyword in text_lower for keyword in keywords):
            matched_categories += 1
        if matched_categories >= 2:
            return True

    return False


def extract_duration_hours(text):
    text_lower = text.lower()
    total_hours = 0.0

    day_matches = re.findall(r"(\d+)\s*(day|days|din|dino|dinon)", text_lower)
    hour_matches = re.findall(r"(\d+)\s*(hour|hours|hr|hrs|ghanta|ghante)", text_lower)

    for value, _ in day_matches:
        total_hours += float(value) * 24.0

    for value, _ in hour_matches:
        total_hours += float(value)

    return total_hours


def detect_severity_score(text):
    text_lower = text.lower()
    score = 0

    for word in SEVERITY_KEYWORDS:
        if word in text_lower:
            score += 1

    return score


def decision_engine(text, priority_label, category_conf, priority_conf):
    final_conf = (0.6 * float(category_conf)) + (0.4 * float(priority_conf))
    duration_hours = extract_duration_hours(text)
    multiple_issues = detect_multiple_issues(text)
    severity_hits = detect_severity_score(text)
    severity_score = min(severity_hits / 3.0, 1.0)
    priority_numeric = PRIORITY_MAP.get(priority_label, 2)
    final_priority_score = priority_numeric * (1 + severity_score)

    if final_priority_score < 1.5:
        final_priority = "Low"
    elif final_priority_score < 2.5:
        final_priority = "Medium"
    else:
        final_priority = "High"

    if multiple_issues:
        decision = "SEND_TO_OMBUDSMAN"
    elif duration_hours > 48:
        decision = "SEND_TO_OMBUDSMAN"
    elif severity_score >= 0.7:
        decision = "SEND_TO_OMBUDSMAN"
    elif final_conf < 0.6:
        decision = "SEND_TO_OMBUDSMAN"
    else:
        decision = "AUTO_PROCESS"

    return {
        "priority": final_priority,
        "duration_hours": round(duration_hours, 2),
        "decision": decision
    }
