import csv
import random
import re

random.seed(26042026)

categories = [
    "Sanitation & Garbage",
    "Electricity",
    "Water Supply",
    "Roads & Infrastructure",
    "Public Transport",
    "Healthcare",
    "Government Services",
]

department_map = {
    "Sanitation & Garbage": "Municipal Corporation",
    "Electricity": "Electricity Board",
    "Water Supply": "Municipal Corporation",
    "Roads & Infrastructure": "PWD",
    "Public Transport": "Transport Department",
    "Healthcare": "Health Department",
    "Government Services": "Revenue Department",
}

locations = [
    "Karol Bagh Delhi", "Indiranagar Bengaluru", "Kondapur Hyderabad", "Anna Nagar Chennai",
    "Andheri East Mumbai", "Salt Lake Kolkata", "Mansarovar Jaipur", "Bopal Ahmedabad",
    "Kothrud Pune", "Aliganj Lucknow", "Patia Bhubaneswar", "Lalpur Ranchi",
    "Civil Lines Prayagraj", "Dwarka Delhi", "Mayur Vihar Delhi", "Shivajinagar Pune",
    "T Nagar Chennai", "Miyapur Hyderabad", "Ameerpet Hyderabad", "Banashankari Bengaluru",
    "Whitefield Bengaluru", "Nerul Navi Mumbai", "Thane West", "Noida Sector 62",
    "Vaishali Ghaziabad", "Gomti Nagar Lucknow", "Laxmi Nagar Delhi", "Rajendra Nagar Patna",
    "Kankarbagh Patna", "Saket Delhi", "Perungudi Chennai", "Gachibowli Hyderabad",
    "Ashok Vihar Delhi", "Hinjewadi Pune", "Behala Kolkata", "Howrah Maidan",
    "Kakadeo Kanpur", "Adyar Chennai", "Viman Nagar Pune", "Sector 21 Gurugram",
]

synonyms = {
    "Sanitation & Garbage": ["garbage", "trash", "waste", "litter", "littering", "dump", "dumping", "dirty", "unhygienic"],
    "Electricity": ["power cut", "blackout", "no electricity", "outage", "voltage issue"],
    "Water Supply": ["water shortage", "no water", "supply problem", "pipeline issue"],
    "Roads & Infrastructure": ["potholes", "broken roads", "road damage", "bad roads"],
    "Public Transport": ["bus delay", "late buses", "no buses", "transport issue"],
    "Healthcare": ["hospital issue", "no doctor", "medical problem"],
}

service_terms = [
    "certificate not updated", "application pending", "portal status stuck", "record mismatch",
    "token not generated", "appointment slot delay", "verification not completed",
]

high_patterns = [
    "{loc} facing {term} continuously for {d} days, urgent action needed.",
    "Since {d} days in {loc}, {term} and service is completely down.",
    "In {loc} we have severe {term} for {d} days, please fix urgently.",
    "{term} not working in {loc} from {d} days, situation is very high impact.",
]

medium_patterns = [
    "In {loc}, {term} is moderate but regular from last {d} days.",
    "{loc} has {term} sometimes, response is slow and there is delay.",
    "Average level {term} seen in {loc}, issue pending from {d} days.",
    "There is transport style delay in service at {loc} due to {term}.",
]

low_patterns = [
    "Please improve {term} in {loc}, this is a minor suggestion.",
    "Small inconvenience in {loc} because of {term}, kindly review.",
    "Routine complaint from {loc} on {term}, not urgent but needs attention.",
    "When possible, improve {term} at {loc}; issue is low priority now.",
]

typos = {
    "please": "plese",
    "urgent": "urgnt",
    "service": "servce",
    "problem": "problm",
    "issue": "isssue",
    "there": "ther",
    "because": "becuse",
    "action": "aciton",
    "delay": "dely",
    "supply": "suply",
}

def maybe_typo(text):
    if random.random() > 0.08:
        return text
    keys = list(typos.keys())
    random.shuffle(keys)
    for k in keys:
        if re.search(r"\b" + re.escape(k) + r"\b", text.lower()):
            return re.sub(r"\b" + re.escape(k) + r"\b", typos[k], text, flags=re.IGNORECASE, count=1)
    return text


def pick_term(cat):
    if cat in synonyms:
        return random.choice(synonyms[cat])
    return random.choice(service_terms)


# Balanced priorities: 500 each
priority_targets = {"High": 500, "Medium": 500, "Low": 500}
priority_rows = []

for priority in ["High", "Medium", "Low"]:
    for i in range(priority_targets[priority]):
        cat = categories[i % len(categories)]
        loc = random.choice(locations)
        term = pick_term(cat)

        if priority == "High":
            d = random.randint(3, 9)  # >2 days as requested
            pattern = random.choice(high_patterns)
            res_days = random.randint(1, 3)
        elif priority == "Medium":
            d = random.randint(1, 2)
            pattern = random.choice(medium_patterns)
            res_days = random.randint(3, 7)
        else:
            d = random.randint(1, 2)
            pattern = random.choice(low_patterns)
            res_days = random.randint(7, 15)

        text = pattern.format(loc=loc, term=term, d=d)
        text = maybe_typo(text)

        # force uniqueness with compact suffix id
        text = f"{text} Ref{priority[0]}{i+1}"

        priority_rows.append({
            "complaint_text": text,
            "category": cat,
            "department": department_map[cat],
            "priority": priority,
            "resolution_time_days": res_days,
        })

# Shuffle rows for natural mix
random.shuffle(priority_rows)

output_path = "improved_grievances_dataset.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["complaint_text", "category", "department", "priority", "resolution_time_days"],
    )
    writer.writeheader()
    writer.writerows(priority_rows)

print(f"Generated {len(priority_rows)} rows at {output_path}")
