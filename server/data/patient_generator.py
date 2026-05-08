"""
PoshanEnv — Synthetic Patient Generator
Generates realistic rural India household data for episodes.
All data is procedurally generated — no real patient data used.
"""

import random
import uuid
from typing import List, Tuple

from env.models import PatientProfile, HouseholdContext
from data.clinical_protocols import classify_muac, classify_maternal_risk


# ---------------------------------------------------------------------------
# Rural India name pools
# ---------------------------------------------------------------------------

MOTHER_NAMES = [
    "Savitri", "Kamla", "Sunita", "Meena", "Rekha", "Geeta",
    "Parvati", "Anita", "Shanti", "Pushpa", "Radha", "Usha",
    "Lalita", "Saroj", "Kiran", "Nirmala", "Sushila", "Pratima",
]

CHILD_NAMES = [
    "Raju", "Priya", "Sonu", "Munni", "Pappu", "Guddi",
    "Chintu", "Pinki", "Bablu", "Rinki", "Chhotu", "Sweetie",
]

VILLAGES = [
    ("Rampur", "Sitapur", "Uttar Pradesh"),
    ("Dhangaon", "Yavatmal", "Maharashtra"),
    ("Kodalipara", "Banka", "Bihar"),
    ("Sundarpada", "Ganjam", "Odisha"),
    ("Vellore Kuppam", "Vellore", "Tamil Nadu"),
    ("Kherwa", "Barwani", "Madhya Pradesh"),
    ("Nandpur", "Kandhamal", "Odisha"),
    ("Tinsukia village", "Tinsukia", "Assam"),
]

MATERNAL_SYMPTOM_POOL = [
    "mild_headache", "severe_headache", "ankle_swelling",
    "blurred_vision", "fatigue", "dizziness", "vaginal_bleeding",
    "reduced_fetal_movement", "swelling_face", "convulsions",
    "nausea", "back_pain",
]

CHILD_SYMPTOM_POOL = [
    "cough", "fever", "diarrhoea", "poor_appetite",
    "unable_to_feed", "lethargic", "chest_indrawing",
    "vomiting_everything", "convulsions",
]


# ---------------------------------------------------------------------------
# Household context generator
# ---------------------------------------------------------------------------

def generate_household(visit_number: int = 1) -> HouseholdContext:
    village, district, state = random.choice(VILLAGES)
    return HouseholdContext(
        household_id=str(uuid.uuid4())[:8],
        village=village,
        district=district,
        state=state,
        distance_to_phc_km=round(random.uniform(2.0, 18.0), 1),
        has_toilet=random.random() > 0.55,
        monthly_income_inr=random.randint(2500, 9000),
        visit_number=visit_number,
    )


# ---------------------------------------------------------------------------
# Mother generator
# ---------------------------------------------------------------------------

def generate_mother(risk_level: str = "random") -> PatientProfile:
    """
    Generate a pregnant woman profile.
    risk_level: "low" | "medium" | "high" | "random"
    """
    if risk_level == "random":
        risk_level = random.choice(["low", "low", "medium", "high"])

    weeks = random.randint(8, 40)
    anc   = random.randint(0, min(4, weeks // 8))

    if risk_level == "low":
        systolic  = random.randint(100, 125)
        diastolic = random.randint(60, 82)
        hb        = round(random.uniform(10.0, 13.5), 1)
        symptoms  = random.sample(["nausea", "back_pain", "fatigue"], k=random.randint(0, 2))

    elif risk_level == "medium":
        systolic  = random.randint(125, 138)
        diastolic = random.randint(82, 89)
        hb        = round(random.uniform(7.5, 10.0), 1)
        symptoms  = random.sample(
            ["mild_headache", "ankle_swelling", "dizziness", "fatigue"],
            k=random.randint(1, 3)
        )

    else:  # high
        systolic  = random.choice([random.randint(140, 170), random.randint(100, 125)])
        diastolic = random.choice([random.randint(90, 110), random.randint(60, 80)])
        hb        = round(random.uniform(4.5, 7.5), 1)
        symptoms  = random.sample(
            ["severe_headache", "blurred_vision", "vaginal_bleeding",
             "swelling_face", "reduced_fetal_movement"],
            k=random.randint(1, 3)
        )
        # Ensure at least one high-risk BP or Hb
        if systolic < 140 and hb >= 7.0:
            systolic = random.randint(140, 160)

    weight = round(random.uniform(42.0, 68.0), 1)
    height = round(random.uniform(148.0, 165.0), 1)

    return PatientProfile(
        patient_id=str(uuid.uuid4())[:6],
        patient_type="mother",
        name=random.choice(MOTHER_NAMES),
        age_years=random.randint(18, 38),
        weight_kg=weight,
        height_cm=height,
        gestational_weeks=weeks,
        anc_visits_completed=anc,
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        hemoglobin=hb,
        symptoms=symptoms,
    )


# ---------------------------------------------------------------------------
# Child generator
# ---------------------------------------------------------------------------

def generate_child(malnutrition_grade: str = "random") -> PatientProfile:
    """
    Generate a child under 5 profile.
    malnutrition_grade: "SAM" | "MAM" | "Normal" | "random"
    """
    if malnutrition_grade == "random":
        malnutrition_grade = random.choice(["Normal", "Normal", "MAM", "SAM"])

    age_months = random.randint(6, 59)

    if malnutrition_grade == "SAM":
        muac   = round(random.uniform(9.5, 11.4), 1)
        oedema = random.random() > 0.6
        weight = round(random.uniform(4.5, 7.5), 1)
        symptoms = random.sample(
            ["poor_appetite", "lethargic", "unable_to_feed"], k=random.randint(1, 2)
        )

    elif malnutrition_grade == "MAM":
        muac   = round(random.uniform(11.5, 12.4), 1)
        oedema = False
        weight = round(random.uniform(7.0, 10.0), 1)
        symptoms = random.sample(
            ["poor_appetite", "cough", "diarrhoea"], k=random.randint(0, 2)
        )

    else:  # Normal
        muac   = round(random.uniform(12.5, 15.5), 1)
        oedema = False
        weight = round(random.uniform(8.0, 16.0), 1)
        symptoms = random.sample(["cough", "fever"], k=random.randint(0, 1))

    height = round(random.uniform(55.0, 105.0), 1)

    diets = [
        "Rice and dal once a day",
        "Biscuits and tea, no vegetables",
        "Breast milk only",
        "Rice, dal, vegetables twice daily",
        "Khichdi once daily, no fruits",
    ]

    return PatientProfile(
        patient_id=str(uuid.uuid4())[:6],
        patient_type="child",
        name=random.choice(CHILD_NAMES),
        age_years=round(age_months / 12, 1),
        age_months=age_months,
        weight_kg=weight,
        height_cm=height,
        muac_cm=muac,
        oedema=oedema,
        symptoms=symptoms,
        dietary_recall=random.choice(diets),
    )


# ---------------------------------------------------------------------------
# Task-specific generators
# ---------------------------------------------------------------------------

def generate_task1_patients(n: int = 5) -> List[PatientProfile]:
    """
    Task 1: Generate n patients with mixed urgency levels.
    Ensures variety — at least one high, one medium, one low.
    """
    patients = []
    # Guarantee variety
    patients.append(generate_mother("high"))
    patients.append(generate_mother("medium"))
    patients.append(generate_mother("low"))
    patients.append(generate_child("SAM"))
    patients.append(generate_child("Normal"))
    # Fill remaining
    while len(patients) < n:
        if random.random() > 0.5:
            patients.append(generate_mother("random"))
        else:
            patients.append(generate_child("random"))
    random.shuffle(patients[:n])
    return patients[:n]


def generate_task2_household() -> Tuple[PatientProfile, PatientProfile, HouseholdContext]:
    """Task 2: One mother + one child in one household."""
    mother = generate_mother(random.choice(["medium", "high"]))
    child  = generate_child(random.choice(["MAM", "SAM"]))
    hh     = generate_household(visit_number=1)
    return mother, child, hh


def generate_task3_progression() -> List[Tuple[PatientProfile, PatientProfile, HouseholdContext, str]]:
    """
    Task 3: 4-visit sequence where condition deteriorates.
    Returns list of (mother, child, household, visit_notes) per visit.
    """
    visits = []

    # Visit 1 — borderline
    m1 = generate_mother("medium")
    c1 = generate_child("MAM")
    hh = generate_household(visit_number=1)
    visits.append((m1, c1, hh, "First visit. Noted mild symptoms."))

    # Visit 2 — slight worsening
    m2 = generate_mother("medium")
    m2.hemoglobin = round(m1.hemoglobin - random.uniform(0.5, 1.5), 1)  # dropping Hb
    m2.symptoms = m1.symptoms + ["dizziness"]
    c2 = generate_child("MAM")
    c2.muac_cm = round(c1.muac_cm - random.uniform(0.3, 0.7), 1)        # MUAC dropping
    hh2 = generate_household(visit_number=2)
    hh2.household_id = hh.household_id
    visits.append((m2, c2, hh2, "Visit 2. Hemoglobin dropped. Child MUAC declining."))

    # Visit 3 — clear deterioration
    m3 = generate_mother("high")
    m3.hemoglobin = round(m2.hemoglobin - random.uniform(1.0, 2.0), 1)
    m3.systolic_bp = random.randint(138, 150)
    m3.symptoms = m2.symptoms + ["severe_headache"]
    c3 = generate_child("SAM")
    c3.muac_cm = round(c2.muac_cm - random.uniform(0.5, 1.0), 1)
    hh3 = generate_household(visit_number=3)
    hh3.household_id = hh.household_id
    visits.append((m3, c3, hh3, "Visit 3. Mother BP rising. Child crossed SAM threshold."))

    # Visit 4 — critical
    m4 = generate_mother("high")
    m4.hemoglobin = max(4.5, round(m3.hemoglobin - 1.0, 1))
    m4.systolic_bp = random.randint(148, 165)
    m4.symptoms = ["severe_headache", "blurred_vision", "swelling_face"]
    c4 = generate_child("SAM")
    c4.oedema = True
    hh4 = generate_household(visit_number=4)
    hh4.household_id = hh.household_id
    visits.append((m4, c4, hh4, "Visit 4. Critical. Immediate escalation required."))

    return visits
