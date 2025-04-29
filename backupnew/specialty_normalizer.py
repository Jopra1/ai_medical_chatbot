import re
from fuzzywuzzy import process

specialty_normalization = {
    "dermatology": "dermatology",
    "a dermatology specialist": "dermatology",
    "neurology": "neurology",
    "a neurology specialist": "neurology",
    "neurology or neurosurgery": "neurology & neurosurgery",
    "a neurology or neurosurgery specialist": "neurology & neurosurgery",
    "pulmonology": "pulmonology",
    "a pulmonology specialist": "pulmonology",
    "otolaryngology": "otolaryngology",
    "otolaryngology (ent)": "otolaryngology",
    "ent": "otolaryngology",
    "an otolaryngology (ent) specialist": "otolaryngology",
    "vascular surgery": "vascular surgery",
    "a vascular surgery specialist": "vascular surgery",
    "cardiology": "cardiology",
    "a cardiology specialist": "cardiology",
    "urology": "urology",
    "a urology specialist": "urology",
    "orthopedics": "orthopedics",
    "a orthopedics specialist": "orthopedics",
    "gastroenterology": "gastroenterology",
    "a gastroenterology specialist": "gastroenterology",
    "anesthesiology": "anesthesiology",
    "a anesthesiology specialist": "anesthesiology",
    "infectious disease": "emergency medicine",
    "a infectious disease specialist": "emergency medicine",
    "emergency medicine": "emergency medicine",
    "a emergency medicine specialist": "emergency medicine"
}

def clean_specialty(specialty, logger):
    try:
        cleaned = re.sub(r'\b(a|an)\b', '', specialty, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\bspecialist\b', '', cleaned, flags=re.IGNORECASE).strip()
        if "(ent)" in cleaned.lower():
            cleaned_base = cleaned.replace("(ENT)", "").strip()
            return [cleaned_base.lower(), "ent"]
        return [cleaned.lower()]
    except Exception as e:
        logger.error(f"Error cleaning specialty '{specialty}': {str(e)}")
        return [specialty.lower()]

def extract_specialty_from_response(response):
    match = re.search(r'(?:consult|recommend)\s+(?:a|an)?\s*([\w\s&]+(?:\s\(ent\))?)\s*specialist', response, re.IGNORECASE)
    if match:
        specialty = match.group(1).strip()
        return specialty
    for specialty in specialty_normalization.keys():
        if specialty in response.lower():
            return specialty
    return None

def get_all_specialties(doctors_data, logger):
    try:
        specialties = [doctor["specialty"] for doctor in doctors_data if doctor["specialty"] != "N/A"]
        return list(set(specialties))
    except Exception as e:
        logger.error(f"Error fetching specialties: {str(e)}")
        return []