from specialty_normalizer import clean_specialty, specialty_normalization, get_all_specialties
from fuzzywuzzy import process

def search_doctors_by_specialty(specialty, doctors_data, logger):
    try:
        logger.info(f"Searching for specialty: {specialty}")
        cleaned_specialties = clean_specialty(specialty, logger)
        normalized_specialties = []
        for cleaned in cleaned_specialties:
            normalized = specialty_normalization.get(cleaned, cleaned)
            normalized_specialties.append(normalized)
        logger.info(f"Normalized specialties: {normalized_specialties}")

        all_specialties = get_all_specialties(doctors_data, logger)
        if not all_specialties:
            logger.warning("No specialties found in the data")
            return ["No specialties found in the data. Please check the CSV file."]
        logger.info(f"All available specialties: {all_specialties}")

        matched_specialty = None
        for norm_specialty in normalized_specialties:
            if norm_specialty in all_specialties:
                matched_specialty = norm_specialty
                break

        if not matched_specialty:
            best_score = 0
            for norm_specialty in normalized_specialties:
                best_match = process.extractOne(norm_specialty, all_specialties, score_cutoff=80)
                if best_match and best_match[1] > best_score:
                    matched_specialty = best_match[0]
                    best_score = best_match[1]
            logger.info(f"Fuzzy match result: matched_specialty={matched_specialty}, score={best_score}")

        if not matched_specialty:
            logger.warning(f"No matching specialty found for '{specialty}' in the data")
            if "emergency medicine" in all_specialties:
                logger.info("Falling back to Emergency Medicine")
                fallback_results = search_doctors_by_specialty("emergency medicine", doctors_data, logger)
                if fallback_results and not fallback_results[0].startswith("No doctors found"):
                    return [
                        f"No {specialty} specialists found in the data. However, you can consult an Emergency Medicine specialist for initial evaluation:"
                    ] + fallback_results
            return [f"No {specialty} specialists found in the data. Try contacting a general practitioner for a referral."]

        logger.info(f"Matched specialty: {matched_specialty}")
        doctor_list = []
        for doc in doctors_data:
            doc_specialty = doc.get('specialty', 'N/A')
            doc_name = doc.get('name', 'N/A')
            if doc_name == "N/A":
                continue
            if doc_specialty == matched_specialty:
                doctor_info = (
                    f"Dr. {doc_name} - {doc_specialty}, "
                    f"Contact: {doc.get('contact', 'N/A')}, "
                    f"Hospital: {doc.get('hospital', 'N/A')}, "
                    f"Consultation Time: {doc.get('consultation_time', 'N/A')}, "
                    f"Rating: {doc.get('rating', 'N/A')}"
                )
                doctor_list.append(doctor_info)

        if not doctor_list:
            logger.warning(f"No doctors found for matched specialty '{matched_specialty}'")
            if "emergency medicine" in all_specialties and matched_specialty != "emergency medicine":
                logger.info("No doctors found for specialty, falling back to Emergency Medicine")
                fallback_results = search_doctors_by_specialty("emergency medicine", doctors_data, logger)
                if fallback_results and not fallback_results[0].startswith("No doctors found"):
                    return [
                        f"No {specialty} specialists found in the data. However, you can consult an Emergency Medicine specialist for initial evaluation:"
                    ] + fallback_results
            return [f"No doctors found for specialty '{matched_specialty}'. Try contacting a general practitioner for a referral."]
        return doctor_list
    except Exception as e:
        logger.error(f"Error in search_doctors_by_specialty: {str(e)}")
        return [f"Error searching doctors: {str(e)}"]