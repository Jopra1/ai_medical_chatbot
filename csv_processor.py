import os
import csv

def load_doctors_data(csv_file, logger):
    doctors_data = []
    try:
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                logger.error("No header row found in CSV file")
                raise SystemExit("Exiting due to missing header row in CSV file")

            logger.info(f"CSV headers: {headers}")
            header_indices = {header: idx for idx, header in enumerate(headers)}

            required_fields = ['name', 'speciality', 'contact', 'hospital', 'consultation_time', 'rating', 'qualification']
            for field in required_fields:
                if field not in header_indices:
                    logger.warning(f"Header '{field}' not found in CSV headers. Using default 'N/A' for this field.")

            for i, row in enumerate(reader, start=2):
                if not row or not any(field.strip() for field in row):
                    logger.warning(f"Skipping empty row at line {i}")
                    continue

                logger.info(f"Processing row at line {i}: {row}")
                while len(row) < len(headers):
                    row.append('')

                doctor = {
                    "name": row[header_indices['name']] if 'name' in header_indices and row[header_indices['name']].strip() else 'N/A',
                    "specialty": row[header_indices['speciality']].lower() if 'speciality' in header_indices and row[header_indices['speciality']].strip() else 'N/A',
                    "contact": row[header_indices['contact']] if 'contact' in header_indices and row[header_indices['contact']].strip() else 'N/A',
                    "hospital": row[header_indices['hospital']] if 'hospital' in header_indices and row[header_indices['hospital']].strip() else 'N/A',
                    "consultation_time": row[header_indices['consultation_time']] if 'consultation_time' in header_indices and row[header_indices['consultation_time']].strip() else 'N/A',
                    "rating": float(row[header_indices['rating']]) if 'rating' in header_indices and row[header_indices['rating']].replace('.', '').isdigit() else 'N/A',
                    "qualifications": row[header_indices['qualification']] if 'qualification' in header_indices and row[header_indices['qualification']].strip() else 'N/A'
                }

                if doctor["name"] != "N/A":
                    doctors_data.append(doctor)
                else:
                    logger.warning(f"Skipping row at line {i} with no name field: {row}")

        if not doctors_data:
            logger.error("No valid doctors found in CSV file after parsing")
            raise SystemExit("Exiting due to empty CSV file after parsing")

        logger.info(f"Loaded {len(doctors_data)} doctors from CSV")
        specialties_found = list(set(doctor["specialty"] for doctor in doctors_data if doctor["specialty"] != "N/A"))
        logger.info(f"Specialties in CSV data: {specialties_found}")
        return doctors_data
    except Exception as e:
        logger.error(f"Failed to load CSV file: {str(e)}")
        raise SystemExit("Exiting due to CSV file loading failure")