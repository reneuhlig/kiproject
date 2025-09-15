#!/bin/bash
# Beispiel-Script für Cronjob

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="aiuser"  
DB_PASSWORD="geheimespasswort"
DB_NAME="ai_detection_db"

# Datenpfad
DATA_DIR="/path/to/your/classified/images"

# Ultralytics Run
python3 run_person_detection.py \
    --model ultralytics \
    --db-host $DB_HOST \
    --db-user $DB_USER \
    --db-password $DB_PASSWORD \
    --db-name $DB_NAME \
    --data-dir $DATA_DIR \
    --max-images 100 \
    --confidence-threshold 0.6 \
    --run-name "daily_ultralytics_scan" \
    --job-id "$(date +%Y%m%d_%H%M%S)"

# DeepFace Run  
python3 run_person_detection.py \
    --model deepface \
    --db-host $DB_HOST \
    --db-user $DB_USER \
    --db-password $DB_PASSWORD \
    --db-name $DB_NAME \
    --data-dir $DATA_DIR \
    --max-images 50 \
    --deepface-backend mtcnn \
    --confidence-threshold 0.5 \
    --run-name "daily_deepface_scan" \
    --job-id "$(date +%Y%m%d_%H%M%S)"

# Gemma Run (nur bei ausreichend GPU-Memory)
python3 run_person_detection.py \
    --model gemma \
    --db-host $DB_HOST \
    --db-user $DB_USER \
    --db-password $DB_PASSWORD \
    --db-name $DB_NAME \
    --data-dir $DATA_DIR \
    --max-images 20 \
    --confidence-threshold 0.7 \
    --run-name "daily_gemma_scan" \
    --job-id "$(date +%Y%m%d_%H%M%S)"
