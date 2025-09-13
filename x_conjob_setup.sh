#!/bin/bash
# Cronjob Setup für automatische Personenerkennung
# Führt die verschiedenen KI-Modelle zeitversetzt aus

# =============================================================================
# KONFIGURATION - BITTE ANPASSEN
# =============================================================================

# Projektpfad
PROJECT_DIR="/path/to/your/person_detection_project"

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="ai_detection_user"
DB_PASSWORD="your_secure_password"
DB_NAME="ai_detection"

# Datenpfad zu klassifizierten Bildern
DATA_DIR="/path/to/your/classified/images"

# Log-Verzeichnis
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Maximale Anzahl Bilder pro Run (zur Performance-Kontrolle)
MAX_IMAGES_ULTRALYTICS=200
MAX_IMAGES_DEEPFACE=100
MAX_IMAGES_GEMMA=50

# Konfidenz-Schwellenwerte
CONFIDENCE_ULTRALYTICS=0.6
CONFIDENCE_DEEPFACE=0.5
CONFIDENCE_GEMMA=0.7

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/cronjob.log"
}

run_detection() {
    local model=$1
    local job_id=$2
    local max_images=$3
    local confidence=$4
    local extra_args=$5
    
    log_message "Starte $model Detection (Job: $job_id)"
    
    cd "$PROJECT_DIR" || {
        log_message "FEHLER: Kann nicht in Projektverzeichnis wechseln"
        return 1
    }
    
    # Logdatei für diesen Run
    local run_log="$LOG_DIR/${model}_${job_id}.log"
    
    # Python-Befehl ausführen
    python3 run_person_detection.py \
        --model "$model" \
        --db-host "$DB_HOST" \
        --db-user "$DB_USER" \
        --db-password "$DB_PASSWORD" \
        --db-name "$DB_NAME" \
        --data-dir "$DATA_DIR" \
        --max-images "$max_images" \
        --confidence-threshold "$confidence" \
        --run-name "cronjob_${model}_$(date +%Y%m%d)" \
        --job-id "$job_id" \
        $extra_args \
        >> "$run_log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "$model Detection erfolgreich abgeschlossen (Job: $job_id)"
    else
        log_message "FEHLER: $model Detection fehlgeschlagen (Job: $job_id, Exit Code: $exit_code)"
    fi
    
    return $exit_code
}

# =============================================================================
# HAUPTFUNKTIONEN FÜR VERSCHIEDENE ZEITPUNKTE
# =============================================================================

run_morning_scan() {
    # Morgendlicher Vollscan mit Ultralytics (schnell und zuverlässig)
    local job_id="morning_$(date +%Y%m%d_%H%M%S)"
    
    log_message "=== MORGENDLICHER SCAN GESTARTET ==="
    
    run_detection "ultralytics" "$job_id" "$MAX_IMAGES_ULTRALYTICS" "$CONFIDENCE_ULTRALYTICS" ""
    
    log_message "=== MORGENDLICHER SCAN BEENDET ==="
}

run_afternoon_detailed() {
    # Nachmittags detaillierte Analyse mit DeepFace und Ultralytics
    local job_id="afternoon_$(date +%Y%m%d_%H%M%S)"
    
    log_message "=== DETAILLIERTE NACHMITTAGS-ANALYSE GESTARTET ==="
    
    # Erst Ultralytics
    run_detection "ultralytics" "${job_id}_ultra" "$MAX_IMAGES_ULTRALYTICS" "$CONFIDENCE_ULTRALYTICS" ""
    
    # Dann DeepFace mit MTCNN (bessere Gesichtserkennung)
    run_detection "deepface" "${job_id}_deep" "$MAX_IMAGES_DEEPFACE" "$CONFIDENCE_DEEPFACE" "--deepface-backend mtcnn"
    
    log_message "=== DETAILLIERTE NACHMITTAGS-ANALYSE BEENDET ==="
}

run_evening_comprehensive() {
    # Abends umfassende Analyse mit allen Modellen
    local job_id="evening_$(date +%Y%m%d_%H%M%S)"
    
    log_message "=== UMFASSENDE ABEND-ANALYSE GESTARTET ==="
    
    # Ultralytics
    run_detection "ultralytics" "${job_id}_ultra" "$MAX_IMAGES_ULTRALYTICS" "$CONFIDENCE_ULTRALYTICS" ""
    
    # Warte 5 Minuten zwischen den Modellen
    sleep 300
    
    # DeepFace
    run_detection "deepface" "${job_id}_deep" "$MAX_IMAGES_DEEPFACE" "$CONFIDENCE_DEEPFACE" "--deepface-backend retinaface"
    
    # Warte weitere 5 Minuten
    sleep 300
    
    # Gemma (nur wenn GPU verfügbar)
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        log_message "GPU erkannt - starte Gemma Analysis"
        run_detection "gemma" "${job_id}_gemma" "$MAX_IMAGES_GEMMA" "$CONFIDENCE_GEMMA" ""
    else
        log_message "Keine GPU erkannt - überspringe Gemma Analysis"
    fi
    
    log_message "=== UMFASSENDE ABEND-ANALYSE BEENDET ==="
}

run_weekly_full_scan() {
    # Wöchentlicher Vollscan ohne Bildlimit
    local job_id="weekly_$(date +%Y%m%d_%H%M%S)"
    
    log_message "=== WÖCHENTLICHER VOLLSCAN GESTARTET ==="
    
    # Ultralytics ohne Limit
    run_detection "ultralytics" "${job_id}_ultra_full" "999999" "$CONFIDENCE_ULTRALYTICS" ""
    
    log_message "=== WÖCHENTLICHER VOLLSCAN BEENDET ==="
}

cleanup_old_logs() {
    # Lösche Logs älter als 30 Tage
    find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete
    log_message "Alte Logs bereinigt"
}

# =============================================================================
# HAUPTLOGIK BASIEREND AUF ARGUMENTEN
# =============================================================================

case "${1:-help}" in
    "morning")
        run_morning_scan
        ;;
    "afternoon") 
        run_afternoon_detailed
        ;;
    "evening")
        run_evening_comprehensive
        ;;
    "weekly")
        run_weekly_full_scan
        ;;
    "cleanup")
        cleanup_old_logs
        ;;
    "install-cron")
        log_message "Installiere Crontab-Einträge..."
        
        # Backup der aktuellen Crontab
        crontab -l > "$LOG_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true
        
        # Temporäre Crontab-Datei erstellen
        temp_cron=$(mktemp)
        
        # Bestehende Crontab laden (falls vorhanden)
        crontab -l 2>/dev/null > "$temp_cron" || true
        
        # Unsere Jobs hinzufügen
        cat >> "$temp_cron" << EOF

# =============================================================================
# AI Person Detection Cronjobs
# =============================================================================

# Morgendlicher Schnellscan (7:30 Uhr)
30 7 * * * $0 morning

# Nachmittags detaillierte Analyse (14:00 Uhr)  
0 14 * * * $0 afternoon

# Abends umfassende Analyse (20:00 Uhr)
0 20 * * * $0 evening

# Wöchentlicher Vollscan (Sonntag 2:00 Uhr)
0 2 * * 0 $0 weekly

# Tägliche Log-Bereinigung (1:00 Uhr)
0 1 * * * $0 cleanup

# =============================================================================
EOF
        
        # Neue Crontab installieren
        crontab "$temp_cron"
        rm "$temp_cron"
        
        log_message "Crontab-Einträge installiert"
        echo "Aktuelle Crontab:"
        crontab -l
        ;;
    "help"|*)
        echo "Usage: $0 {morning|afternoon|evening|weekly|cleanup|install-cron}"
        echo ""
        echo "Commands:"
        echo "  morning     - Schneller Morgenscan mit Ultralytics"  
        echo "  afternoon   - Detaillierte Analyse mit Ultralytics + DeepFace"
        echo "  evening     - Umfassende Analyse mit allen Modellen"
        echo "  weekly      - Wöchentlicher Vollscan ohne Bildlimit"
        echo "  cleanup     - Bereinige alte Log-Dateien" 
        echo "  install-cron- Installiere alle Cronjobs automatisch"
        echo ""
        echo "Konfiguration:"
        echo "  PROJECT_DIR: $PROJECT_DIR"
        echo "  DATA_DIR: $DATA_DIR"
        echo "  LOG_DIR: $LOG_DIR"
        ;;
esac