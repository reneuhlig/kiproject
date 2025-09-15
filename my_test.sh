#!/bin/bash
# test_models.sh - Separate Testläufe für alle KI-Modelle inkl. Ollama

# =============================================================================
# KONFIGURATION - BITTE ANPASSEN
# =============================================================================

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="DHBW1234!?"
DB_NAME="ai_detection"

# Datenpfad zu klassifizierten Bildern
DATA_DIR="/blob"

# Test-Parameter
TEST_MAX_IMAGES=5  # Kleine Anzahl für Tests
CONFIDENCE_THRESHOLD=0.5

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

test_model() {
    local model=$1
    local extra_args="$2"
    
    log_message "=== Teste $model Modell ==="
    
    python3 run_person_detection.py \
        --model "$model" \
        --db-host "$DB_HOST" \
        --db-user "$DB_USER" \
        --db-password "$DB_PASSWORD" \
        --db-name "$DB_NAME" \
        --data-dir "$DATA_DIR" \
        --max-images "$TEST_MAX_IMAGES" \
        --confidence-threshold "$CONFIDENCE_THRESHOLD" \
        --run-name "test_${model}_$(date +%Y%m%d_%H%M%S)" \
        $extra_args
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "✓ $model Test erfolgreich"
    else
        log_message "✗ $model Test fehlgeschlagen (Exit Code: $exit_code)"
    fi
    
    echo
    return $exit_code
}

# =============================================================================
# EINZELNE TESTFUNKTIONEN
# =============================================================================

test_ultralytics() {
    log_message "Starte Ultralytics YOLO Test..."
    test_model "ultralytics" "--yolo-model-path yolov8n.pt"
}

test_deepface() {
    log_message "Starte DeepFace Test..."
    
    echo "1. Test mit OpenCV Backend:"
    test_model "deepface" "--deepface-backend opencv"
    
    #echo "2. Test mit MTCNN Backend:"
    #test_model "deepface" "--deepface-backend mtcnn"
    
    #echo "3. Test mit SSD Backend:"
    #test_model "deepface" "--deepface-backend ssd"
}

test_ollama() {
    log_message "Starte Ollama Test (Gemma 3 Modelle)..."
    
    # Standardmodell klein zum Testen
    test_model "ollama-gemma3" "--ollama-model gemma3:4b"
}

test_database_connection() {
    log_message "Teste Datenbankverbindung..."
    
    python3 -c "
import mysql.connector
try:
    conn = mysql.connector.connect(
        host='$DB_HOST',
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME'
    )
    if conn.is_connected():
        print('✓ Datenbankverbindung erfolgreich')
        conn.close()
        exit(0)
    else:
        print('✗ Datenbankverbindung fehlgeschlagen')
        exit(1)
except Exception as e:
    print(f'✗ Datenbankverbindung fehlgeschlagen: {e}')
    exit(1)
"
}

check_prerequisites() {
    log_message "Prüfe Voraussetzungen..."
    
    python3 -c "
try:
    import cv2, numpy, PIL, psutil, mysql.connector
    print('✓ Core-Bibliotheken verfügbar')
except ImportError as e:
    print(f'✗ Import-Fehler: {e}')
    exit(1)

try:
    from ultralytics import YOLO
    print('✓ Ultralytics verfügbar')
except ImportError:
    print('✗ Ultralytics nicht verfügbar')

try:
    import deepface
    print('✓ DeepFace verfügbar')
except ImportError:
    print('✗ DeepFace nicht verfügbar')

try:
    import ollama
    print('✓ Ollama Python-Bibliothek verfügbar')
except ImportError:
    print('✗ Ollama Python-Bibliothek nicht verfügbar')
"
    # Datenverzeichnis prüfen
    if [ ! -d "$DATA_DIR" ]; then
        log_message "✗ Datenverzeichnis nicht gefunden: $DATA_DIR"
        return 1
    else
        log_message "✓ Datenverzeichnis gefunden: $DATA_DIR"
        image_count=$(find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        log_message "  Gefunden: $image_count Bilder"
        if [ $image_count -eq 0 ]; then
            log_message "⚠ Keine Bilder gefunden - erstelle Testbilder?"
        fi
    fi
}

create_test_images() {
    log_message "Erstelle Test-Bilderstruktur..."
    
    mkdir -p "$DATA_DIR/test_with_people"
    mkdir -p "$DATA_DIR/test_without_people"
    mkdir -p "$DATA_DIR/test_uncertain"
    
    python3 -c "
import cv2, numpy as np, os
def create_test_image(filename, with_person=False):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    if with_person:
        cv2.circle(img, (200, 120), 30, (255, 200, 150), -1)
        cv2.rectangle(img, (170, 150), (230, 280), (100, 100, 255), -1)
    else:
        cv2.rectangle(img, (0, 300), (400, 400), (50, 150, 50), -1)
        cv2.circle(img, (100, 100), 50, (0, 255, 255), -1)
    cv2.imwrite(filename, img)

base_dir = '$DATA_DIR'
for i in range(3):
    create_test_image(f'{base_dir}/test_with_people/person_{i+1}.jpg', True)
    create_test_image(f'{base_dir}/test_without_people/landscape_{i+1}.jpg', False)
    create_test_image(f'{base_dir}/test_uncertain/mixed_{i+1}.jpg', i % 2 == 0)
print('✓ Test-Bilder erstellt')
"
    
    log_message "✓ Test-Bilderstruktur erstellt"
}

# =============================================================================
# HAUPTLOGIK
# =============================================================================

case "${1:-all}" in
    "prerequisites"|"prereq")
        check_prerequisites
        ;;
    "database"|"db")
        test_database_connection
        ;;
    "ultralytics"|"yolo")
        test_ultralytics
        ;;
    "deepface"|"df")
        test_deepface
        ;;
    "ollama")
        test_ollama
        ;;
    "create-test-images")
        create_test_images
        ;;
    "quick")
        log_message "=== SCHNELLTEST ALLER MODELLE ==="
        TEST_MAX_IMAGES=3
        check_prerequisites
        test_database_connection
        test_ultralytics
        test_deepface
        test_ollama
        ;;
    "all"|"")
        log_message "=== VOLLSTÄNDIGER TEST ALLER MODELLE ==="
        check_prerequisites
        test_database_connection
        if [ $? -eq 0 ]; then
            test_ultralytics
            sleep 5
            test_deepface  
            # sleep 5
            # test_ollama
        else
            log_message "✗ Datenbank-Test fehlgeschlagen - breche ab"
            exit 1
        fi
        ;;
    "help"|*)
        echo "Usage: $0 {prerequisites|database|ultralytics|deepface|ollama|create-test-images|quick|all}"
        echo ""
        echo "Commands:"
        echo "  prerequisites    - Prüfe Voraussetzungen und Bibliotheken"
        echo "  database         - Teste nur Datenbankverbindung"
        echo "  ultralytics      - Teste nur Ultralytics YOLO"
        echo "  deepface         - Teste nur DeepFace (alle Backends)"
        echo "  ollama-gemma3     - Teste nur Ollama (Gemma 3 Modelle)"
        echo "  create-test-images - Erstelle einfache Test-Bilder"
        echo "  quick            - Schnelltest aller Modelle (3 Bilder)"
        echo "  all              - Vollständiger Test aller Modelle"
        ;;
esac
