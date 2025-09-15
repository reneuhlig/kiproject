#!/bin/bash
# test_models.sh - Separate Testläufe für alle KI-Modelle

# =============================================================================
# KONFIGURATION - BITTE ANPASSEN
# =============================================================================

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="geheimespasswort"
DB_NAME="ai_detection"

# Datenpfad zu klassifizierten Bildern
DATA_DIR="/blob"

# Test-Parameter
TEST_MAX_IMAGES=1  # Kleine Anzahl für Tests
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
    
    # Test mit verschiedenen Backends
    echo "1. Test mit OpenCV Backend:"
    test_model "deepface" "--deepface-backend opencv"
    
    echo "2. Test mit MTCNN Backend:"
    test_model "deepface" "--deepface-backend mtcnn"
    
    echo "3. Test mit SSD Backend:"
    test_model "deepface" "--deepface-backend ssd"
}

test_gemma() {
    log_message "Starte Gemma/LLaVA Test..."
    
    # Prüfe ob GPU verfügbar
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        log_message "GPU erkannt - verwende Standard-Modell"
        test_model "gemma" "--gemma-model llava-hf/llava-1.5-7b-hf"
    else
        log_message "Keine GPU erkannt - teste trotzdem (wird langsam sein)"
        test_model "gemma" "--gemma-model llava-hf/llava-1.5-7b-hf"
    fi
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
    
    # Python-Imports testen
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
    import torch, transformers
    print('✓ PyTorch/Transformers verfügbar')
    if torch.cuda.is_available():
        print(f'✓ CUDA verfügbar - GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠ Kein CUDA verfügbar - GPU-Modelle werden CPU verwenden')
except ImportError:
    print('✗ PyTorch/Transformers nicht verfügbar')
"

    # Datenverzeichnis prüfen
    if [ ! -d "$DATA_DIR" ]; then
        log_message "✗ Datenverzeichnis nicht gefunden: $DATA_DIR"
        return 1
    else
        log_message "✓ Datenverzeichnis gefunden: $DATA_DIR"
        
        # Anzahl Bilder zählen
        image_count=$(find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        log_message "  Gefunden: $image_count Bilder"
        
        if [ $image_count -eq 0 ]; then
            log_message "⚠ Keine Bilder gefunden - erstelle Testbilder?"
        fi
    fi
}

create_test_images() {
    log_message "Erstelle Test-Bilderstruktur..."
    
    # Erstelle Ordnerstruktur
    mkdir -p "$DATA_DIR/test_with_people"
    mkdir -p "$DATA_DIR/test_without_people"
    mkdir -p "$DATA_DIR/test_uncertain"
    
    # Erstelle einfache Test-Bilder mit OpenCV
    python3 -c "
import cv2
import numpy as np
import os

# Erstelle einfache Test-Bilder
def create_test_image(filename, with_person=False):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    if with_person:
        # Zeichne einfache Person (Kreis für Kopf, Rechteck für Körper)
        cv2.circle(img, (200, 120), 30, (255, 200, 150), -1)  # Kopf
        cv2.rectangle(img, (170, 150), (230, 280), (100, 100, 255), -1)  # Körper
        cv2.rectangle(img, (160, 280), (190, 350), (50, 50, 200), -1)  # Bein 1
        cv2.rectangle(img, (210, 280), (240, 350), (50, 50, 200), -1)  # Bein 2
    else:
        # Zeichne Landschaft
        cv2.rectangle(img, (0, 300), (400, 400), (50, 150, 50), -1)  # Gras
        cv2.circle(img, (100, 100), 50, (0, 255, 255), -1)  # Sonne
        cv2.rectangle(img, (300, 200), (350, 300), (139, 69, 19), -1)  # Baum
    
    cv2.imwrite(filename, img)

# Erstelle Testbilder
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
    "gemma"|"llava")
        test_gemma
        ;;
    "create-test-images")
        create_test_images
        ;;
    "quick")
        log_message "=== SCHNELLTEST ALLER MODELLE ==="
        TEST_MAX_IMAGES=3  # Nur 3 Bilder für Schnelltest
        check_prerequisites
        test_database_connection
        test_ultralytics
        test_deepface
        test_gemma
        ;;
    "all"|"")
        log_message "=== VOLLSTÄNDIGER TEST ALLER MODELLE ==="
        check_prerequisites
        test_database_connection
        
        if [ $? -eq 0 ]; then
            test_ultralytics
            sleep 5
            test_deepface  
            sleep 5
            test_gemma
        else
            log_message "✗ Datenbank-Test fehlgeschlagen - breche ab"
            exit 1
        fi
        ;;
    "help"|*)
        echo "Usage: $0 {prerequisites|database|ultralytics|deepface|gemma|create-test-images|quick|all}"
        echo ""
        echo "Commands:"
        echo "  prerequisites    - Prüfe Voraussetzungen und Bibliotheken"
        echo "  database        - Teste nur Datenbankverbindung"  
        echo "  ultralytics     - Teste nur Ultralytics YOLO"
        echo "  deepface        - Teste nur DeepFace (alle Backends)"
        echo "  gemma          - Teste nur Gemma/LLaVA"
        echo "  create-test-images - Erstelle einfache Test-Bilder"
        echo "  quick          - Schnelltest aller Modelle (3 Bilder)"
        echo "  all            - Vollständiger Test aller Modelle"
        echo ""
        echo "Konfiguration:"
        echo "  DATA_DIR: $DATA_DIR"  
        echo "  DB_HOST: $DB_HOST"
        echo "  TEST_MAX_IMAGES: $TEST_MAX_IMAGES"
        ;;
esac