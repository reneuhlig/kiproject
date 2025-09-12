#!/usr/bin/env python3
"""
Setup-Script für die Personenerkennungsumgebung
Überprüft Dependencies, lädt Modelle herunter und testet die Konfiguration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import mysql.connector
from mysql.connector import Error


def check_python_version():
    """Überprüft Python-Version"""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 oder höher erforderlich")
        return False
    print(f"✓ Python {sys.version.split()[0]} gefunden")
    return True


def install_requirements():
    """Installiert Requirements"""
    print("Installiere Python-Abhängigkeiten...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Abhängigkeiten installiert")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Fehler bei Installation: {e}")
        return False


def test_imports():
    """Testet wichtige Imports"""
    imports_to_test = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("psutil", "psutil"),
        ("mysql.connector", "MySQL Connector"),
        ("ultralytics", "Ultralytics YOLO"),
        ("deepface", "DeepFace"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers")
    ]
    
    failed_imports = []
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Fehlt!")
            failed_imports.append(name)
    
    return len(failed_imports) == 0


def download_yolo_model():
    """Lädt YOLO-Modell herunter"""
    try:
        from ultralytics import YOLO
        print("Lade YOLO-Modell herunter...")
        model = YOLO('yolov8n.pt')  # Lädt automatisch herunter
        print("✓ YOLO v8n Modell heruntergeladen")
        return True
    except Exception as e:
        print(f"✗ YOLO-Modell Download fehlgeschlagen: {e}")
        return False


def test_database_connection(host, user, password, database):
    """Testet Datenbankverbindung"""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            print("✓ MySQL-Verbindung erfolgreich")
            connection.close()
            return True
    except Error as e:
        print(f"✗ MySQL-Verbindung fehlgeschlagen: {e}")
        return False


def create_test_data_structure(base_dir):
    """Erstellt Testdatenstruktur"""
    test_dir = Path(base_dir) / "test_data"
    
    # Klassifizierungsordner erstellen
    classes = ["with_people", "without_people", "uncertain", "group_photos"]
    
    for class_name in classes:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # README in jeden Ordner
        readme_file = class_dir / "README.txt"
        readme_file.write_text(f"""
Ordner für Bilder der Klasse: {class_name}

Unterstützte Formate: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp

Legen Sie hier Ihre klassifizierten Bilder ab.
        """)
    
    print(f"✓ Testdatenstruktur erstellt: {test_dir}")
    return str(test_dir)


def test_gemma_model():
    """Testet Gemma/LLaVA Modell (optional)"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ Kein CUDA verfügbar - Gemma wird CPU verwenden (sehr langsam)")
        
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        print("✓ LLaVA Processor geladen")
        return True
    except Exception as e:
        print(f"✗ Gemma/LLaVA Test fehlgeschlagen: {e}")
        return False


def create_sample_config():
    """Erstellt Beispielkonfiguration"""
    config_content = """#!/bin/bash
# Beispiel-Script für Cronjob

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="ai_detection_user"  
DB_PASSWORD="your_password_here"
DB_NAME="ai_detection_db"

# Datenpfad
DATA_DIR="/path/to/your/classified/images"

# Ultralytics Run
python3 run_person_detection.py \\
    --model ultralytics \\
    --db-host $DB_HOST \\
    --db-user $DB_USER \\
    --db-password $DB_PASSWORD \\
    --db-name $DB_NAME \\
    --data-dir $DATA_DIR \\
    --max-images 100 \\
    --confidence-threshold 0.6 \\
    --run-name "daily_ultralytics_scan" \\
    --job-id "$(date +%Y%m%d_%H%M%S)"

# DeepFace Run  
python3 run_person_detection.py \\
    --model deepface \\
    --db-host $DB_HOST \\
    --db-user $DB_USER \\
    --db-password $DB_PASSWORD \\
    --db-name $DB_NAME \\
    --data-dir $DATA_DIR \\
    --max-images 50 \\
    --deepface-backend mtcnn \\
    --confidence-threshold 0.5 \\
    --run-name "daily_deepface_scan" \\
    --job-id "$(date +%Y%m%d_%H%M%S)"

# Gemma Run (nur bei ausreichend GPU-Memory)
python3 run_person_detection.py \\
    --model gemma \\
    --db-host $DB_HOST \\
    --db-user $DB_USER \\
    --db-password $DB_PASSWORD \\
    --db-name $DB_NAME \\
    --data-dir $DATA_DIR \\
    --max-images 20 \\
    --confidence-threshold 0.7 \\
    --run-name "daily_gemma_scan" \\
    --job-id "$(date +%Y%m%d_%H%M%S)"
"""
    
    with open("example_run.sh", "w") as f:
        f.write(config_content)
    
    os.chmod("example_run.sh", 0o755)
    print("✓ Beispiel-Script erstellt: example_run.sh")


def main():
    parser = argparse.ArgumentParser(description='Setup Personenerkennungsumgebung')
    parser.add_argument('--install', action='store_true', help='Installiere Abhängigkeiten')
    parser.add_argument('--test-db', action='store_true', help='Teste Datenbankverbindung')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-user', default='root')
    parser.add_argument('--db-password', default='')
    parser.add_argument('--db-name', default='ai_detection')
    parser.add_argument('--create-testdata', help='Erstelle Testdatenstruktur im angegebenen Pfad')
    parser.add_argument('--test-models', action='store_true', help='Teste alle Modelle')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PERSONENERKENNUNG SETUP")
    print("="*60)
    
    success = True
    
    # Python-Version prüfen
    if not check_python_version():
        success = False
    
    # Dependencies installieren
    if args.install:
        if not install_requirements():
            success = False
    
    # Imports testen
    print("\nTeste Imports...")
    if not test_imports():
        success = False
    
    # YOLO-Modell herunterladen
    print("\nTeste YOLO...")
    if not download_yolo_model():
        success = False
    
    # Datenbankverbindung testen
    if args.test_db:
        print("\nTeste Datenbank...")
        if not test_database_connection(args.db_host, args.db_user, 
                                       args.db_password, args.db_name):
            success = False
    
    # Testdaten erstellen
    if args.create_testdata:
        print("\nErstelle Testdatenstruktur...")
        create_test_data_structure(args.create_testdata)
    
    # Modelle testen
    if args.test_models:
        print("\nTeste Gemma/LLaVA...")
        test_gemma_model()  # Nicht kritisch für success
    
    # Beispielkonfiguration erstellen
    create_sample_config()
    
    print("\n" + "="*60)
    if success:
        print("✓ Setup erfolgreich abgeschlossen!")
        print("\nNächste Schritte:")
        print("1. Passe example_run.sh mit deinen Datenbankdaten an")
        print("2. Lege klassifizierte Bilder in die Ordnerstruktur")
        print("3. Starte einen Testlauf mit run_person_detection.py")
    else:
        print("✗ Setup mit Fehlern abgeschlossen!")
        print("Bitte behebe die oben genannten Probleme.")
    print("="*60)


if __name__ == "__main__":
    main()