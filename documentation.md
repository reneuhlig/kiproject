# KI-Personenerkennung System

Ein umfassendes System zur automatisierten Personenerkennung in klassifizierten Bilddaten mit drei verschiedenen KI-Modellen: Ultralytics YOLO, DeepFace und Gemma/LLaVA.

## 🚀 Features

- **Drei KI-Modelle**: Ultralytics YOLO, DeepFace und Gemma/LLaVA für verschiedene Erkennungsansätze
- **Randomisierte Verarbeitung**: Verhindert Bias durch sequenzielle Verarbeitung gleicher Klassifizierungen
- **Konfidenz-Tracking**: Erkennt unsichere Vorhersagen und speichert Wahrscheinlichkeitswerte
- **MySQL Integration**: Vollständige Speicherung aller Ergebnisse mit Run-Tracking
- **Systemmonitoring**: CPU, RAM und GPU Auslastung während der Verarbeitung
- **Cronjob-fähig**: Automatisierte Ausführung über Zeitpläne
- **Performance-Metriken**: Detaillierte Zeitmessungen pro Bild und Run

## 📁 Projektstruktur

```
person_detection/
├── BaseDetector.py              # Abstrakte Basisklasse für Detektoren
├── DataLoader.py                # Lädt und verwaltet klassifizierte Bilddaten
├── DatabaseHandler.py           # MySQL Datenbankoperationen
├── DetectionProcessor.py        # Hauptverarbeitungslogik
├── SystemMonitor.py             # Systemressourcen-Monitoring
├── UltralyticsPersonDetector.py # YOLO-basierte Personenerkennung
├── DeepFacePersonDetector.py    # Gesichtsbasierte Personenerkennung
├── GemmaPersonDetector.py       # LLM-basierte Personenerkennung
├── run_person_detection.py      # Hauptausführungsscript
├── setup_environment.py         # Setup und Konfigurationstool
├── cronjob_setup.sh            # Cronjob-Automatisierung
├── database_schema.sql          # MySQL Datenbankschema
├── requirements.txt             # Python-Abhängigkeiten
└── README.md                    # Diese Dokumentation
```

## 🛠️ Installation

### 1. Repository klonen und Setup ausführen

```bash
git clone <repository-url>
cd person_detection

# Python-Umgebung erstellen (empfohlen)
python3 -m venv venv
source venv/bin/activate

# Setup-Script ausführen
python3 setup_environment.py --install --test-models --create-testdata ./data
```

### 2. MySQL Datenbank einrichten

```bash
# Datenbank und Tabellen erstellen
mysql -u root -p < database_schema.sql

# Oder mit anderem Benutzer:
mysql -u your_user -p your_database < database_schema.sql
```

### 3. Konfiguration anpassen

Bearbeiten Sie die Datenbankverbindung in `example_run.sh`:

```bash
DB_HOST="localhost"
DB_USER="ai_detection_user"
DB_PASSWORD="your_secure_password"
DB_NAME="ai_detection"
DATA_DIR="/path/to/your/classified/images"
```

## 📊 Datenstruktur

### Klassifizierte Bilddaten

Organisieren Sie Ihre Bilder in folgender Struktur:

```
classified_images/
├── with_people/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── without_people/
│   ├── image3.jpg
│   └── ...
├── group_photos/
│   └── ...
└── uncertain/
    └── ...
```

**Unterstützte Formate**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

## 🎯 Verwendung

### Einzelne Ausführung

```bash
# Ultralytics YOLO
python3 run_person_detection.py \
    --model ultralytics \
    --db-host localhost \
    --db-user your_user \
    --db-password your_password \
    --db-name ai_detection \
    --data-dir /path/to/images \
    --max-images 100 \
    --confidence-threshold 0.6

# DeepFace mit MTCNN Backend
python3 run_person_detection.py \
    --model deepface \
    --deepface-backend mtcnn \
    --db-host localhost \
    --db-user your_user \
    --db-password your_password \
    --db-name ai_detection \
    --data-dir /path/to/images \
    --max-images 50 \
    --confidence-threshold 0.5

# Gemma/LLaVA (benötigt GPU)
python3 run_person_detection.py \
    --model gemma \
    --db-host localhost \
    --db-user your_user \
    --db-password your_password \
    --db-name ai_detection \
    --data-dir /path/to/images \
    --max-images 20 \
    --confidence-threshold 0.7
```

### Automatisierte Ausführung (Cronjobs)

```bash
# Cronjob-Script konfigurieren
cp cronjob_setup.sh /path/to/your/cronjob_setup.sh
nano /path/to/your/cronjob_setup.sh  # Pfade anpassen

# Cronjobs installieren
chmod +x /path/to/your/cronjob_setup.sh
/path/to/your/cronjob_setup.sh install-cron
```

**Standard Cronjob-Zeiten:**
- **07:30**: Morgendlicher Schnellscan (Ultralytics)
- **14:00**: Detaillierte Nachmittags-Analyse (Ultralytics + DeepFace)
- **20:00**: Umfassende Abend-Analyse (alle Modelle)
- **So 02:00**: Wöchentlicher Vollscan
- **01:00**: Tägliche Log-Bereinigung

## 🤖 KI-Modelle

### 1. Ultralytics YOLO
- **Typ**: Objekterkennung
- **Stärken**: Schnell, präzise Bounding Boxes, gut für Echtzeit
- **Schwächen**: Nur vortrainierte Klassen
- **Empfohlene Nutzung**: Hauptscan, große Bildmengen

### 2. DeepFace
- **Typ**: Gesichtserkennung
- **Stärken**: Erkennt Personen über Gesichter, verschiedene Backends
- **Schwächen**: Nur frontal sichtbare Gesichter
- **Empfohlene Nutzung**: Detailanalyse, Portrait-Bilder

### 3. Gemma/LLaVA
- **Typ**: Vision-Language Model
- **Stärken**: Versteht Kontext, kann komplexe Szenen analysieren
- **Schwächen**: Langsam, hoher GPU-Bedarf, keine exakten Bounding Boxes
- **Empfohlene Nutzung**: Qualitätskontrolle, komplexe Szenen

## 📈 Datenbankschema

### Tabelle: `ai_runs`
Speichert Informationen über jeden Verarbeitungsdurchlauf:

- `run_id`: Eindeutige Run-ID (UUID)
- `model_name`: Verwendetes KI-Modell
- `start_time`/`end_time`: Zeitstempel
- `total_images`: Anzahl verarbeiteter Bilder
- `successful_detections`/`failed_detections`: Erfolgs-/Fehlerstatistik
- `avg_processing_time`: Durchschnittliche Zeit pro Bild
- `avg_cpu_usage`/`max_cpu_usage`: CPU-Auslastung
- `avg_memory_usage`/`max_memory_usage`: RAM-Auslastung
- `avg_gpu_usage`/`max_gpu_usage`: GPU-Auslastung
- `status`: Run-Status (running/completed/failed/cancelled)
- `config_json`: Verwendete Konfiguration

### Tabelle: `detection_results`
Speichert Einzelergebnisse pro Bild:

- `run_id`: Verknüpfung zum Run
- `image_path`/`image_filename`: Bildidentifikation
- `classification`: Ursprüngliche Klassifizierung
- `model_output`: Vollständige Modellausgabe (JSON)
- `confidence_scores`: Konfidenzwerte als String
- `persons_detected`: Anzahl erkannter Personen
- `avg_confidence`/`max_confidence`/`min_confidence`: Konfidenzstatistiken
- `is_uncertain`: Flag für unsichere Erkennungen
- `processing_time`: Zeit für diese Detection
- `success`: Erfolg/Fehler Flag

## 📊 Auswertung und Monitoring

### Nützliche SQL-Abfragen

```sql
-- Modellvergleich
SELECT * FROM model_comparison;

-- Klassifizierungsanalyse
SELECT * FROM classification_analysis;

-- Run-Übersicht der letzten 7 Tage
SELECT * FROM run_overview 
WHERE start_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY start_time DESC;

-- Top 10 Bilder mit den meisten Personen
SELECT image_filename, classification, persons_detected, avg_confidence, run_id
FROM detection_results 
WHERE success = TRUE 
ORDER BY persons_detected DESC, avg_confidence DESC 
LIMIT 10;

-- Unsichere Erkennungen
SELECT classification, COUNT(*) as total,
       COUNT(CASE WHEN is_uncertain = TRUE THEN 1 END) as uncertain,
       ROUND(COUNT(CASE WHEN is_uncertain = TRUE THEN 1 END) / COUNT(*) * 100, 2) as uncertain_percentage
FROM detection_results 
WHERE success = TRUE
GROUP BY classification
ORDER BY uncertain_percentage DESC;
```

## ⚙️ Konfigurationsoptionen

### Kommandozeilenparameter

```bash
python3 run_person_detection.py --help
```

**Wichtige Parameter:**
- `--model`: KI-Modell (ultralytics/deepface/gemma)
- `--max-images`: Maximale Bildanzahl pro Run
- `--confidence-threshold`: Mindest-Konfidenz (0.0-1.0)
- `--classifications`: Nur bestimmte Klassifizierungen verarbeiten
- `--no-randomize`: Deaktiviert Randomisierung
- `--run-name`: Name für den Run (für Tracking)
- `--job-id`: Job-ID für Cronjob-Zuordnung

### Modell-spezifische Parameter

**Ultralytics:**
- `--yolo-model-path`: Pfad zum YOLO-Modell (Standard: yolov8n.pt)

**DeepFace:**
- `--deepface-backend`: Backend (opencv/ssd/mtcnn/retinaface)

**Gemma:**
- `--gemma-model`: HuggingFace Modellname

## 🐛 Troubleshooting

### Häufige Probleme

1. **CUDA/GPU Probleme**
   ```bash
   # GPU-Status prüfen
   nvidia-smi
   
   # Torch CUDA-Verfügbarkeit testen
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **MySQL Verbindungsfehler**
   ```bash
   # MySQL-Service prüfen
   sudo systemctl status mysql
   
   # Verbindung testen
   mysql -h localhost -u your_user -p
   ```

3. **Speicherprobleme bei Gemma**
   ```bash
   # Reduziere max-images für Gemma
   --max-images 10
   
   # Oder verwende nur Ultralytics/DeepFace
   ```

4. **Fehlende Abhängigkeiten**
   ```bash
   # Re-Installation
   pip install -r requirements.txt
   
   # Spezifische Pakete
   pip install ultralytics deepface transformers torch
   ```

### Log-Analyse

```bash
# Cronjob-Logs
tail -f logs/cronjob.log

# Spezifische Run-Logs  
tail -f logs/ultralytics_*.log
tail -f logs/deepface_*.log
tail -f logs/gemma_*.log
```

## 🔧 Erweiterte Konfiguration

### Performance-Optimierung

1. **GPU-Memory für Gemma begrenzen**
   ```python
   # In GemmaPersonDetector.py
   self.model = LlavaForConditionalGeneration.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       low_cpu_mem_usage=True,
       max_memory={0: "8GB"}  # Begrenze GPU 0 auf 8GB
   )
   ```

2. **Batch-Processing aktivieren**
   ```python
   # Mehrere Bilder gleichzeitig verarbeiten (für Ultralytics)
   results = self.model(image_paths, batch=8)
   ```

3. **Systemmonitoring anpassen**
   ```python
   # In SystemMonitor.py - Monitoring-Intervall ändern
   time.sleep(1.0)  # Statt 0.5 Sekunden
   ```

### Cronjob-Anpassungen

```bash
# Verschiedene Zeitpläne für verschiedene Modelle
# Ultralytics: Alle 2 Stunden
0 */2 * * * /path/to/cronjob_setup.sh morning

# DeepFace: Zweimal täglich  
0 8,20 * * * /path/to/cronjob_setup.sh afternoon

# Gemma: Einmal täglich nachts
0 2 * * * /path/to/cronjob_setup.sh evening
```

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committe deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

## 📝 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe `LICENSE` Datei für Details.

## 🙏 Danksagungen

- [Ultralytics](https://ultralytics.com/) für YOLO
- [DeepFace](https://github.com/serengil/deepface) für Gesichtserkennung
- [HuggingFace](https://huggingface.co/) für LLaVA/Gemma Modelle
- [MySQL](https://mysql.com/) für die Datenbank

## 📞 Support

Bei Problemen oder Fragen:

1. Prüfe die [Troubleshooting](#-troubleshooting) Sektion
2. Schaue in die [Issues](https://github.com/your-repo/issues)
3. Erstelle ein neues Issue mit detaillierter Beschreibung

---

**Hinweis**: Dieses System wurde für Debian-Server optimiert und mit Python 3.8+ getestet.