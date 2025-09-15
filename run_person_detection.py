#!/usr/bin/env python3
"""
Verbessertes Hauptscript für die Personenerkennung mit verschiedenen KI-Modellen
Jetzt mit CSV-Export und verbesserter Fehlerbehandlung
"""

import argparse
import sys
import os
from typing import Optional, List
import logging

# Import der Detector-Implementierungen
from UltralyticsPersonDetector import UltralyticsPersonDetector
from DeepFacePersonDetector import DeepFacePersonDetector
from GemmaPersonDetector import GemmaPersonDetector

try:
    from OllamaGemma3PersonDetector import OllamaGemma3PersonDetector
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠ OllamaGemma3PersonDetector nicht verfügbar")

from DetectionProcessor import DetectionProcessor

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_detector(model_name: str, **kwargs):
    """
    Verbesserte Factory function für Detector-Erstellung mit Fehlerbehandlung
    
    Args:
        model_name: Name des Modells ('ultralytics', 'deepface', 'gemma', 'ollama-gemma3')
        **kwargs: Zusätzliche Argumente für den Detector
    """
    try:
        if model_name.lower() == 'ultralytics':
            return UltralyticsPersonDetector(
                model_path=kwargs.get('model_path', 'yolov8n.pt'),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5)
            )
        
        elif model_name.lower() == 'deepface':
            return DeepFacePersonDetector(
                detector_backend=kwargs.get('detector_backend', 'opencv'),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5)
            )
        
        elif model_name.lower() == 'gemma':
            return GemmaPersonDetector(
                model_name=kwargs.get('hf_model_name', 'llava-hf/llava-1.5-7b-hf'),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5)
            )
        
        elif model_name.lower() == 'ollama-gemma3':
            if not OLLAMA_AVAILABLE:
                raise ImportError("OllamaGemma3PersonDetector nicht verfügbar")
            
            return OllamaGemma3PersonDetector(
                model_name=kwargs.get('ollama_model', 'gemma3:4b'),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5),
                ollama_host=kwargs.get('ollama_host', 'http://localhost:11434')
            )
        
        else:
            available_models = ['ultralytics', 'deepface', 'gemma']
            if OLLAMA_AVAILABLE:
                available_models.append('ollama-gemma3')
            raise ValueError(f"Unbekanntes Modell: {model_name}. Verfügbar: {', '.join(available_models)}")
            
    except Exception as e:
        logger.error(f"Fehler bei der Detector-Erstellung für {model_name}: {e}")
        raise


def validate_arguments(args) -> bool:
    """Validiert die Kommandozeilenargumente"""
    # Datenverzeichnis prüfen
    if not os.path.exists(args.data_dir):
        logger.error(f"Datenverzeichnis nicht gefunden: {args.data_dir}")
        return False
    
    if not os.path.isdir(args.data_dir):
        logger.error(f"Datenverzeichnis ist keine Datei: {args.data_dir}")
        return False
    
    # CSV-Export-Verzeichnis erstellen falls nötig
    if hasattr(args, 'csv_export_dir') and args.csv_export_dir:
        try:
            os.makedirs(args.csv_export_dir, exist_ok=True)
            logger.info(f"CSV-Export-Verzeichnis bereit: {args.csv_export_dir}")
        except Exception as e:
            logger.warning(f"CSV-Export-Verzeichnis konnte nicht erstellt werden: {e}")
    
    # Confidence-Threshold prüfen
    if not 0.0 <= args.confidence_threshold <= 1.0:
        logger.error(f"Confidence-Threshold muss zwischen 0.0 und 1.0 liegen: {args.confidence_threshold}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Personenerkennung mit verschiedenen KI-Modellen (mit CSV-Export)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Modell-Auswahl
    model_choices = ['ultralytics', 'deepface', 'gemma']
    if OLLAMA_AVAILABLE:
        model_choices.append('ollama-gemma3')
    
    parser.add_argument(
        '--model', 
        choices=model_choices, 
        required=True,
        help='KI-Modell für die Personenerkennung'
    )
    
    # Datenbank-Konfiguration
    parser.add_argument('--db-host', default='localhost', help='MySQL Host')
    parser.add_argument('--db-user', required=True, help='MySQL Benutzername')
    parser.add_argument('--db-password', required=True, help='MySQL Passwort')
    parser.add_argument('--db-name', required=True, help='MySQL Datenbankname')
    
    # Daten-Konfiguration
    parser.add_argument('--data-dir', required=True, help='Pfad zu klassifizierten Bilddaten')
    parser.add_argument('--max-images', type=int, help='Maximale Anzahl zu verarbeitender Bilder')
    parser.add_argument('--classifications', nargs='+', help='Nur bestimmte Klassifizierungen verarbeiten')
    parser.add_argument('--no-randomize', action='store_true', help='Deaktiviert Randomisierung der Bilderreihenfolge')
    
    # CSV-Export-Konfiguration
    parser.add_argument('--csv-export-dir', default='./csv_exports', help='Verzeichnis für CSV-Exports')
    parser.add_argument('--no-csv-export', action='store_true', help='Deaktiviert CSV-Export')
    
    # Modell-spezifische Optionen
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Mindest-Konfidenz für Detections')
    
    # Ultralytics-spezifisch
    parser.add_argument('--yolo-model-path', default='yolov8n.pt', help='Pfad zum YOLO Modell')
    
    # DeepFace-spezifisch
    parser.add_argument('--deepface-backend', 
                       choices=['opencv', 'ssd', 'mtcnn', 'retinaface'], 
                       default='opencv',
                       help='DeepFace Detector Backend')
    
    # Gemma (HuggingFace)-spezifisch
    parser.add_argument('--gemma-model', default='llava-hf/llava-1.5-7b-hf', 
                       help='HuggingFace Modellname für Gemma/LLaVA')
    
    # Ollama Gemma3-spezifisch
    parser.add_argument('--ollama-model', 
                       choices=['gemma3:270m', 'gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'gemma3:27b'],
                       default='gemma3:4b',
                       help='Ollama Gemma 3 Modell')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                       help='Ollama Server URL')
    
    # Run-Konfiguration
    parser.add_argument('--run-name', help='Name für diesen Run (für Tracking)')
    parser.add_argument('--job-id', help='Job-ID für Cronjob-Tracking')
    
    # Debug-Optionen
    parser.add_argument('--verbose', '-v', action='store_true', help='Detaillierte Ausgabe')
    parser.add_argument('--dry-run', action='store_true', help='Testlauf ohne echte Verarbeitung')
    
    args = parser.parse_args()
    
    # Logging-Level anpassen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Argumente validieren
    if not validate_arguments(args):
        sys.exit(1)
    
    # Dry-run Modus
    if args.dry_run:
        print("🧪 DRY-RUN Modus aktiviert - keine echte Verarbeitung")
        print(f"  Modell: {args.model}")
        print(f"  Datenverzeichnis: {args.data_dir}")
        print(f"  Max. Bilder: {args.max_images or 'Alle'}")
        print(f"  CSV-Export: {'Deaktiviert' if args.no_csv_export else args.csv_export_dir}")
        return
    
    # Datenbankverbindung konfigurieren
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name
    }
    
    # Run-Konfiguration erstellen
    run_config = {
        'run_name': args.run_name,
        'job_id': args.job_id,
        'script_args': vars(args),
        'confidence_threshold': args.confidence_threshold
    }
    
    try:
        # Detector erstellen
        detector_kwargs = {
            'confidence_threshold': args.confidence_threshold
        }
        
        if args.model == 'ultralytics':
            detector_kwargs['model_path'] = args.yolo_model_path
        elif args.model == 'deepface':
            detector_kwargs['detector_backend'] = args.deepface_backend
        elif args.model == 'gemma':
            detector_kwargs['hf_model_name'] = args.gemma_model
        elif args.model == 'ollama-gemma3':
            detector_kwargs['ollama_model'] = args.ollama_model
            detector_kwargs['ollama_host'] = args.ollama_host
        
        print(f"🚀 Initialisiere {args.model} Detektor...")
        detector = create_detector(args.model, **detector_kwargs)
        
        # Detection Processor erstellen
        csv_export_dir = None if args.no_csv_export else args.csv_export_dir
        
        processor = DetectionProcessor(
            detector=detector,
            db_config=db_config,
            data_dir=args.data_dir,
            run_config=run_config,
            csv_export_dir=csv_export_dir
        )
        
        # Verarbeitung starten
        print("\n" + "="*80)
        print("🔍 PERSONENERKENNUNG GESTARTET")
        print("="*80)
        
        run_id = processor.process_images(
            max_images=args.max_images,
            classifications=args.classifications,
            randomize=not args.no_randomize
        )
        
        print(f"\n✅ Run erfolgreich abgeschlossen: {run_id}")
        
        # CSV-Export-Info
        if not args.no_csv_export:
            summary = processor.csv_exporter.get_export_summary()
            print(f"📊 CSV-Exports in: {summary['export_directory']}")
        
    except KeyboardInterrupt:
        print("\n❌ Verarbeitung durch Benutzer abgebrochen")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()