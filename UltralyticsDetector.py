from typing import Any, Dict

import numpy as np
import BaseDetector
import DetectionProcessor


class UltralyticsDetector(BaseDetector):
    """Ultralytics YOLO Detektor"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        from ultralytics import YOLO
        super().__init__("Ultralytics-YOLO", model_path)
        self.model = YOLO(model_path)
        self.person_class_id = 0  # Person hat Class ID 0 in COCO
        
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Erkennt Personen in einem Bild"""
        results = self.model(image_path, verbose=False)
        
        person_confidences = []
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.person_class_id:
                        person_confidences.append(confidence)
                        bbox = box.xyxy[0].tolist()
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'person'
                        })
        
        return {
            'persons_detected': len(person_confidences),
            'confidences': person_confidences,
            'detections': detections,
            'avg_confidence': np.mean(person_confidences) if person_confidences else 0,
            'max_confidence': max(person_confidences) if person_confidences else 0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das YOLO Modell zurück"""
        return {
            'model_type': 'YOLO',
            'model_file': self.model_version,
            'task': 'object_detection',
            'classes': ['person'],  # Wir interessieren uns nur für Personen
            'input_size': 640
        }


# Hauptfunktion für Ultralytics
def run_ultralytics_detection(config: Dict[str, Any]):
    """Führt Ultralytics Detection aus"""
    
    # Detektor initialisieren
    detector = UltralyticsDetector(config['model_path'])
    
    # Processor initialisieren
    processor = DetectionProcessor(
        detector=detector,
        db_config=config['database'],
        data_dir=config['data_directory'],
        run_config={'model_path': config['model_path']}
    )
    
    # Verarbeitung starten
    run_id = processor.process_images(
        max_images=config.get('max_images'),
        classifications=config.get('classifications'),
        randomize=config.get('randomize', True)
    )
    
    return run_id


def main():
    """Hauptfunktion - Beispiel für Ultralytics"""
    
    # Konfiguration
    CONFIG = {
        'data_directory': '/path/to/classified/data',  # Anpassen!
        'model_path': 'yolov8n.pt',  # oder 'yolov8s.pt', 'yolov8m.pt', etc.
        'max_images': None,  # None für alle Bilder, oder Zahl für Limit
        'classifications': None,  # None für alle, oder ['class1', 'class2']
        'randomize': True,  # Reihenfolge randomisieren
        'database': {
            'host': 'localhost',
            'user': 'your_username',     # Anpassen!
            'password': 'your_password', # Anpassen!
            'database': 'ai_detection'   # Anpassen!
        }
    }
    
    try:
        run_id = run_ultralytics_detection(CONFIG)
        print(f"\n🎉 Erfolgreich abgeschlossen! Run-ID: {run_id}")
        
    except KeyboardInterrupt:
        print("\n❌ Verarbeitung durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()