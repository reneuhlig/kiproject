import time
import uuid
from typing import List, Optional, Dict, Any
import numpy as np
from BaseDetector import BaseDetector
from DataLoader import DataLoader
from DatabaseHandler import DatabaseHandler
from SystemMonitor import SystemMonitor


class DetectionProcessor:
    """Hauptklasse für die Verarbeitung mit beliebigen Detektoren"""
    
    def __init__(self, detector: BaseDetector, db_config: Dict[str, str], 
                 data_dir: str, run_config: Dict[str, Any] = None):
        """
        Initialisiert den Detection Processor
        
        Args:
            detector: Instanz eines BaseDetector
            db_config: MySQL Konfiguration
            data_dir: Pfad zu klassifizierten Daten
            run_config: Zusätzliche Konfiguration für den Run
        """
        self.detector = detector
        self.db = DatabaseHandler(**db_config)
        self.data_loader = DataLoader(data_dir)
        self.monitor = SystemMonitor()
        self.run_config = run_config or {}
        
    def process_images(self, max_images: Optional[int] = None,
                      classifications: List[str] = None,
                      randomize: bool = True) -> str:
        """
        Verarbeitet Bilder mit dem konfigurierten Detektor
        
        Args:
            max_images: Maximale Anzahl zu verarbeitender Bilder
            classifications: Nur bestimmte Klassifizierungen verarbeiten
            randomize: Reihenfolge randomisieren
            
        Returns:
            Run-ID für diesen Durchlauf
        """
        # Run-ID generieren
        run_id = str(uuid.uuid4())
        
        # Datenbankverbindung und Setup
        if not self.db.connect():
            raise Exception("Datenbankverbindung fehlgeschlagen")
            
        if not self.db.create_tables():
            raise Exception("Tabellenerstellung fehlgeschlagen")
            
        # Run in Datenbank erstellen
        model_info = self.detector.get_model_info()
        full_config = {
            **self.run_config,
            'max_images': max_images,
            'classifications': classifications,
            'randomize': randomize,
            'model_info': model_info
        }
        
        if not self.db.insert_run(run_id, self.detector.model_name, 
                                 self.detector.model_version, full_config):
            raise Exception("Run-Erstellung fehlgeschlagen")
        
        print(f"✓ Neuer Run gestartet: {run_id}")
        print(f"  Modell: {self.detector.model_name} v{self.detector.model_version}")
        
        # Bilder laden
        images = self.data_loader.get_classified_images(
            randomize=randomize, 
            classifications=classifications
        )
        
        if not images:
            print("✗ Keine Bilder zum Verarbeiten gefunden")
            return run_id
            
        if max_images:
            images = images[:max_images]
            print(f"  Limitiert auf {max_images} Bilder")
            
        # Monitoring starten
        self.monitor.start_monitoring()
        
        # Statistiken
        processing_times = []
        successful_detections = 0
        failed_detections = 0
        start_time = time.time()
        
        print(f"\nStarte Verarbeitung von {len(images)} Bildern...")
        print("-" * 80)
        
        status = 'completed'  # Initialize status
        
        try:
            for i, (image_path, classification) in enumerate(images, 1):
                detection_start = time.time()
                
                try:
                    # Detection durchführen
                    result = self.detector.detect(image_path)
                    processing_time = time.time() - detection_start
                    processing_times.append(processing_time)
                    
                    # Bildinformationen
                    image_info = self.data_loader.get_image_info(image_path)
                    
                    # Ergebnis in Datenbank speichern
                    confidence_str = self._format_confidences(result.get('confidences', []))
                    
                    success = self.db.insert_result(
                        run_id=run_id,
                        image_path=image_path,
                        image_filename=image_info['filename'],
                        classification=classification,
                        model_output=result,
                        confidence_scores=confidence_str,
                        processing_time=processing_time,
                        success=True
                    )
                    
                    if success:
                        successful_detections += 1
                    else:
                        failed_detections += 1
                        
                    # Status ausgeben
                    self._print_detection_result(i, len(images), image_info['filename'],
                                               classification, result, processing_time)
                    
                except Exception as e:
                    processing_time = time.time() - detection_start
                    processing_times.append(processing_time)
                    failed_detections += 1
                    
                    # Fehler in Datenbank speichern
                    image_info = self.data_loader.get_image_info(image_path)
                    self.db.insert_result(
                        run_id=run_id,
                        image_path=image_path,
                        image_filename=image_info['filename'],
                        classification=classification,
                        model_output=None,
                        confidence_scores="",
                        processing_time=processing_time,
                        success=False,
                        error_message=str(e)
                    )
                    
                    print(f"[{i:4d}/{len(images)}] {image_info['filename']} "
                          f"({classification}) -> FEHLER: {e}")
                
                # Kurze Pause zwischen den Bildern
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n❌ Verarbeitung durch Benutzer abgebrochen")
            status = 'cancelled'
        except Exception as e:
            print(f"\n❌ Kritischer Fehler: {e}")
            status = 'failed'
        
        # Monitoring stoppen
        self.monitor.stop_monitoring()
        
        # Statistiken berechnen
        total_time = time.time() - start_time
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        system_stats = self.monitor.get_average_usage()
        
        # Run-Informationen aktualisieren
        self.db.update_run_completion(
            run_id=run_id,
            total_images=len(images),
            successful_detections=successful_detections,
            failed_detections=failed_detections,
            avg_processing_time=avg_processing_time,
            total_processing_time=total_time,
            system_stats=system_stats,
            status=status
        )
        
        # Zusammenfassung ausgeben
        self._print_summary(run_id, len(images), successful_detections, 
                           failed_detections, total_time, avg_processing_time,
                           system_stats)
        
        # Datenbankverbindung schließen
        self.db.close()
        
        return run_id
    
    def _format_confidences(self, confidences: List[float]) -> str:
        """Formatiert Konfidenzwerte als String"""
        if not confidences:
            return ""
        return ",".join([f"{c:.3f}" for c in confidences])
    
    def _print_detection_result(self, current: int, total: int, filename: str,
                               classification: str, result: Dict[str, Any],
                               processing_time: float):
        """Gibt Erkennungsergebnis auf Konsole aus"""
        persons = result.get('persons_detected', 0)
        conf = result.get('avg_confidence', 0.0)
        uncertain = "⚠" if result.get('uncertain', False) else "✓"
        
        print(f"[{current:4d}/{total}] {filename} ({classification}) -> "
              f"{persons} Personen, Konfidenz: {conf:.3f} {uncertain}, Zeit: {processing_time:.3f}s")
    
    def _print_summary(self, run_id: str, total_images: int, successful: int,
                      failed: int, total_time: float, avg_time: float,
                      system_stats: Dict[str, float]):
        """Gibt Zusammenfassung auf Konsole aus"""
        print("-" * 80)
        print(f"✓ Verarbeitung abgeschlossen!")
        print(f"  Run-ID: {run_id}")
        print(f"  Modell: {self.detector.model_name} v{self.detector.model_version}")
        print(f"  Verarbeitete Bilder: {successful + failed}/{total_images}")
        print(f"  Erfolgreiche Detections: {successful}")
        print(f"  Fehlgeschlagene Detections: {failed}")
        print(f"  Gesamtzeit: {total_time:.1f}s")
        print(f"  Durchschnittliche Zeit pro Bild: {avg_time:.3f}s")
        print(f"  Durchschnittliche CPU-Auslastung: {system_stats['avg_cpu']:.1f}%")
        print(f"  Maximale CPU-Auslastung: {system_stats['max_cpu']:.1f}%")
        print(f"  Durchschnittliche RAM-Auslastung: {system_stats['avg_memory']:.1f}%")
        if system_stats['avg_gpu'] > 0:
            print(f"  Durchschnittliche GPU-Auslastung: {system_stats['avg_gpu']:.1f}%")