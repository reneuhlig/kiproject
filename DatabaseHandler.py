from datetime import datetime
from typing import  Dict


import mysql.connector
from mysql.connector import Error
import json


class DatabaseHandler:
    """Handhabt MySQL Datenbankoperationen für alle KI-Modelle"""
    
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self) -> bool:
        """Verbindet zur MySQL Datenbank"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=True
            )
            print(f"✓ Erfolgreich mit MySQL Datenbank verbunden ({self.host})")
            return True
        except Error as e:
            print(f"✗ Fehler bei Datenbankverbindung: {e}")
            return False
            
    def create_tables(self) -> bool:
        """Erstellt die benötigten Tabellen falls sie nicht existieren"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        
        # Tabelle für Run-Informationen (erweitert für alle Modelle)
        create_runs_table = """
        CREATE TABLE IF NOT EXISTS ai_runs (
            run_id VARCHAR(36) PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            total_images INT DEFAULT 0,
            successful_detections INT DEFAULT 0,
            failed_detections INT DEFAULT 0,
            avg_processing_time FLOAT,
            total_processing_time FLOAT,
            avg_cpu_usage FLOAT,
            max_cpu_usage FLOAT,
            avg_memory_usage FLOAT,
            max_memory_usage FLOAT,
            avg_gpu_usage FLOAT,
            max_gpu_usage FLOAT,
            status ENUM('running', 'completed', 'failed', 'cancelled') DEFAULT 'running',
            error_message TEXT,
            config_json TEXT
        )
        """
        
        # Tabelle für Erkennungsergebnisse (generisch für alle Modelle)
        # KORREKTUR: is_uncertain als BOOLEAN hinzugefügt
        create_results_table = """
        CREATE TABLE IF NOT EXISTS detection_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(36) NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            image_filename VARCHAR(255) NOT NULL,
            classification VARCHAR(100),
            model_output JSON,
            confidence_scores TEXT,
            processing_time FLOAT NOT NULL,
            success BOOLEAN DEFAULT TRUE,
            is_uncertain BOOLEAN DEFAULT FALSE,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES ai_runs(run_id),
            INDEX idx_run_id (run_id),
            INDEX idx_classification (classification),
            INDEX idx_timestamp (timestamp)
        )
        """
        
        try:
            cursor.execute(create_runs_table)
            cursor.execute(create_results_table)
            print("✓ Datenbanktabellen erstellt/überprüft")
            return True
        except Error as e:
            print(f"✗ Fehler beim Erstellen der Tabellen: {e}")
            return False
        finally:
            cursor.close()
            
    def insert_run(self, run_id: str, model_name: str, model_version: str = None, 
                   config: Dict = None) -> bool:
        """Fügt einen neuen Run in die Datenbank ein"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO ai_runs (run_id, model_name, model_version, start_time, config_json)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        config_json = json.dumps(config) if config else None
        
        try:
            cursor.execute(query, (run_id, model_name, model_version, 
                                 datetime.now(), config_json))
            return True
        except Error as e:
            print(f"✗ Fehler beim Einfügen des Runs: {e}")
            return False
        finally:
            cursor.close()
            
    def insert_result(self, run_id: str, image_path: str, image_filename: str,
                     classification: str, model_output: Dict, confidence_scores: str,
                     processing_time: float, success: bool = True, 
                     error_message: str = None) -> bool:
        """
        Fügt ein Erkennungsergebnis in die Datenbank ein
        
        KORREKTUR: Angepasst an bestehende Tabellenstruktur mit allen Spalten
        """
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO detection_results 
        (run_id, image_path, image_filename, classification, model_output, 
         confidence_scores, processing_time, success, error_message, 
         persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        model_output_json = json.dumps(model_output) if model_output else None
        
        # Extrahiere Werte aus model_output
        persons_detected = 0
        avg_confidence = None
        max_confidence = None
        min_confidence = None
        is_uncertain = False
        
        if model_output and isinstance(model_output, dict):
            # Personen-Anzahl
            persons_detected = model_output.get('persons_detected', 0)
            if not isinstance(persons_detected, int):
                persons_detected = 0
            
            # Konfidenz-Werte
            avg_confidence = model_output.get('avg_confidence', None)
            max_confidence = model_output.get('max_confidence', None)
            min_confidence = model_output.get('min_confidence', None)
            
            # Konvertiere zu Float oder None
            for conf_name, conf_value in [('avg', avg_confidence), ('max', max_confidence), ('min', min_confidence)]:
                if conf_value is not None:
                    try:
                        if conf_name == 'avg':
                            avg_confidence = float(conf_value) if conf_value > 0 else None
                        elif conf_name == 'max':
                            max_confidence = float(conf_value) if conf_value > 0 else None
                        elif conf_name == 'min':
                            min_confidence = float(conf_value) if conf_value > 0 else None
                    except (ValueError, TypeError):
                        if conf_name == 'avg':
                            avg_confidence = None
                        elif conf_name == 'max':
                            max_confidence = None
                        elif conf_name == 'min':
                            min_confidence = None

                            
            is_uncertain_db = 0

            if model_output and isinstance(model_output, dict):
                persons_detected = model_output.get('persons_detected', 0)            
                def uncertain_to_int(value):
                    if isinstance(value, bool):
                        return int(value)
                    elif isinstance(value, str):
                        return 1 if value.lower() in ('true', '1', 'yes', 'on') else 0
                    elif isinstance(value, (int, float)):
                        return 1 if value else 0
                    else:
                        return 0

                is_uncertain_db = uncertain_to_int(model_output.get('uncertain', False))

        try:
            cursor.execute(query, (
                run_id, image_path, image_filename, classification,
                model_output_json, confidence_scores, processing_time,
                success, error_message,
                persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain_db
            ))
            return True
        except Error as e:
            print(f"✗ Fehler beim Einfügen des Ergebnisses: {e}")
            print(f"   Daten: persons={persons_detected}, avg_conf={avg_confidence}, max_conf={max_confidence}, min_conf={min_confidence}, uncertain={is_uncertain_db}")
            return False
        finally:
            cursor.close()
            
    def update_run_completion(self, run_id: str, total_images: int, 
                            successful_detections: int, failed_detections: int,
                            avg_processing_time: float, total_processing_time: float,
                            system_stats: Dict, status: str = 'completed',
                            error_message: str = None) -> bool:
        """Aktualisiert Run-Informationen nach Abschluss"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        UPDATE ai_runs SET 
        end_time = %s,
        total_images = %s,
        successful_detections = %s,
        failed_detections = %s,
        avg_processing_time = %s,
        total_processing_time = %s,
        avg_cpu_usage = %s,
        max_cpu_usage = %s,
        avg_memory_usage = %s,
        max_memory_usage = %s,
        avg_gpu_usage = %s,
        max_gpu_usage = %s,
        status = %s,
        error_message = %s
        WHERE run_id = %s
        """
        
        try:
            cursor.execute(query, (
                datetime.now(),
                total_images,
                successful_detections,
                failed_detections,
                avg_processing_time,
                total_processing_time,
                system_stats['avg_cpu'],
                system_stats['max_cpu'],
                system_stats['avg_memory'],
                system_stats['max_memory'],
                system_stats['avg_gpu'],
                system_stats['max_gpu'],
                status,
                error_message,
                run_id
            ))
            return True
        except Error as e:
            print(f"✗ Fehler beim Aktualisieren des Runs: {e}")
            return False
        finally:
            cursor.close()
    
    def fix_existing_table(self) -> bool:
        """
        Fügt fehlende Spalte zu bestehender Tabelle hinzu
        Aufruf falls Tabelle bereits existiert aber Spalte fehlt
        """
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        
        try:
            # Prüfe ob Spalte bereits existiert
            cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'detection_results' 
                AND COLUMN_NAME = 'is_uncertain'
            """, (self.database,))
            
            if not cursor.fetchone():
                # Spalte existiert nicht - füge sie hinzu
                cursor.execute("""
                    ALTER TABLE detection_results 
                    ADD COLUMN is_uncertain BOOLEAN DEFAULT FALSE
                """)
                print("✓ Spalte 'is_uncertain' hinzugefügt")
            else:
                print("✓ Spalte 'is_uncertain' bereits vorhanden")
                
            return True
        except Error as e:
            print(f"✗ Fehler beim Korrigieren der Tabelle: {e}")
            return False
        finally:
            cursor.close()
            
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ Datenbankverbindung geschlossen")