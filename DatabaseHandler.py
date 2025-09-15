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
            persons_detected INT DEFAULT 0,
            avg_confidence FLOAT,
            max_confidence FLOAT,
            min_confidence FLOAT,
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

    def _convert_to_boolean_int(self, value):
        """
        Konvertiert verschiedene Werte zu Integer für MySQL BOOLEAN Spalten
        MySQL BOOLEAN ist eigentlich TINYINT(1), daher 0 oder 1
        """
        if isinstance(value, bool):
            return int(value)
        elif isinstance(value, str):
            return 1 if value.lower() in ('true', '1', 'yes', 'on') else 0
        elif isinstance(value, (int, float)):
            return 1 if value else 0
        else:
            return 0
            
    def insert_result(self, run_id: str, image_path: str, image_filename: str,
                     classification: str, model_output: Dict, confidence_scores: str,
                     processing_time: float, success: bool = True, 
                     error_message: str = None) -> bool:
        """
        Fügt ein Erkennungsergebnis in die Datenbank ein
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
        
        # Initialisiere Standardwerte
        persons_detected = 0
        avg_confidence = None
        max_confidence = None
        min_confidence = None
        is_uncertain_int = 0
        
        if model_output and isinstance(model_output, dict):
            # Personen-Anzahl
            persons_detected = model_output.get('persons_detected', 0)
            if not isinstance(persons_detected, int):
                persons_detected = 0
            
            # Konfidenz-Werte - sicher konvertieren
            for conf_key, conf_var in [('avg_confidence', 'avg_confidence'), 
                                     ('max_confidence', 'max_confidence'), 
                                     ('min_confidence', 'min_confidence')]:
                conf_value = model_output.get(conf_key)
                if conf_value is not None:
                    try:
                        conf_value = float(conf_value)
                        if conf_value > 0:
                            if conf_key == 'avg_confidence':
                                avg_confidence = conf_value
                            elif conf_key == 'max_confidence':
                                max_confidence = conf_value
                            elif conf_key == 'min_confidence':
                                min_confidence = conf_value
                    except (ValueError, TypeError):
                        pass  # Bleibt None
            
            # Uncertain-Wert sicher konvertieren
            uncertain_value = model_output.get('uncertain', False)
            is_uncertain_int = self._convert_to_boolean_int(uncertain_value)

        # Konvertiere success zu Integer für MySQL BOOLEAN
        success_int = self._convert_to_boolean_int(success)

        print(f""+ query, (
                run_id, image_path, image_filename, classification,
                model_output_json, confidence_scores, processing_time,
                success_int, error_message,
                persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain_int
            ))

        try:
            cursor.execute(query, (
                run_id, image_path, image_filename, classification,
                model_output_json, confidence_scores, processing_time,
                success_int, error_message,
                persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain_int
            ))
            return True
        except Error as e:
            print(f"✗ Fehler beim Einfügen des Ergebnisses: {e}")
            print(f"   Daten: persons={persons_detected}, avg_conf={avg_confidence}, max_conf={max_confidence}, min_conf={min_confidence}, uncertain={is_uncertain_int}")
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
        Fügt fehlende Spalten zu bestehender Tabelle hinzu
        Aufruf falls Tabelle bereits existiert aber Spalten fehlen
        """
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        
        try:
            # Prüfe welche Spalten fehlen
            cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'detection_results'
            """, (self.database,))
            
            existing_columns = {row[0] for row in cursor.fetchall()}
            
            # Liste der benötigten Spalten mit ihren Definitionen
            required_columns = {
                'persons_detected': 'INT DEFAULT 0',
                'avg_confidence': 'FLOAT',
                'max_confidence': 'FLOAT', 
                'min_confidence': 'FLOAT',
                'is_uncertain': 'BOOLEAN DEFAULT FALSE'
            }
            
            # Füge fehlende Spalten hinzu
            for column_name, column_def in required_columns.items():
                if column_name not in existing_columns:
                    cursor.execute(f"""
                        ALTER TABLE detection_results 
                        ADD COLUMN {column_name} {column_def}
                    """)
                    print(f"✓ Spalte '{column_name}' hinzugefügt")
                else:
                    print(f"✓ Spalte '{column_name}' bereits vorhanden")
                
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