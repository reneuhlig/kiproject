from datetime import datetime
from typing import Dict, Any, Optional, Union
import mysql.connector
from mysql.connector import Error
import json
import logging


class DatabaseHandler:
    """Verbesserte MySQL Datenbankoperationen für alle KI-Modelle"""
    
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
        # Logging konfigurieren
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Verbindet zur MySQL Datenbank"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=True,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
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
            config_json TEXT,
            INDEX idx_model_name (model_name),
            INDEX idx_start_time (start_time),
            INDEX idx_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
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
            success TINYINT(1) DEFAULT 1,
            persons_detected INT DEFAULT 0,
            avg_confidence FLOAT,
            max_confidence FLOAT,
            min_confidence FLOAT,
            is_uncertain TINYINT(1) DEFAULT 0,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES ai_runs(run_id) ON DELETE CASCADE,
            INDEX idx_run_id (run_id),
            INDEX idx_classification (classification),
            INDEX idx_timestamp (timestamp),
            INDEX idx_persons_detected (persons_detected),
            INDEX idx_success (success)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
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
        
        try:
            # Parameter sicher vorbereiten
            model_version = str(model_version) if model_version else 'unknown'
            config_json = json.dumps(config, ensure_ascii=False) if config else None
            start_time = datetime.now()
            
            cursor.execute(query, (run_id, model_name, model_version, start_time, config_json))
            return True
        except Error as e:
            self.logger.error(f"Fehler beim Einfügen des Runs: {e}")
            return False
        finally:
            cursor.close()

    def _safe_convert_to_int(self, value: Any) -> int:
        """Sichere Konvertierung zu Integer für MySQL TINYINT(1) - gibt IMMER int zurück"""
        if value is None:
            return 0
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, str):
            # Explizite String-zu-Boolean-Konvertierung
            lower_val = value.lower().strip()
            if lower_val in ('true', '1', 'yes', 'on'):
                return 1
            elif lower_val in ('false', '0', 'no', 'off', ''):
                return 0
            else:
                # Unbekannter String-Wert - versuche als Zahl zu interpretieren
                try:
                    return 1 if float(value) != 0 else 0
                except (ValueError, TypeError):
                    return 0
        if isinstance(value, (int, float)):
            return 1 if value != 0 else 0
        # Für alle anderen Typen: Wahrheitswert verwenden
        return 1 if value else 0
    
    def _safe_convert_to_float(self, value: Any) -> Optional[float]:
        """Sichere Konvertierung zu Float"""
        if value is None:
            return None
        try:
            result = float(value)
            # Prüfe auf ungültige Werte
            if result != result:  # NaN check
                return None
            return result
        except (ValueError, TypeError):
            return None
    
    def _safe_convert_to_int_nullable(self, value: Any) -> Optional[int]:
        """Sichere Konvertierung zu Integer (nullable)"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
            
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
        
        try:
            # Sichere JSON-Serialisierung
            model_output_json = json.dumps(model_output, ensure_ascii=False) if model_output else None
            
            # Sichere Parameter-Extraktion und -Konvertierung
            persons_detected = 0
            avg_confidence = None
            max_confidence = None
            min_confidence = None
            is_uncertain_int = 0
            
            if model_output and isinstance(model_output, dict):
                # Personen-Anzahl sicher extrahieren
                persons_detected = self._safe_convert_to_int_nullable(
                    model_output.get('persons_detected', 0)
                ) or 0
                
                # Konfidenz-Werte sicher konvertieren
                avg_confidence = self._safe_convert_to_float(model_output.get('avg_confidence'))
                max_confidence = self._safe_convert_to_float(model_output.get('max_confidence'))
                min_confidence = self._safe_convert_to_float(model_output.get('min_confidence'))
                
                # Uncertain-Wert sicher konvertieren - KRITISCHER FIX!
                uncertain_raw = model_output.get('uncertain', False)
                is_uncertain_int = self._safe_convert_to_int(uncertain_raw)
            
            # Success-Wert sicher konvertieren - KRITISCHER FIX!
            success_int = self._safe_convert_to_int(success)
            
            # Processing time sicher konvertieren
            processing_time_safe = self._safe_convert_to_float(processing_time) or 0.0
            
            # KRITISCHER FIX: Nochmalige Validierung der Boolean-Werte
            # Falls MySQL trotzdem Strings erhält, forcieren wir Integer
            if not isinstance(success_int, int):
                success_int = 1 if str(success_int).lower() == 'true' else 0
            if not isinstance(is_uncertain_int, int):
                is_uncertain_int = 1 if str(is_uncertain_int).lower() == 'true' else 0
            
            # Parameter-Array explizit aufbauen - keine implizite Konvertierung!
            params = [
                str(run_id),                                     # str
                str(image_path),                                 # str
                str(image_filename),                             # str
                str(classification) if classification else '',   # str
                model_output_json,                               # str or None
                str(confidence_scores) if confidence_scores else '', # str
                float(processing_time_safe),                     # float (explizit)
                int(success_int),                                # int (explizit)
                str(error_message) if error_message else None,   # str or None
                int(persons_detected),                           # int (explizit)
                float(avg_confidence) if avg_confidence is not None else None,  # float or None
                float(max_confidence) if max_confidence is not None else None,  # float or None
                float(min_confidence) if min_confidence is not None else None,  # float or None
                int(is_uncertain_int)                            # int (explizit)
            ]
            
            # Debug vor dem Execute (nur die kritischen Werte)
            self.logger.debug(f"Final params - success: {params[7]} (type: {type(params[7])}), "
                            f"is_uncertain: {params[13]} (type: {type(params[13])})")
            
            cursor.execute(query, params)
            return True
            
        except Error as e:
            # Noch detaillierteres Debug
            debug_params = {
                'persons_detected': (persons_detected, type(persons_detected)),
                'avg_confidence': (avg_confidence, type(avg_confidence)),
                'max_confidence': (max_confidence, type(max_confidence)),
                'min_confidence': (min_confidence, type(min_confidence)),
                'is_uncertain_raw': (model_output.get('uncertain') if model_output else None,),
                'is_uncertain_converted': (is_uncertain_int, type(is_uncertain_int)),
                'success_raw': (success,),
                'success_converted': (success_int, type(success_int)),
                'processing_time': (processing_time_safe, type(processing_time_safe))
            }
            self.logger.error(f"Fehler beim Einfügen des Ergebnisses: {e}")
            self.logger.error(f"Detaillierte Debug-Parameter: {debug_params}")
            
            # Zusätzlich: MySQL Connector Version prüfen
            import mysql.connector
            self.logger.error(f"MySQL Connector Version: {mysql.connector.__version__}")
            
            return False
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler beim Einfügen des Ergebnisses: {e}")
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
            # Parameter sicher konvertieren
            end_time = datetime.now()
            total_images = self._safe_convert_to_int_nullable(total_images) or 0
            successful_detections = self._safe_convert_to_int_nullable(successful_detections) or 0
            failed_detections = self._safe_convert_to_int_nullable(failed_detections) or 0
            
            avg_processing_time = self._safe_convert_to_float(avg_processing_time)
            total_processing_time = self._safe_convert_to_float(total_processing_time)
            
            # System-Stats sicher extrahieren
            avg_cpu = self._safe_convert_to_float(system_stats.get('avg_cpu'))
            max_cpu = self._safe_convert_to_float(system_stats.get('max_cpu'))
            avg_memory = self._safe_convert_to_float(system_stats.get('avg_memory'))
            max_memory = self._safe_convert_to_float(system_stats.get('max_memory'))
            avg_gpu = self._safe_convert_to_float(system_stats.get('avg_gpu'))
            max_gpu = self._safe_convert_to_float(system_stats.get('max_gpu'))
            
            cursor.execute(query, (
                end_time,
                total_images,
                successful_detections,
                failed_detections,
                avg_processing_time,
                total_processing_time,
                avg_cpu,
                max_cpu,
                avg_memory,
                max_memory,
                avg_gpu,
                max_gpu,
                str(status),
                str(error_message) if error_message else None,
                run_id
            ))
            return True
        except Error as e:
            self.logger.error(f"Fehler beim Aktualisieren des Runs: {e}")
            return False
        finally:
            cursor.close()
    
    def fix_existing_table(self) -> bool:
        """
        Fügt fehlende Spalten zu bestehender Tabelle hinzu
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
                'is_uncertain': 'TINYINT(1) DEFAULT 0'
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
    
    def get_run_statistics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt Statistiken für einen bestimmten Run zurück
        
        Args:
            run_id: Run-ID
            
        Returns:
            Dictionary mit Statistiken oder None bei Fehler
        """
        if not self.connection:
            return None
            
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            query = """
            SELECT 
                r.*,
                COUNT(dr.id) as result_count,
                AVG(dr.persons_detected) as avg_persons_per_image,
                SUM(dr.persons_detected) as total_persons_found,
                COUNT(CASE WHEN dr.is_uncertain = 1 THEN 1 END) as uncertain_count
            FROM ai_runs r
            LEFT JOIN detection_results dr ON r.run_id = dr.run_id
            WHERE r.run_id = %s
            GROUP BY r.run_id
            """
            
            cursor.execute(query, (run_id,))
            result = cursor.fetchone()
            
            return result
        except Error as e:
            self.logger.error(f"Fehler beim Abrufen der Run-Statistiken: {e}")
            return None
        finally:
            cursor.close()
            
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ Datenbankverbindung geschlossen")