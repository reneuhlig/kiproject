#!/usr/bin/env python3
"""
Alternative Lösung für das MySQL Boolean-Problem
Ersetzt die insert_result Methode mit einer robusteren Version
"""

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime


def alternative_insert_result(connection, run_id: str, image_path: str, 
                            image_filename: str, classification: str, 
                            model_output: dict, confidence_scores: str,
                            processing_time: float, success: bool = True, 
                            error_message: str = None) -> bool:
    """
    Alternative insert_result Methode mit expliziter Typ-Kontrolle
    """
    if not connection:
        return False
        
    cursor = connection.cursor()
    
    try:
        # 1. JSON sicher serialisieren
        model_output_json = json.dumps(model_output, ensure_ascii=False) if model_output else None
        
        # 2. Werte extrahieren und konvertieren
        persons_detected = 0
        avg_confidence = None
        max_confidence = None  
        min_confidence = None
        is_uncertain = False
        
        if model_output and isinstance(model_output, dict):
            persons_detected = int(model_output.get('persons_detected', 0))
            
            if model_output.get('avg_confidence') is not None:
                avg_confidence = float(model_output.get('avg_confidence'))
            if model_output.get('max_confidence') is not None:
                max_confidence = float(model_output.get('max_confidence'))
            if model_output.get('min_confidence') is not None:
                min_confidence = float(model_output.get('min_confidence'))
                
            # Boolean-Wert explizit behandeln
            uncertain_raw = model_output.get('uncertain', False)
            is_uncertain = bool(uncertain_raw)  # Explizit zu bool konvertieren
        
        success_bool = bool(success)  # Explizit zu bool konvertieren
        processing_time_float = float(processing_time)
        
        # 3. SQL mit expliziten CAST-Operationen
        query = """
        INSERT INTO detection_results 
        (run_id, image_path, image_filename, classification, model_output, 
         confidence_scores, processing_time, success, error_message, 
         persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS UNSIGNED), %s, %s, %s, %s, %s, CAST(%s AS UNSIGNED))
        """
        
        # 4. Parameter explizit definieren
        params = [
            str(run_id),
            str(image_path), 
            str(image_filename),
            str(classification) if classification else '',
            model_output_json,
            str(confidence_scores) if confidence_scores else '',
            processing_time_float,
            1 if success_bool else 0,  # Als Integer für CAST
            str(error_message) if error_message else None,
            persons_detected,
            avg_confidence,
            max_confidence,
            min_confidence,
            1 if is_uncertain else 0   # Als Integer für CAST
        ]
        
        print(f"🔍 Debug - success: {success_bool} -> {1 if success_bool else 0}")
        print(f"🔍 Debug - uncertain: {is_uncertain} -> {1 if is_uncertain else 0}")
        
        cursor.execute(query, params)
        return True
        
    except Error as e:
        print(f"❌ MySQL-Fehler: {e}")
        print(f"   Parameter-Typen: {[type(p) for p in params]}")
        return False
    except Exception as e:
        print(f"❌ Allgemeiner Fehler: {e}")
        return False
    finally:
        cursor.close()


def test_alternative_solution():
    """Testet die alternative Lösung"""
    
    db_config = {
        'host': 'localhost',
        'user': 'aiuser', 
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'autocommit': True,
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci'
    }
    
    try:
        connection = mysql.connector.connect(**db_config)
        
        # Dummy-Daten wie aus Ultralytics
        test_model_output = {
            'persons_detected': 2,
            'persons': [],
            'confidences': [0.85, 0.72],
            'avg_confidence': 0.785,
            'max_confidence': 0.85,
            'min_confidence': 0.72,
            'uncertain': True,  # Das problematische Boolean-Feld
            'model_output': {}
        }
        
        print("🧪 Teste alternative Lösung...")
        
        success = alternative_insert_result(
            connection=connection,
            run_id="test-alternative-123",
            image_path="/test/image.jpg", 
            image_filename="image.jpg",
            classification="test",
            model_output=test_model_output,
            confidence_scores="0.850,0.720",
            processing_time=1.23,
            success=True,
            error_message=None
        )
        
        if success:
            print("✅ Alternative Lösung funktioniert!")
            
            # Prüfe eingefügte Daten
            cursor = connection.cursor()
            cursor.execute("""
                SELECT success, is_uncertain, persons_detected 
                FROM detection_results 
                WHERE run_id = 'test-alternative-123'
            """)
            result = cursor.fetchone()
            
            if result:
                success_val, uncertain_val, persons_val = result
                print(f"📊 Eingefügte Werte: success={success_val}, uncertain={uncertain_val}, persons={persons_val}")
            
            # Aufräumen
            cursor.execute("DELETE FROM detection_results WHERE run_id = 'test-alternative-123'")
            cursor.close()
        else:
            print("❌ Alternative Lösung fehlgeschlagen!")
            
    except Error as e:
        print(f"❌ Datenbankfehler: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    print("=" * 60)
    print("ALTERNATIVE MYSQL BOOLEAN LÖSUNG")
    print("=" * 60)
    
    # Erst Connector-Debug
    print("1. MySQL Connector Debug ausführen? (j/n): ", end="")
    if input().lower() in ['j', 'ja', 'y', 'yes']:
        exec(open('mysql_connector_debug.py').read())
    
    print("\n2. Alternative Lösung testen? (j/n): ", end="")
    if input().lower() in ['j', 'ja', 'y', 'yes']:
        test_alternative_solution()