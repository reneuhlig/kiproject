#!/usr/bin/env python3
"""
Debug-Script für Datenbank-Probleme
"""

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

def test_database_types():
    """Testet verschiedene Datentypen in der Datenbank"""
    
    # Datenbankverbindung (anpassen!)
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
        cursor = connection.cursor()
        
        print("🔍 Teste Datenbank-Datentypen...")
        
        # 1. Tabellenschema prüfen
        cursor.execute("""
            DESCRIBE detection_results
        """)
        
        schema = cursor.fetchall()
        print("\n📋 Tabellenschema:")
        for column in schema:
            field_name, field_type, null, key, default, extra = column
            print(f"  {field_name}: {field_type} (Null: {null}, Default: {default})")
        
        # 2. Test-Insert mit verschiedenen Boolean-Werten
        test_run_id = "debug-test-12345"
        test_values = [
            ("Test 1", True, False),     # Python Boolean
            ("Test 2", 1, 0),           # Integer
            ("Test 3", False, True),    # Python Boolean umgekehrt
        ]
        
        print(f"\n🧪 Teste verschiedene Boolean-Werte...")
        
        for i, (test_name, success_val, uncertain_val) in enumerate(test_values, 1):
            # Werte konvertieren wie in der echten Anwendung
            success_int = 1 if success_val else 0
            uncertain_int = 1 if uncertain_val else 0
            
            print(f"  {test_name}:")
            print(f"    success: {success_val} -> {success_int} (Typ: {type(success_int)})")
            print(f"    uncertain: {uncertain_val} -> {uncertain_int} (Typ: {type(uncertain_int)})")
            
            query = """
            INSERT INTO detection_results 
            (run_id, image_path, image_filename, classification, processing_time, 
             success, persons_detected, avg_confidence, max_confidence, 
             min_confidence, is_uncertain)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            try:
                cursor.execute(query, (
                    f"{test_run_id}-{i}",        # run_id
                    f"/test/image{i}.jpg",       # image_path  
                    f"image{i}.jpg",             # image_filename
                    "debug_test",                # classification
                    1.23,                        # processing_time
                    success_int,                 # success (TINYINT)
                    1,                          # persons_detected
                    0.85,                       # avg_confidence
                    0.92,                       # max_confidence
                    0.78,                       # min_confidence
                    uncertain_int               # is_uncertain (TINYINT)
                ))
                print(f"    ✅ Insert erfolgreich")
                
            except Error as e:
                print(f"    ❌ Insert fehlgeschlagen: {e}")
        
        # 3. Prüfe eingefügte Werte
        print(f"\n📊 Prüfe eingefügte Test-Werte...")
        cursor.execute("""
            SELECT run_id, success, is_uncertain 
            FROM detection_results 
            WHERE run_id LIKE %s
            ORDER BY run_id
        """, (f"{test_run_id}%",))
        
        results = cursor.fetchall()
        for run_id, success, uncertain in results:
            print(f"  {run_id}: success={success} (Typ: {type(success)}), uncertain={uncertain} (Typ: {type(uncertain)})")
        
        # 4. Aufräumen
        cursor.execute("""
            DELETE FROM detection_results WHERE run_id LIKE %s
        """, (f"{test_run_id}%",))
        print(f"\n🧹 Test-Daten aufgeräumt")
        
        print("\n✅ Datenbank-Test abgeschlossen")
        
    except Error as e:
        print(f"❌ Datenbankfehler: {e}")
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()


def test_model_output_parsing():
    """Testet die model_output Parsing-Logik"""
    
    print("\n🔬 Teste model_output Parsing...")
    
    # Beispiel model_output wie vom Ultralytics Detector
    test_model_outputs = [
        {
            'persons_detected': 1,
            'persons': [{'bbox': [100, 100, 200, 200], 'confidence': 0.902}],
            'confidences': [0.902],
            'avg_confidence': 0.902,
            'max_confidence': 0.902,
            'min_confidence': 0.902,
            'uncertain': False,  # Boolean
            'model_output': {'total_detections': 1}
        },
        {
            'persons_detected': 0,
            'persons': [],
            'confidences': [],
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0,
            'uncertain': False,  # Boolean
            'model_output': {'total_detections': 0}
        },
        {
            'persons_detected': 5,
            'persons': [],
            'confidences': [0.92, 0.85, 0.78, 0.65, 0.52],
            'avg_confidence': 0.744,
            'max_confidence': 0.92,
            'min_confidence': 0.52,
            'uncertain': True,  # Boolean - sollte zu 1 werden
            'model_output': {'total_detections': 5}
        }
    ]
    
    from DatabaseHandler import DatabaseHandler
    
    # Dummy-Handler für Tests
    handler = DatabaseHandler('localhost', 'user', 'pass', 'db')
    
    for i, model_output in enumerate(test_model_outputs, 1):
        print(f"\n  Test {i}:")
        print(f"    Original uncertain: {model_output['uncertain']} (Typ: {type(model_output['uncertain'])})")
        
        # Konvertierung testen
        uncertain_converted = handler._safe_convert_to_int(model_output.get('uncertain', False))
        print(f"    Konvertiert: {uncertain_converted} (Typ: {type(uncertain_converted)})")
        
        # JSON-Serialisierung testen
        json_str = json.dumps(model_output, ensure_ascii=False)
        print(f"    JSON serialisiert: ...uncertain\":{json.loads(json_str)['uncertain']}...")


if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE DEBUG SCRIPT")
    print("=" * 60)
    
    test_model_output_parsing()
    
    print("\n" + "=" * 60)
    
    # Frage vor Datenbank-Test
    db_test = input("🔍 Datenbank-Test ausführen? (j/n): ").lower().strip()
    if db_test in ['j', 'ja', 'y', 'yes']:
        test_database_types()
    else:
        print("Datenbank-Test übersprungen")