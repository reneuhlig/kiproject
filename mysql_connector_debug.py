#!/usr/bin/env python3
"""
Spezielles Debug-Script für MySQL Connector Boolean-Problem
"""

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

def test_mysql_boolean_handling():
    """Testet wie MySQL Connector Boolean-Werte behandelt"""
    
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
        
        print("🔍 MySQL Connector Boolean-Debug...")
        print(f"MySQL Connector Version: {mysql.connector.__version__}")
        
        # Temporäre Test-Tabelle erstellen
        cursor.execute("""
            CREATE TEMPORARY TABLE test_booleans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bool_col TINYINT(1) NOT NULL,
                name VARCHAR(50)
            )
        """)
        
        test_cases = [
            ("Python True", True),
            ("Python False", False), 
            ("Integer 1", 1),
            ("Integer 0", 0),
            ("String '1'", '1'),
            ("String '0'", '0'),
            ("String 'true'", 'true'),
            ("String 'false'", 'false'),
        ]
        
        print("\n🧪 Teste verschiedene Boolean-Werte:")
        
        for name, value in test_cases:
            try:
                print(f"\n  {name}: {value} (Typ: {type(value)})")
                
                # Direkte Übergabe
                cursor.execute("""
                    INSERT INTO test_booleans (bool_col, name) VALUES (%s, %s)
                """, (value, f"{name}_direct"))
                
                print(f"    ✅ Direkte Übergabe erfolgreich")
                
                # Mit expliziter Konvertierung
                converted = 1 if str(value).lower() in ('true', '1') else 0
                cursor.execute("""
                    INSERT INTO test_booleans (bool_col, name) VALUES (%s, %s)
                """, (converted, f"{name}_converted"))
                
                print(f"    ✅ Konvertierte Übergabe erfolgreich (Wert: {converted})")
                
            except Error as e:
                print(f"    ❌ Fehler: {e}")
        
        # Ergebnisse prüfen
        print(f"\n📊 Eingefügte Werte:")
        cursor.execute("SELECT name, bool_col FROM test_booleans ORDER BY id")
        results = cursor.fetchall()
        
        for name, bool_val in results:
            print(f"  {name}: {bool_val} (Typ: {type(bool_val)})")
        
        # JSON-Parameter testen (wie im echten Code)
        print(f"\n🧪 JSON-Parameter Test:")
        
        test_model_output = {
            'persons_detected': 1,
            'avg_confidence': 0.85,
            'uncertain': True  # Das problematische Feld
        }
        
        model_output_json = json.dumps(test_model_output, ensure_ascii=False)
        print(f"  JSON: {model_output_json}")
        
        # Extrahiere uncertain Wert wie im echten Code
        uncertain_raw = test_model_output.get('uncertain', False)
        print(f"  uncertain_raw: {uncertain_raw} (Typ: {type(uncertain_raw)})")
        
        # Konvertierungsfunktion testen
        def safe_convert_to_int(value):
            if value is None:
                return 0
            if isinstance(value, bool):
                return 1 if value else 0
            if isinstance(value, str):
                lower_val = value.lower().strip()
                if lower_val in ('true', '1', 'yes', 'on'):
                    return 1
                elif lower_val in ('false', '0', 'no', 'off', ''):
                    return 0
                else:
                    try:
                        return 1 if float(value) != 0 else 0
                    except (ValueError, TypeError):
                        return 0
            if isinstance(value, (int, float)):
                return 1 if value != 0 else 0
            return 1 if value else 0
        
        uncertain_converted = safe_convert_to_int(uncertain_raw)
        print(f"  uncertain_converted: {uncertain_converted} (Typ: {type(uncertain_converted)})")
        
        # Parameter-Array wie im echten Code
        params_array = [
            "test-run-id",           # run_id
            "/test/path.jpg",        # image_path
            "test.jpg",              # filename
            "test_class",            # classification
            model_output_json,       # model_output_json
            "0.850",                 # confidence_scores
            1.23,                    # processing_time
            1,                       # success (int)
            None,                    # error_message
            1,                       # persons_detected
            0.85,                    # avg_confidence
            0.85,                    # max_confidence
            0.85,                    # min_confidence
            int(uncertain_converted) # is_uncertain (explizit int)
        ]
        
        print(f"\n📋 Parameter-Array:")
        for i, param in enumerate(params_array):
            print(f"  [{i:2d}]: {param} (Typ: {type(param)})")
        
        # Test mit echtem Query (ohne tatsächliches Insert)
        print(f"\n🔍 SQL-Query-Simulation:")
        query = """
        SELECT %s as run_id, %s as success_val, %s as uncertain_val
        """
        
        cursor.execute(query, (params_array[0], params_array[7], params_array[13]))
        result = cursor.fetchone()
        print(f"  Query-Ergebnis: {result}")
        
        print(f"\n✅ MySQL Connector Debug abgeschlossen")
        
    except Error as e:
        print(f"❌ MySQL-Fehler: {e}")
    except Exception as e:
        print(f"❌ Allgemeiner Fehler: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

if __name__ == "__main__":
    test_mysql_boolean_handling()