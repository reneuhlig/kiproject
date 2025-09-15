#!/usr/bin/env python3
"""
Quick Fix für das Database Handler Problem
Führen Sie diesen vor dem nächsten Test aus!
"""

import mysql.connector
from mysql.connector import Error

def fix_database_columns():
    """Ändert die Spalten-Definitionen um String-zu-Boolean-Probleme zu lösen"""
    
    # WICHTIG: Anpassen an Ihre Datenbankverbindung!
    db_config = {
        'host': 'localhost',
        'user': 'aiuser', 
        'password': 'DHBW1234!?',
        'database': 'ai_detection'
    }
    
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        print("🔧 Repariere Datenbank-Spalten...")
        
        # Option 1: Spalten neu definieren als TINYINT mit expliziter Konvertierung
        fixes = [
            """
            ALTER TABLE detection_results 
            MODIFY COLUMN success TINYINT(1) NOT NULL DEFAULT 1
            """,
            """
            ALTER TABLE detection_results 
            MODIFY COLUMN is_uncertain TINYINT(1) NOT NULL DEFAULT 0
            """
        ]
        
        for fix_sql in fixes:
            try:
                cursor.execute(fix_sql)
                print(f"✅ Spalte erfolgreich angepasst")
            except Error as e:
                print(f"⚠ Spalten-Anpassung: {e}")
        
        # Option 2: Problematische Datensätze bereinigen (falls vorhanden)
        cleanup_queries = [
            """
            UPDATE detection_results 
            SET success = CASE 
                WHEN success = 'true' THEN 1 
                WHEN success = 'false' THEN 0 
                ELSE success 
            END
            """,
            """
            UPDATE detection_results 
            SET is_uncertain = CASE 
                WHEN is_uncertain = 'true' THEN 1 
                WHEN is_uncertain = 'false' THEN 0 
                ELSE is_uncertain 
            END
            """
        ]
        
        for cleanup_sql in cleanup_queries:
            try:
                cursor.execute(cleanup_sql)
                affected = cursor.rowcount
                if affected > 0:
                    print(f"🧹 {affected} Datensätze bereinigt")
            except Error as e:
                print(f"⚠ Daten-Bereinigung: {e}")
        
        connection.commit()
        print("✅ Datenbank-Reparatur abgeschlossen")
        
    except Error as e:
        print(f"❌ Datenbankfehler: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

def verify_table_structure():
    """Überprüft die aktuelle Tabellenstruktur"""
    
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?', 
        'database': 'ai_detection'
    }
    
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        cursor.execute("DESCRIBE detection_results")
        columns = cursor.fetchall()
        
        print("\n📋 Aktuelle Tabellenstruktur:")
        for column in columns:
            field, dtype, null, key, default, extra = column
            if field in ['success', 'is_uncertain']:
                print(f"  🔍 {field}: {dtype} (Null: {null}, Default: {default})")
        
    except Error as e:
        print(f"❌ Strukturprüfung fehlgeschlagen: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

if __name__ == "__main__":
    print("=" * 50)
    print("DATABASE QUICK FIX")
    print("=" * 50)
    
    print("WARNUNG: Dieses Script ändert Ihre Datenbank!")
    print("Stellen Sie sicher, dass Sie ein Backup haben.")
    
    proceed = input("\nFortfahren? (j/n): ").lower().strip()
    if proceed in ['j', 'ja', 'y', 'yes']:
        verify_table_structure()
        fix_database_columns()
        verify_table_structure()
    else:
        print("Abgebrochen.")