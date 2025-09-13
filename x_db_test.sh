#!/bin/bash

# ==============================================
# Testscript: Prüft die Verbindung zur Datenbank
# ==============================================

# Konfiguration
DB_HOST="localhost"
DB_PORT="3306"
DB_USER="aiuser"
DB_PASS="aiuser"
DB_NAME="ai_detection"

# Verbindung testen
echo "Prüfe Verbindung zu $DB_NAME auf $DB_HOST:$DB_PORT ..."

ERROR_MSG=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" -e "USE $DB_NAME;" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Verbindung erfolgreich!"
    exit 0
else
    echo "❌ Verbindung fehlgeschlagen!"
    echo "Original-Fehlermeldung:"
    echo "$ERROR_MSG"
    echo

    # Genauere Ursachenanalyse
    if echo "$ERROR_MSG" | grep -q "Access denied"; then
        echo "👉 Ursache: Benutzername oder Passwort ist falsch, oder Benutzer hat keine Rechte."
    elif echo "$ERROR_MSG" | grep -q "Unknown database"; then
        echo "👉 Ursache: Die angegebene Datenbank '$DB_NAME' existiert nicht."
    elif echo "$ERROR_MSG" | grep -q "Can't connect"; then
        echo "👉 Ursache: Host oder Port ist falsch, oder die Datenbank ist nicht erreichbar."
    elif echo "$ERROR_MSG" | grep -q "Lost connection"; then
        echo "👉 Ursache: Verbindung wurde unerwartet getrennt (Netzwerkproblem oder Server down)."
    else
        echo "👉 Ursache: Unbekannter Fehler – bitte Original-Fehlermeldung prüfen."
    fi

    exit 1
fi
