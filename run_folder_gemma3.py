#!/usr/bin/env python3
"""
Ordner → DB → Gemma3 (Ollama) → Ergebnisse in DB

Ablauf:
1) Alle Bilder aus IMAGE_FOLDER in die Tabelle `Images` importieren (falls noch nicht vorhanden).
2) Für alle Images ohne Ergebnis dieses Modells: Gemma3 (Vision) prompten.
3) Ergebnisse in `AI_Results` speichern (prediction JSON, predicted_label, confidence, processed_at, latency_ms, prompt).
4) Personenanzahl zusätzlich in `Analysis` (metric_type='count') speichern.
5) Fehler in `Errors` loggen.

Anpassbar per Umgebungsvariablen:
  IMAGE_FOLDER=/pfad/zum/ordner
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
  MODEL_NAME (z. B. 'gemma3' oder 'llama3.2-vision'), MODEL_VERSION, MODEL_FRAMEWORK
  BATCH_SIZE (Default 64)

Wichtig: Test
- Verwende ein Vision-fähiges Modell in Ollama. Falls `gemma3` in deiner Installation kein Bild akzeptiert,
  nutze z. B. `llama3.2-vision` oder `llava` und setze MODEL_NAME entsprechend.
"""

from __future__ import annotations
import os
import io
import time
import json
import base64
import datetime as dt
from urllib.parse import urlparse

import requests
import mysql.connector
from mysql.connector import errorcode
from dateutil.tz import tzlocal
import ollama  # pip install ollama

# =========================
# Konfiguration
# =========================
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "/path/to/images")  # <--- ANPASSEN ODER per env setzen

DB_CFG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "aiuser"),
    "password": os.getenv("DB_PASSWORD", "geheimespasswort"),
    "database": os.getenv("DB_NAME", "AIModelAuswertung"),
}

MODEL_NAME = os.getenv("MODEL_NAME", "gemma3")       # ggf. 'llama3.2-vision' / 'llava'
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MODEL_FRAMEWORK = os.getenv("MODEL_FRAMEWORK", "Ollama")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
ALLOWED_EXT = (".jpg", ".jpeg", ".png")

PROMPT_INSTRUCTIONS = """
Du bekommst genau EIN Bild. Antworte NUR mit gültigem JSON (ohne Markdown, ohne Kommentar).
Aufgabe: Erkenne, ob auf dem Bild Menschen/Personen zu sehen sind und wie viele.

Gib GENAU dieses JSON-Schema zurück:
{
  "has_person": true/false,
  "person_count": <integer>=0,
  "probability": <float von 0.0 bis 1.0>,
  "notes": "<kurzer Hinweis, ein Satz>"
}

Regeln:
- Wenn unsicher: probability ~0.5
- person_count = geschätzte Anzahl sichtbarer Personen (0 wenn keine/nicht erkennbar)
- Gib KEINEN Fließtext, KEINE zusätzlichen Felder aus, NUR das JSON-Objekt.
"""

# =========================
# DB & Schema
# =========================
CREATE_TABLES_SQL = [
    # Images
    """
    CREATE TABLE IF NOT EXISTS Images (
        image_id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        blob_path VARCHAR(1024) NOT NULL,
        source VARCHAR(255),
        fps_filter BOOLEAN DEFAULT TRUE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Models
    """
    CREATE TABLE IF NOT EXISTS Models (
        model_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        version VARCHAR(255),
        framework VARCHAR(255),
        params JSON NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # AI_Results
    """
    CREATE TABLE IF NOT EXISTS AI_Results (
        result_id INT AUTO_INCREMENT PRIMARY KEY,
        image_id INT NOT NULL,
        model_id INT NOT NULL,
        prediction JSON NOT NULL,
        predicted_label VARCHAR(255),
        confidence FLOAT,
        processed_at DATETIME NOT NULL,
        latency_ms INT NULL,
        prompt TEXT NULL,
        CONSTRAINT fk_res_image FOREIGN KEY (image_id) REFERENCES Images(image_id) ON DELETE CASCADE,
        CONSTRAINT fk_res_model FOREIGN KEY (model_id) REFERENCES Models(model_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Analysis
    """
    CREATE TABLE IF NOT EXISTS Analysis (
        analysis_id INT AUTO_INCREMENT PRIMARY KEY,
        image_id INT NOT NULL,
        metric_type VARCHAR(255) NOT NULL,
        metric_value JSON NOT NULL,
        created_at DATETIME NOT NULL,
        CONSTRAINT fk_ana_image FOREIGN KEY (image_id) REFERENCES Images(image_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Errors
    """
    CREATE TABLE IF NOT EXISTS Errors (
        log_id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        component VARCHAR(255) NOT NULL,
        message TEXT NOT NULL,
        severity ENUM('INFO','WARNING','ERROR') NOT NULL DEFAULT 'ERROR'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # GroundTruth (optional hier nur, damit Schema komplett ist – wird nicht beschrieben)
    """
    CREATE TABLE IF NOT EXISTS GroundTruth (
        truth_id INT AUTO_INCREMENT PRIMARY KEY,
        image_id INT NOT NULL,
        actual_label VARCHAR(255) NOT NULL,
        validated_at DATETIME,
        validator VARCHAR(255),
        CONSTRAINT fk_gt_image FOREIGN KEY (image_id) REFERENCES Images(image_id) ON DELETE CASCADE,
        UNIQUE KEY ux_gt_image (image_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

MIGRATIONS_SQL = [
    # Zusätzliche Spalten/Indizes
    # AI_Results.latency_ms & prompt (falls nicht vorhanden) – robust via try
    "ALTER TABLE AI_Results ADD COLUMN latency_ms INT NULL",
    "ALTER TABLE AI_Results ADD COLUMN prompt TEXT NULL",
    # Eindeutigkeit: pro Bild/Modell nur ein Eintrag
    "CREATE UNIQUE INDEX ux_ai_results_image_model ON AI_Results (image_id, model_id)",
    # Für schnelle Abfragen
    "CREATE INDEX ix_ai_results_processed_at ON AI_Results (processed_at)",
    # Praktisch: keine doppelten Pfade importieren
    "CREATE UNIQUE INDEX ux_images_blob_path ON Images (blob_path(255))"
]

def connect_db():
    return mysql.connector.connect(**DB_CFG)

def ensure_schema(conn):
    cur = conn.cursor()
    for sql in CREATE_TABLES_SQL:
        cur.execute(sql)
    # Migrations tolerant ausführen
    for sql in MIGRATIONS_SQL:
        try:
            cur.execute(sql)
        except mysql.connector.Error:
            pass
    conn.commit()
    cur.close()

def get_or_create_model_id(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT model_id FROM Models WHERE name=%s AND version=%s LIMIT 1",
                (MODEL_NAME, MODEL_VERSION))
    row = cur.fetchone()
    if row:
        mid = row[0]
        cur.close()
        return mid
    cur.execute("""
        INSERT INTO Models (name, version, framework, params)
        VALUES (%s, %s, %s, %s)
    """, (MODEL_NAME, MODEL_VERSION, MODEL_FRAMEWORK, json.dumps({"source": "run_folder_gemma3.py"})))
    conn.commit()
    mid = cur.lastrowid
    cur.close()
    return mid

# =========================
# Ordner → Images importieren
# =========================
def import_images_from_folder(conn, folder: str, source_label: str = None, fps_filter: bool = True) -> int:
    """
    Liest alle Bilddateien aus `folder` und legt (falls noch nicht vorhanden) Einträge in `Images` an.
    Vermeidet Dubletten über UNIQUE-Index auf blob_path.
    """
    source_label = source_label or f"folder:{os.path.basename(os.path.abspath(folder))}"
    now = dt.datetime.now(tzlocal()).replace(tzinfo=None)

    cur = conn.cursor()
    inserted = 0

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(ALLOWED_EXT):
            continue
        fullpath = os.path.join(folder, name)

        try:
            cur.execute("""
                INSERT INTO Images (timestamp, blob_path, source, fps_filter)
                VALUES (%s, %s, %s, %s)
            """, (now, fullpath, source_label, fps_filter))
            inserted += 1
        except mysql.connector.Error as e:
            # Duplicate -> ignorieren
            if e.errno in (errorcode.ER_DUP_ENTRY,):
                continue
            else:
                raise
    conn.commit()
    cur.close()
    return inserted

# =========================
# Gemma3 Inferenz
# =========================
def load_image_bytes(path_or_url: str) -> bytes:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        return resp.content
    with open(path_or_url, "rb") as f:
        return f.read()

def ollama_chat_vision(prompt: str, image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = [{
        "role": "user",
        "content": prompt,
        "images": [b64],  # Vision-fähige Modelle in Ollama akzeptieren base64-Bilder
    }]
    res = ollama.chat(model=MODEL_NAME, messages=messages)
    return res["message"]["content"]

def insert_error(conn, component: str, message: str, severity: str = "ERROR"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO Errors (timestamp, component, message, severity)
        VALUES (%s, %s, %s, %s)
    """, (dt.datetime.now(tzlocal()).replace(tzinfo=None), component, message[:65000], severity))
    conn.commit()
    cur.close()

def insert_result_and_analysis(conn, image_id: int, model_id: int, prediction: dict,
                               prompt_text: str, latency_ms: int):
    now = dt.datetime.now(tzlocal()).replace(tzinfo=None)

    has_person = bool(prediction.get("has_person", False))
    person_count = int(prediction.get("person_count", 0) or 0)
    prob = float(prediction.get("probability", 0.5) or 0.5)

    predicted_label = "Person" if has_person else "NoPerson"
    confidence = prob

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO AI_Results
            (image_id, model_id, prediction, predicted_label, confidence, processed_at, latency_ms, prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        image_id, model_id, json.dumps(prediction, ensure_ascii=False),
        predicted_label, confidence, now, latency_ms, prompt_text
    ))
    conn.commit()
    cur.close()

    # Analysis: Personenanzahl
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO Analysis (image_id, metric_type, metric_value, created_at)
        VALUES (%s, %s, %s, %s)
    """, (image_id, "count", json.dumps({"persons": person_count}), now))
    conn.commit()
    cur.close()

def list_images_without_result(conn, model_id: int, limit: int):
    cur = conn.cursor()
    sql = """
        SELECT i.image_id, i.blob_path
        FROM Images i
        WHERE NOT EXISTS (
            SELECT 1 FROM AI_Results r
            WHERE r.image_id = i.image_id AND r.model_id = %s
        )
        ORDER BY i.timestamp ASC
        LIMIT %s
    """
    cur.execute(sql, (model_id, limit))
    rows = cur.fetchall()
    cur.close()
    return rows

# =========================
# Main
# =========================
def main():
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"[ABBRUCH] IMAGE_FOLDER nicht gefunden: {IMAGE_FOLDER}")
        return

    try:
        conn = connect_db()
    except mysql.connector.Error as e:
        print(f"[ABBRUCH] DB-Verbindung fehlgeschlagen: {e}")
        return

    try:
        ensure_schema(conn)
        model_id = get_or_create_model_id(conn)

        # 1) Ordner → Images importieren
        added = import_images_from_folder(conn, IMAGE_FOLDER)
        print(f"[INFO] {added} neue Bildpfade importiert aus {IMAGE_FOLDER}.")

        # 2) Bilder ohne Ergebnis dieses Modells abarbeiten (in Batches)
        total_ok = 0
        while True:
            todo = list_images_without_result(conn, model_id, BATCH_SIZE)
            if not todo:
                break

            for image_id, blob_path in todo:
                try:
                    img_bytes = load_image_bytes(blob_path)
                except Exception as e:
                    insert_error(conn, "Loader", f"image_id={image_id} path={blob_path} load_error={repr(e)}")
                    continue

                try:
                    t0 = time.perf_counter()
                    raw = ollama_chat_vision(PROMPT_INSTRUCTIONS.strip(), img_bytes)
                    t1 = time.perf_counter()
                    latency_ms = int((t1 - t0) * 1000)

                    cleaned = raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.strip("`")
                        cleaned = cleaned.split("\n", 1)[-1].strip()
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3].strip()

                    prediction = json.loads(cleaned)

                    insert_result_and_analysis(conn, image_id, model_id, prediction,
                                               PROMPT_INSTRUCTIONS.strip(), latency_ms)
                    total_ok += 1
                    print(f"[OK] image_id={image_id} latency={latency_ms}ms has_person={prediction.get('has_person')} count={prediction.get('person_count')}")

                except Exception as ex:
                    insert_error(conn, "AI", f"image_id={image_id} path={blob_path} infer_error={repr(ex)}")

        print(f"[FERTIG] Erfolgreich verarbeitet: {total_ok} Bilder.")

    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
