#!/usr/bin/env python3
"""
Schnelles Testscript für einzelne KI-Modelle
Kann ohne Datenbank verwendet werden
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Import der Detector-Implementierungen
try:
    from UltralyticsPersonDetector import UltralyticsPersonDetector
    from DeepFacePersonDetector import DeepFacePersonDetector
    from GemmaPersonDetector import GemmaPersonDetector
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Stelle sicher, dass alle Dateien im gleichen Verzeichnis sind.")
    sys.exit(1)


def find_test_images(data_dir: str, max_images: int = 5) -> List[str]:
    """Findet Testbilder im Datenverzeichnis"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Datenverzeichnis nicht gefunden: {data_dir}")
        return []
    
    # Unterstützte Bildformate
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Sammle alle Bilder
    images = []
    for ext in image_extensions:
        images.extend(data_path.rglob(f"*{ext}"))
        images.extend(data_path.rglob(f"*{ext.upper()}"))
    
    # Begrenze auf max_images
    images = [str(img) for img in images[:max_images]]
    
    print(f"✓ {len(images)} Testbilder gefunden")
    for img in images:
        print(f"  - {Path(img).name}")
    
    return images


def test_ultralytics(images: List[str], confidence: float = 0.5):
    """Testet Ultralytics YOLO Detektor"""
    print("\n" + "="*60)
    print("ULTRALYTICS YOLO TEST")
    print("="*60)
    
    try:
        detector = UltralyticsPersonDetector(confidence_threshold=confidence)
        print(f"✓ Detektor initialisiert: {detector.get_model_info()}")
        
        for i, image_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Teste: {Path(image_path).name}")
            start_time = time.time()
            
            result = detector.detect(image_path)
            processing_time = time.time() - start_time
            
            print(f"  Personen erkannt: {result.get('persons_detected', 0)}")
            print(f"  Durchschn. Konfidenz: {result.get('avg_confidence', 0):.3f}")
            print(f"  Verarbeitungszeit: {processing_time:.3f}s")
            if result.get('error'):
                print(f"  ⚠ Fehler: {result['error']}")
                
    except Exception as e:
        print(f"✗ Ultralytics Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


def test_deepface(images: List[str], confidence: float = 0.5, backend: str = 'opencv'):
    """Testet DeepFace Detektor"""
    print("\n" + "="*60)
    print(f"DEEPFACE TEST (Backend: {backend})")
    print("="*60)
    
    try:
        detector = DeepFacePersonDetector(
            detector_backend=backend, 
            confidence_threshold=confidence
        )
        print(f"✓ Detektor initialisiert: {detector.get_model_info()}")
        
        for i, image_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Teste: {Path(image_path).name}")
            start_time = time.time()
            
            result = detector.detect(image_path)
            processing_time = time.time() - start_time
            
            print(f"  Personen erkannt: {result.get('persons_detected', 0)}")
            print(f"  Durchschn. Konfidenz: {result.get('avg_confidence', 0):.3f}")
            print(f"  Verarbeitungszeit: {processing_time:.3f}s")
            if result.get('error'):
                print(f"  ⚠ Fehler: {result['error']}")
                
    except Exception as e:
        print(f"✗ DeepFace Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


def test_gemma(images: List[str], confidence: float = 0.5, model_name: str = "llava-hf/llava-1.5-7b-hf"):
    """Testet Gemma/LLaVA Detektor"""
    print("\n" + "="*60)
    print("GEMMA/LLAVA TEST")
    print("="*60)
    print("⚠ Warnung: Dieser Test kann sehr lange dauern und viel GPU-Speicher benötigen!")
    
    try:
        detector = GemmaPersonDetector(
            model_name=model_name,
            confidence_threshold=confidence
        )
        print(f"✓ Detektor initialisiert: {detector.get_model_info()}")
        
        for i, image_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Teste: {Path(image_path).name}")
            start_time = time.time()
            
            result = detector.detect(image_path)
            processing_time = time.time() - start_time
            
            print(f"  Personen erkannt: {result.get('persons_detected', 0)}")
            print(f"  Durchschn. Konfidenz: {result.get('avg_confidence', 0):.3f}")
            print(f"  Verarbeitungszeit: {processing_time:.3f}s")
            print(f"  LLM Antwort: {result.get('model_output', {}).get('raw_response', 'N/A')[:100]}...")
            if result.get('error'):
                print(f"  ⚠ Fehler: {result['error']}")
                
    except Exception as e:
        print(f"✗ Gemma Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


def create_simple_test_image():
    """Erstellt ein einfaches Testbild wenn keine vorhanden sind"""
    try:
        import cv2
        import numpy as np
        
        # Erstelle einfaches Bild mit "Person"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Zeichne einfache Person
        cv2.circle(img, (200, 120), 30, (255, 200, 150), -1)  # Kopf
        cv2.rectangle(img, (170, 150), (230, 280), (100, 100, 255), -1)  # Körper
        cv2.rectangle(img, (160, 280), (190, 350), (50, 50, 200), -1)  # Bein 1
        cv2.rectangle(img, (210, 280), (240, 350), (50, 50, 200), -1)  # Bein 2
        
        # Speichere Testbild
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        test_image_path = test_dir / "simple_person.jpg"
        
        cv2.imwrite(str(test_image_path), img)
        print(f"✓ Testbild erstellt: {test_image_path}")
        
        return [str(test_image_path)]
        
    except ImportError:
        print("✗ OpenCV nicht verfügbar - kann kein Testbild erstellen")
        return []


def main():
    parser = argparse.ArgumentParser(description='Schnelltest für KI-Personenerkennung')
    parser.add_argument('--model', 
                       choices=['ultralytics', 'deepface', 'gemma', 'all'], 
                       default='all',
                       help='Zu testendes Modell')
    parser.add_argument('--data-dir', 
                       help='Pfad zu Testbildern (optional)')
    parser.add_argument('--max-images', type=int, default=3,
                       help='Maximale Anzahl Testbilder')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Konfidenz-Schwellenwert')
    parser.add_argument('--deepface-backend', 
                       choices=['opencv', 'ssd', 'mtcnn', 'retinaface'],
                       default='opencv',
                       help='DeepFace Backend')
    parser.add_argument('--create-test-image', action='store_true',
                       help='Erstelle einfaches Testbild')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KI-PERSONENERKENNUNG SCHNELLTEST")
    print("="*60)
    
    # Finde Testbilder
    images = []
    if args.data_dir:
        images = find_test_images(args.data_dir, args.max_images)
    
    # Erstelle Testbild falls keine gefunden
    if not images and args.create_test_image:
        images = create_simple_test_image()
    
    if not images:
        print("✗ Keine Testbilder gefunden!")
        print("Verwende --data-dir um ein Bildverzeichnis anzugeben")
        print("oder --create-test-image um ein einfaches Testbild zu erstellen")
        sys.exit(1)
    
    # Teste Modelle
    try:
        if args.model in ['ultralytics', 'all']:
            test_ultralytics(images, args.confidence)
        
        if args.model in ['deepface', 'all']:
            test_deepface(images, args.confidence, args.deepface_backend)
        
        if args.model in ['gemma', 'all']:
            # Frage nach Bestätigung für Gemma (dauert lange)
            if args.model == 'all':
                response = input("\n⚠ Gemma-Test kann sehr lange dauern. Fortfahren? (y/N): ")
                if response.lower() != 'y':
                    print("Gemma-Test übersprungen.")
                else:
                    test_gemma(images[:1], args.confidence)  # Nur 1 Bild für Gemma
            else:
                test_gemma(images, args.confidence)
        
        print("\n" + "="*60)
        print("✓ Alle Tests abgeschlossen!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Tests durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()