#!/usr/bin/env python3
"""
Spezielles Testscript für Ollama Gemma 3 Vision
Kann ohne Datenbank verwendet werden
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from typing import List

try:
    from OllamaGemma3PersonDetector import OllamaGemma3PersonDetector
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Stelle sicher, dass OllamaGemma3PersonDetector.py im gleichen Verzeichnis ist.")
    sys.exit(1)


def check_ollama_status(host: str = "http://localhost:11434") -> bool:
    """Prüft ob Ollama läuft"""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(host: str = "http://localhost:11434") -> List[str]:
    """Listet verfügbare Ollama Modelle auf"""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except:
        pass
    return []


def create_test_images():
    """Erstellt verschiedene Testbilder"""
    try:
        import cv2
        import numpy as np
        
        test_dir = Path("ollama_test_images")
        test_dir.mkdir(exist_ok=True)
        
        images = []
        
        # 1. Einzelperson
        img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img1, (150, 80), 25, (255, 200, 150), -1)  # Kopf
        cv2.rectangle(img1, (125, 105), (175, 200), (100, 100, 255), -1)  # Körper
        cv2.rectangle(img1, (115, 200), (135, 250), (50, 50, 200), -1)  # Bein 1
        cv2.rectangle(img1, (165, 200), (185, 250), (50, 50, 200), -1)  # Bein 2
        path1 = test_dir / "one_person.jpg"
        cv2.imwrite(str(path1), img1)
        images.append((str(path1), "1 Person"))
        
        # 2. Zwei Personen
        img2 = np.zeros((300, 400, 3), dtype=np.uint8)
        # Person 1
        cv2.circle(img2, (120, 80), 20, (255, 200, 150), -1)
        cv2.rectangle(img2, (100, 100), (140, 180), (100, 100, 255), -1)
        cv2.rectangle(img2, (95, 180), (110, 220), (50, 50, 200), -1)
        cv2.rectangle(img2, (130, 180), (145, 220), (50, 50, 200), -1)
        # Person 2
        cv2.circle(img2, (280, 90), 18, (255, 180, 120), -1)
        cv2.rectangle(img2, (262, 108), (298, 170), (120, 80, 200), -1)
        cv2.rectangle(img2, (258, 170), (270, 210), (60, 30, 150), -1)
        cv2.rectangle(img2, (290, 170), (302, 210), (60, 30, 150), -1)
        path2 = test_dir / "two_people.jpg"
        cv2.imwrite(str(path2), img2)
        images.append((str(path2), "2 Personen"))
        
        # 3. Keine Person (Landschaft)
        img3 = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img3, (0, 200), (400, 300), (50, 150, 50), -1)  # Gras
        cv2.circle(img3, (80, 80), 30, (0, 255, 255), -1)  # Sonne
        cv2.rectangle(img3, (300, 120), (330, 200), (139, 69, 19), -1)  # Baum
        cv2.circle(img3, (315, 110), 20, (34, 139, 34), -1)  # Baumkrone
        path3 = test_dir / "landscape.jpg"
        cv2.imwrite(str(path3), img3)
        images.append((str(path3), "0 Personen"))
        
        # 4. Drei Personen (Gruppe)
        img4 = np.zeros((350, 450, 3), dtype=np.uint8)
        positions = [(120, 100), (225, 110), (330, 105)]
        colors = [(255, 200, 150), (200, 255, 150), (150, 200, 255)]
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            x, y = pos
            cv2.circle(img4, (x, y), 18, color, -1)  # Kopf
            cv2.rectangle(img4, (x-20, y+18), (x+20, y+70), tuple(int(c*0.8) for c in color), -1)  # Körper
            cv2.rectangle(img4, (x-25, y+70), (x-10, y+110), (50, 50, 150), -1)  # Bein 1
            cv2.rectangle(img4, (x+10, y+70), (x+25, y+110), (50, 50, 150), -1)  # Bein 2
        
        path4 = test_dir / "group_three.jpg"
        cv2.imwrite(str(path4), img4)
        images.append((str(path4), "3 Personen"))
        
        print(f"✓ {len(images)} Testbilder erstellt in {test_dir}/")
        return images
        
    except ImportError:
        print("✗ OpenCV nicht verfügbar - kann keine Testbilder erstellen")
        return []


def test_ollama_models(images: List, models_to_test: List[str], host: str):
    """Testet verschiedene Ollama Modelle"""
    print(f"\nTeste {len(models_to_test)} Modelle mit {len(images)} Bildern...")
    print("="*80)
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🤖 TESTE MODELL: {model_name}")
        print("-" * 60)
        
        try:
            detector = OllamaGemma3PersonDetector(
                model_name=model_name,
                confidence_threshold=0.5,
                ollama_host=host
            )
            
            model_results = []
            total_time = 0
            
            for image_path, expected in images:
                print(f"📷 Teste: {Path(image_path).name} (Erwartet: {expected})")
                
                start_time = time.time()
                result = detector.detect(image_path)
                processing_time = time.time() - start_time
                total_time += processing_time
                
                detected = result.get('persons_detected', 0)
                confidence = result.get('avg_confidence', 0.0)
                uncertain = result.get('uncertain', False)
                raw_response = result.get('model_output', {}).get('raw_response', '')
                
                # Bewertung
                expected_num = int(expected.split()[0]) if expected.split()[0].isdigit() else 0
                correct = detected == expected_num
                
                status = "✓" if correct else "✗"
                uncertainty_icon = "⚠" if uncertain else ""
                
                print(f"   {status} Erkannt: {detected}, Konfidenz: {confidence:.3f} {uncertainty_icon}")
                print(f"   ⏱ Zeit: {processing_time:.2f}s")
                print(f"   💬 LLM: '{raw_response.strip()[:50]}{'...' if len(raw_response) > 50 else ''}'")
                
                if result.get('error'):
                    print(f"   ❌ Fehler: {result['error']}")
                
                model_results.append({
                    'image': Path(image_path).name,
                    'expected': expected_num,
                    'detected': detected,
                    'correct': correct,
                    'confidence': confidence,
                    'time': processing_time,
                    'uncertain': uncertain,
                    'raw_response': raw_response
                })
                
                print()
            
            # Zusammenfassung für dieses Modell
            correct_count = sum(1 for r in model_results if r['correct'])
            avg_time = total_time / len(images)
            avg_confidence = sum(r['confidence'] for r in model_results) / len(model_results)
            
            results[model_name] = {
                'accuracy': correct_count / len(images),
                'avg_time': avg_time,
                'avg_confidence': avg_confidence,
                'results': model_results
            }
            
            print(f"📊 {model_name} Zusammenfassung:")
            print(f"   Genauigkeit: {correct_count}/{len(images)} ({100*correct_count/len(images):.1f}%)")
            print(f"   Ø Zeit: {avg_time:.2f}s")
            print(f"   Ø Konfidenz: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Fehler bei {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def print_comparison(results: dict):
    """Druckt Vergleich aller getesteten Modelle"""
    print("\n" + "="*80)
    print("📈 MODELLVERGLEICH")
    print("="*80)
    
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_models:
        print("❌ Keine erfolgreichen Modelle zu vergleichen")
        return
    
    print(f"{'Modell':<15} {'Genauigkeit':<12} {'Ø Zeit':<10} {'Ø Konfidenz':<12}")
    print("-" * 55)
    
    # Sortiere nach Genauigkeit
    sorted_models = sorted(successful_models.items(), 
                          key=lambda x: x[1]['accuracy'], 
                          reverse=True)
    
    for model_name, stats in sorted_models:
        accuracy = f"{stats['accuracy']*100:.1f}%"
        avg_time = f"{stats['avg_time']:.2f}s"
        avg_conf = f"{stats['avg_confidence']:.3f}"
        
        print(f"{model_name:<15} {accuracy:<12} {avg_time:<10} {avg_conf:<12}")
    
    # Empfehlung
    best_model = sorted_models[0] if sorted_models else None
    if best_model:
        print(f"\n🏆 Beste Genauigkeit: {best_model[0]} ({best_model[1]['accuracy']*100:.1f}%)")
    
    fastest_model = min(successful_models.items(), key=lambda x: x[1]['avg_time'])
    print(f"⚡ Schnellstes Modell: {fastest_model[0]} ({fastest_model[1]['avg_time']:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description='Ollama Gemma 3 Vision Test')
    parser.add_argument('--host', default='http://localhost:11434',
                       help='Ollama Server URL')
    parser.add_argument('--models', nargs='+',
                       choices=['gemma3:270m', 'gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'gemma3:27b'],
                       default=['gemma3:270m'],
                       help='Zu testende Modelle')
    parser.add_argument('--create-images', action='store_true',
                       help='Erstelle Testbilder')
    parser.add_argument('--image-dir', 
                       help='Verwende eigene Bilder aus Verzeichnis')
    parser.add_argument('--single-test', action='store_true',
                       help='Nur ein schneller Test')
    
    args = parser.parse_args()
    
    print("🦙 OLLAMA GEMMA 3 VISION TEST")
    print("="*50)
    
    # Prüfe Ollama Status
    if not check_ollama_status(args.host):
        print(f"❌ Ollama läuft nicht auf {args.host}")
        print("Starte Ollama mit: ollama serve")
        sys.exit(1)
    
    print(f"✅ Ollama läuft auf {args.host}")
    
    # Liste verfügbare Modelle
    available_models = list_available_models(args.host)
    print(f"📋 Verfügbare Modelle: {', '.join(available_models)}")
    
    # Prüfe ob gewünschte Modelle verfügbar sind
    missing_models = [m for m in args.models if m not in available_models]
    if missing_models:
        print(f"⚠ Fehlende Modelle: {', '.join(missing_models)}")
        print("Lade sie mit: ollama pull <model_name>")
    
    # Testbilder vorbereiten
    images = []
    
    if args.image_dir:
        # Verwende eigene Bilder
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"❌ Bildverzeichnis nicht gefunden: {image_dir}")
            sys.exit(1)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for ext in extensions:
            images.extend([(str(p), "Unbekannt") for p in image_dir.rglob(f"*{ext}")])
        
        if not images:
            print(f"❌ Keine Bilder in {image_dir} gefunden")
            sys.exit(1)
            
        print(f"📁 Verwende {len(images)} Bilder aus {image_dir}")
        
    elif args.create_images:
        # Erstelle Testbilder
        images = create_test_images()
        if not images:
            sys.exit(1)
    else:
        print("❌ Keine Bilder spezifiziert!")
        print("Verwende --create-images oder --image-dir")
        sys.exit(1)
    
    # Schnelltest
    if args.single_test:
        args.models = args.models[:1]  # Nur erstes Modell
        images = images[:2]  # Nur erste 2 Bilder
        print("⚡ Schnelltest-Modus aktiviert")
    
    # Teste Modelle
    try:
        results = test_ollama_models(images, args.models, args.host)
        
        if len(args.models) > 1:
            print_comparison(results)
        
        print("\n✅ Tests abgeschlossen!")
        
    except KeyboardInterrupt:
        print("\n❌ Tests durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()