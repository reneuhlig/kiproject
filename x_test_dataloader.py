#!/usr/bin/env python3
"""
Test-Script für den erweiterten DataLoader mit Auto-Discovery
"""

import sys
from pathlib import Path
from DataLoader import DataLoader


def test_dataloader_discovery(data_dir: str):
    """Testet die Auto-Discovery Funktionalität des DataLoaders"""
    
    print(f"🧪 Teste DataLoader Auto-Discovery für: {data_dir}")
    print("=" * 80)
    
    # DataLoader initialisieren
    loader = DataLoader(data_dir)
    
    # 1. Struktur-Vorschau
    print("1️⃣ STRUKTUR-ANALYSE")
    print("-" * 40)
    structure_info = loader.preview_structure(max_examples=3)
    
    print(f"📊 Gesamt: {structure_info['total_images']} Bilder in {structure_info['classifications']} Klassifizierungen")
    print()
    
    for classification, info in sorted(structure_info['structure'].items()):
        print(f"📁 {classification}: {info['count']} Bilder")
        for example in info['examples']:
            print(f"   └── {example}")
        if info['count'] > len(info['examples']):
            print(f"   └── ... und {info['count'] - len(info['examples'])} weitere")
        print()
    
    # 2. Test mit begrenzter Anzahl
    print("2️⃣ TEST-LAUF (max 10 Bilder)")
    print("-" * 40)
    test_images = loader.get_classified_images(randomize=False, max_depth=5)
    
    if test_images:
        print(f"✅ {len(test_images)} Bilder gefunden")
        
        # Zeige erste paar Beispiele
        print("\n📋 Erste Beispiele:")
        for i, (image_path, classification) in enumerate(test_images[:10]):
            image_info = loader.get_image_info(image_path)
            size_mb = image_info['size_bytes'] / (1024 * 1024) if image_info['size_bytes'] > 0 else 0
            print(f"  {i+1:2d}. [{classification}] {image_info['filename']} "
                  f"({image_info['width']}x{image_info['height']}, {size_mb:.1f}MB)")
        
        if len(test_images) > 10:
            print(f"       ... und {len(test_images) - 10} weitere")
            
    else:
        print("❌ Keine Bilder gefunden!")
        return False
    
    # 3. Test verschiedener Filter
    print("\n3️⃣ FILTER-TESTS")
    print("-" * 40)
    
    # Verfügbare Klassifizierungen
    available_classifications = list(set(classification for _, classification in test_images))
    print(f"Verfügbare Klassifizierungen: {', '.join(sorted(available_classifications))}")
    
    # Test mit spezifischer Klassifizierung (erste verfügbare)
    if available_classifications:
        test_classification = available_classifications[0]
        filtered_images = loader.get_classified_images(
            randomize=False, 
            classifications=[test_classification]
        )
        print(f"✅ Filter '{test_classification}': {len(filtered_images)} Bilder")
    
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_dataloader.py /path/to/data")
        print("Example: python test_dataloader.py /blob")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    if not Path(data_dir).exists():
        print(f"❌ Verzeichnis nicht gefunden: {data_dir}")
        sys.exit(1)
    
    try:
        success = test_dataloader_discovery(data_dir)
        if success:
            print("\n🎉 Test erfolgreich abgeschlossen!")
        else:
            print("\n❌ Test fehlgeschlagen!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Fehler beim Test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()