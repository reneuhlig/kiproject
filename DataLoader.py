import random

from pathlib import Path
from typing import List, Tuple, Dict, Any

from mysql.connector import Error


class DataLoader:
    """Lädt und verwaltet klassifizierte Bilddaten"""
    
    def __init__(self, data_dir: str, supported_formats: set = None):
        """
        Initialisiert den DataLoader
        
        Args:
            data_dir: Pfad zum Hauptordner mit klassifizierten Unterordnern
            supported_formats: Unterstützte Dateiformate
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = supported_formats or {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
        }
        
    def get_classified_images(self, randomize: bool = True, 
                            classifications: List[str] = None) -> List[Tuple[str, str]]:
        """
        Sammelt alle Bilder aus klassifizierten Ordnern
        
        Args:
            randomize: Ob die Reihenfolge randomisiert werden soll
            classifications: Liste spezifischer Klassifizierungen (None = alle)
            
        Returns:
            Liste von (image_path, classification) Tupeln
        """
        images = []
        
        if not self.data_dir.exists():
            print(f"✗ Datenordner nicht gefunden: {self.data_dir}")
            return images
            
        classification_counts = {}
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            classification = class_dir.name
            
            # Filter nach spezifischen Klassifizierungen falls angegeben
            if classifications and classification not in classifications:
                continue
                
            count = 0
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.supported_formats:
                    images.append((str(img_file), classification))
                    count += 1
                    
            if count > 0:
                classification_counts[classification] = count
        
        # Randomisieren der Reihenfolge falls gewünscht
        if randomize:
            random.shuffle(images)
        
        print(f"✓ {len(images)} Bilder aus {len(classification_counts)} Klassen geladen:")
        for class_name, count in sorted(classification_counts.items()):
            print(f"  - {class_name}: {count} Bilder")
            
        return images
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Gibt Informationen über ein Bild zurück
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Bildinformationen
        """
        path_obj = Path(image_path)
        info = {
            'filename': path_obj.name,
            'size_bytes': 0,
            'width': 0,
            'height': 0,
            'format': path_obj.suffix.lower(),
            'exists': path_obj.exists()
        }
        
        if path_obj.exists():
            info['size_bytes'] = path_obj.stat().st_size
            
            # Bildabmessungen ermitteln
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    info['height'], info['width'] = img.shape[:2]
            except Exception:
                pass
                
        return info

