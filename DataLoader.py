import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from mysql.connector import Error


class DataLoader:
    """Lädt und verwaltet klassifizierte Bilddaten mit automatischer Erkennung"""
    
    def __init__(self, data_dir: str, supported_formats: set = None):
        """
        Initialisiert den DataLoader
        
        Args:
            data_dir: Pfad zum Hauptordner (wird rekursiv durchsucht)
            supported_formats: Unterstützte Dateiformate
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = supported_formats or {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
        }
        
    def get_classified_images(self, randomize: bool = True, 
                            classifications: List[str] = None,
                            max_depth: int = 10) -> List[Tuple[str, str]]:
        """
        Sammelt alle Bilder rekursiv aus allen Unterverzeichnissen
        
        Args:
            randomize: Ob die Reihenfolge randomisiert werden soll
            classifications: Liste spezifischer Klassifizierungen (None = alle)
            max_depth: Maximale Suchtiefe für rekursive Suche
            
        Returns:
            Liste von (image_path, classification) Tupeln
        """
        images = []
        
        if not self.data_dir.exists():
            print(f"✗ Datenordner nicht gefunden: {self.data_dir}")
            return images
            
        classification_counts = {}
        
        print(f"🔍 Durchsuche {self.data_dir} rekursiv nach Bildern...")
        
        # Rekursive Suche nach allen Bildern
        for img_file in self._find_images_recursive(self.data_dir, max_depth):
            # Klassifizierung aus Pfad ableiten
            classification = self._extract_classification_from_path(img_file)
            
            # Filter nach spezifischen Klassifizierungen falls angegeben
            if classifications and classification not in classifications:
                continue
            
            images.append((str(img_file), classification))
            
            # Statistik aktualisieren
            if classification in classification_counts:
                classification_counts[classification] += 1
            else:
                classification_counts[classification] = 1
        
        # Randomisieren der Reihenfolge falls gewünscht
        if randomize:
            random.shuffle(images)
        
        print(f"✓ {len(images)} Bilder aus {len(classification_counts)} Klassifizierungen gefunden:")
        for class_name, count in sorted(classification_counts.items()):
            print(f"  - {class_name}: {count} Bilder")
        
        # Zeige Beispiel-Pfade für jede Klassifizierung
        if len(classification_counts) <= 10:  # Nur bei wenigen Klassifizierungen
            print("📁 Beispiel-Pfade:")
            shown_classifications = set()
            for img_path, classification in images[:20]:  # Erste 20 Bilder
                if classification not in shown_classifications:
                    rel_path = Path(img_path).relative_to(self.data_dir)
                    print(f"  - {classification}: .../{rel_path}")
                    shown_classifications.add(classification)
                    if len(shown_classifications) >= 5:  # Maximal 5 Beispiele
                        break
            
        return images
    
    def _find_images_recursive(self, directory: Path, max_depth: int) -> List[Path]:
        """
        Findet alle Bilder rekursiv in einem Verzeichnis
        
        Args:
            directory: Zu durchsuchendes Verzeichnis
            max_depth: Maximale Suchtiefe
            
        Returns:
            Liste aller gefundenen Bilddateien
        """
        images = []
        
        try:
            # Verwende rglob für rekursive Suche mit Tiefenbegrenzung
            for img_file in directory.rglob("*"):
                # Prüfe Suchtiefe
                try:
                    relative_path = img_file.relative_to(directory)
                    depth = len(relative_path.parts)
                    if depth > max_depth:
                        continue
                except ValueError:
                    continue
                
                # Prüfe ob es eine Datei mit unterstütztem Format ist
                if (img_file.is_file() and 
                    img_file.suffix.lower() in self.supported_formats):
                    images.append(img_file)
                    
        except Exception as e:
            print(f"⚠ Fehler beim Durchsuchen von {directory}: {e}")
            
        return images
    
    def _extract_classification_from_path(self, img_path: Path) -> str:
        """
        Extrahiert Klassifizierung aus dem Dateipfad
        
        Strategien (in dieser Reihenfolge):
        1. Direkter Eltern-Ordner als Klassifizierung
        2. Tiefster nicht-root Ordner
        3. Kombination aus mehreren Ordnern
        4. Fallback basierend auf Pfad-Elementen
        
        Args:
            img_path: Pfad zur Bilddatei
            
        Returns:
            Klassifizierungs-String
        """
        try:
            # Relativer Pfad zum Datenverzeichnis
            rel_path = img_path.relative_to(self.data_dir)
            path_parts = rel_path.parts[:-1]  # Ohne Dateiname
            
            if not path_parts:
                # Datei direkt im Root-Verzeichnis
                return "root_level"
            
            # Strategie 1: Direkter Eltern-Ordner
            if len(path_parts) == 1:
                return self._normalize_classification(path_parts[0])
            
            # Strategie 2: Tiefster Ordner (meist spezifischste Klassifizierung)
            deepest_folder = path_parts[-1]
            
            # Strategie 3: Kombiniere letzte 2 Ordner falls sinnvoll
            if len(path_parts) >= 2:
                parent_folder = path_parts[-2]
                
                # Kombiniere wenn parent_folder generisch erscheint
                if self._is_generic_folder_name(parent_folder):
                    combined = f"{parent_folder}_{deepest_folder}"
                    return self._normalize_classification(combined)
            
            return self._normalize_classification(deepest_folder)
            
        except Exception as e:
            print(f"⚠ Fehler beim Extrahieren der Klassifizierung von {img_path}: {e}")
            return "unknown"
    
    def _is_generic_folder_name(self, folder_name: str) -> bool:
        """
        Prüft ob ein Ordnername generisch ist (dann sollte kombiniert werden)
        """
        generic_names = {
            'data', 'images', 'pics', 'pictures', 'photos', 'files',
            'dataset', 'training', 'test', 'validation', 'samples',
            'input', 'output', 'raw', 'processed', 'temp', 'tmp',
            'archive', 'backup', 'export', 'import', 'uploads',
            'year', 'month', 'day', 'batch', 'set', 'group'
        }
        
        folder_lower = folder_name.lower()
        
        # Exakte Übereinstimmung
        if folder_lower in generic_names:
            return True
        
        # Enthält Jahr/Datum
        if (folder_lower.isdigit() and 
            (len(folder_lower) == 4 or len(folder_lower) == 2)):
            return True
        
        # Enthält generische Begriffe
        for generic in generic_names:
            if generic in folder_lower:
                return True
        
        return False
    
    def _normalize_classification(self, classification: str) -> str:
        """
        Normalisiert Klassifizierungsnamen für Konsistenz
        
        Args:
            classification: Roher Klassifizierungsname
            
        Returns:
            Normalisierter Klassifizierungsname
        """
        # Basis-Normalisierung
        normalized = classification.lower().strip()
        
        # Ersetze häufige Zeichen
        replacements = {
            ' ': '_',
            '-': '_',
            '.': '_',
            '(': '',
            ')': '',
            '[': '',
            ']': '',
            '{': '',
            '}': '',
            '&': 'and',
            '+': 'plus'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Mehrfache Unterstriche reduzieren
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        
        # Führende/folgende Unterstriche entfernen
        normalized = normalized.strip('_')
        
        # Leer oder nur Sonderzeichen? -> Fallback
        if not normalized or not any(c.isalnum() for c in normalized):
            normalized = "unclassified"
        
        # Länge begrenzen
        if len(normalized) > 50:
            normalized = normalized[:50].rstrip('_')
        
        return normalized
    
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
            'exists': path_obj.exists(),
            'relative_path': str(path_obj.relative_to(self.data_dir)) if path_obj.is_relative_to(self.data_dir) else str(path_obj)
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
    
    def preview_structure(self, max_examples: int = 5) -> Dict[str, Any]:
        """
        Gibt eine Vorschau der Datenstruktur zurück ohne alle Bilder zu laden
        
        Args:
            max_examples: Maximale Anzahl Beispiele pro Klassifizierung
            
        Returns:
            Dictionary mit Struktur-Informationen
        """
        structure = {}
        total_images = 0
        
        print(f"🔍 Analysiere Struktur von {self.data_dir}...")
        
        for img_file in self._find_images_recursive(self.data_dir, max_depth=10):
            classification = self._extract_classification_from_path(img_file)
            
            if classification not in structure:
                structure[classification] = {
                    'count': 0,
                    'examples': []
                }
            
            structure[classification]['count'] += 1
            total_images += 1
            
            # Beispiele sammeln
            if len(structure[classification]['examples']) < max_examples:
                rel_path = img_file.relative_to(self.data_dir)
                structure[classification]['examples'].append(str(rel_path))
        
        return {
            'total_images': total_images,
            'classifications': len(structure),
            'structure': structure
        }