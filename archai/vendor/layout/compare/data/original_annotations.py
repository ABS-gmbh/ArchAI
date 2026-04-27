"""
Parse CVAT XML annotations and convert to COCO format for evaluation.
"""
import xml.etree.ElementTree as ET
import json
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pycocotools.mask as mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")


def parse_rle(rle_string, width, height):
    """
    Parse RLE (Run-Length Encoding) string from CVAT format.
    CVAT RLE format is a simple list of counts: "count1, count2, count3, ..."
    This represents a flattened binary mask where counts alternate between
    runs of 0s and 1s.
    """
    if not rle_string or not rle_string.strip():
        return None
    
    try:
        # Split by comma and convert to integers
        counts = [int(x.strip()) for x in rle_string.split(',') if x.strip()]
        
        if len(counts) == 0:
            return None
        
        # Create binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Parse RLE: counts alternate between 0s and 1s
        # First count is typically 0s, then 1s, then 0s, etc.
        pos = 0
        is_foreground = False  # Start with background (0s)
        
        for count in counts:
            if is_foreground:
                # Fill foreground pixels
                for _ in range(count):
                    y = pos // width
                    x = pos % width
                    if y < height and x < width:
                        mask[y, x] = 1
                    pos += 1
            else:
                # Skip background pixels
                pos += count
            
            is_foreground = not is_foreground
        
        # Convert to COCO RLE format
        try:
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
        except ImportError:
            # Fallback if pycocotools not available
            print("Warning: pycocotools not available, using bbox only")
            return None
    except Exception as e:
        print(f"Warning: Failed to parse RLE: {e}")
        return None


def bbox_from_mask(rle, width, height):
    """Extract bounding box from RLE mask."""
    if rle is None or not HAS_PYCOCOTOOLS:
        return None
    
    try:
        # Decode RLE to get mask
        rle_decoded = rle.copy()
        rle_decoded['counts'] = rle_decoded['counts'].encode('utf-8')
        mask = mask_util.decode(rle_decoded)
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # COCO format: [x, y, width, height]
        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    except Exception as e:
        print(f"Warning: Failed to extract bbox from mask: {e}")
        return None


def parse_cvat_xml(xml_path, images_dir):
    """
    Parse CVAT XML file and convert to COCO format.
    Handles <box>, <polygon>, and <mask> annotations.
    
    Args:
        xml_path: Path to CVAT annotations.xml file
        images_dir: Directory containing the images
    
    Returns:
        COCO format dictionary
    """
    # 1. Load the XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: Could not find XML file: {xml_path}")
        return None
    
    # 2. Initialize COCO structure
    coco = {
        "info": {
            "description": "Converted from CVAT XML",
            "year": 2024,
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 3. Create Category (Label) Map
    # First, try to get labels from <label> tags in meta section
    labels = set()
    for label in root.findall('.//label'):
        label_name = label.find('name')
        if label_name is not None and label_name.text:
            labels.add(label_name.text)
    
    # Also scan images for any labels used in annotations
    for image in root.findall('image'):
        for child in image:
            if child.tag in ['box', 'polygon', 'mask']:
                label = child.get('label')
                if label:
                    labels.add(label)
    
    # Sort labels to ensure consistent IDs
    label_map = {}
    for i, label_name in enumerate(sorted(list(labels))):
        category_id = i + 1  # COCO IDs start at 1
        label_map[label_name] = category_id
        coco["categories"].append({
            "id": category_id,
            "name": label_name,
            "supercategory": "object"
        })
    
    print(f"Found Categories: {label_map}")
    
    # 4. Parse Images and Annotations
    annotation_id = 1
    image_id = 1
    
    # CVAT images are stored in <image> tags
    for img_tag in root.findall('image'):
        file_name = img_tag.get('name')
        
        # Check if file exists
        full_image_path = Path(images_dir) / file_name
        if not full_image_path.exists():
            print(f"Warning: Image {file_name} mentioned in XML not found in folder. Processing anyway.")
        
        width = int(img_tag.get('width'))
        height = int(img_tag.get('height'))
        
        # Add Image to COCO
        coco_image = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        }
        coco["images"].append(coco_image)
        
        # Process Bounding Boxes (<box>)
        for box in img_tag.findall('box'):
            label = box.get('label')
            if label not in label_map:
                continue
            
            # CVAT uses Top-Left (xtl, ytl) and Bottom-Right (xbr, ybr)
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            # Convert to COCO format: [x_min, y_min, width, height]
            w = xbr - xtl
            h = ybr - ytl
            bbox = [xtl, ytl, w, h]
            area = w * h
            
            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label_map[label],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # Empty for simple boxes
            }
            coco["annotations"].append(ann)
            annotation_id += 1
        
        # Process Polygons (<polygon>)
        for poly in img_tag.findall('polygon'):
            label = poly.get('label')
            if label not in label_map:
                continue
            
            points_str = poly.get('points')  # "x1,y1;x2,y2;..."
            
            # Parse points into flat list [x1, y1, x2, y2, ...]
            points = []
            for pair in points_str.split(';'):
                if not pair.strip():
                    continue
                x, y = map(float, pair.split(','))
                points.extend([x, y])
            
            if len(points) < 6:  # Need at least 3 points (6 coordinates)
                continue
            
            # Calculate bounding box from polygon
            x_coords = points[0::2]
            y_coords = points[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            w = max(x_coords) - x_min
            h = max(y_coords) - y_min
            
            # Calculate polygon area using shoelace formula
            area = 0.5 * abs(sum(x_coords[i] * y_coords[(i + 1) % len(x_coords)] - 
                                x_coords[(i + 1) % len(x_coords)] * y_coords[i] 
                                for i in range(len(x_coords))))
            
            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label_map[label],
                "bbox": [x_min, y_min, w, h],
                "area": area,
                "iscrowd": 0,
                "segmentation": [points]
            }
            coco["annotations"].append(ann)
            annotation_id += 1
        
        # Process Masks (<mask>)
        for mask_elem in img_tag.findall('mask'):
            label_name = mask_elem.get('label')
            if label_name not in label_map:
                continue
            
            # Get RLE data
            rle_string = mask_elem.text
            left = int(mask_elem.get('left', 0))
            top = int(mask_elem.get('top', 0))
            mask_width = int(mask_elem.get('width', width))
            mask_height = int(mask_elem.get('height', height))
            
            # Parse RLE
            rle = parse_rle(rle_string, mask_width, mask_height)
            if rle is None:
                # Fallback: try to create bbox from mask attributes
                bbox = [left, top, mask_width, mask_height]
                area = mask_width * mask_height
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_map[label_name],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                }
                coco["annotations"].append(ann)
                annotation_id += 1
                continue
            
            # Get bounding box from mask
            bbox = bbox_from_mask(rle, mask_width, mask_height)
            if bbox is None:
                continue
            
            # Adjust bbox coordinates if mask has offset
            bbox[0] += left
            bbox[1] += top
            
            # Calculate area
            if HAS_PYCOCOTOOLS:
                area = mask_util.area(rle)
            else:
                # Approximate area from bbox
                area = bbox[2] * bbox[3]
            
            # Create COCO annotation
            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label_map[label_name],
                "segmentation": rle,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            }
            coco["annotations"].append(ann)
            annotation_id += 1
        
        image_id += 1
    
    return coco


def load_ground_truth(xml_path, images_dir):
    """
    Load ground truth annotations from CVAT XML.
    
    Args:
        xml_path: Path to annotations.xml
        images_dir: Directory containing images
    
    Returns:
        COCO format dictionary
    """
    return parse_cvat_xml(xml_path, images_dir)


if __name__ == "__main__":
    # Test parsing
    xml_path = "Aleyna 1 (2024)/Annotations/annotations.xml"
    images_dir = "Aleyna 1 (2024)/Images"
    output_json = "ground_truth_coco.json"
    
    coco = load_ground_truth(xml_path, images_dir)
    
    if coco:
        print(f"\nSuccess! Converted {len(coco['images'])} images and {len(coco['annotations'])} annotations.")
        print(f"Categories: {[c['name'] for c in coco['categories']]}")
        
        # Save to JSON for inspection
        with open(output_json, "w") as f:
            json.dump(coco, f, indent=4)
        print(f"Saved to: {output_json}")
    else:
        print("Error: Failed to parse XML file")

