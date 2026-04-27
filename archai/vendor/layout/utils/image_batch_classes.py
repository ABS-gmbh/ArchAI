from pathlib import Path
import json
import numpy as np
from PIL import Image as PILImage
import os
from rtree import index
from shapely.geometry import box

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Constants for category mappings
catmus_zones_mapping = {
    'DefaultLine': 'Main script black',
    'InterlinearLine': 'Gloss',
        'MainZone': 'Column',
    'DropCapitalZone': 'Plain initial- coloured',
     'StampZone': 'Illustrations',
    'GraphicZone': 'Illustrations',
    'MarginTextZone': 'Gloss',
    'MusicZone': 'Music',
    'NumberingZone': 'Page Number',
    'QuireMarksZone': 'Quire Mark',
    'RunningTitleZone': 'Running header',
    'TitlePageZone': 'Column'
}

coco_class_mapping = {
    'Border': 1,
    'Table': 2,
    'Diagram': 3,
    'Main script black': 4,
    'Main script coloured': 5,
    'Variant script black': 6,
    'Variant script coloured': 7,
    'Historiated': 8,
    'Inhabited': 9,
    'Zoo - Anthropomorphic': 10,
    'Embellished': 11,
    'Plain initial- coloured': 12,
    'Plain initial - Highlighted': 13,
    'Plain initial - Black': 14,
    'Page Number': 15,
    'Quire Mark': 16,
    'Running header': 17,
    'Catchword': 18,
    'Gloss': 19,
    'Illustrations': 20,
    'Column': 21,
    'GraphicZone': 22,
    'MusicLine': 23,
    'MusicZone': 24,
    'Music': 25
}


class Annotation:
    def __init__(self, annotation, image):
        self.name = annotation['name']
        self.cls = annotation['class']
        self.confidence = annotation['confidence']
        self.bbox = annotation['box']
        self.segments = annotation['segments'] if 'segments' in annotation else None
        #Annotation contains name, class, confidence, bbox and segments
        self.image = image
    
    def set_id(self, id):
        self.id = id

    def fix_empty_segments(self,x_coords,y_coords):
        self.segments = {'x': x_coords, 'y': y_coords}

    def segments_to_coco_format(self, segment_dict):
        coco_segment = []
        for x, y in zip(segment_dict['x'], segment_dict['y']):
            coco_segment.append(x)
            coco_segment.append(y)
        return [coco_segment]

    def bbox_to_coco_format(self, box):
        x = box['x1']
        y = box['y1']
        width = box['x2'] - box['x1']
        height = box['y2'] - box['y1']
        return [x, y, width, height]

    def polygon_area(self, segment_dict):
        #Showlace formula for area of polygon
        x = segment_dict['x']
        y = segment_dict['y']
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def unify_names(self):
        self.name = catmus_zones_mapping.get(self.name, self.name)

    def to_coco_format(self, current_annotation_id):
        cls_string = catmus_zones_mapping.get(self.name, self.name)
        cls_int = coco_class_mapping[cls_string]

        if self.segments:
            segmentation = self.segments_to_coco_format(self.segments)
            area = self.polygon_area(self.segments)

        else:
            segmentation = []
            width = self.bbox['x2'] - self.bbox['x1']
            height = self.bbox['y2'] - self.bbox['y1']
            area = width * height
        
        annotation_dict = {
            "id": current_annotation_id,
            "image_id": self.image.id,
            "category_id": cls_int,
            "segmentation": segmentation,
            "area": area,
            "bbox": self.bbox_to_coco_format(self.bbox),
            "iscrowd": 0,
            "attributes": {"occluded": False}
        }
        return annotation_dict
    

class Image:
    def __init__(self, image_path, image_id):
        self.path = image_path
        self.id = image_id
        self.filename = os.path.basename(image_path)
        self.width, self.height = self._get_image_dimensions()
        self.annotations = []
        self.spatial_index = index.Index()
        self.deleted_indices = []
        self.annotations_dict = {}

    def _get_image_dimensions(self):
        with PILImage.open(self.path) as img:
            return img.size
        

    def process_intersection(self, new_box, relevant_classes, overlap_threshold, percentage_dividend, index_to_remove=-1):
        """
        Processes intersection of a new bounding box with existing bounding boxes in the spatial index.

        :param new_box: The new bounding box to check for intersections.
        :param relevant_classes: List of relevant classes to consider for processing.
        :param overlap_threshold: Minimum overlap percentage threshold to consider an intersection.
        :param percentage_dividend: Criterion for calculating percentage overlap ('new_box', 'match_bbox', 'symmetric').
        :param index_to_remove: Index to remove from self.deleted_indices; if -1, remove the intersecting box.
        """
        # Find possible matches using spatial index
        possible_matches = self.spatial_index.intersection(new_box.bounds, objects=True)

        # Iterate over possible matches
        for match in possible_matches:
            # Filter matches based on relevant classes
            if match.object['class'] not in relevant_classes:
                continue

            # Create bounding box for the matched object
            match_bbox = box(*match.bbox)

            # Calculate the intersection area
            intersection_area = new_box.intersection(match_bbox).area

            # Calculate percentage intersection based on the specified dividend
            if percentage_dividend == 'new_box':
                percentage_intersection = intersection_area / new_box.area
            elif percentage_dividend == 'match_bbox':
                percentage_intersection = intersection_area / match_bbox.area
            elif percentage_dividend == 'symmetric':
                # Ensure that both percentages meet the threshold
                percentage_intersection = min(intersection_area / new_box.area, intersection_area / match_bbox.area)
            else:
                raise ValueError("Invalid percentage_dividend value. Must be 'new_box', 'match_bbox', or 'symmetric'.")

            # Append to deleted indices if conditions are met and avoid duplicates
            if percentage_intersection > overlap_threshold:
                to_remove = index_to_remove if index_to_remove != -1 else match.id
                if to_remove not in self.deleted_indices:
                    self.deleted_indices.append(to_remove)


    def process_defaultline(self,new_box,index):
        
        possible_matches = list(self.spatial_index.intersection(new_box.bounds, objects=True))
        #Remove default line if it intersects with any of the following
        variant_colored_matches = [match for match in possible_matches if match.object['class'] in ['Variant script coloured',
        'Variant script black','Main script coloured','NumberingZone','Diagram','MarginTextZone','RunningTitleZone','Table',
        'Quire Mark']]

        if variant_colored_matches:
            self.deleted_indices.append(index)
        else:
            for match in possible_matches:      
                #Remove Main Script Black if its area overlaps with the default line
                if match.object['class']=='Main script black':
                    match_bbox= box(*match.bbox)
                    intersection_area = new_box.intersection(match_bbox).area
                    percentage_intersection = (intersection_area / match_bbox.area)
                    if percentage_intersection > 0.6:
                        self.deleted_indices.append(match.id)
        
    
    def add_annotation(self, annotation):
        #Store indices to remove to remove them at the end
        pos = len(self.annotations)
        #Correct annotations with segments with empty coordinates
        minx,miny,maxx,maxy=annotation.bbox['x1'],annotation.bbox['y1'],annotation.bbox['x2'],annotation.bbox['y2']
        new_box = box(minx,miny,maxx,maxy)

        if annotation.segments: # Execute validations for segmentation models

            if not annotation.segments['x']:
                x_coords = [minx, minx, maxx, maxx, minx]  
                y_coords = [miny, maxy, maxy, miny, miny]
                annotation.fix_empty_segments(x_coords, y_coords)

            if annotation.name in ['Main script black','Main script coloured','Variant script black','Variant script coloured','Plain initial- coloured','Plain initial - Highlighted','Plain initial - Black']:
                self.process_intersection(new_box,['MarginTextZone','NumberingZone'],0.7,'new_box',pos)
            
            if annotation.name in ['Embellished','Plain initial- coloured','Plain initial - Highlighted','Plain initial - Black','Inhabited']:
                self.process_intersection(new_box,['DropCapitalZone','GraphicZone'],0.4,'symmetric')
                
            if annotation.name=='Page Number':
                self.process_intersection(new_box,['NumberingZone'],0.8,'new_box',pos)

            if annotation.name=='Music':
                self.process_intersection(new_box,['MusicZone','GraphicZone'],0.7,'new_box')

            if annotation.name=='Table':
                self.process_intersection(new_box,['MainZone','MarginTextZone'],0.4,'match_bbox')

            if annotation.name in ['Diagram','Illustrations']:
                self.process_intersection(new_box,['GraphicZone'],0.5,'new_box')

            if annotation.name=='DefaultLine':
                
                self.process_defaultline(new_box,pos)


        self.annotations.append(annotation)
        
        annotation.set_id(pos)
        self.spatial_index.insert(pos, new_box.bounds,obj={'class':annotation.name})

    def filter_annotations(self):
    # Convert delete_indices to a set for faster lookup
        delete_indices_set = set(self.deleted_indices)
        filtered_annotations = [item for index, item in enumerate(self.annotations) if index not in delete_indices_set]
        return filtered_annotations
    
    def unify_names(self):
        overlapping_classes = ['MainZone','MarginTextZone']
        for index, annotation in enumerate(self.annotations):
            if index not in self.deleted_indices and annotation.name in overlapping_classes:
                minx,miny,maxx,maxy=annotation.bbox['x1'],annotation.bbox['y1'],annotation.bbox['x2'],annotation.bbox['y2']
                new_box = box(minx,miny,maxx,maxy)

                possible_matches = self.spatial_index.intersection(new_box.bounds, objects=True)

                for match in possible_matches:

                    if match.id not in self.deleted_indices and match.object['class']==annotation.name and match.id!=index:
                        match_bbox= box(*match.bbox)


                        # Calculate the intersection area as a percentage of the smaller box area
                        if new_box.area > match_bbox.area:
                            intersection_area = new_box.intersection(match_bbox).area / match_bbox.area
                        else:
                            intersection_area = match_bbox.intersection(new_box).area / new_box.area

                        if intersection_area > 0.80:
                            delete_index = index if new_box.area < match_bbox.area else match.id
                            self.deleted_indices.append(delete_index)
                        
            annotation.unify_names()



    

    def to_coco_image_dict(self):
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
    
    def plot_annotations(self):
    # Load the image
        with PILImage.open(self.path) as img:
            fig, ax = plt.subplots(1, figsize=(self.width / 100, self.height / 100), dpi=100)
            ax.imshow(img)
            
            for annotation in self.filter_annotations():
                if annotation.segments:
                    
                    # Plot polygon segments
                    x = annotation.segments['x']
                    y = annotation.segments['y']
                    # Close the polygon by appending the first point to the end
                    x.append(x[0])
                    y.append(y[0])

                    polygon = patches.Polygon(xy=list(zip(x, y)), closed=True, edgecolor='r', facecolor='none')
                    ax.add_patch(polygon)
                    # Annotate the polygon with the name
                    plt.text(x[0], y[0], annotation.name, color='red', fontsize=25, verticalalignment='top')
                else:
                    # Plot bounding box if no segments
                    bbox = annotation.bbox
                    x1, y1 = bbox['x1'], bbox['y1']
                    x2, y2 = bbox['x2'], bbox['y2']
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    # Annotate the bounding box with the name
                    plt.text(x1, y1, annotation.name, color='red', fontsize=25, verticalalignment='top')
            
            plt.title(f"Image ID: {self.id} - {self.filename}")
            plt.axis('off')  # Hide axes
            plt.show()



class ImageBatch:
    def __init__(self, image_folder, catmus_labels_folder, emanuskript_labels_folder,zone_labels_folder):
        self.image_folder = image_folder
        self.catmus_labels_folder = catmus_labels_folder
        self.emanuskript_labels_folder = emanuskript_labels_folder
        self.zone_labels_folder = zone_labels_folder
        self.images = []



    def load_images(self):
        image_paths = [
            str(path).replace('\\', '/') 
            for path in Path(self.image_folder).glob('*') 
            if path.is_file()  # Ensure only files are processed
        ]
        image_paths = sorted(image_paths)
        
        for image_id, image_path in enumerate(image_paths, start=1):
            print(f"Processing image: {image_path}")  # Print the image path
            self.images.append(Image(image_path, image_id))


    def load_annotations(self):
        for image in self.images:
            image_basename = os.path.splitext(image.filename)[0]

            catmus_json_path = f'{self.catmus_labels_folder}/{image_basename}.json'
            emanuskript_json_path = f'{self.emanuskript_labels_folder}/{image_basename}.json'
            zone_json_path = f'{self.zone_labels_folder}/{image_basename}.json'

            with open(catmus_json_path) as f:
                catmus_predictions = json.load(f)

            with open(emanuskript_json_path) as f:
                emanuskripts_predictions = json.load(f)

            with open(zone_json_path) as f:
                zone_predictions = json.load(f)

            for annotation_data in zone_predictions + emanuskripts_predictions + catmus_predictions :
                
                if annotation_data['name'] =='Variant script black' and len(annotation_data['segments']['x'])<3:
                    pass
                else:
                    annotation = Annotation(annotation_data, image)
                    image.add_annotation(annotation)

    def unify_names(self):
        for image in self.images:
            image.unify_names()

    def create_coco_dict(self):
        coco_dict = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": [
                {"id": coco_id, "name": cls_name, "supercategory": ""}
                for cls_name, coco_id in coco_class_mapping.items()
            ],
            "annotations": [annotation.to_coco_format(annotation_id) for image in self.images for annotation_id, annotation in enumerate(image.filter_annotations(), start=1)],
            "images": [image.to_coco_image_dict() for image in self.images]
        }
        return coco_dict

    def save_coco_file(self, output_file):
        coco_dict = self.create_coco_dict()
        with open(output_file, 'w') as f:
            json.dump(coco_dict, f, indent=4)

    def return_coco_file(self):
        coco_dict = self.create_coco_dict()
        return coco_dict