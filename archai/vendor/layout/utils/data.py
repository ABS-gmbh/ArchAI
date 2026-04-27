from database import (
    fix_ids,
    ImageModel,
    CategoryModel,
    AnnotationModel,
    DatasetModel,
    TaskModel,
    ExportModel
)

# import pycocotools.mask as mask
import numpy as np
import time
import json
import os
import gc


from celery import shared_task
from ..socket import create_socket
from mongoengine import Q



@shared_task
def export_annotations(task_id, dataset_id, categories, with_empty_images=False):

    task = TaskModel.objects.get(id=task_id)
    dataset = DatasetModel.objects.get(id=dataset_id)

    task.update(status="PROGRESS")
    socket = create_socket()

    task.info("Beginning Export (COCO Format)")

    db_categories = CategoryModel.objects(id__in=categories, deleted=False) \
        .only(*CategoryModel.COCO_PROPERTIES)
    db_images = ImageModel.objects(
        deleted=False, dataset_id=dataset.id).only(
        *ImageModel.COCO_PROPERTIES)
    db_annotations = AnnotationModel.objects(
        deleted=False, category_id__in=categories)

    total_items = db_categories.count()

    coco = {
        'images': [],
        'categories': [],
        'annotations': []
    }

    total_items += db_images.count()
    progress = 0

    # iterate though all ccategories
    category_names = []
    for category in fix_ids(db_categories):

        if len(category.get('keypoint_labels', [])) > 0:
            category['keypoints'] = category.pop('keypoint_labels', [])
            category['skeleton'] = category.pop('keypoint_edges', [])
        else:
            if 'keypoint_edges' in category:
                del category['keypoint_edges']
            if 'keypoint_labels' in category:
                del category['keypoint_labels']

        task.info(f"Adding category: {category.get('name')}")
        coco.get('categories').append(category)
        category_names.append(category.get('name'))

        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

    total_annotations = db_annotations.count()
    total_images = db_images.count()
    for image in db_images:
        image = fix_ids(image)

        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

        annotations = db_annotations.filter(image_id=image.get('id'))\
            .only(*AnnotationModel.COCO_PROPERTIES)
        annotations = fix_ids(annotations)

        if len(annotations) == 0:
            if with_empty_images:
                coco.get('images').append(image)
            continue

        num_annotations = 0
        for annotation in annotations:

            has_keypoints = len(annotation.get('keypoints', [])) > 0
            has_segmentation = len(annotation.get('segmentation', [])) > 0

            if has_keypoints or has_segmentation:

                if not has_keypoints:
                    if 'keypoints' in annotation:
                        del annotation['keypoints']
                else:
                    arr = np.array(annotation.get('keypoints', []))
                    arr = arr[2::3]
                    annotation['num_keypoints'] = len(arr[arr > 0])

                num_annotations += 1
                coco.get('annotations').append(annotation)

        task.info(
            f"Exporting {num_annotations} annotations for image {image.get('id')}")
        coco.get('images').append(image)

    task.info(
        f"Done export {total_annotations} annotations and {total_images} images from {dataset.name}")

    timestamp = time.time()
    directory = f"{dataset.directory}.exports/"
    file_path = f"{directory}coco-{timestamp}.json"

    if not os.path.exists(directory):
        os.makedirs(directory)

    task.info(f"Writing export to file {file_path}")
    with open(file_path, 'w') as fp:
        json.dump(coco, fp)

    task.info("Creating export object")
    export = ExportModel(dataset_id=dataset.id, path=file_path, tags=[
                         "COCO", *category_names])
    export.save()

    task.set_progress(100, socket=socket)


def process_coco_file(coco_json,task,socket,dataset,images,categories):
    coco_images = coco_json.get('images', [])
    coco_annotations = coco_json.get('annotations', [])
    coco_categories = coco_json.get('categories', [])

    task.info(f"Importing {len(coco_categories)} categories, "
              f"{len(coco_images)} images, and "
              f"{len(coco_annotations)} annotations")

    total_items = sum([
        len(coco_categories),
        len(coco_annotations),
        len(coco_images)
    ])
    progress = 0

    task.info("===== Importing Categories =====")
    # category id mapping  ( file : database )
    categories_id = {}

    # Create any missing categories
    for category in coco_categories:

        category_name = category.get('name')
        category_id = category.get('id')
        category_model = categories.filter(name__iexact=category_name).first()

        if category_model is None:
            task.warning(
                f"{category_name} category not found (creating a new one)")

            new_category = CategoryModel(
                name=category_name,
                keypoint_edges=category.get('skeleton', []),
                keypoint_labels=category.get('keypoints', [])
            )
            new_category.save()

            category_model = new_category
            dataset.categories.append(new_category.id)

        task.info(f"{category_name} category found")
        # map category ids
        categories_id[category_id] = category_model.id

        # update progress
        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

    dataset.update(set__categories=dataset.categories)

    task.info("===== Loading Images =====")
    # image id mapping ( file: database )
    images_id = {}
    categories_by_image = {}

    # Find all images
    for image in coco_images:
        image_id = image.get('id')
        image_filename = image.get('file_name')

        # update progress
        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

        image_model = images.filter(file_name__exact=image_filename).all()

        if len(image_model) == 0:
            task.warning(f"Could not find image {image_filename}")
            continue

        if len(image_model) > 1:
            task.error(
                f"Too many images found with the same file name: {image_filename}")
            continue

        task.info(f"Image {image_filename} found")
        image_model = image_model[0]
        images_id[image_id] = image_model
        categories_by_image[image_id] = list()

    task.info("===== Import Annotations =====")
    for annotation in coco_annotations:

        image_id = annotation.get('image_id')
        category_id = annotation.get('category_id')
        segmentation = annotation.get('segmentation', [])
        keypoints = annotation.get('keypoints', [])
        # is_crowd = annotation.get('iscrowed', False)
        area = annotation.get('area', 0)
        bbox = annotation.get('bbox', [0, 0, 0, 0])
        isbbox = annotation.get('isbbox', False)

        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

        has_segmentation = len(segmentation) > 0
        has_keypoints = len(keypoints) > 0
        if not has_segmentation and not has_keypoints:
           task.warning(
               f"Annotation {annotation.get('id')} has no segmentation or keypoints, but bbox {bbox}")
           #continue

        try:
            image_model = images_id[image_id]
            category_model_id = categories_id[category_id]
            image_categories = categories_by_image[image_id]
        except KeyError:
            task.warning(
                f"Could not find image assoicated with annotation {annotation.get('id')}")
            continue

        annotation_model = AnnotationModel.objects(
            image_id=image_model.id,
            category_id=category_model_id,
            segmentation=segmentation,
            keypoints=keypoints
        ).first()

        if annotation_model is None:
            task.info(f"Creating annotation data ({image_id}, {category_id})")

            annotation_model = AnnotationModel(image_id=image_model.id)
            annotation_model.category_id = category_model_id

            annotation_model.color = annotation.get('color')
            annotation_model.metadata = annotation.get('metadata', {})
            annotation_model.area = area
            annotation_model.bbox = bbox
            
            if has_segmentation:
                annotation_model.segmentation = segmentation
            else:
                task.warning(
               f"Annotation {annotation.get('id')} has no segmentation. Creating one from bbox {bbox}")

                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                segments = [
                    x_max, y_min,  # Top-right corner
                    x_max, y_max,  # Bottom-right corner
                    x_min, y_max,  # Bottom-left corner
                    x_min, y_min   # Top-left corner
                ]

                annotation_model.segmentation = segments

            if has_keypoints:
                annotation_model.keypoints = keypoints

            annotation_model.isbbox = isbbox
            annotation_model.save()

            image_categories.append(category_id)
        else:
            annotation_model.update(deleted=False, isbbox=isbbox)
            task.info(
                f"Annotation already exists (i:{image_id}, c:{category_id})")

    for image_id in images_id:
        image_model = images_id[image_id]
        category_ids = categories_by_image[image_id]
        all_category_ids = list(image_model.category_ids)
        all_category_ids += category_ids

        num_annotations = AnnotationModel.objects(
            Q(image_id=image_id) & Q(deleted=False) &
            (Q(area__gt=0) | Q(keypoints__size__gt=0))
        ).count()

        image_model.update(
            set__annotated=True,
            set__category_ids=list(set(all_category_ids)),
            set__num_annotations=num_annotations
        )

    task.set_progress(100, socket=socket)


@shared_task
def import_annotations(task_id, dataset_id, coco_json):

    task = TaskModel.objects.get(id=task_id)
    dataset = DatasetModel.objects.get(id=dataset_id)

    task.update(status="PROGRESS")
    socket = create_socket()

    task.info("Beginning Import")

    images = ImageModel.objects(dataset_id=dataset.id)
    categories = CategoryModel.objects

    process_coco_file(coco_json,task,socket,dataset,images,categories)


@shared_task
def predict_annotations(task_id, model_name, image_path,image_id,dict_labels_folders):
    from ultralytics import YOLO

    if model_name=='emanuskript':
        emanuskript_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]
        model = YOLO("workers/best_emanuskript_segmentation.onnx",task='segment')
        results = model.predict(image_path,classes = emanuskript_classes,
                                                    iou=0.3,device='cpu',augment=False,stream=False)
    
    elif model_name=='catmus':
        catmus_classes=[1,7]
        model = YOLO("workers/best_catmus.onnx",task='segment')
        results = model.predict(image_path,classes = catmus_classes,
                                iou=0.3,device='cpu',augment=False,stream=False)
    elif model_name=='zone':
        model = YOLO("workers/best_zone_detection.pt")
        results = model.predict(image_path,device='cpu',
                                                iou=0.3,
                                                augment=False,stream=False)
    else:
        raise Exception('Model name must be one of emanuskript, catmus or zone')
    
    # get the images to apply the model
    task = TaskModel.objects.get(id=task_id)
    
    # Save labels
    result = results[0]
    prediction_path = f'{dict_labels_folders[model_name]}/{image_id}.json'
    with open(prediction_path,'w') as f:
        f.write(result.tojson())

    task.info(f'Labels predicted in : {prediction_path}')
    task.update(status="COMPLETED")
    del model
    del result
    del results
    gc.collect() 
    return 1



@shared_task
def unify_predictions(results, task_id, dataset_id, images_path,dict_labels_folders):
    
    #Results is unused by necessary for Celery Chord
    from .image_batch_classes import ImageBatch

    task = TaskModel.objects.get(id=task_id)
    task.info(f'Starts prediction unification')
    dataset = DatasetModel.objects.get(id=dataset_id)

    image_batch = ImageBatch(
        image_folder=images_path,
        catmus_labels_folder=dict_labels_folders['catmus'],
        emanuskript_labels_folder=dict_labels_folders['emanuskript'],
        zone_labels_folder=dict_labels_folders['zone']
    )
    image_batch.load_images()
    image_batch.load_annotations()
    image_batch.unify_names()
    coco_json = image_batch.return_coco_file()
    task.info(f'COCO Json file created')

    # Update task status
    task.update(status="PROGRESS")
    socket = create_socket()

    images = ImageModel.objects(dataset_id=dataset_id)
    categories = CategoryModel.objects

    total_images = images.count()
    task.info(f"Found {total_images} images to process")


    process_coco_file(coco_json,task,socket,dataset,images,categories)







__all__ = ["export_annotations", "import_annotations","predict_annotations","unify_predictions"]