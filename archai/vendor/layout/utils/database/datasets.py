
from flask_login import current_user
from mongoengine import *
from config import Config

from .tasks import TaskModel

import os


class DatasetModel(DynamicDocument):
    
    id = SequenceField(primary_key=True)
    name = StringField(required=True, unique=True)
    directory = StringField()
    thumbnails = StringField()
    categories = ListField(default=[])

    owner = StringField(required=True)
    users = ListField(default=[])

    annotate_url = StringField(default="")

    default_annotation_metadata = DictField(default={})

    deleted = BooleanField(default=False)
    deleted_date = DateTimeField()

    def save(self, *args, **kwargs):

        directory = os.path.join(Config.DATASET_DIRECTORY, self.name + '/')
        os.makedirs(directory, mode=0o777, exist_ok=True)

        self.directory = directory
        self.owner = current_user.username if current_user else 'system'

        return super(DatasetModel, self).save(*args, **kwargs)

    def get_users(self):
        from .users import UserModel
    
        members = self.users
        members.append(self.owner)

        return UserModel.objects(username__in=members)\
            .exclude('password', 'id', 'preferences')

    def import_coco(self, coco_json):

        from workers.tasks import import_annotations

        task = TaskModel(
            name="Import COCO format into {}".format(self.name),
            dataset_id=self.id,
            group="Annotation Import"
        )
        task.save()

        cel_task = import_annotations.delay(task.id, self.id, coco_json)

        return {
            "celery_id": cel_task.id,
            "id": task.id,
            "name": task.name
        }


    def predict_coco(self):

        from workers.tasks import predict_annotations,unify_predictions
        from celery import chord

        # Setup
        #TODO Get images from the image model
        images_path = self.directory

        catmus_labels_folder = os.path.join(images_path, 'labels', 'catmus')
        emanuskript_labels_folder = os.path.join(images_path, 'labels', 'emanuskript')
        zone_detection_labels_folder = os.path.join(images_path, 'labels', 'zone_detection')

        dict_labels_folders = {'catmus':catmus_labels_folder,
                            'emanuskript':emanuskript_labels_folder,
                            'zone':zone_detection_labels_folder}

        for label_path in [dict_labels_folders['catmus'],dict_labels_folders['emanuskript'],dict_labels_folders['zone']]:
            os.makedirs(label_path,exist_ok=True)

        #Predict 
        
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        prediction_tasks = []

        for image_path in image_files:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            image_full_path = os.path.join(images_path, image_path)
            for model in dict_labels_folders.keys():

                task = TaskModel(
                    name=f"Predicting {model} annotations for {image_id}",
                    dataset_id=self.id,
                    group="Annotation Prediction"
                )
                task.save()
                prediction_tasks.append(predict_annotations.s(task.id, model, image_full_path,image_id,dict_labels_folders))

        # List to hold the task details for each image
        
        unify_task = TaskModel(
            name=f"Unifying annotations for dataset {self.name}",
            dataset_id=self.id,
            group="Annotation Prediction"
        )
        unify_task.save()

        # This task will be triggered after all image predictions are completed
        unify_task_signature = unify_predictions.s(unify_task.id, self.id, images_path, dict_labels_folders)

        # Use Celery `chord` to handle the parallel predictions and trigger unification

        chord(prediction_tasks)(unify_task_signature)

        return {
            "unify_task_id": unify_task.id,
            
        }



    def export_coco(self, categories=None, style="COCO", with_empty_images=False):

        from workers.tasks import export_annotations

        if categories is None or len(categories) == 0:
            categories = self.categories

        task = TaskModel(
            name=f"Exporting {self.name} into {style} format",
            dataset_id=self.id,
            group="Annotation Export"
        )
        task.save()

        cel_task = export_annotations.delay(task.id, self.id, categories, with_empty_images)

        return {
            "celery_id": cel_task.id,
            "id": task.id,
            "name": task.name
        }

    def scan(self):

        from workers.tasks import scan_dataset
        
        task = TaskModel(
            name=f"Scanning {self.name} for new images",
            dataset_id=self.id,
            group="Directory Image Scan"
        )
        task.save()
        
        cel_task = scan_dataset.delay(task.id, self.id)

        return {
            "celery_id": cel_task.id,
            "id": task.id,
            "name": task.name
        }

    def is_owner(self, user):

        if user.is_admin:
            return True
        
        return user.username.lower() == self.owner.lower()

    def can_download(self, user):
        return self.is_owner(user)

    def can_delete(self, user):
        return self.is_owner(user)
    
    def can_share(self, user):
        return self.is_owner(user)
    
    def can_generate(self, user):
        return self.is_owner(user)

    def can_edit(self, user):
        return user.username in self.users or self.is_owner(user)
    
    def permissions(self, user):
        return {
            'owner': self.is_owner(user),
            'edit': self.can_edit(user),
            'share': self.can_share(user),
            'generate': self.can_generate(user),
            'delete': self.can_delete(user),
            'download': self.can_download(user)
        }


__all__ = ["DatasetModel"]
