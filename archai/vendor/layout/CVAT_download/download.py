from cvat_sdk import make_client
from cvat_sdk.core.client import Config
import os
from pathlib import Path
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HOST = "http://134.76.21.30:8080"
USERNAME = "XXXXXX"
PASSWORD = "XXXXXXX"
PROJECT_ID = 7

# Base output directory
OUTPUT_ROOT = Path(f"cvat_project_{PROJECT_ID}_export")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Connect to CVAT
    with make_client(HOST, credentials=(USERNAME, PASSWORD)) as client:
        # Disable SSL verification - CVAT returns HTTPS URLs for downloads even when connecting via HTTP
        client.config.verify_ssl = False
        # Optional: if you use organizations, set it here:
        # client.config.org_slug = "eManusKript"

        project = client.projects.retrieve(PROJECT_ID)
        print(f"Project: {project.name} (ID={project.id})")

        # Get all tasks belonging to this project
        tasks = project.get_tasks()
        print(f"Found {len(tasks)} tasks in project {PROJECT_ID}")

        for t in tasks:
            task_id = t.id
            task_name = t.name
            task_name_sanitized = "".join(c if c.isalnum() or c in "-_ " else "_" for c in task_name)
            task_dir = OUTPUT_ROOT / f"task_{task_id}_{task_name_sanitized}"
            task_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Task {task_id}: {task_name} ===")

            # Retrieve the full Task proxy object (not just TaskRead model)
            task = client.tasks.retrieve(task_id)

            # 1) Download images with original filenames
            images_dir = task_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            from PIL import Image
            from io import BytesIO
            
            # Get frames info
            frames_info = task.get_frames_info()
            if not frames_info:
                print(f"  No frames found in task {task_id}")
            else:
                # Check if images already downloaded
                existing_images = list(images_dir.glob("*"))
                if len(existing_images) == len(frames_info):
                    print(f"  Images already exist in {images_dir} ({len(frames_info)} images)")
                else:
                    print(f"  Downloading {len(frames_info)} images to {images_dir} ...")
                    
                    for idx, frame_info in enumerate(frames_info):
                        frame_id = idx  # Frame IDs are 0-indexed
                        # frame_info is a dict with 'name', 'height', 'width', etc.
                        original_name = frame_info.get('name', f'frame_{frame_id:06d}.jpg')
                        # Ensure we have an extension
                        if '.' not in original_name:
                            original_name += '.jpg'
                        
                        output_path = images_dir / original_name
                        if output_path.exists():
                            continue
                        
                        try:
                            frame_bytes = task.get_frame(frame_id, quality="original")
                            # get_frame returns a response object, read it
                            img_data = frame_bytes.read()
                            img = Image.open(BytesIO(img_data))
                            img.save(output_path)
                            if (idx + 1) % 10 == 0 or (idx + 1) == len(frames_info):
                                print(f"    Downloaded {idx + 1}/{len(frames_info)} images...")
                        except Exception as e:
                            print(f"    Error downloading frame {frame_id} ({original_name}): {e}")

            # 2) Export annotations in COCO 1.0 (without images since we download them separately)
            anno_zip = task_dir / f"task_{task_id}_coco1.0.zip"
            if not anno_zip.exists():
                print(f"  Exporting COCO 1.0 annotations to {anno_zip} ...")
                # Replace pool manager BEFORE export_dataset call to handle HTTPS downloads
                import ssl
                from urllib3.poolmanager import PoolManager
                from cvat_sdk.core.downloading import Downloader
                
                old_pool = client.api_client.rest_client.pool_manager
                # Replace pool manager to disable SSL verification for HTTPS downloads
                client.api_client.rest_client.pool_manager = PoolManager(
                    cert_reqs=ssl.CERT_NONE
                )
                try:
                    # Use the downloader directly to have more control
                    downloader = Downloader(client)
                    
                    # Prepare the export using the same endpoint as export_dataset
                    print(f"    Preparing export...")
                    export_request = downloader.prepare_file(
                        task.api.create_dataset_export_endpoint,
                        url_params={"id": task_id},
                        query_params={
                            "format": "COCO 1.0",
                            "save_images": "false"
                        }
                    )
                    
                    if not export_request.result_url:
                        raise Exception("Export completed but no result URL returned")
                    
                    # Convert HTTPS URL to HTTP if needed
                    result_url = export_request.result_url
                    if result_url.startswith("https://"):
                        result_url = result_url.replace("https://", "http://", 1)
                        print(f"    Converted HTTPS URL to HTTP: {result_url[:80]}...")
                    
                    # Download the file
                    print(f"    Downloading from result URL...")
                    downloader.download_file(result_url, output_path=Path(anno_zip))
                    print(f"    Successfully downloaded annotations")
                    
                except Exception as e:
                    print(f"  Error exporting annotations: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try with images included as fallback
                    print(f"  Retrying with images included...")
                    try:
                        export_request = downloader.prepare_file(
                            task.api.create_dataset_export_endpoint,
                            url_params={"id": task_id},
                            query_params={
                                "format": "COCO 1.0",
                                "save_images": "true"
                            }
                        )
                        result_url = export_request.result_url
                        if result_url and result_url.startswith("https://"):
                            result_url = result_url.replace("https://", "http://", 1)
                        downloader.download_file(result_url, output_path=Path(anno_zip))
                        print(f"    Successfully downloaded annotations with images")
                    except Exception as e2:
                        print(f"  Failed again: {e2}")
                        raise
                finally:
                    # Restore original pool manager after export completes
                    client.api_client.rest_client.pool_manager = old_pool
            else:
                print(f"  Annotations already exist: {anno_zip}")

    print(f"\nDone. All data saved under: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()