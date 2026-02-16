#!/usr/bin/env python3
"""
Test file for OpenCV, ChromaDB, and NumPy packages
"""

import cv2
import numpy as np
import chromadb
from chromadb.utils import embedding_functions


def test_numpy():
    """Test NumPy basic operations"""
    print("=" * 50)
    print("Testing NumPy")
    print("=" * 50)
    
    # Create arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([10, 20, 30, 40, 50])
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    print(f"Sum: {arr1 + arr2}")
    print(f"Mean of arr1: {np.mean(arr1)}")
    print(f"Standard deviation of arr2: {np.std(arr2)}")
    
    # Create a 2D array
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\nMatrix:\n{matrix}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix transpose:\n{matrix.T}")
    
    print("✓ NumPy test passed!\n")


def test_opencv():
    """Test OpenCV image operations"""
    print("=" * 50)
    print("Testing OpenCV")
    print("=" * 50)
    
    # Create a blank image (black)
    height, width = 400, 600
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)
    print(f"Created blank image with shape: {blank_image.shape}")
    
    # Draw some shapes on the image
    # Draw a blue rectangle
    cv2.rectangle(blank_image, (50, 50), (250, 150), (255, 0, 0), 3)
    
    # Draw a green circle
    cv2.circle(blank_image, (400, 200), 75, (0, 255, 0), -1)
    
    # Draw a red line
    cv2.line(blank_image, (100, 250), (500, 350), (0, 0, 255), 5)
    
    # Add text
    cv2.putText(blank_image, 'OpenCV Test', (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the image
    output_path = '/Users/mobasuony/Desktop/Thesis project/test_opencv_output.png'
    cv2.imwrite(output_path, blank_image)
    print(f"Image saved to: {output_path}")
    
    # Read the image back
    loaded_image = cv2.imread(output_path)
    print(f"Loaded image shape: {loaded_image.shape}")
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale image shape: {gray_image.shape}")
    
    print("✓ OpenCV test passed!\n")


def test_chromadb():
    """Test ChromaDB vector database operations"""
    print("=" * 50)
    print("Testing ChromaDB")
    print("=" * 50)
    
    # Create a ChromaDB client (in-memory)
    client = chromadb.Client()
    print("ChromaDB client created (in-memory)")
    
    # Create a collection
    collection = client.create_collection(name="test_collection")
    print(f"Collection '{collection.name}' created")
    
    # Add some documents
    documents = [
        "This is a document about Python programming",
        "Machine learning is a subset of artificial intelligence",
        "OpenCV is a computer vision library",
        "NumPy is used for numerical computing",
        "ChromaDB is a vector database"
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"source": f"source_{i}"} for i in range(len(documents))]
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"Added {len(documents)} documents to the collection")
    print(f"Collection count: {collection.count()}")
    
    # Query the collection
    query_text = "What is computer vision?"
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    
    print(f"\nQuery: '{query_text}'")
    print("Top results:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"  {i+1}. {doc} (distance: {distance:.4f})")
    
    # Get all documents
    all_docs = collection.get()
    print(f"\nTotal documents in collection: {len(all_docs['ids'])}")
    
    print("✓ ChromaDB test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("PACKAGE TESTING SUITE")
    print("="*50 + "\n")
    
    try:
        test_numpy()
        test_opencv()
        test_chromadb()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
