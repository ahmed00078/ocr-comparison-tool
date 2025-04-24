
import os
import argparse
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# Import the OCRModelComparator class
from ocr_comparator import OCRModelComparator

def test_single_model(model_name, image_path, ground_truth_path=None):
    """
    Test a single OCR model on a specific image and display the results.
    
    Args:
        model_name (str): Name of the OCR model to test
        image_path (str): Path to the image file
        ground_truth_path (str, optional): Path to the ground truth text file
    """
    print(f"\n{'='*80}")
    print(f"TESTING MODEL: {model_name}")
    print(f"IMAGE: {image_path}")
    print(f"{'='*80}\n")
    
    # Initialize comparator with only the specified model
    comparator = OCRModelComparator(
        models_to_test=[model_name],
        ground_truth_dir=os.path.dirname(ground_truth_path) if ground_truth_path else None,
        test_images_dir=os.path.dirname(image_path)
    )
    
    # Get the OCR function for the model
    ocr_func = comparator.available_models.get(model_name)
    if not ocr_func:
        print(f"Model {model_name} not available or not properly initialized")
        return
    
    # Read ground truth if available
    ground_truth = ""
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        print(f"\nGROUND TRUTH TEXT:")
        print(f"{'-'*40}")
        print(ground_truth)
        print(f"{'-'*40}\n")
    
    # Process the image with the model
    try:
        print(f"Processing image with {model_name}...")
        start_time = datetime.now()
        result = ocr_func(image_path)
        end_time = datetime.now()
        
        # Display results
        print(f"\nEXTRACTED TEXT:")
        print(f"{'-'*40}")
        print(result['text'])
        print(f"{'-'*40}\n")
        
        processing_time = result['processing_time']
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Calculate metrics if ground truth is available
        if ground_truth:
            metrics = comparator._calculate_metrics(result['text'], ground_truth)
            print("\nMETRICS:")
            print(f"{'-'*40}")
            print(f"Character Error Rate (CER): {metrics['character_error_rate']:.4f} (lower is better)")
            print(f"Word Error Rate (WER): {metrics['word_error_rate']:.4f} (lower is better)")
            print(f"Similarity Ratio: {metrics['similarity_ratio']:.4f} (higher is better)")
            print(f"{'-'*40}")
        
        # Save the results to a text file
        output_dir = "individual_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, f"{base_name}_{model_name}_result.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n\n")
            f.write("EXTRACTED TEXT:\n")
            f.write(f"{'-'*40}\n")
            f.write(result['text'])
            f.write(f"\n{'-'*40}\n\n")
            
            if ground_truth:
                f.write("GROUND TRUTH TEXT:\n")
                f.write(f"{'-'*40}\n")
                f.write(ground_truth)
                f.write(f"\n{'-'*40}\n\n")
                
                f.write("METRICS:\n")
                f.write(f"{'-'*40}\n")
                f.write(f"Character Error Rate (CER): {metrics['character_error_rate']:.4f}\n")
                f.write(f"Word Error Rate (WER): {metrics['word_error_rate']:.4f}\n")
                f.write(f"Similarity Ratio: {metrics['similarity_ratio']:.4f}\n")
                f.write(f"{'-'*40}\n")
        
        print(f"\nResults saved to {os.path.join(output_dir, f'{base_name}_{model_name}_result.txt')}")
        
        # Optional: Display the image
        try:
            print("\nDisplaying the image (close window to continue)...")
            img = Image.open(image_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Test Image: {os.path.basename(image_path)}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")
        
        return {
            "model": model_name,
            "image": image_path,
            "text": result['text'],
            "processing_time": processing_time,
            "metrics": metrics if ground_truth else None
        }
    
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        return None

def test_all_models(image_path, ground_truth_path=None):
    """
    Test all available OCR models on a single image and compare their results.
    
    Args:
        image_path (str): Path to the image file
        ground_truth_path (str, optional): Path to the ground truth text file
    """
    # Initialize a full comparator to get all available models
    comparator = OCRModelComparator()
    available_models = list(comparator.available_models.keys())
    
    print(f"\n{'='*80}")
    print(f"TESTING ALL AVAILABLE MODELS: {', '.join(available_models)}")
    print(f"IMAGE: {image_path}")
    print(f"{'='*80}\n")
    
    # Test each model and collect results
    results = {}
    for model_name in available_models:
        try:
            print(f"\nTesting {model_name}...")
            result = test_single_model(model_name, image_path, ground_truth_path)
            if result:
                results[model_name] = result
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Save comparative results
    output_dir = "individual_test_results"
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(output_dir, f"{base_name}_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparative results saved to {os.path.join(output_dir, f'{base_name}_comparison.json')}")

def main():
    parser = argparse.ArgumentParser(description='Test individual OCR models')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, help='Name of the OCR model to test (omit to test all models)')
    parser.add_argument('--ground-truth', type=str, help='Path to the ground truth text file')
    
    args = parser.parse_args()
    
    if args.model:
        test_single_model(args.model, args.image, args.ground_truth)
    else:
        test_all_models(args.image, args.ground_truth)

if __name__ == "__main__":
    main()