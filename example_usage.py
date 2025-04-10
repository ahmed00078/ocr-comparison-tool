import os
import sys
import argparse
from OCRModelComparator import OCRModelComparator, download_test_data
import matplotlib.pyplot as plt
import pandas as pd

def prepare_environmental_test_data(source_dir, output_dir):
    """
    Prepare test data specifically for environmental documents.
    This function extracts sections from PDFs that contain carbon footprint information.
    
    Args:
        source_dir: Directory containing source environmental PDFs
        output_dir: Directory to save prepared test images
    """
    print(f"Preparing environmental test data from {source_dir}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
    
    # In a real implementation, this would:
    # 1. Convert PDF pages to images
    # 2. Extract regions containing environmental data
    # 3. Generate ground truth by manual annotation
    
    print(f"Environmental test data prepared in {output_dir}")
    return output_dir

def analyze_carbon_content(results_dir):
    """
    Analyze the OCR results specifically for carbon footprint data.
    
    Args:
        results_dir: Directory containing OCR comparison results
    """
    print(f"Analyzing carbon footprint data extraction quality from {results_dir}")
    
    # Load OCR results
    metrics_df = pd.read_csv(os.path.join(results_dir, "ocr_metrics.csv"))
    
    # Analysis specific to environmental/carbon data
    # In a real implementation, this would:
    # 1. Check accuracy of numerical values (CO2e, kg, etc.)
    # 2. Evaluate extraction of units and measurements
    # 3. Assess table structure preservation
    
    # Generate carbon-specific visualization
    plt.figure(figsize=(10, 6))
    
    # Simple example visualization
    models = metrics_df['model'].unique()
    avg_similarity = metrics_df.groupby('model')['similarity'].mean()
    
    plt.bar(models, avg_similarity)
    plt.title('OCR Model Accuracy for Carbon Data Extraction')
    plt.xlabel('OCR Model')
    plt.ylabel('Average Text Similarity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, "carbon_data_extraction_quality.png"))
    print(f"Carbon data analysis saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='Environmental Document OCR Comparison Example')
    parser.add_argument('--source', type=str, help='Directory with environmental PDF documents')
    parser.add_argument('--download-samples', action='store_true', help='Download sample environmental data')
    parser.add_argument('--output', type=str, default='environmental_ocr_results', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Step 1: Prepare test data
    test_data_dir = "environmental_test_data"
    if args.download_samples:
        print("Downloading sample environmental document data...")
        download_test_data(test_data_dir)
    elif args.source:
        test_data_dir = prepare_environmental_test_data(args.source, test_data_dir)
    else:
        print("Using existing test data (if available)")
    
    # Step 2: Run OCR comparison with all available models
    print("Running OCR comparison on environmental documents...")
    comparator = OCRModelComparator(
        models_to_test=['tesseract', 'easyocr', 'kraken', 'trocr', 'paddleocr'],
        ground_truth_dir=os.path.join(test_data_dir, "ground_truth"),
        test_images_dir=os.path.join(test_data_dir, "images")
    )
    
    comparator.run_tests()
    comparator.save_results(args.output)
    comparator.generate_visualizations(args.output)
    
    # Step 3: Perform environmental-specific analysis
    analyze_carbon_content(args.output)
    
    print(f"""
    =================================================================
    OCR COMPARISON FOR ENVIRONMENTAL DATA EXTRACTION COMPLETE
    =================================================================
    
    Results saved to: {args.output}
    
    Next steps:
    1. Review the visualizations to identify the best model for your needs
    2. Check accuracy specifically for numerical environmental data
    3. Use the chosen model in your carbon database automation pipeline
    
    For your internship project on creating a carbon database for ecological
    electronics, we recommend focusing on models that performed best on:
      - Table structure preservation
      - Numerical accuracy
      - Processing speed (for large-scale automation)
    =================================================================
    """)

if __name__ == "__main__":
    main()