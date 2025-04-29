import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import easyocr
from difflib import SequenceMatcher
import Levenshtein
import requests
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import seaborn as sns
from tqdm import tqdm
import argparse

class OCRModelComparator:
    def __init__(self, models_to_test=None, ground_truth_dir=None, test_images_dir=None):
        """
        Initialize the OCR model comparator.
        
        Args:
            models_to_test (list): List of OCR models to test
            ground_truth_dir (str): Directory containing ground truth text files
            test_images_dir (str): Directory containing test images
        """
        self.available_models = {
            'tesseract': self._tesseract_ocr,
            'easyocr': self._easyocr_ocr,
            'trocr': self._trocr_ocr,
            'paddleocr': self._paddleocr_ocr
        }
        
        # Default to all models if none specified
        self.models_to_test = models_to_test if models_to_test else list(self.available_models.keys())
        self.ground_truth_dir = ground_truth_dir
        self.test_images_dir = test_images_dir
        self.results = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all required OCR models"""
        self.initialized_models = {}
        
        for model_name in self.models_to_test:
            if model_name == 'easyocr':
                try:
                    print(f"Initializing {model_name}...")
                    self.initialized_models[model_name] = easyocr.Reader(['en'])
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
                    self.models_to_test.remove(model_name)
            
            elif model_name == 'trocr':
                try:
                    print(f"Initializing {model_name}...")
                    self.initialized_models[model_name] = {
                        'processor': TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed'),
                        'model': VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
                    }
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
                    self.models_to_test.remove(model_name)
            
            elif model_name == 'paddleocr':
                try:
                    from paddleocr import PaddleOCR
                    print(f"Initializing {model_name}...")
                    self.initialized_models[model_name] = PaddleOCR(use_angle_cls=True, lang='en')
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
                    if model_name in self.models_to_test:
                        self.models_to_test.remove(model_name)
        
        print(f"Successfully initialized models: {list(self.initialized_models.keys())}")
    
    def _tesseract_ocr(self, image_path):
        """Run Tesseract OCR on an image"""
        try:
            start_time = time.time()
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            end_time = time.time()
            
            return {
                'text': text,
                'processing_time': end_time - start_time
            }
        except Exception as e:
            print(f"Error with Tesseract OCR: {e}")
            return {'text': '', 'processing_time': 0}
    
    def _easyocr_ocr(self, image_path):
        """Run EasyOCR on an image"""
        try:
            start_time = time.time()
            reader = self.initialized_models.get('easyocr')
            if not reader:
                reader = easyocr.Reader(['en'])
                
            result = reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            end_time = time.time()
            
            return {
                'text': text,
                'processing_time': end_time - start_time
            }
        except Exception as e:
            print(f"Error with EasyOCR: {e}")
            return {'text': '', 'processing_time': 0}
    
    def _trocr_ocr(self, image_path):
        """Run TrOCR on an image"""
        try:
            start_time = time.time()
            
            processor = self.initialized_models.get('trocr', {}).get('processor')
            model = self.initialized_models.get('trocr', {}).get('model')
            
            if not processor or not model:
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            end_time = time.time()
            
            return {
                'text': text,
                'processing_time': end_time - start_time
            }
        except Exception as e:
            print(f"Error with TrOCR: {e}")
            return {'text': '', 'processing_time': 0}
    
    def _paddleocr_ocr(self, image_path):
        """Run PaddleOCR on an image"""
        try:
            start_time = time.time()
            
            ocr = self.initialized_models.get('paddleocr')
            if not ocr:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
            
            result = ocr.ocr(image_path, cls=True)
            
            # Extract text from result
            text = ""
            for line in result:
                for word in line:
                    text += word[1][0] + " "
            
            end_time = time.time()
            
            return {
                'text': text,
                'processing_time': end_time - start_time
            }
        except Exception as e:
            print(f"Error with PaddleOCR: {e}")
            return {'text': '', 'processing_time': 0}
    
    def _calculate_metrics(self, extracted_text, ground_truth):
        """Calculate various metrics to evaluate OCR performance"""
        # Clean up texts
        extracted_text = extracted_text.strip().lower()
        ground_truth = ground_truth.strip().lower()

        print(f"Extracted text: {extracted_text}")
        print(f"Ground truth: {ground_truth}")
        
        # Character Error Rate (CER)
        edit_distance = Levenshtein.distance(extracted_text, ground_truth)
        cer = edit_distance / max(len(ground_truth), 1)
        
        # Word Error Rate (WER)
        extracted_words = extracted_text.split()
        ground_truth_words = ground_truth.split()
        
        wer_distance = Levenshtein.distance(" ".join(extracted_words), " ".join(ground_truth_words))
        wer = wer_distance / max(len(ground_truth_words), 1)
        
        # Similarity ratio
        similarity = SequenceMatcher(None, extracted_text, ground_truth).ratio()
        
        return {
            'character_error_rate': cer,
            'word_error_rate': wer,
            'similarity_ratio': similarity
        }
    
    def run_tests(self):
        """Run OCR tests on all images with all models"""
        if not self.test_images_dir:
            raise ValueError("Test images directory not specified")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        # Dictionary to store results
        results = {model: {'texts': {}, 'metrics': {}, 'processing_times': {}} 
                  for model in self.models_to_test}
        
        # Process each image with each model
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(self.test_images_dir, image_file)
            base_name = os.path.splitext(image_file)[0]
            
            # Get ground truth if available
            ground_truth = ""
            if self.ground_truth_dir:
                gt_path = os.path.join(self.ground_truth_dir, f"{base_name}.txt")
                if os.path.exists(gt_path):
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        ground_truth = f.read()
            print(f"Ground truth for {image_file}: {ground_truth}")
            
            # Process with each model
            for model_name in self.models_to_test:
                ocr_func = self.available_models.get(model_name)
                if not ocr_func:
                    print(f"Model {model_name} not available")
                    continue
                
                try:
                    print(f"Processing {image_file} with {model_name}...")
                    result = ocr_func(image_path)
                    extracted_text = result['text']
                    processing_time = result['processing_time']
                    
                    # Store results
                    results[model_name]['texts'][image_file] = extracted_text
                    results[model_name]['processing_times'][image_file] = processing_time
                    
                    # Calculate metrics if ground truth is available
                    if ground_truth:
                        metrics = self._calculate_metrics(extracted_text, ground_truth)
                        results[model_name]['metrics'][image_file] = metrics
                
                except Exception as e:
                    print(f"Error processing {image_file} with {model_name}: {e}")
        
        self.results = results
        return results
    
    def save_results(self, output_dir='results'):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        with open(os.path.join(output_dir, 'ocr_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV with metrics
        csv_data = []
        for model_name, model_results in self.results.items():
            for image_file, metrics in model_results['metrics'].items():
                row = {
                    'model': model_name,
                    'image': image_file,
                    'processing_time': model_results['processing_times'][image_file],
                    'cer': metrics['character_error_rate'],
                    'wer': metrics['word_error_rate'],
                    'similarity': metrics['similarity_ratio']
                }
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(output_dir, 'ocr_metrics.csv'), index=False)
        
        # Save extracted texts
        texts_dir = os.path.join(output_dir, 'extracted_texts')
        os.makedirs(texts_dir, exist_ok=True)
        
        for model_name, model_results in self.results.items():
            model_dir = os.path.join(texts_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            for image_file, text in model_results['texts'].items():
                base_name = os.path.splitext(image_file)[0]
                with open(os.path.join(model_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)
        
        print(f"Results saved to {output_dir}")
    
    def generate_visualizations(self, output_dir='results'):
        """Generate visualizations of the results"""
        if not self.results:
            print("No results to visualize")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        model_names = list(self.results.keys())
        metrics_data = {
            'cer': [],
            'wer': [],
            'similarity': [],
            'processing_time': []
        }
        
        for model_name in model_names:
            # Average metrics across all images
            avg_cer = np.mean([metrics['character_error_rate'] 
                              for metrics in self.results[model_name]['metrics'].values()])
            avg_wer = np.mean([metrics['word_error_rate'] 
                              for metrics in self.results[model_name]['metrics'].values()])
            avg_similarity = np.mean([metrics['similarity_ratio'] 
                                    for metrics in self.results[model_name]['metrics'].values()])
            avg_time = np.mean(list(self.results[model_name]['processing_times'].values()))
            
            metrics_data['cer'].append(avg_cer)
            metrics_data['wer'].append(avg_wer)
            metrics_data['similarity'].append(avg_similarity)
            metrics_data['processing_time'].append(avg_time)
        
        # Create visualizations
        plt.figure(figsize=(12, 10))
        
        # 1. Error rates bar chart
        plt.subplot(2, 2, 1)
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, metrics_data['cer'], width, label='CER')
        plt.bar(x + width/2, metrics_data['wer'], width, label='WER')
        plt.xlabel('OCR Models')
        plt.ylabel('Error Rate (lower is better)')
        plt.title('Character and Word Error Rates')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. Similarity ratio bar chart
        plt.subplot(2, 2, 2)
        plt.bar(model_names, metrics_data['similarity'], color='green')
        plt.xlabel('OCR Models')
        plt.ylabel('Similarity Ratio (higher is better)')
        plt.title('Text Similarity Ratio')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Processing time bar chart
        plt.subplot(2, 2, 3)
        plt.bar(model_names, metrics_data['processing_time'], color='orange')
        plt.xlabel('OCR Models')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Average Processing Time')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Radar chart for overall comparison
        plt.subplot(2, 2, 4, polar=True)
        categories = ['Accuracy (1-CER)', 'Speed (1/Time)', 'Similarity']
        
        # Normalize metrics for radar chart
        max_time = max(metrics_data['processing_time'])
        normalized_metrics = []
        for i, model in enumerate(model_names):
            accuracy = 1 - metrics_data['cer'][i]  # Convert error to accuracy
            speed = max_time / max(metrics_data['processing_time'][i], 0.001)  # Convert time to speed (inverted)
            similarity = metrics_data['similarity'][i]
            normalized_metrics.append([accuracy, speed, similarity])
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.gca()
        for i, metrics in enumerate(normalized_metrics):
            metrics = metrics + metrics[:1]  # Close the loop
            ax.plot(angles, metrics, linewidth=2, label=model_names[i])
            ax.fill(angles, metrics, alpha=0.1)
        
        plt.xticks(angles[:-1], categories)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Overall OCR Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ocr_comparison.png'), dpi=300)
        
        # Generate additional detailed visualizations
        self._generate_detailed_visualizations(output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _generate_detailed_visualizations(self, output_dir):
        """Generate more detailed visualizations"""
        # Create a DataFrame with all metrics for all images
        data = []
        for model_name, model_results in self.results.items():
            for image_file in model_results['metrics']:
                data.append({
                    'Model': model_name,
                    'Image': image_file,
                    'CER': model_results['metrics'][image_file]['character_error_rate'],
                    'WER': model_results['metrics'][image_file]['word_error_rate'],
                    'Similarity': model_results['metrics'][image_file]['similarity_ratio'],
                    'Processing Time': model_results['processing_times'][image_file]
                })
        
        df = pd.DataFrame(data)
        
        # 1. Boxplots of metrics distributions
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(x='Model', y='CER', data=df)
        plt.title('Character Error Rate Distribution')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Model', y='WER', data=df)
        plt.title('Word Error Rate Distribution')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Model', y='Similarity', data=df)
        plt.title('Similarity Distribution')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 4)
        sns.boxplot(x='Model', y='Processing Time', data=df)
        plt.title('Processing Time Distribution')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ocr_distributions.png'), dpi=300)
        
        # 2. Heatmap of average metrics by model
        plt.figure(figsize=(12, 8))
        
        # Define the columns you actually want to average
        numeric_cols = ['CER', 'WER', 'Similarity', 'Processing Time']

        # Group by 'Model', select only the numeric columns, then calculate the mean
        heatmap_data = df.groupby('Model')[numeric_cols].mean()
        
        # Normalize for better visualization
        normalized_data = heatmap_data.copy()
        for col in normalized_data.columns:
            if col == 'Similarity':  # Higher is better
                normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                      (normalized_data[col].max() - normalized_data[col].min())
            else:  # Lower is better
                normalized_data[col] = 1 - (normalized_data[col] - normalized_data[col].min()) / \
                                      (normalized_data[col].max() - normalized_data[col].min())
        
        # Create heatmap
        sns.heatmap(normalized_data, annot=heatmap_data, fmt=".3f", cmap="YlGnBu", linewidths=.5)
        plt.title('OCR Models Performance Heatmap')
        plt.savefig(os.path.join(output_dir, 'ocr_heatmap.png'), dpi=300)

def download_test_data(output_dir='test_data'):
    """Download sample test data if needed"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ground_truth'), exist_ok=True)
    
    # Example URLs for test images (replace with actual URLs)
    test_images = [
        ('sample1.png', 'https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/test/eurotext.png'),
        ('sample2.png', 'https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/test/phototest.png')
    ]
    
    # Download test images
    for filename, url in test_images:
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(os.path.join(output_dir, 'images', filename), 'wb') as f:
                f.write(response.content)
            
            # Create a simple ground truth for testing
            with open(os.path.join(output_dir, 'ground_truth', f"{os.path.splitext(filename)[0]}.txt"), 'w') as f:
                f.write("This is a sample ground truth text for testing purposes.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    print(f"Test data downloaded to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='OCR Model Comparison Tool')
    parser.add_argument('--models', type=str, nargs='+', choices=['tesseract', 'easyocr', 'trocr', 'paddleocr'],
                        default=['tesseract', 'easyocr'], help='OCR models to test')
    parser.add_argument('--images', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--ground-truth', type=str, help='Directory containing ground truth text files')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--download-samples', action='store_true', help='Download sample test data')
    
    args = parser.parse_args()
    
    if args.download_samples:
        download_test_data()
        return
    
    # Run OCR comparison
    comparator = OCRModelComparator(
        models_to_test=args.models,
        ground_truth_dir=args.ground_truth,
        test_images_dir=args.images
    )
    
    comparator.run_tests()
    comparator.save_results(args.output)
    comparator.generate_visualizations(args.output)

if __name__ == "__main__":
    main()