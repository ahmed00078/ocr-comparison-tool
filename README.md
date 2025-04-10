# OCR Model Comparison Tool for Environmental Data Extraction

## Overview

This tool helps you systematically evaluate and compare open-source OCR (Optical Character Recognition) models to find the optimal solution for extracting environmental data from electronic product documents. This directly supports your internship project *"Création d'uné basé dé donnéés carboné pour uné éléctroniqué plus écologiqué"* at INSA Rennes.

## Features

- 🔎 Tests 5 popular open-source OCR engines:
  - Tesseract OCR
  - EasyOCR
  - Kraken OCR
  - TrOCR (Transformer-based OCR)
  - PaddleOCR

- 📊 Comprehensive metrics:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Text similarity ratio
  - Processing time

- 📈 Detailed visualizations:
  - Comparison charts
  - Performance distributions
  - Metric heatmaps
  - Radar charts for overall performance

- 🧪 Specialized for environmental data:
  - Optimized for carbon footprint documents
  - Focused on accurate extraction of numerical values
  - Tailored for technical specifications from manufacturers

## Project Structure

```
ocr-comparison-tool/
├── ocr_comparator.py         # Main comparison tool
├── example_usage.py          # Example specific to environmental data
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── test_data/                # Test data directory
│   ├── images/               # Test document images
│   └── ground_truth/         # Ground truth text files
└── results/                  # Output directory
    ├── ocr_comparison.png    # Visualization of results
    ├── ocr_metrics.csv       # Detailed metrics in CSV format
    ├── ocr_results.json      # Complete results in JSON format
    └── extracted_texts/      # Extracted text from each model
```

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ocr-comparison-tool.git
   cd ocr-comparison-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install OCR engines (see `requirements.md` for detailed instructions)

### Running the Tool

Basic usage:
```bash
python ocr_comparator.py --images test_data/images --ground-truth test_data/ground_truth --output results
```

For environmental document analysis:
```bash
python example_usage.py --source your_environmental_pdfs/ --output environmental_results
```

## How It Works

1. **Document Preparation**: 
   - The tool takes environmental document images as input
   - You provide ground truth text for comparison

2. **OCR Processing**:
   - Each model processes the same set of images
   - Text extraction is performed with default parameters
   - Processing time is measured for each model

3. **Metrics Calculation**:
   - Error rates and similarity scores are calculated
   - Special attention to environmental data extraction accuracy

4. **Result Analysis**:
   - Results are saved in multiple formats (JSON, CSV)
   - Visualizations show comparative performance
   - Model rankings help you choose the best OCR for your needs

## Adapting for Your Carbon Database Project

This tool is specifically designed to support your internship project on creating a carbon database for more ecological electronics:

1. **Use real product documentation**:
   - Add manufacturer datasheets as test images
   - Create ground truth files with key carbon metrics

2. **Focus on critical data**:
   - Carbon footprint values (kg CO2e)
   - Energy consumption metrics
   - Material composition percentages
   - Lifecycle analysis figures

3. **Integration with your project**:
   - Use the best model(s) in your data extraction pipeline
   - Apply specialized post-processing for environmental units
   - Build validation rules for extracted carbon metrics

## Contributing

Contributions to improve the tool are welcome! Please consider:

1. Adding support for more OCR engines
2. Implementing specialized metrics for environmental data
3. Enhancing preprocessing for technical documents
4. Developing post-processing optimized for carbon metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool was developed to support the ESOS (Electronique Soutenable Ouverte et Souveraine) research group at INSA Rennes
- References in the internship document guided the approach to environmental data extraction