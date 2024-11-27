# Visual Question Answering (VQA) System for Real-World Image Understanding

## Project Overview
This project implements a Visual Question Answering (VQA) system using multimodal transformers in PyTorch to enable real-world image understanding. The system combines image and text data for feature extraction, fusion, and prediction, leveraging advanced transformer architectures to achieve robust results. Two approaches—classification and generation—are explored using the DAQUAR dataset.

---

## Contents

1. **Abstract**
    - Overview of the project goals, methods, and key findings.
2. **Introduction**
    - Background, objectives, and scope of the project.
3. **Methodology**
    - Feature extraction techniques, multimodal fusion, and implementation tools.
4. **Datasets**
    - Description of the DAQUAR dataset used in this project.
5. **Assessment Methodology**
    - Metrics and evaluation techniques.
6. **Literature Review**
    - Thematic and comparative analyses of existing approaches.
7. **Critical Analysis**
    - Gaps, limitations, and implications of the study.
8. **Conclusion**
    - Summary of findings and future directions.
9. **References**
    - Cited sources and resources.

---

## Key Features

### Models and Techniques
- **Image Feature Extraction**: Vision Transformers (ViT) for tokenizing images into spatial representations.
- **Text Feature Extraction**: BERT for encoding natural language questions.
- **Multimodal Fusion**: Late fusion, bilinear pooling, and attention mechanisms to integrate visual and textual data.
- **Generation Model**: Combining BERT, ViT, and GPT2 for sequence generation tasks.

### Regularization Techniques
- Dropout layers to mitigate overfitting.
- Gradient clipping to stabilize backpropagation.

### Tools and Frameworks
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NLTK

---

## Dataset
**DAQUAR (DAtaset for QUestion Answering on Real-world images):**
- **Size**: 12,500 question-answer pairs.
- **Focus**: Indoor scenes and basic object recognition.
- **Applications**: Ideal for single-word/phrase-answer modeling.

---

## Evaluation Metrics
- **Accuracy**: Measures correctness of predictions.
- **Macro F1 Score**: Evaluates model balance across classes.
- **Wu and Palmer Similarity (WUPS)**: Captures semantic similarity between predicted answers and ground truths.

### Ablation Studies
- **Input Dimensions**: Effect of image patch and token embedding sizes.
- **Pre-processing**: Analysis of normalization, resizing, and tokenization methods.
- **Fusion Mechanisms**: Comparing concatenation and bilinear pooling.
- **Attention Mechanisms**: Evaluating different attention models.

---

## Findings
- **Classification Model**: BERT + ViT achieved a WUPS score of 0.26.
- **Generation Model**: BERT + ViT + GPT2 achieved superior performance with a WUPS score of 0.27.
- **Challenges**: Limited dataset size and high computational requirements.
- **Future Directions**:
  - Transfer learning for diverse datasets.
  - Integration of external knowledge graphs.
  - Optimization for computational efficiency.

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RobuRishabh/Multimodal-Visual-Question-Answering-VQA-with-Generative-AI-utilizing-LLM-and-Vision-Language-Model.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the DAQUAR dataset and place it in the `data/` folder.

### Running the Project
1. **Training the Classification Model**:
   ```bash
   python VQA_Classification.ipynb
   ```
2. **Training the Generation Model**:
   ```bash
   python VQA_Generation.ipynb
   ```

---

## References
- [Paperspace Blog: Vision Transformers](https://blog.paperspace.com/vision-transformers/)
- [Hugging Face Documentation](https://huggingface.co/docs/transformers/v4.15.0/en/index)
- [DAQUAR Dataset](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset)

---

## Author
**Rishabh Singh**
- Course: CS 6120 (Natural Language Processing)
- Instructor: Prof. Uzair Ahmad
