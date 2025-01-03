# Whale & Dolphin Species Classifier

This project is a machine learning-based classifier designed to identify whale and dolphin species from dorsal fin images. It leverages state-of-the-art deep learning architectures and integrates generative AI (Gemini) to enhance the user experience.
### Demo: https://huggingface.co/spaces/Galii/Whale_Dolphin
![屏幕截图_29-12-2024_21152_huggingface co](https://github.com/user-attachments/assets/ee341d27-a9e7-45bd-b79c-cb4abbba767f)
---

## **Overview**

The Whale & Dolphin Species Classifier is a computer vision project that uses transfer learning to classify dorsal fin images into various whale and dolphin species. The project achieves high accuracy and provides real-time predictions through an interactive web interface.

### **Key Features**
- **High Accuracy**: Achieved 92.12% classification accuracy using the EfficientNetB0 architecture.
- **Generative AI Integration**: Automatically generates structured information about species, including scientific classifications, behavior patterns, and conservation status, using Google’s Gemini API.
- **Interactive Web Interface**: Deployed on Hugging Face with Gradio, allowing users to upload dorsal fin images for real-time classification results.

---

## **Dataset**

The dataset used in this project was sourced from Kaggle and contains over 15,000 labeled images of dorsal fins from 30 different species. 
Data source: https://www.kaggle.com/datasets/andrewgustyjrstudent/happywhaleimagessortedbyspecies/data

### **Data Preprocessing**
- **Error Correction**: Corrected spelling mistakes in species labels (e.g., "kiler_whale" → "killer_whale").
- **Category Merging**: Standardized species names by merging synonymous labels.
- **Class Filtering**: Removed species categories with fewer than 100 images to avoid overfitting and underperformance.
- **Augmentation**: Applied real-time image augmentations (e.g., rotation, scaling) using TensorFlow's `ImageDataGenerator` to improve generalization.

### **Dataset Splitting**
- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

---

## **Model Architectures**

Two pre-trained convolutional neural networks (CNNs) were employed for transfer learning:

### **1. EfficientNetB0**
- A state-of-the-art architecture known for its balance between efficiency and accuracy.
- Scalable design that optimally adjusts depth, width, and resolution.
- Achieved a test accuracy of 92.12%.

### **2. VGG16**
- A deeper architecture with 16 layers, widely used for feature extraction in image classification tasks.
- Achieved a test accuracy of 87.79%.

#### **Comparison**
| Metric                | EfficientNetB0 | VGG16    |
|-----------------------|----------------|----------|
| Test Accuracy         | 92.12%        | 87.79%   |
| Training Time         | Faster         | Slower   |
| Generalization        | Stronger       | Moderate |
| Confusion Matrix      | More focused   | More spread out |

EfficientNetB0 outperformed VGG16 in terms of accuracy, training efficiency, and generalization capability.

---

## **Methodology**

### **1. Transfer Learning**
Both models were fine-tuned as follows:
- The convolutional base was frozen to retain pre-trained weights.
- Custom classification layers were added for the specific task of identifying whale and dolphin species.
- Fine-tuned only the added layers to adapt the models to the dataset.

### **2. Model Compilation**
The models were compiled with:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy (for multi-class classification)
- Metrics: Accuracy

### **3. Training**
- Used callbacks like `ModelCheckpoint` to save the best-performing model during training.
- Hyperparameters were tuned based on validation set performance.

---

## **Results**

### **Performance Metrics**
EfficientNetB0 demonstrated superior performance:
- Test Accuracy: 92.12%
- Precision, Recall, F1-score: Higher across all classes compared to VGG16.

### **Insights**
- EfficientNetB0 showed better classification of underrepresented species due to its advanced feature extraction capabilities.
- Both models occasionally confused visually similar species, highlighting challenges in fine-grained classification.

---

## **Deployment**

The trained model was deployed as an interactive web application using Gradio on Hugging Face:
1. Users can upload dorsal fin images.
2. The model provides real-time classification results along with structured species information generated by Google’s Gemini API.

---

## **Future Work**

To further improve the classifier:
1. Increase dataset size for underrepresented species.
2. Experiment with other architectures like EfficientNetV2 or Vision Transformers.
3. Enhance the web interface for better user experience.

---

## **Acknowledgments**

This project utilized datasets from Kaggle and leveraged TensorFlow/Keras for model development. Special thanks to Google’s Gemini API for enabling generative AI capabilities.

