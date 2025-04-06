# Deep-learning_project
Skin Cancer detection using Grad_CAM
🩺 Skin Cancer Detection using MobileNetV2 & Grad-CAM 🔬
📌 Project Overview
This deep learning project focuses on automated skin cancer detection using MobileNetV2, a lightweight and efficient convolutional neural network, combined with Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of model predictions.

Using dermatoscopic images from the HAM10000 and ISIC 2019 datasets, the model classifies various types of skin lesions and highlights the regions influencing its decision, helping build trust and interpretability in AI-assisted medical diagnosis.

🎯 Key Features
📱 MobileNetV2-based model for fast and accurate classification

🧠 Fine-tuning with pretrained weights (Transfer Learning)

🧼 Advanced preprocessing and data augmentation

⚖️ Dataset balancing to handle class imbalance

📊 Performance evaluation: Accuracy, F1 Score, Confusion Matrix

🌈 Grad-CAM heatmap visualizations to show model attention

💻 Built to run smoothly on Google Colab

🧰 Tech Stack
Python

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib / Seaborn

Grad-CAM (custom Keras-based implementation)

Google Colab (recommended environment)

📁 Folder Structure
bash
Copy
Edit
skin-cancer-mobilenetv2/
│
├── data/                        # Dataset directories (HAM10000, ISIC2019)
├── models/                      # Trained MobileNetV2 models
├── gradcam_outputs/            # Saved Grad-CAM heatmaps
├── notebooks/                  
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_mobilenetv2_training.ipynb
│   ├── 3_gradcam_explainer.ipynb
├── utils/
│   ├── gradcam.py               # Grad-CAM helper functions
│   └── helpers.py               # Miscellaneous utilities
├── README.md
└── requirements.txt
📊 Evaluation Metrics
✅ Accuracy

📈 Precision, Recall, F1-score

📉 Confusion Matrix

🔥 Grad-CAM Visuals for explainability

🗃️ Dataset Sources
HAM10000 - Kaggle Link

ISIC 2019 - ISIC Archive

Both datasets include multiple classes of skin lesions such as:

Melanoma (mel)

Melanocytic nevi (nv)

Basal cell carcinoma (bcc)

Benign keratosis (bkl)

Actinic keratoses (akiec)

Dermatofibroma (df)

Vascular lesions (vasc)

🧪 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/skin-cancer-mobilenetv2.git
cd skin-cancer-mobilenetv2
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Upload your dataset
Place images in the /data/ directory or load from Google Drive in Colab.

Run the notebooks

1_data_preprocessing.ipynb – Load, clean, and balance the data

2_mobilenetv2_training.ipynb – Train & evaluate the MobileNetV2 model

3_gradcam_explainer.ipynb – Visualize predictions using Grad-CAM



