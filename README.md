🩺 BioFusion-AI: A Multi-Modal Breast Cancer Detection System

BioFusion-AI is a hybrid AI system designed to enhance the accuracy and reliability of breast cancer diagnosis by fusing multiple data modalities — mammograms, ultrasound images, and synthetic structured clinical history. This project aims to bridge critical gaps in existing diagnostic tools by leveraging both visual and non-visual data in a unified pipeline.

🔬 Motivation

While mammography and ultrasound are widely used for breast cancer screening, imaging alone can be insufficient for definitive diagnosis, especially in complex or subtle cases. To address this, BioFusion-AI incorporates structured clinical features such as breast density, family history, hormone therapy, and other historical risk factors to make more informed predictions. Our goal was to build a smart, data-driven tool that could assist radiologists in making faster, more accurate decisions — particularly in resource-limited settings.

🧠 Core Components

Image Feature Extraction: Fine-tuned MobileNetV2 CNNs were used to extract features from mammogram and ultrasound images.

Synthetic Clinical History: A structured dataset was generated to simulate real-world patient histories, including categorical and numerical medical variables.

Feature Fusion: All three modalities (mammogram, ultrasound, and clinical history) were processed using PCA and scaling before being merged into a single feature vector.

Classifier Ensemble: Multiple ML classifiers (Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting, etc.) were evaluated, with Gradient Boosting achieving the best performance with 92% accuracy.

Data Augmentation: Applied on image datasets to improve generalization and reduce overfitting.

Model Evaluation: Train-test split strategy with sanity checks using label shuffling to validate model integrity.

✅ Results

Accuracy: 92%

Precision: High precision and recall on both benign and malignant cases

Sanity Check: Model showed degraded performance when labels were shuffled, indicating valid learning
