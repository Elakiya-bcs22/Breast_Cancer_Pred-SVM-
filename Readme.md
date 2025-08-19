# Breast Cancer Prediction — SVM Classifier

A Python-based machine learning project that uses a **Support Vector Machine (SVM)** to predict whether a tumor is **benign** or **malignant**.  
This project demonstrates data processing, model training, evaluation, and making predictions using real-world datasets.

---

##  Project Structure

```
Breast_Cancer_Pred-SVM/
├── app.py              # Script to run predictions using the trained model
├── train.py            # Script to train the SVM classifier
├── data/               # Dataset files (raw or preprocessed)
├── model/              # Saved model file (e.g., `.pkl`)
├── requirements.txt    # Required dependencies
└── README.md           # Documentation file
```

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Elakiya-bcs22/Breast_Cancer_Pred-SVM-.git
   cd Breast_Cancer_Pred-SVM-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

1. **Train the SVM model**:
   ```bash
   python train.py
   ```

2. **Run predictions** using `app.py` with new input samples:
   ```bash
   python app.py
   ```

---

##  Features

- Implements a **Support Vector Machine** for classifying breast tumors.
- Includes full workflow: data preprocessing → model training → inference.
- Clean and modular structure—great for learning or extending with new models.
- Supports quick experimentation and future enhancements (e.g., hyperparameter tuning or using different kernels).

---

##  Conclusion

The **SVM model** provides strong performance and reliable predictions for classifying tumors as benign or malignant.  
With more data, feature engineering, or kernel tuning, you can further improve its accuracy and robustness.
