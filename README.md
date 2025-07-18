Hand Gesture Recognition using Machine Learning
### 📌 Prodigy InfoTech – Internship Task 4

---

## 🎯 Task Objective

Develop a machine learning model that accurately identifies and classifies different **hand gestures** from images or video data, enabling intuitive **human-computer interaction** and **gesture-based control systems**.

---

## 📂 Dataset

**Source:** Kaggle  
🔗 [Leap Hand Gesture Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)

- The dataset contains 10 different hand gesture classes captured with a Leap Motion sensor.
- Each class includes several images in grayscale, with consistent hand positions and backgrounds.

---

## ⚙️ Technologies Used

- Python
- Jupyter Notebook
- OpenCV (for image processing)
- Scikit-learn
- TensorFlow / Keras (optional – if using deep learning)
- NumPy / Pandas
- Matplotlib / Seaborn

---

## 📁 Project Structure

📦 Hand-Gesture-Recognition
│
├── HandGesture_Recognition.ipynb # Main code notebook
├── README.md # Project documentation
├── /data # Folder with gesture images
│ ├── 00_palm/
│ ├── 01_fist/
│ └── ... (Other gesture folders)
└── /models # Trained model (optional)

markdown
Copy
Edit

---

## 🧪 Workflow Overview

1. **Data Collection & Preprocessing**
   - Loaded gesture image dataset
   - Converted to grayscale (if not already)
   - Resized all images to consistent dimensions (e.g., 64x64)
   - Flattened image data into feature vectors

2. **Label Encoding**
   - Assigned labels to each gesture class

3. **Feature Scaling**
   - Normalized pixel values between 0 and 1

4. **Model Building**
   - Used models like:
     - SVM
     - KNN
     - CNN (optional, for better accuracy)

5. **Training & Evaluation**
   - Trained model with train/test split
   - Evaluated accuracy, precision, and confusion matrix

6. **Gesture Prediction**
   - Predicted gestures for unseen test images

---

## 🔍 Sample Code Snippet

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
