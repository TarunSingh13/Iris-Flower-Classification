# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Algorithm-KNN-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge">
</p>

> **Task 2** of my **Artificial Intelligence Internship** at **DecodeLabs**.

This project demonstrates a complete **Machine Learning Classification Pipeline** using the **K-Nearest Neighbors (KNN)** algorithm on the famous **Iris Dataset**. The project follows a modular architecture and covers every major step of a supervised learning workflow—from data loading to model evaluation and interactive prediction.

---

# 🚀 Project Overview

The objective of this project is to classify Iris flowers into one of the following species:

- 🌸 Setosa
- 🌼 Versicolor
- 🌺 Virginica

The classifier is trained using the **K-Nearest Neighbors (KNN)** algorithm and includes preprocessing, evaluation, visualization, model persistence, and user-driven prediction.

---

# ✨ Features

- ✅ Modular Python project structure
- ✅ Iris Dataset loading
- ✅ Dataset exploration
- ✅ Data preprocessing
- ✅ Train-Test Split
- ✅ Feature Scaling using StandardScaler
- ✅ KNN Classification Model
- ✅ Model Persistence using Joblib
- ✅ Accuracy Evaluation
- ✅ Classification Report
- ✅ Confusion Matrix
- ✅ Accuracy vs K Value Graph
- ✅ Interactive Flower Prediction
- ✅ Prediction Confidence
- ✅ Professional Project Structure

---

# 🏗️ Project Structure

```text
Task-02-Data-Classification/
│
├── classification.py        # Main application
├── config.py                # Project configuration
├── data_loader.py           # Dataset loading & exploration
├── preprocessing.py         # Data preprocessing
├── model.py                 # Model training & prediction
├── evaluation.py            # Evaluation & visualization
│
├── models/
│   └── iris_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── accuracy_vs_k.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧠 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Preprocess Data
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Train KNN Model
      │
      ▼
Save Model
      │
      ▼
Evaluate Model
      │
      ▼
Generate Visualizations
      │
      ▼
Predict New Flower
```

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

# 📚 Machine Learning Concepts Covered

- Supervised Learning
- Classification
- Iris Dataset
- Feature Engineering
- Feature Scaling
- Train-Test Split
- K-Nearest Neighbors (KNN)
- Model Training
- Model Persistence
- Model Evaluation
- Hyperparameter Tuning
- Confusion Matrix
- Classification Report
- Prediction Probability

---

# 📊 Model Performance

The trained KNN model achieved excellent performance on the Iris dataset.

Example Output:

```text
Accuracy : 100%

Precision : 100%

Recall : 100%

F1 Score : 100%
```

> *Results may vary slightly depending on the train-test split and K value.*

---

# 📈 Generated Outputs

The project automatically generates:

- 📊 Confusion Matrix
- 📈 Accuracy vs K Value Graph

These files are saved inside the **outputs/** directory.

---

# 🌸 Interactive Prediction

The application allows users to enter custom flower measurements and predicts the corresponding Iris species.

Example:

```text
Sepal Length : 5.1
Sepal Width  : 3.5
Petal Length : 1.4
Petal Width  : 0.2

Predicted Flower

🌸 Setosa
```

The project also displays the prediction confidence for each class.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Iris-Flower-Classification.git
```

Move into the project folder:

```bash
cd Iris-Flower-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python classification.py
```

---

# 🎯 Future Improvements

- Support additional classification algorithms
- Hyperparameter optimization
- GUI using Tkinter
- Flask Web Application
- Streamlit Dashboard
- Deploy using Docker
- Model API using FastAPI
- Cross Validation
- Automated Testing

---

# 💼 Internship Information

**Organization:** DecodeLabs

**Internship Domain:** Artificial Intelligence

**Project:** Task 2 - Data Classification Using AI

**Algorithm Used:** K-Nearest Neighbors (KNN)

---

# 🙋 About Me

**Tarun Singh**

B.Tech Computer Science Engineering (AI & ML)

Passionate about Machine Learning, Backend Development, Cloud Computing, and Problem Solving.

- 💼 LinkedIn: *(Add your profile link)*
- 💻 GitHub: *(Add your GitHub profile link)*

---

# ⭐ If you found this project helpful...

Please consider giving this repository a **⭐ Star**.

It motivates me to build more Machine Learning and AI projects throughout my internship journey.

---

<p align="center">
Made with ❤️ by <b>Tarun Singh</b>
</p>
