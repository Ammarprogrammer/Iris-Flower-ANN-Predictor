# Iris Flower ANN Prediction 🌸

This project predicts the species of Iris flowers using **Artificial Neural Networks (ANN)**. The model is trained on the classic Iris dataset, utilizing features like Sepal Length, Sepal Width, Petal Length, and Petal Width.  

The project combines **data preprocessing**, **machine learning with Perceptron**, and **deep learning with TensorFlow/Keras** to create a robust predictive model.

---

## Features

- Data visualization with **Seaborn** (pairplots for species distribution)
- Preprocessing with **StandardScaler** and **LabelEncoder**
- Train-test split (80% training, 20% testing)
- ANN model using **Keras Sequential API**
  - 3 Dense layers:  
    - 16 neurons with ReLU (input layer)  
    - 8 neurons with ReLU (hidden layer)  
    - 3 neurons with Softmax (output layer)
- Model evaluation with accuracy metrics
- Compare **training & validation accuracy** through plots
- Save and load trained models with `.keras` format
- Predict flower species from user input

---

## Libraries Used

- `pandas`  
- `matplotlib`  
- `seaborn`  
- `numpy`  
- `warnings`  
- `scikit-learn` (`StandardScaler`, `train_test_split`, `LabelEncoder`, `Perceptron`, `accuracy_score`)  
- `tensorflow` (`Sequential`, `Dense`, `to_categorical`)  

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/Iris-Flower-ANN-Prediction.git
cd Iris-Flower-ANN-Prediction
