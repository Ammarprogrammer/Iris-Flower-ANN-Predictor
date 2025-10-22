import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('DeepLearning\Iris.csv')

sns.pairplot(df,hue ='Species')
plt.show()

X = df.drop(columns=['Species','Id'])
Y = df['Species']

encoder = LabelEncoder()
Y_int = encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_int , test_size=0.2 , random_state=42, stratify=Y_int)

scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

per = Perceptron(max_iter=1000, random_state=42)
per.fit(X_train_Scaled, Y_train)
y_pred_percep = per.predict(X_test_scaled)

accura = accuracy_score(Y_test, y_pred_percep)
print(accura)#Checking perceptron accuracy

Y_train_cat = to_categorical(Y_train, num_classes=3)
Y_test_cat = to_categorical(Y_test, num_classes=3)

model = Sequential([
    Dense(16,input_dim=4,activation='relu'),
# → 4 input features
# → 16 neurons
# → ReLU adds non-linearity
    Dense(8,activation='relu'),
# → 8 neurons
# → ReLU adds non-linearity
    Dense(3,activation='softmax')
# → 3 output neurons (e.g., for 3 classes)
# → softmax gives probabilities that sum to 1   
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_Scaled,Y_train_cat,
                    epochs = 100,batch_size= 8, validation_split = 0.2,verbose = 1)

loss, acc = model.evaluate(X_test_scaled, Y_test_cat, verbose=1)
print(acc)

plt.figure(figsize = (10,4))
plt.plot(history.history['accuracy'],label = "train Acc")
plt.plot(history.history['val_accuracy'],label = "val Acc")
plt.show()

# 39: save final model
from tensorflow import keras
model.save("DeepLearning/iris_flower_pred.keras")
model = keras.models.load_model("DeepLearning/iris_flower_pred.keras")
print('----Predict your result----')
try:
    SepalLengthCm = float(input('Enter your SepalLengthCm: '))
    SepalWidthCm= float(input('Enter your SepalWidthCm '))
    PetalLengthCm = float(input('Enter your PetalLengthCm: '))
    PetalWidthCm= float(input('Enter your PetalWidthCm '))

    user_input_df = pd.DataFrame([{
        'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm
    }])

    user_input_scaled = scaler.transform(user_input_df)
    prediction = model.predict(user_input_scaled)
    print(prediction, prediction.shape)
    if prediction.ndim > 1:  # softmax output
       pred_class = prediction.argmax(axis=1)[0]
    else:                    # already class index
       pred_class = int(prediction[0])



    if pred_class == 0:
       print('Prediction based on Iris data: Iris-setosa')
    elif pred_class == 1:
       print('Prediction based on Iris data: Iris-versicolor')
    else:  
       print('Prediction based on Iris data: Iris-virginica')
except Exception as e:
    print(e)

