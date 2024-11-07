import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epochs')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
file_path = "datasets/cleaned_auto-mpg.csv"
model_path = "models/fuel_efficiency.keras"

if not os.path.isfile(file_path):
    # Reading the dataset and dropping None values
    df = pd.read_csv(url, names=column_names,
                     na_values='?', comment='\t',
                     sep=' ', skipinitialspace=True).dropna()

    # Converting numeric codes to meaningful names for readibility
    df['Origin'] = df['Origin'].map({1: "USA", 2: "Europe", 3: "Japan"})
    os.makedirs(os.path.dirname(file_path), exist_ok=1)
    df.to_csv(file_path)
else:
    df = pd.read_csv("datasets/cleaned_auto-mpg.csv")

df = pd.get_dummies(df, columns=['Origin'], prefix="", prefix_sep="")

train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)
train_features = train_df.copy().drop(columns=['Unnamed: 0'], errors='ignore')
test_features = test_df.copy().drop(columns=['Unnamed: 0'], errors='ignore')
train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

print(train_features)
print(test_features)


if not os.path.isfile(model_path):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    dnn_model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    dnn_model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100
    )

    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100
    )

    plot_loss(history)

    os.makedirs(os.path.dirname(model_path), exist_ok=1)
    dnn_model.save(model_path)
else:
    dnn_model = tf.keras.models.load_model(model_path)

print(
    f'Evaluation metrics: {dnn_model.evaluate(test_features, test_labels, verbose=0)}'
)
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Value [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()

print("\n" * 3, 'Input Test in the following form: ')
print("Cylinders,Displacement,Horsepower,Weight,Acceleration,Model Year,[USA|Europe|Japan]")
Origin = ['USA', 'Europe', 'Japan']
test_list_values = input().split(',')

print(dnn_model.predict(pd.DataFrame([{value: test_list_values[-1] == value if value in Origin else float(test_list_values[index]) for index, value in enumerate(test_features.columns)}])))
