# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and output required plot and loss
    
    df_train = pd.read_csv('/Users/kittyqu/Desktop/Assignment 3/Problem 2/train_data_RNN.csv')
    df_test = pd.read_csv('/Users/kittyqu/Desktop/Assignment 3/Problem 2/test_data_RNN.csv')
    del df_train[df_train.columns[0]]
    del df_test[df_test.columns[0]]
    
    # Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(df_train.to_numpy())
    scaled_test = scaler.fit_transform(df_test.to_numpy())

    scaled_X_train = scaled_train[:, 0:12]
    scaled_y_train = scaled_train[:, 12]
    scaled_X_test = scaled_test[:, 0:12]
    scaled_y_test = scaled_test[:, 12]

    # Reshape train data to 3D arrary for LSTM model
    scaled_X_train = np.reshape(scaled_X_train, (scaled_X_train.shape[0], scaled_X_train.shape[1], 1))
    scaled_y_train = np.reshape(scaled_y_train, (scaled_y_train.shape[0],1))
    scaled_X_test = np.reshape(scaled_X_test, (scaled_X_test.shape[0], scaled_X_test.shape[1], 1))
    scaled_y_test = np.reshape(scaled_y_test, (scaled_y_test.shape[0],1))
    scaled_X_train, x_val,scaled_y_train,y_val = train_test_split(scaled_X_train,scaled_y_train,test_size=0.15,random_state=42)
    
    # model = tf.keras.saving.load_model("/Users/kittyqu/Desktop/Assignment 3/Problem 2/models/21015968_RNN_model.keras")
    #Model predictions for train dataset
    scores = model.evaluate(scaled_X_test,scaled_y_test)
    print(scores)
    y_pred = model.predict(scaled_X_test)  

    # visualisation
    plt.figure(figsize=(20,10))
    plt.plot(y_pred, marker='o', linestyle='dashed', color = "b", label = "y_pred" )
    plt.plot(scaled_y_test, marker='o', linestyle='dashed', color = "g", label = "y_test")
    plt.xlabel("Random Dates")
    plt.ylabel("Normalized open price")
    plt.title("Simple RNN model, Stock price prediction")
    plt.legend()
    plt.show()