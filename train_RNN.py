# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# import dataset
df = pd.read_csv('/Users/kittyqu/Desktop/Assignment 3/Problem 2/q2_dataset.csv')
# convert dataframe to numpy array
df_np = df.to_numpy()
# delete column "date" and "close/last"
df_np = np.delete(df_np, [0,1], 1)
new_dataset = []
for i in range(len(df_np)-3):
    feature = np.concatenate((df_np[i+1],df_np[i+2],df_np[i+3]))
    feature_target = np.append(feature,df_np[i][1])
    new_dataset.append(feature_target.tolist())
new_dataset = np.array(new_dataset)
np.random.shuffle(new_dataset)
new_df = pd.DataFrame(new_dataset)
new_df.columns = ['f1','f2','f3','f4','f5','f6','f7','f8','f9',
                  'f10','f11','f12','target']
features = new_df[['f1','f2','f3','f4','f5','f6','f7','f8','f9',
                  'f10','f11','f12']]

target = new_df.loc[:, 'target']

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0)
df_train = pd.concat([X_train,y_train],axis=1)
df_test =pd.concat([X_test,y_test],axis=1)  
df_train.to_csv('/Users/kittyqu/Desktop/Assignment 3/Problem 2/train_data_RNN.csv')
df_test.to_csv('/Users/kittyqu/Desktop/Assignment 3/Problem 2/test_data_RNN.csv')

if __name__ == "__main__": 
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model
    
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

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(scaled_X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compiling RNN
    model.compile(optimizer='adam',loss='mean_squared_error')
    # Fitting RNN
    history=model.fit(scaled_X_train,scaled_y_train,epochs=300,batch_size = 32,validation_data=[x_val,y_val])

    # Plotting train & val Loss vs. epochs
    plt.plot(history.history['val_loss'],label="val loss")
    plt.plot(history.history['loss'],label="train loss")
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title("LSTM model, Loss vs Epoch")
    plt.legend()
    plt.show()
    
    # save model
    model.save("/Users/kittyqu/Desktop/Assignment 3/Problem 2/models/21015968_RNN_model.keras")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    