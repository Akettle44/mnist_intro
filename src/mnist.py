from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#following tutorial on kaggle + help from keras textbook

def explore_data(train, test):
	print(train.head())
	print(test.head())
	#Arranaged in 28x28 images = 785 total pixels per image
	#Training set contains image in each row (where the columns are arranged in a 1d fashion)
	print(len(train))
	print(len(test))

	#check data for nulls (only 1 unique value and it is false therefore no values are missing)
	print(train.isnull().any().describe())
	print(test.isnull().any().describe())

### SETTING UP DATA
def setup_data(train, test): 
	#the training labels
	Y_train = train["label"]
	#training features (axis 1 is columns)
	X_train = train.drop(labels = ["label"], axis=1)

	#data set looks fairly balanced, each label frequency is similar
	#sns.countplot(Y_train)
	#plt.show()

	#normalize data (makes model perform better)
	X_train /= 255.0
	test /= 255.0

	#reshape data to prepare it for model, (can also use -1 instead of X_train.index)
	#reshape is 42k 28x28 gray scale images
	X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
	test = test.values.reshape(test.shape[0], 28, 28, 1)
	#one hot encoding
	Y_train = to_categorical(Y_train, num_classes=10)

	#splitting up into training and testing data
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=3)

	#visualizing each of the images
	#plt.imshow(X_train[1], cmap=pyplot.get_cmap('gray'))

	model = models.Sequential()

	model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

	return X_train, X_val, Y_train, Y_val, model, test

def plot_history(history, graph_type):
	history_dic = history.history
	print(history_dic)
	train = None
	validation = None
	train_label = "Training " + graph_type
	val_label = "Validation " + graph_type

	if(graph_type == "loss"):
		train = history_dic['loss']
		validation = history_dic['val_loss']
		plt.ylabel("Loss")
	else:
		train = history_dic['accuracy']
		validation = history_dic['val_accuracy']
		plt.ylabel("Accuracy")

	epochs = range(1, len(train) + 1)

	plt.plot(epochs, train, 'bo', label=train_label)
	plt.plot(epochs, validation, 'b', label=val_label)
	plt.title("Training and Validation Loss")
	plt.xlabel("Epochs")
	plt.legend()
	plt.show()

def main():

	train = pd.read_csv("~/projects/deep_learning/mnist/datasets/train.csv")
	test = pd.read_csv("~/projects/deep_learning/mnist/datasets/test.csv")
	#explore_data(train, test)
	X_train, X_val, Y_train, Y_val, model, test = setup_data(train, test)
	history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_val, Y_val))
	#plot_history(history, "acc")
	#plot_history(history, "loss")

	#softmax therefore predictions are max of each row's tensor
	pred = model.predict(test)
	#max axis refers to axis that it is calculated along: Calc along columns = row max
	output = np.argmax(pred, axis = 1)
	
	#create submission file
	submission = pd.DataFrame({'ImageId': range(1, len(output)+1), 'Label': output})
	submission.to_csv("submission.csv", index = False)

if __name__ == "__main__":
	main()



