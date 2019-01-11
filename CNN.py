from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from WBDSascii_data import X,Y
import matplotlib.pyplot as plt

# Train-test split of the data
train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size=0.2)

batch_size = 64
epochs = 10
num_classes = 3

# Building the model with keras
model = Sequential()
# Layer 1
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(250,66,1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2),padding='same'))
# Layer 2
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# Layer 3
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(Activation('relu'))               
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
# Layer 4
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.2))
model.add(Activation('relu'))               
model.add(Dense(num_classes, activation='softmax'))

# Initializing the loss function, optimizer used and the metrics for our model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Training the model
model_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data = (valid_X, valid_label))

# Plot for accuracy
fig = plt.figure()
plt.plot(model_train.history['acc'])
plt.plot(model_train.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot for loss
fig1 = plt.figure()
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
