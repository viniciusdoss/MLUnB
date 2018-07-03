import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint


np.random.seed(15)
train_many_configurations = False

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

print(train_features.shape)
print(test_features.shape)

_, img_rows, img_cols = train_features.shape
num_classes = len(np.unique(train_labels))
num_input_nodes = img_rows*img_cols

def preprocess_data(X):
    m = np.mean(X, axis=1)
    m = m.reshape((m.shape[0], 1))
    m = m*np.ones(X.shape)
    s = np.std(X, axis=1)
    s = s.reshape((s.shape[0], 1))
    out = (X-m)/s
    return out


# reshape images
train_features = train_features.reshape(train_features.shape[0], img_rows*img_cols)
train_features = train_features/255
train_features = preprocess_data(train_features)

test_features = test_features.reshape(test_features.shape[0], img_rows*img_cols)
test_features = test_features/255
test_features = preprocess_data(test_features)


# convert class labels to one-hot encoded
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

# Create model MLP
def MLP(num_neurons_hidden_layer):

    model = Sequential()
    model.add(Dense(num_neurons_hidden_layer, input_dim=num_input_nodes, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    return model
list_num_neurons = [20, 50, 70]
list_lr = [0.1, 0.01, 0.001]
num_epochs = 50
list_results = []

if train_many_configurations:

    for num_neurons in list_num_neurons:
        for alpha in list_lr:

            print('\n')
            print(num_neurons, ' in hidden layer..')
            print('Learning rate: ', alpha)
            print('\n')

            model = MLP(num_neurons)

            model.compile(optimizer=SGD(lr=alpha), loss='categorical_crossentropy', metrics=['accuracy'])

            # Show the model architecture
            print('#############################################')
            model.summary()
            print('#############################################')

            start = time.time()
            history = model.fit(train_features, train_labels, batch_size=64, \
                                   nb_epoch=num_epochs, verbose=2, validation_split=0.2)
            end = time.time()
            print("%0.2f seconds to train" % (end - start))
            list_results.append(history.history)

    count = 1
    for history in list_results:

        plt.figure()
        plt.plot(history['acc'])
        plt.plot(history['val_acc'], 'g')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./figures/accurary_' + str(count) + '.png')


        plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'], 'g')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./figures/loss_' + str(count) + '.png')

        count += 1

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy(model, X_test, y_test):
    out = model.predict(X_test)
    predicted_label = np.argmax(out, axis=1)
    true_label = np.argmax(y_test, axis=1)
    num_correct = np.sum(predicted_label == true_label)
    accuracy = float(num_correct)/out.shape[0]
    return 100*accuracy


# Train with early-stopping
mlp_weights = 'weightsMLP.h5'

model = MLP(70)

#model.load_weights('weightsMLP.h5')

model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

#  Early stopping
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(mlp_weights, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
callbacks_list = [earlyStopping, checkpoint]


# Show the model architecture
print('#############################################')
model.summary()
print('#############################################')

history = model.fit(train_features, train_labels, batch_size=64, \
                                   nb_epoch=100, callbacks=callbacks_list, verbose=2, validation_split=0.2)


plot_model_history(history)

print("Accuracy on test samples is: %0.2f" % accuracy(model, test_features, test_labels))









