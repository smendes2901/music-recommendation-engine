from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(arr1, arr2):
    return (np.sum(arr1 * arr2) / (np.sqrt(np.sum(arr1**2)) * np.sqrt(np.sum(arr2**2))))


def init_dict():
    return np.zeros((1, 32))


def plot_graph(history, epochs_to_show=29):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'][:epochs_to_show])
    plt.plot(history.history['val_accuracy'][:epochs_to_show])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation F1 score values
    plt.plot(history.history['f1_m'][:epochs_to_show])
    plt.plot(history.history['val_f1_m'][:epochs_to_show])
    plt.title('Model F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation Precision values
    plt.plot(history.history['precision_m'][:epochs_to_show])
    plt.plot(history.history['val_precision_m'][:epochs_to_show])
    plt.title('Model Precision Score')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation Recall score values
    plt.plot(history.history['recall_m'][:epochs_to_show])
    plt.plot(history.history['val_recall_m'][:epochs_to_show])
    plt.title('Model Recall Score')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'][:epochs_to_show])
    plt.plot(history.history['val_loss'][:epochs_to_show])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
