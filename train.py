import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from model import create_model

def train():
    maxlen = 25
    train_pad, y, test_pad, vocab_size, test = preprocess_data(maxlen=maxlen)

    X_train, X_val, y_train, y_val = train_test_split(train_pad, y, test_size=0.3, random_state=123)

    model = create_model(vocab_size=vocab_size, maxlen=maxlen)
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=True, epochs=20, batch_size=32)
    
    model.save('model.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, 21)

    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('loss.png')

if __name__ == '__main__':
    train()
