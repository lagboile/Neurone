import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Charger les données
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Afficher la taille des ensembles
print(f"Train set: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test set: {x_test.shape}, Labels: {y_test.shape}")

# Normalisation des valeurs de pixels
x_train = x_train / 255.0
x_test = x_test / 255.0

# Affichage de quelques exemples d'images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Convertir l'image 28x28 en vecteur 1D
    Dense(128, activation='relu'),  # Couche cachée avec 128 neurones
    Dense(10, activation='softmax')  # Couche de sortie (10 classes)
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Courbes de performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Prédiction
predictions = model.predict(x_test)

# Exemple d'affichage d'une prédiction
plt.figure(figsize=(6, 3))
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Predicted: {class_names[predictions[0].argmax()]}")
plt.show()


