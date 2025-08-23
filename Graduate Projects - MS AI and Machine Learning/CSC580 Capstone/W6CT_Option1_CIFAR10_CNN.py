#TF Warnings Suppressor import
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()

#Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#Set random seeds for academic reproducibility
np.random.seed(13)
tf.random.set_seed(13)

def load_and_preprocess_data():
    print("Loading CIFAR-10 dataset")
    
    #Load the dataset from Keras' internal data repository
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    #Print dataset information
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    #Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    #Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        #First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        #Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        #Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        #Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=128):
    #Define optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    #Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    #Print model summary
    model.summary()
    
    #Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    #Train the model
    print(f"\nTraining model for {epochs} epochs")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, x_test, y_test):
    print("\nEvaluating model on test data")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    #Make predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    return test_loss, test_accuracy, predicted_classes, true_classes

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    #Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.close()

def plot_sample_predictions(model, x_test, y_test, num_samples=10):
    #CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    #Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        #Get prediction
        img = x_test[idx]
        true_label = np.argmax(y_test[idx])
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = np.argmax(pred)
        confidence = np.max(pred)
        
        #Plot image
        axes[i].imshow(img)
        axes[i].set_title(f'True: {class_names[true_label]}\n'
                         f'Pred: {class_names[pred_label]} ({confidence:.2f})',
                         fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300)
    plt.close()

def create_confusion_matrix(true_classes, predicted_classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    #CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    #Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    return cm

def main():
    print("~~~")
    print("CIFAR-10 CNN Classification")
    print("~~~")
    
    #Step 1: Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    #Step 2: Create CNN model
    model = create_cnn_model()
    
    #Step 3: Train the model
    history = train_model(model, x_train, y_train, epochs=50)
    
    #Step 4: Evaluate the model
    test_loss, test_accuracy, predicted_classes, true_classes = evaluate_model(model, x_test, y_test)
    
    #Step 5: Create visualizations
    plot_training_history(history)
    plot_sample_predictions(model, x_test, y_test)
    confusion_matrix = create_confusion_matrix(true_classes, predicted_classes)
    
    #Step 6: Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_h5 = f'cifar10_cnn_model_{timestamp}.h5'
    model_filename_keras = f'cifar10_cnn_model_{timestamp}.keras'
    
    model.save(model_filename_h5)
    model.save(model_filename_keras)
    print(f"\nModel saved as: {model_filename_h5}")
    print(f"Model saved as: {model_filename_keras}")
    
    #Print final results
    print("\n~~~")
    print("FINAL RESULTS")
    print("~~~")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    #Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        class_accuracy = np.mean(predicted_classes[class_mask] == i)
        print(f"{class_name}: {class_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()