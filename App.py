"""
Fashion MNIST Clothing Categorization Model Classification - Interactive ML Learning Application
This is my ML Model better visualized to help understand what is going on, not only does it help predict and categorize clothing, 
but helps users understand Machine Learning as in Todays and age ML models can sometimes be frustrating to understand
Note Click out of each tab to move on to the next step

See my Jupyter Notebook for a less App Interactive Environment
https://www.overleaf.com/read/dkdqxsrvtpcn#f21845 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend for VS Code compatibility
# Use 'Qt5Agg' if TkAgg doesn't work on your system
plt.switch_backend('TkAgg')

class FashionMNISTClassifier:
    """
    Interactive Fashion MNIST Classifier for Machine Learning Education
    
    This class provides a complete pipeline for:
    - Loading and preprocessing Fashion MNIST data
    - Building and training a neural network
    - Visualizing results and model performance
    - Understanding ML concepts through interactive examples
    
    The Fashion MNIST dataset contains 70,000 grayscale images of 10 different
    clothing categories, each image is 28x28 pixels.
    """
    
    def __init__(self):
        """
        Initialize the Fashion MNIST Classifier
        
        Sets up the class with necessary attributes and configurations:
        - Class names for the 10 clothing categories
        - Placeholder variables for model and data
        - Random seed for reproducible results
        """
        print("Initializing Fashion MNIST Classifier...")
        print("=" * 50)
        
        # Define the 10 clothing categories in Fashion MNIST
        # These correspond to labels 0-9 in the dataset
        self.class_names = [
            "T-shirt/top",  # Label 0
            "Trouser",      # Label 1
            "Pullover",     # Label 2
            "Dress",        # Label 3
            "Coat",         # Label 4
            "Sandal",       # Label 5
            "Shirt",        # Label 6
            "Sneaker",      # Label 7
            "Bag",          # Label 8
            "Ankle boot"    # Label 9
        ]
        
        # Initialize placeholders for model and data
        # These will be populated during the pipeline execution
        self.model = None           # TensorFlow/Keras model
        self.history = None         # Training history object
        self.X_train = None         # Training images
        self.y_train = None         # Training labels
        self.X_valid = None         # Validation images
        self.y_valid = None         # Validation labels
        self.X_test = None          # Test images
        self.y_test = None          # Test labels
        
        # Set random seeds for reproducible results
        # This ensures that random operations give the same results each time
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print("Classifier initialized successfully!")
        print(f"Dataset contains {len(self.class_names)} types of clothing items")
        print("Categories:", ", ".join(self.class_names))
        
    def load_and_preprocess_data(self):
        """
        Load Fashion MNIST dataset and prepare for training
        
        This method:
        1. Downloads the Fashion MNIST dataset (if not already cached)
        2. Splits the data into training, validation, and test sets
        3. Normalizes pixel values from 0-255 to 0-1 range
        4. Provides information about the dataset structure
        """
        print("\nLoading Fashion MNIST Dataset...")
        print("-" * 30)
        
        # Load the Fashion MNIST dataset from TensorFlow/Keras
        # This automatically downloads the dataset if not already present
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        
        print("Dataset loaded successfully!")
        print(f"Training images: {X_train_full.shape[0]:,}")
        print(f"Test images: {X_test.shape[0]:,}")
        print(f"Image dimensions: {X_train_full.shape[1]} x {X_train_full.shape[2]} pixels")
        
        # Split the training data into training and validation sets
        # We use the last 5,000 images for validation
        # This gives us: 55,000 training + 5,000 validation + 10,000 test
        self.X_train = X_train_full[:-5000]  # First 55,000 images for training
        self.y_train = y_train_full[:-5000]  # Corresponding labels
        self.X_valid = X_train_full[-5000:]  # Last 5,000 images for validation
        self.y_valid = y_train_full[-5000:]  # Corresponding labels
        self.X_test = X_test                 # Test set (10,000 images)
        self.y_test = y_test                 # Test labels
        
        print("Data split completed:")
        print(f"   Training: {self.X_train.shape[0]:,} images")
        print(f"   Validation: {self.X_valid.shape[0]:,} images")
        print(f"   Test: {self.X_test.shape[0]:,} images")
        
        # Normalize pixel values to 0-1 range
        # Original values are 0-255 (8-bit grayscale)
        # Normalization helps with model training stability and convergence
        self.X_train = self.X_train / 255.0
        self.X_valid = self.X_valid / 255.0
        self.X_test = self.X_test / 255.0
        
        print("Pixel values normalized to [0, 1] range")
        
        # Display dataset statistics for educational purposes
        print("\nDataset Statistics:")
        print(f"   Original pixel range: 0-255")
        print(f"   Normalized pixel range: {self.X_train.min():.1f}-{self.X_train.max():.1f}")
        print(f"   Data type: {self.X_train.dtype}")
        
    def visualize_sample_data(self, num_samples=25):
        """
        Display sample images from the dataset
        
        This method creates a visual grid showing random samples from the training set
        along with their corresponding labels. This helps understand the data we're working with.
        
        Args:
            num_samples (int): Number of sample images to display (default: 25)
        """
        print(f"\nDisplaying {num_samples} sample images...")
        
        # Create a figure with subplots arranged in a 5x5 grid
        plt.figure(figsize=(12, 12))
        plt.suptitle("Fashion MNIST Sample Images", fontsize=16, fontweight='bold')
        
        # Display each sample image with its label
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)  # Create subplot in 5x5 grid
            plt.imshow(self.X_train[i], cmap='gray')  # Display as grayscale
            plt.title(f'{self.class_names[self.y_train[i]]}', fontsize=10)
            plt.axis('off')  # Hide axes for cleaner appearance
        
        plt.tight_layout()
        plt.show()
        
        # Also show class distribution to understand data balance
        self.plot_class_distribution()
        
    def plot_class_distribution(self):
        """
        Plot the distribution of classes in the dataset
        
        This visualization shows how many images belong to each clothing category.
        A balanced dataset (equal distribution) is generally preferable for training.
        """
        print("\nAnalyzing class distribution...")
        
        # Count occurrences of each class in the training set
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        # Create a bar chart showing class distribution
        plt.figure(figsize=(12, 6))
        
        # Create bars with styling
        bars = plt.bar(range(len(self.class_names)), counts, 
                      color='skyblue', edgecolor='navy')
        plt.xlabel('Clothing Categories', fontweight='bold')
        plt.ylabel('Number of Images', fontweight='bold')
        plt.title('Distribution of Clothing Categories in Training Data', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.class_names)), self.class_names, 
                  rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("Class distribution analysis complete!")
        print("Note: Fashion MNIST has a relatively balanced distribution across classes")
        
    def build_model(self, hidden_layers=[300, 100], learning_rate=0.01):
        """
        Build the neural network model architecture
        
        This method creates a feedforward neural network with:
        - Input layer: Flattened 28x28 images (784 features)
        - Hidden layers: Two dense layers with ReLU activation
        - Output layer: 10 neurons with softmax activation (one per class)
        
        Args:
            hidden_layers (list): Number of neurons in each hidden layer
            learning_rate (float): Learning rate for the optimizer
        """
        print("\nBuilding Neural Network Model...")
        print("-" * 30)
        
        # Create a sequential model (layers stacked one after another)
        self.model = tf.keras.Sequential([
            # Input layer: Define input shape (28x28 images)
            tf.keras.Input(shape=[28, 28], name='input_layer'),
            
            # Flatten layer: Convert 28x28 images to 784-dimensional vectors
            # This is necessary because Dense layers expect 1D input
            tf.keras.layers.Flatten(name='flatten_layer'),
            
            # First hidden layer: 300 neurons with ReLU activation
            # ReLU (Rectified Linear Unit) helps with gradient flow and learning
            tf.keras.layers.Dense(hidden_layers[0], activation='relu', 
                                name='hidden_layer_1'),
            
            # Second hidden layer: 100 neurons with ReLU activation
            tf.keras.layers.Dense(hidden_layers[1], activation='relu', 
                                name='hidden_layer_2'),
            
            # Output layer: 10 neurons (one per class) with softmax activation
            # Softmax converts outputs to probabilities that sum to 1
            tf.keras.layers.Dense(len(self.class_names), activation='softmax', 
                                name='output_layer')
        ])
        
        # Configure the model for training
        # SGD: Stochastic Gradient Descent optimizer
        # sparse_categorical_crossentropy: Loss function for integer labels
        # accuracy: Metric to monitor during training
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture created successfully!")
        print(f"Architecture: Input(784) -> Dense({hidden_layers[0]}) -> Dense({hidden_layers[1]}) -> Output({len(self.class_names)})")
        print(f"Optimizer: SGD with learning rate {learning_rate}")
        print(f"Loss function: Sparse Categorical Crossentropy")
        print("Activation functions: ReLU (hidden layers), Softmax (output)")
        
        # Display detailed model summary
        print("\nDetailed Model Summary:")
        self.model.summary()
        
        # Calculate and display total parameters
        total_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {total_params:,}")
        print("Note: More parameters = more model capacity, but also more risk of overfitting")
        
    def train_model(self, epochs=30, batch_size=32, verbose=1):
        """
        Train the neural network model
        
        This method trains the model using the training data and validates
        performance on the validation set. It includes early stopping to
        prevent overfitting.
        
        Args:
            epochs (int): Maximum number of training epochs
            batch_size (int): Number of samples processed before updating weights
            verbose (int): Training progress display level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        print(f"\nTraining the model for up to {epochs} epochs...")
        print("-" * 40)
        
        # Set up training callbacks
        # Early stopping prevents overfitting by stopping training when
        # validation accuracy stops improving
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',      # Monitor validation accuracy
                patience=5,                  # Wait 5 epochs without improvement
                restore_best_weights=True,   # Restore best weights when stopping
                verbose=1                    # Print when early stopping occurs
            )
        ]
        
        # Train the model
        # The model learns by:
        # 1. Making predictions on training data
        # 2. Calculating loss (how wrong the predictions are)
        # 3. Updating weights to minimize loss
        # 4. Validating on separate validation set
        self.history = self.model.fit(
            self.X_train, self.y_train,           # Training data
            batch_size=batch_size,                # Process 32 samples at a time
            epochs=epochs,                        # Maximum training iterations
            validation_data=(self.X_valid, self.y_valid),  # Validation data
            callbacks=callbacks,                  # Early stopping callback
            verbose=verbose                       # Progress display
        )
        
        print("\nTraining completed!")
        
        # Display final training results
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        print("Final Training Results:")
        print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        
        # Check for overfitting
        if final_train_acc - final_val_acc > 0.05:  # 5% difference threshold
            print("   Warning: Possible overfitting detected (training >> validation accuracy)")
        else:
            print("   Good: Training and validation accuracies are similar")
        
    def plot_training_history(self):
        """
        Visualize training progress over epochs
        
        This method creates plots showing how loss and accuracy changed
        during training. These plots help understand:
        - Whether the model is learning (decreasing loss, increasing accuracy)
        - Whether overfitting is occurring (training vs validation divergence)
        - When training should be stopped
        """
        if self.history is None:
            print("No training history available. Train the model first!")
            return
            
        print("\nVisualizing training progress...")
        
        # Create side-by-side plots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get epoch numbers for x-axis
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        # Plot 1: Accuracy over time
        ax1.plot(epochs, self.history.history['accuracy'], 'b-', 
                label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, self.history.history['val_accuracy'], 'r--', 
                label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss over time
        ax2.plot(epochs, self.history.history['loss'], 'b-', 
                label='Training Loss', linewidth=2)
        ax2.plot(epochs, self.history.history['val_loss'], 'r--', 
                label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Provide interpretation guidance
        print("Training History Analysis:")
        print("- Accuracy should generally increase over time")
        print("- Loss should generally decrease over time")
        print("- Training and validation metrics should track closely")
        print("- Large gaps suggest overfitting")
        
    def evaluate_model(self):
        """
        Evaluate model performance on the test set
        
        This method provides a comprehensive evaluation of the trained model:
        - Overall accuracy and loss on unseen test data
        - Detailed per-class performance metrics
        - Confusion matrix visualization
        
        Returns:
            float: Test accuracy score
        """
        print("\nEvaluating model on test set...")
        print("-" * 35)
        
        # Evaluate model on test set (data it has never seen)
        # This gives us an unbiased estimate of model performance
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print("Test Set Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Generate predictions for detailed analysis
        # predict() returns probabilities for each class
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Generate detailed classification report
        # This shows precision, recall, and F1-score for each class
        print("\nDetailed Classification Report:")
        print("(Precision: % of predicted class that are correct)")
        print("(Recall: % of actual class that are correctly predicted)")
        print("(F1-score: Harmonic mean of precision and recall)")
        print("-" * 60)
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        # Create confusion matrix visualization
        self.plot_confusion_matrix(self.y_test, y_pred)
        
        return test_accuracy
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Create and display a confusion matrix
        
        A confusion matrix shows:
        - True positives: Correctly predicted items for each class
        - False positives: Items incorrectly predicted as each class
        - False negatives: Items of each class that were missed
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
        """
        print("\nGenerating confusion matrix...")
        
        # Calculate confusion matrix
        # Each row represents true class, each column represents predicted class
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Fashion MNIST Classification', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontweight='bold')
        plt.ylabel('True Labels', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        print("Confusion Matrix Interpretation:")
        print("- Diagonal elements: Correct predictions")
        print("- Off-diagonal elements: Misclassifications")
        print("- Darker colors indicate higher values")
        print("- Perfect classifier would have all values on the diagonal")
        
    def visualize_predictions(self, num_samples=9, show_correct=True):
        """
        Visualize model predictions on test samples
        
        This method shows actual images alongside the model's predictions
        and confidence scores. Green titles indicate correct predictions,
        red titles indicate incorrect predictions.
        
        Args:
            num_samples (int): Number of samples to visualize
            show_correct (bool): Whether to show correct predictions
        """
        print("\nVisualizing model predictions...")
        
        # Get predictions for the first num_samples test images
        y_pred_proba = self.model.predict(self.X_test[:num_samples], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Create a 3x3 grid of subplots
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('Model Predictions vs True Labels', fontsize=16, fontweight='bold')
        
        # Display each sample with its prediction
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Display the image
                ax.imshow(self.X_test[i], cmap='gray')
                
                # Get labels and confidence
                true_label = self.class_names[self.y_test[i]]
                pred_label = self.class_names[y_pred[i]]
                confidence = np.max(y_pred_proba[i]) * 100
                
                # Color based on correctness
                color = 'green' if true_label == pred_label else 'red'
                
                # Set title with prediction information
                ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                           color=color, fontsize=10, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Prediction Visualization Guide:")
        print("- Green titles: Correct predictions")
        print("- Red titles: Incorrect predictions")
        print("- Confidence: Model's certainty (higher = more confident)")
        
    def analyze_misclassifications(self, num_samples=9):
        """
        Analyze and visualize misclassified examples
        
        This method helps understand what types of errors the model makes
        by showing examples where it was wrong. This is crucial for
        improving model performance.
        
        Args:
            num_samples (int): Number of misclassified examples to show
        """
        print("\nAnalyzing misclassifications...")
        
        # Get predictions for all test samples
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Find indices where predictions don't match true labels
        misclassified_indices = np.where(y_pred != self.y_test)[0]
        
        print(f"Misclassification Analysis:")
        print(f"   Total errors: {len(misclassified_indices)} out of {len(self.y_test)}")
        print(f"   Error rate: {len(misclassified_indices)/len(self.y_test)*100:.2f}%")
        print(f"   Accuracy: {(1 - len(misclassified_indices)/len(self.y_test))*100:.2f}%")
        
        if len(misclassified_indices) == 0:
            print("Perfect classification! No errors found.")
            return
            
        # Visualize misclassified examples
        plt.figure(figsize=(15, 10))
        plt.suptitle('Misclassified Examples - Learning from Mistakes', 
                    fontsize=16, fontweight='bold')
        
        # Show the first num_samples misclassified examples
        for i, idx in enumerate(misclassified_indices[:num_samples]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.X_test[idx], cmap='gray')
            
            # Get labels and confidence for this misclassified example
            true_label = self.class_names[self.y_test[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx]) * 100
            
            plt.title(f'True: {true_label}\nPredicted: {pred_label}\nConfidence: {confidence:.1f}%',
                     color='red', fontsize=10, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nMisclassification Analysis Tips:")
        print("- Look for patterns in errors (e.g., similar-looking items)")
        print("- High confidence errors are particularly concerning")
        print("- Consider if humans would make similar mistakes")
        print("- This information can guide model improvements")
        
    def interactive_single_prediction(self, image_index=0):
        """
        Analyze a single prediction in detail
        
        This method provides an in-depth look at how the model makes
        predictions for a single image, showing probabilities for all classes.
        
        Args:
            image_index (int): Index of the test image to analyze
        """
        print(f"\nAnalyzing detailed prediction for image {image_index}...")
        
        # Get the specific image and its true label
        single_image = self.X_test[image_index:image_index+1]
        true_label = self.y_test[image_index]
        
        # Make prediction and get probabilities for all classes
        prediction_proba = self.model.predict(single_image, verbose=0)
        predicted_class = np.argmax(prediction_proba)
        
        # Create side-by-side visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Show the input image
        ax1.imshow(self.X_test[image_index], cmap='gray')
        ax1.set_title(f'Input Image\nTrue Label: {self.class_names[true_label]}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Right plot: Show prediction probabilities for all classes
        proba_percentages = prediction_proba[0] * 100
        bars = ax2.bar(range(len(self.class_names)), proba_percentages)
        ax2.set_title('Prediction Probabilities for All Classes', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Clothing Categories', fontweight='bold')
        ax2.set_ylabel('Probability (%)', fontweight='bold')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Highlight the predicted class (red) and true class (green)
        bars[predicted_class].set_color('red')    # Model's prediction
        bars[true_label].set_color('green')       # True class
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed numerical analysis
        print("\nDetailed Prediction Analysis:")
        print(f"   True Label: {self.class_names[true_label]}")
        print(f"   Predicted: {self.class_names[predicted_class]}")
        print(f"   Confidence: {np.max(prediction_proba)*100:.2f}%")
        print(f"   Correct: {'Yes' if predicted_class == true_label else 'No'}")
        
        # Show top 3 most likely predictions
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        print("\nTop 3 Most Likely Predictions:")
        for i, idx in enumerate(top_3_indices):
            print(f"   {i+1}. {self.class_names[idx]}: {prediction_proba[0][idx]*100:.2f}%")
        
        # Educational notes
        print("\nWhat this tells us:")
        print("- The model outputs probabilities for ALL classes")
        print("- The sum of all probabilities equals 100%")
        print("- Higher probabilities indicate more confidence")
        print("- Second-highest probability shows the model's 'second guess'")
            
    def run_complete_pipeline(self):
        """
        Execute the complete machine learning pipeline
        
        This method runs through all steps of the ML process:
        1. Data loading and preprocessing
        2. Data visualization and exploration
        3. Model architecture design
        4. Model training
        5. Training progress visualization
        6. Model evaluation
        7. Prediction visualization
        8. Error analysis
        9. Interactive prediction demonstration
        """
        print("Running Complete Fashion MNIST Classification Pipeline")
        print("=" * 60)
        
        # Step 1: Load and preprocess the data
        print("\n[STEP 1] DATA LOADING AND PREPROCESSING")
        self.load_and_preprocess_data()
        
        # Step 2: Visualize and explore the data
        print("\n[STEP 2] DATA VISUALIZATION AND EXPLORATION")
        self.visualize_sample_data()
        
        # Step 3: Build the neural network model
        print("\n[STEP 3] MODEL ARCHITECTURE DESIGN")
        self.build_model()
        
        # Step 4: Train the model
        print("\n[STEP 4] MODEL TRAINING")
        self.train_model(epochs=30)
        
        # Step 5: Visualize training progress
        print("\n[STEP 5] TRAINING PROGRESS ANALYSIS")
        self.plot_training_history()
        
        # Step 6: Evaluate model performance
        print("\n[STEP 6] MODEL EVALUATION")
        test_accuracy = self.evaluate_model()
        
        # Step 7: Visualize predictions
        print("\n[STEP 7] PREDICTION VISUALIZATION")
        self.visualize_predictions()
        
        # Step 8: Analyze errors and misclassifications
        print("\n[STEP 8] ERROR ANALYSIS")
        self.analyze_misclassifications()
        
        # Step 9: Interactive single prediction analysis
        print("\n[STEP 9] INTERACTIVE PREDICTION ANALYSIS")
        # Select a random test image for detailed analysis
        random_index = np.random.randint(0, len(self.X_test))
        self.interactive_single_prediction(random_index)
        
        print(f"\n Pipeline Complete!")
        print(f" Final Test Accuracy: {test_accuracy*100:.2f}%")
        print(" Great job exploring machine learning with Fashion MNIST!")

def main():
    """Main function to run the Fashion MNIST classifier"""
    print("Welcome to the Fashion MNIST Classification Learning Tool!")
    print("This application will teach you ML concepts through hands-on examples.")
    print("\n" + "="*60)
    
    # Create classifier instance
    classifier = FashionMNISTClassifier()
    
    # Run the complete pipeline
    classifier.run_complete_pipeline()
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode Available!")
    print("You can now use the classifier object to explore further:")
    print("   - classifier.interactive_single_prediction(index)")
    print("   - classifier.visualize_predictions(num_samples=9)")
    print("   - classifier.analyze_misclassifications()")
    print("   - classifier.plot_training_history()")
    
    return classifier

if __name__ == "__main__":
    # Run the main application
    print("""
    #########################
    CLICK OUT OF EACH WINDOW TO MOVE TO THE NEXT STEP
    ##########################
    """)
    classifier = main()
    
    # Keep the application running for interaction
    print("\n🔄 Application ready for interaction!")
  
