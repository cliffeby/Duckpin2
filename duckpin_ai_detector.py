# Duckpin Bowling Pin Detection AI System
# This program creates an AI model to identify standing duckpin bowling pins in images
# Uses transfer learning with a pre-trained CNN model and connects to Azure Blob Storage

import os
import cv2
import numpy as np
import json
import glob
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow imports with proper error handling
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Try standalone Keras first (preferred for TF 2.16+)
    try:
        import keras
        from keras import layers
        from keras.metrics import Precision, Recall
        KERAS_BACKEND = "standalone"
        print(f"✅ Using standalone Keras {keras.__version__}")
    except ImportError:
        # Fallback to tensorflow.keras
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.metrics import Precision, Recall
        KERAS_BACKEND = "tensorflow"
        print(f"✅ Using TensorFlow.Keras")
        
except ImportError as e:
    print("❌ TensorFlow is not installed or not properly configured.")
    print("Please install TensorFlow using: pip install tensorflow")
    print(f"Error: {e}")
    exit(1)

from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
import credentials

def check_tensorflow_installation():
    """Check if TensorFlow is properly installed and configured"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic TensorFlow operations
        test_tensor = tf.constant([1, 2, 3, 4])
        print(f"✅ TensorFlow basic operations working: {test_tensor}")
        
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            print("✅ GPU support available")
        else:
            print("ℹ️  Running on CPU (no GPU detected)")
        
        return True
    except ImportError as e:
        print("❌ TensorFlow is not installed!")
        print("Please install TensorFlow using one of these commands:")
        print("  pip install tensorflow")
        print("  pip install tensorflow-cpu  (CPU only)")
        print("  pip install tensorflow-gpu  (if you have CUDA)")
        print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorFlow installation issue: {e}")
        return False

class DuckpinPinDetector:
    """
    AI model for detecting standing duckpin bowling pins in images.
    Uses transfer learning with a pre-trained CNN model.
    """
    
    def __init__(self, image_size=(224, 224), num_pins=10):
        """
        Initialize the DuckpinPinDetector
        
        Args:
            image_size: Input image size for the model
            num_pins: Number of pins to detect (10 for duckpin bowling)
        """
        self.image_size = image_size
        self.num_pins = num_pins
        self.model = None
        self.history = None
        self.class_names = [f'pin_{i}' for i in range(num_pins)]
        
        # Azure connection setup
        self.setup_azure_connection()
        
    def setup_azure_connection(self):
        """Setup Azure Blob Storage connection using credentials"""
        try:
            account_name = credentials.STORAGE_ACCOUNT_NAME
            account_key = credentials.STORAGE_ACCOUNT_KEY
            account_url = "https://duckpinjson.blob.core.windows.net"
            
            self.azure_credential = AzureNamedKeyCredential(account_name, account_key)
            self.blob_service_client = BlobServiceClient(account_url, credential=self.azure_credential)
            self.container_name = 'jsoncontdp'
            print("✅ Azure connection established successfully")
            
        except Exception as e:
            print(f"❌ Error setting up Azure connection: {e}")
            self.blob_service_client = None
    
    def parse_filename_to_pin_state(self, filename):
        """
        Parse filename to extract pin configuration
        
        Args:
            filename: e.g., "finalframe_pins_1023_to_512_20251105_143052_834567.jpg"
            
        Returns:
            tuple: (beginning_pins, ending_pins) as binary arrays
        """
        try:
            # Extract pin values from filename
            if "finalframe_pins_" in filename:
                # Format: finalframe_pins_1023_to_512_timestamp.jpg
                parts = filename.replace("finalframe_pins_", "").split("_")
                if len(parts) >= 3 and parts[1] == "to":
                    beginning_decimal = int(parts[0])
                    ending_decimal = int(parts[2])
                    
                    # Convert to 10-bit binary (pad with leading zeros)
                    beginning_binary = [int(bit) for bit in format(beginning_decimal, '010b')]
                    ending_binary = [int(bit) for bit in format(ending_decimal, '010b')]
                    
                    return beginning_binary, ending_binary
            
            # Fallback: try to extract from any numeric pattern
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) <= 4:
                    decimal_val = int(part)
                    if decimal_val <= 1023:  # Valid pin configuration
                        binary_state = [int(bit) for bit in format(decimal_val, '010b')]
                        return binary_state, binary_state
            
            print(f"⚠️  Cannot parse filename: {filename}")
            return [0] * self.num_pins, [0] * self.num_pins
                
        except Exception as e:
            print(f"❌ Error parsing filename {filename}: {e}")
            return [0] * self.num_pins, [0] * self.num_pins
    
    def download_images_from_azure(self, local_dir='data/raw_images', limit=None):
        """
        Download images from Azure blob storage
        
        Args:
            local_dir: Local directory to save images
            limit: Maximum number of images to download (None for all)
        """
        if not self.blob_service_client:
            print("❌ Azure connection not available")
            return []
        
        os.makedirs(local_dir, exist_ok=True)
        downloaded_files = []
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with="Lane4Free/images/")
            
            count = 0
            for blob in blob_list:
                if limit and count >= limit:
                    break
                    
                if blob.name.endswith(('.jpg', '.jpeg', '.png')):
                    local_filename = os.path.join(local_dir, os.path.basename(blob.name))
                    
                    # Download blob
                    with open(local_filename, "wb") as download_file:
                        download_stream = container_client.download_blob(blob.name)
                        download_file.write(download_stream.readall())
                    
                    downloaded_files.append(local_filename)
                    count += 1
                    print(f"📥 Downloaded: {blob.name}")
            
            print(f"✅ Downloaded {len(downloaded_files)} images")
            return downloaded_files
            
        except Exception as e:
            print(f"❌ Error downloading from Azure: {e}")
            return []
    
    def create_training_dataset(self, image_dir='data/raw_images', output_dir='data/processed'):
        """
        Create training dataset with labels from downloaded images
        
        Args:
            image_dir: Directory containing raw images
            output_dir: Directory to save processed dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        dataset = []
        image_files = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpeg")
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Parse pin states from filename
            beginning_pins, ending_pins = self.parse_filename_to_pin_state(filename)
            
            # Use ending pins as the target (final state after ball impact)
            target_pins = ending_pins
            
            # Load and preprocess image
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                # Resize image
                image = cv2.resize(image, self.image_size)
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values
                image = image.astype(np.float32) / 255.0
                
                dataset.append({
                    'image': image,
                    'pin_states': target_pins,
                    'beginning_pins': beginning_pins,
                    'ending_pins': ending_pins,
                    'filename': filename,
                    'standing_pins': sum(target_pins)  # Count of standing pins
                })
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue
        
        if not dataset:
            print("❌ No valid images found in dataset")
            return []
        
        # Save dataset
        dataset_file = f"{output_dir}/dataset.json"
        images_array = np.array([item['image'] for item in dataset])
        labels_array = np.array([item['pin_states'] for item in dataset])
        
        np.save(f"{output_dir}/images.npy", images_array)
        np.save(f"{output_dir}/labels.npy", labels_array)
        
        # Save metadata
        metadata = [{
            'filename': item['filename'],
            'pin_states': item['pin_states'],
            'beginning_pins': item['beginning_pins'],
            'ending_pins': item['ending_pins'],
            'standing_pins': item['standing_pins']
        } for item in dataset]
        
        with open(dataset_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Created dataset with {len(dataset)} samples")
        print(f"📊 Average standing pins: {np.mean([item['standing_pins'] for item in dataset]):.2f}")
        
        return dataset
    
    def get_custom_objects(self):
        """Get custom objects dictionary for model saving/loading"""
        return {
            'RandomRotation': layers.RandomRotation,
            'RandomTranslation': layers.RandomTranslation,
            'RandomZoom': layers.RandomZoom,
            'RandomContrast': layers.RandomContrast,
            'Precision': Precision,
            'Recall': Recall
        }
    
    def create_data_augmentation(self):
        """Create data augmentation using keras.Sequential for modern TensorFlow"""
        return keras.Sequential([
            layers.RandomRotation(0.08),  # 15 degrees = ~0.08 in factor
            layers.RandomTranslation(0.05, 0.05),  # Small shifts
            layers.RandomZoom(0.05),  # Minimal zoom
            layers.RandomContrast(0.1),  # Add contrast variation
        ], name='data_augmentation')
    
    def build_model(self):
        """
        Build the CNN model using transfer learning
        """
        # Use MobileNetV2 as base model (good for this type of detection)
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_pins, activation='sigmoid')  # Sigmoid for multi-label classification
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # For multi-label classification
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        print("✅ Model built successfully")
        print(f"📊 Model has {model.count_params():,} parameters")
        return model
    
    def train_model(self, dataset_dir='data/processed', epochs=50, validation_split=0.2, batch_size=16):
        """
        Train the model on the dataset
        
        Args:
            dataset_dir: Directory containing processed dataset
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            batch_size: Training batch size
        """
        # Load dataset
        if not os.path.exists(f"{dataset_dir}/images.npy"):
            print("❌ Dataset not found. Create dataset first.")
            return None
            
        images = np.load(f"{dataset_dir}/images.npy")
        labels = np.load(f"{dataset_dir}/labels.npy")
        
        print(f"📊 Dataset shape: Images {images.shape}, Labels {labels.shape}")
        
        if len(images) == 0:
            print("❌ Empty dataset")
            return None
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42
        )
        
        print(f"📊 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create data augmentation
        data_augmentation = self.create_data_augmentation()
        
        # Apply augmentation to training data
        def augment_data(images, labels):
            augmented_images = data_augmentation(images, training=True)
            return augmented_images, labels
        
        # Create tf.data.Dataset for efficient training
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model_savedmodel',
                monitor='val_loss',
                save_best_only=True,
                save_format='tf',  # Use SavedModel format for better compatibility
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train model
        print("🚀 Starting training...")
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def fine_tune_model(self, epochs=20):
        """
        Fine-tune the model by unfreezing some base model layers
        """
        if not self.model:
            print("❌ No model found. Train the model first.")
            return
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        print("🔧 Fine-tuning model...")
        print(f"🔓 Unfroze {len([l for l in base_model.layers[fine_tune_at:] if l.trainable])} layers")
    
    def evaluate_model(self, dataset_dir='data/processed'):
        """
        Evaluate the model performance
        """
        if not self.model:
            print("❌ No model found. Train the model first.")
            return None
        
        # Load test data
        images = np.load(f"{dataset_dir}/images.npy")
        labels = np.load(f"{dataset_dir}/labels.npy")
        
        # Make predictions
        predictions = self.model.predict(images, verbose=1)
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate accuracy per pin
        pin_accuracies = []
        print("\n📊 Pin-by-pin accuracy:")
        for pin_idx in range(self.num_pins):
            pin_accuracy = np.mean(labels[:, pin_idx] == binary_predictions[:, pin_idx])
            pin_accuracies.append(pin_accuracy)
            print(f"Pin {pin_idx + 1:2d}: {pin_accuracy:.3f}")
        
        # Overall accuracy
        overall_accuracy = np.mean(labels == binary_predictions)
        
        # Exact match accuracy (all pins correct)
        exact_matches = np.mean(np.all(labels == binary_predictions, axis=1))
        
        print(f"\n🎯 Overall pin accuracy: {overall_accuracy:.3f}")
        print(f"🎯 Exact frame accuracy: {exact_matches:.3f}")
        
        return {
            'pin_accuracies': pin_accuracies,
            'overall_accuracy': overall_accuracy,
            'exact_match_accuracy': exact_matches,
            'predictions': predictions,
            'binary_predictions': binary_predictions
        }
    
    def visualize_predictions(self, dataset_dir='data/processed', num_samples=10):
        """
        Visualize model predictions on sample images
        """
        if not self.model:
            print("❌ No model found. Train the model first.")
            return
        
        # Load data
        images = np.load(f"{dataset_dir}/images.npy")
        labels = np.load(f"{dataset_dir}/labels.npy")
        
        with open(f"{dataset_dir}/dataset.json", 'r') as f:
            metadata = json.load(f)
        
        # Select random samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        # Make predictions
        predictions = self.model.predict(images[indices])
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Visualize
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Display image
            ax.imshow(images[idx])
            
            # Create title with prediction vs actual
            true_pins = labels[idx]
            pred_pins = binary_predictions[i]
            
            title = f"File: {metadata[idx]['filename']}\n"
            title += f"True pins: {true_pins} (Standing: {sum(true_pins)})\n"
            title += f"Pred pins: {pred_pins} (Standing: {sum(pred_pins)})\n"
            title += f"Match: {'✅' if np.array_equal(true_pins, pred_pins) else '❌'}"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/predictions_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved to models/predictions_visualization.png")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("❌ No training history found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0,0].plot(self.history.history['loss'], label='Training Loss')
        axes[0,0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        
        # Accuracy
        axes[0,1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1,0].plot(self.history.history['precision'], label='Training Precision')
            axes[1,0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1,0].set_title('Model Precision')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1,1].plot(self.history.history['recall'], label='Training Recall')
            axes[1,1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1,1].set_title('Model Recall')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Recall')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Training history saved to models/training_history.png")
    
    def save_model(self, filepath='models/duckpin_detector.h5', use_savedmodel=False):
        """Save the trained model with custom object scope"""
        if not self.model:
            print("❌ No model found. Train the model first.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if use_savedmodel:
            # Use SavedModel format for better compatibility
            savedmodel_path = filepath.replace('.h5', '_savedmodel')
            self.model.save(savedmodel_path, save_format='tf')
            print(f"💾 Model saved to {savedmodel_path} (SavedModel format)")
        else:
            # Save with custom objects in H5 format
            with keras.utils.custom_object_scope(self.get_custom_objects()):
                self.model.save(filepath)
            print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='models/duckpin_detector.h5'):
        """Load a trained model with custom object scope"""
        try:
            # Check if it's a SavedModel directory
            if os.path.isdir(filepath):
                self.model = keras.models.load_model(filepath)
                print(f"📁 Model loaded from {filepath} (SavedModel format)")
                return True
            else:
                # Load H5 format with custom objects
                with keras.utils.custom_object_scope(self.get_custom_objects()):
                    self.model = keras.models.load_model(filepath)
                print(f"📁 Model loaded from {filepath}")
                return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try using SavedModel format instead of H5 for better compatibility")
            return False
    
    def predict_single_image(self, image_path):
        """
        Predict pin states for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results
        """
        if not self.model:
            print("❌ No model found. Train or load a model first.")
            return None
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # Preprocess
            image = cv2.resize(image, self.image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(image_batch)[0]
            binary_prediction = (prediction > 0.5).astype(int)
            
            return {
                'probabilities': prediction.tolist(),
                'binary_prediction': binary_prediction.tolist(),
                'standing_pins': int(sum(binary_prediction)),
                'confidence_scores': prediction.tolist(),
                'pin_decimal': sum(binary_prediction[i] * (2**(9-i)) for i in range(10))
            }
            
        except Exception as e:
            print(f"❌ Error predicting image {image_path}: {e}")
            return None

def main():
    """Main function to demonstrate the DuckpinPinDetector usage"""
    print("🎳 Duckpin Pin Detection AI System")
    print("=" * 50)
    
    # Check TensorFlow installation first
    if not check_tensorflow_installation():
        print("\n❌ Cannot proceed without TensorFlow. Please install it first.")
        return
    
    # Initialize detector
    detector = DuckpinPinDetector()
    
    # Download images from Azure (limit for testing)
    print("\n📥 Downloading images from Azure...")
    downloaded_files = detector.download_images_from_azure(limit=50)  # Start small
    
    if not downloaded_files:
        print("❌ No images downloaded. Check Azure connection and data.")
        return
    
    # Create training dataset
    print("\n🔄 Creating training dataset...")
    dataset = detector.create_training_dataset()
    
    if not dataset:
        print("❌ Failed to create dataset.")
        return
    
    # Build model
    print("\n🏗️  Building model...")
    detector.build_model()
    
    # Train model
    print("\n🚀 Training model...")
    detector.train_model(epochs=30, batch_size=8)  # Smaller batch for limited data
    
    # Plot training history
    print("\n📈 Plotting training history...")
    detector.plot_training_history()
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    results = detector.evaluate_model()
    
    # Visualize predictions
    print("\n🖼️  Visualizing predictions...")
    detector.visualize_predictions()
    
    # Save model
    print("\n💾 Saving model...")
    detector.save_model()
    
    print("\n✅ Training complete! Model saved and ready for use.")
    print("\nTo use the model for prediction:")
    print("detector.load_model('models/duckpin_detector.h5')")
    print("result = detector.predict_single_image('path/to/image.jpg')")

if __name__ == "__main__":
    main()