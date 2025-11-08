# Duckpin AI Model Testing and Validation Program
# This program tests the accuracy of the model created in duckpin_ai_detector.py
# by comparing predictions against ground truth from Azure Blob Storage images
# and generating comprehensive accuracy reports.

import os
import cv2
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from matplotlib.patches import Rectangle
import credentials

# Import TensorFlow with proper error handling
try:
    import tensorflow as tf
    try:
        import keras
        from keras import layers
        from keras.metrics import Precision, Recall
        print(f"✅ Using Keras {keras.__version__}")
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.metrics import Precision, Recall
        print(f"✅ Using TensorFlow.Keras {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    exit(1)

class DuckpinModelTester:
    """
    Test and validate the accuracy of the trained duckpin pin detection model
    """
    
    def __init__(self, model_path='models/duckpin_detector_compatible.h5'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.image_size = (224, 224)
        self.num_pins = 10
        
        # Azure connection
        self.setup_azure_connection()
        
        # Results storage
        self.test_results = []
        self.accuracy_metrics = {}
        
    def setup_azure_connection(self):
        """Setup Azure Blob Storage connection"""
        try:
            account_name = credentials.STORAGE_ACCOUNT_NAME
            account_key = credentials.STORAGE_ACCOUNT_KEY
            account_url = "https://duckpinjson.blob.core.windows.net"
            
            self.azure_credential = AzureNamedKeyCredential(account_name, account_key)
            self.blob_service_client = BlobServiceClient(account_url, credential=self.azure_credential)
            self.container_name = 'jsoncontdp'
            print("✅ Azure connection established")
            
        except Exception as e:
            print(f"❌ Error setting up Azure connection: {e}")
            self.blob_service_client = None
    
    def get_custom_objects(self):
        """Get custom objects for model loading"""
        custom_objects = {
            'RandomRotation': layers.RandomRotation,
            'RandomTranslation': layers.RandomTranslation,
            'RandomZoom': layers.RandomZoom,
            'RandomContrast': layers.RandomContrast,
            'Precision': Precision,
            'Recall': Recall
        }
        
        # Add compatibility for DepthwiseConv2D groups parameter
        try:
            # Try both import paths
            try:
                from tensorflow.keras.layers import DepthwiseConv2D
            except ImportError:
                from keras.layers import DepthwiseConv2D
            
            class CompatibleDepthwiseConv2D(DepthwiseConv2D):
                def __init__(self, *args, **kwargs):
                    # Remove the 'groups' parameter if it exists (not supported in older versions)
                    kwargs.pop('groups', None)
                    super().__init__(*args, **kwargs)
            
            custom_objects['DepthwiseConv2D'] = CompatibleDepthwiseConv2D
            
        except ImportError:
            pass
            
        return custom_objects
    
    def load_model(self):
        """Load the trained model with multiple fallback strategies"""
        try:
            # Strategy 1: Try loading as SavedModel format
            if os.path.isdir(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"📁 Model loaded from {self.model_path} (SavedModel)")
                return True
            
            # Strategy 2: Try loading H5 with custom objects
            try:
                with keras.utils.custom_object_scope(self.get_custom_objects()):
                    self.model = keras.models.load_model(self.model_path)
                print(f"📁 Model loaded from {self.model_path} (H5 with custom objects)")
                return True
            except Exception as e1:
                print(f"⚠️  Custom objects loading failed: {e1}")
                
                # Strategy 3: Try loading without custom objects
                try:
                    self.model = keras.models.load_model(self.model_path, compile=False)
                    print(f"📁 Model loaded from {self.model_path} (H5 without compilation)")
                    
                    # Recompile the model with current TensorFlow version
                    self.model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', Precision(), Recall()]
                    )
                    print("🔧 Model recompiled with current TensorFlow version")
                    return True
                    
                except Exception as e2:
                    print(f"⚠️  Standard loading failed: {e2}")
                    
                    # Strategy 4: Load weights only (requires recreating architecture)
                    try:
                        print("🔄 Attempting to load weights only...")
                        return self._load_weights_only()
                    except Exception as e3:
                        print(f"⚠️  Weights-only loading failed: {e3}")
            
            return False
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try regenerating the model or check the file path")
            return False
    
    def _load_weights_only(self):
        """Fallback method to load weights into a new model architecture"""
        try:
            # Import the DuckpinPinDetector class to recreate the model
            import importlib.util
            spec = importlib.util.spec_from_file_location("duckpin_ai", "duckpin_ai_detector.py")
            if spec and spec.loader:
                duckpin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(duckpin_module)
                
                # Create a new detector instance
                detector = duckpin_module.DuckpinPinDetector()
                detector.create_model()
                
                # Try to load weights
                weights_path = self.model_path.replace('.h5', '_weights.h5')
                if os.path.exists(weights_path):
                    detector.model.load_weights(weights_path)
                    self.model = detector.model
                    print(f"📁 Weights loaded from {weights_path}")
                    return True
                else:
                    # Try loading weights from the main model file
                    detector.model.load_weights(self.model_path)
                    self.model = detector.model
                    print(f"📁 Weights extracted from {self.model_path}")
                    return True
                    
        except Exception as e:
            print(f"❌ Weights-only loading failed: {e}")
            return False
    
    def parse_filename_to_pin_state(self, filename):
        """
        Parse filename to extract pin configuration
        
        Args:
            filename: e.g., "finalframe_pins_1023_to_512_20251106_143052_834567.jpg"
            
        Returns:
            tuple: (beginning_pins, ending_pins, timestamp) as binary arrays and datetime
        """
        try:
            if "finalframe_pins_" in filename:
                # Remove extension and prefix
                name_part = filename.replace("finalframe_pins_", "").replace(".jpg", "").replace(".png", "")
                parts = name_part.split("_")
                
                if len(parts) >= 5 and parts[1] == "to":
                    beginning_decimal = int(parts[0])
                    ending_decimal = int(parts[2])
                    
                    # Extract timestamp
                    date_part = parts[3]  # YYYYMMDD
                    time_part = parts[4]  # HHMMSS
                    
                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                    
                    # Convert to binary
                    beginning_binary = [int(bit) for bit in format(beginning_decimal, '010b')]
                    ending_binary = [int(bit) for bit in format(ending_decimal, '010b')]
                    
                    return beginning_binary, ending_binary, timestamp
            
            print(f"⚠️  Cannot parse filename: {filename}")
            return [0] * self.num_pins, [0] * self.num_pins, datetime.now()
            
        except Exception as e:
            print(f"❌ Error parsing filename {filename}: {e}")
            return [0] * self.num_pins, [0] * self.num_pins, datetime.now()
    
    def download_test_images(self, after_date=None, limit=None, local_dir='test_data/images'):
        """
        Download images from Azure for testing
        
        Args:
            after_date: Only download images created after this date (datetime object)
            limit: Maximum number of images to download
            local_dir: Local directory to save images
            
        Returns:
            list: Downloaded image file paths with metadata
        """
        if not self.blob_service_client:
            print("❌ Azure connection not available")
            return []
        
        os.makedirs(local_dir, exist_ok=True)
        downloaded_files = []
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with="Lane4Free/images/")
            
            print(f"🔍 Scanning images after {after_date if after_date else 'any date'}...")
            
            count = 0
            for blob in blob_list:
                if limit and count >= limit:
                    break
                
                if blob.name.endswith(('.jpg', '.jpeg', '.png')):
                    filename = os.path.basename(blob.name)
                    
                    # Parse filename to get timestamp
                    _, _, timestamp = self.parse_filename_to_pin_state(filename)
                    
                    # Filter by date if specified
                    if after_date and timestamp < after_date:
                        continue
                    
                    local_filepath = os.path.join(local_dir, filename)
                    
                    # Download if not already exists
                    if not os.path.exists(local_filepath):
                        with open(local_filepath, "wb") as download_file:
                            download_stream = container_client.download_blob(blob.name)
                            download_file.write(download_stream.readall())
                    
                    downloaded_files.append({
                        'filepath': local_filepath,
                        'filename': filename,
                        'timestamp': timestamp,
                        'azure_path': blob.name
                    })
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"📥 Downloaded {count} images...")
            
            print(f"✅ Downloaded {len(downloaded_files)} images for testing")
            return downloaded_files
            
        except Exception as e:
            print(f"❌ Error downloading test images: {e}")
            return []
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array: Preprocessed image ready for prediction
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Resize and convert color space
            image = cv2.resize(image, self.image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_image(self, image_path):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results
        """
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return None
        
        try:
            # Add batch dimension and predict
            image_batch = np.expand_dims(image, axis=0)
            prediction = self.model.predict(image_batch, verbose=0)[0]
            binary_prediction = (prediction > 0.5).astype(int)
            
            return {
                'probabilities': prediction,
                'binary_prediction': binary_prediction,
                'standing_pins': int(sum(binary_prediction)),
                'confidence': float(np.mean(prediction))
            }
            
        except Exception as e:
            print(f"❌ Error predicting image {image_path}: {e}")
            return None
    
    def run_test_suite(self, after_date=None, limit=100):
        """
        Run comprehensive test suite on images
        
        Args:
            after_date: Test images created after this date
            limit: Maximum number of images to test
            
        Returns:
            dict: Test results and metrics
        """
        print("🧪 Starting model test suite...")
        
        # Load model
        if not self.load_model():
            return None
        
        # Download test images
        test_images = self.download_test_images(after_date=after_date, limit=limit)
        
        if not test_images:
            print("❌ No test images available")
            return None
        
        print(f"🎯 Testing on {len(test_images)} images...")
        
        # Run predictions
        self.test_results = []
        
        for i, image_info in enumerate(test_images):
            if i % 20 == 0:
                print(f"🔄 Processing image {i+1}/{len(test_images)}")
            
            # Get ground truth from filename
            beginning_pins, ending_pins, timestamp = self.parse_filename_to_pin_state(
                image_info['filename']
            )
            
            # Make prediction
            prediction = self.predict_image(image_info['filepath'])
            
            if prediction is None:
                continue
            
            # Calculate accuracy metrics for this image
            true_pins = np.array(ending_pins)
            pred_pins = prediction['binary_prediction']
            
            # Pin-by-pin accuracy
            pin_accuracy = np.mean(true_pins == pred_pins)
            
            # Exact match (all pins correct)
            exact_match = np.array_equal(true_pins, pred_pins)
            
            # Standing pin count accuracy
            true_standing = sum(true_pins)
            pred_standing = prediction['standing_pins']
            count_accuracy = true_standing == pred_standing
            
            # Store results
            result = {
                'filename': image_info['filename'],
                'timestamp': timestamp,
                'true_beginning': beginning_pins,
                'true_ending': ending_pins,
                'predicted_ending': pred_pins.tolist(),
                'true_standing_count': true_standing,
                'pred_standing_count': pred_standing,
                'probabilities': prediction['probabilities'].tolist(),
                'confidence': prediction['confidence'],
                'pin_accuracy': pin_accuracy,
                'exact_match': exact_match,
                'count_accuracy': count_accuracy
            }
            
            self.test_results.append(result)
        
        # Calculate overall metrics
        self.calculate_accuracy_metrics()
        
        print(f"✅ Test suite completed on {len(self.test_results)} images")
        return self.accuracy_metrics
    
    def show_mismatched_predictions(self, max_images=10):
        """Display images where predictions don't match ground truth"""
        if not self.test_results:
            print("❌ No test results available")
            return
        
        print(f"\n🔍 SHOWING MISMATCHED PREDICTIONS (up to {max_images} images)")
        print("=" * 60)
        
        mismatches = [result for result in self.test_results if not result['exact_match']]
        
        if not mismatches:
            print("✅ No mismatches found - all predictions were correct!")
            return
        
        print(f"Found {len(mismatches)} mismatched predictions out of {len(self.test_results)} total")
        
        # Show up to max_images mismatches
        for i, result in enumerate(mismatches[:max_images]):
            self._display_mismatch(result, i + 1)
            
        if len(mismatches) > max_images:
            print(f"\n... and {len(mismatches) - max_images} more mismatches (increase max_images to see more)")
    
    def _display_mismatch(self, result, mismatch_num):
        """Display a single mismatched prediction with image"""
        print(f"\n🖼️  MISMATCH #{mismatch_num}")
        print("-" * 40)
        
        # Extract info
        filename = result['filename']
        timestamp = result['timestamp']
        true_pins = result['true_ending']
        pred_pins = result['predicted_ending']
        confidence = result['confidence']
        
        # Find the image file
        image_path = None
        for test_dir in ['test_data/images', 'quick_test_images']:
            potential_path = os.path.join(test_dir, filename)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"⚠️  Image file not found: {filename}")
            return
        
        print(f"📅 Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 File: {filename}")
        print(f"🎯 True pins:  {true_pins} (standing: {sum(true_pins)})")
        print(f"🤖 Pred pins:  {pred_pins} (standing: {sum(pred_pins)})")
        print(f"📊 Confidence: {confidence:.3f}")
        
        # Calculate differences
        differences = []
        for i, (true_val, pred_val) in enumerate(zip(true_pins, pred_pins)):
            if true_val != pred_val:
                status = "Wrong Standing" if pred_val == 1 else "Wrong Fallen"
                differences.append(f"Pin {i+1}: {status}")
        
        print(f"❌ Errors: {', '.join(differences)}")
        
        # Display the image with annotations
        self._show_annotated_image(image_path, true_pins, pred_pins, filename, timestamp)
    
    def _show_annotated_image(self, image_path, true_pins, pred_pins, filename, timestamp):
        """Show image with pin state annotations"""
        try:
            # Load and display image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not load image: {image_path}")
                return
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Left subplot: Original image
            ax1.imshow(image_rgb)
            ax1.set_title(f'Original Image\n{filename}\n{timestamp.strftime("%Y-%m-%d %H:%M:%S")}', 
                         fontsize=10, pad=10)
            ax1.axis('off')
            
            # Right subplot: Pin state visualization
            self._create_pin_diagram(ax2, true_pins, pred_pins)
            
            plt.tight_layout()
            plt.show()
            
            # Wait for user input before showing next image
            input("Press Enter to continue to next mismatch...")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
    
    def _create_pin_diagram(self, ax, true_pins, pred_pins):
        """Create a visual diagram showing pin states"""
        # Standard duckpin layout (triangular arrangement)
        # Pin 1 is centered in bottom row, then 2-3, 4-5-6, 7-8-9-10
        visual_positions = [
            (2.0, 0.5),    # Pin 1 (bottom row, center - only pin)
            (1.5, 1.5),    # Pin 2 (second row, left)
            (2.5, 1.5),    # Pin 3 (second row, right)
            (1.0, 2.5),    # Pin 4 (third row, left)
            (2.0, 2.5),    # Pin 5 (third row, center)
            (3.0, 2.5),    # Pin 6 (third row, right)
            (0.5, 3.5),    # Pin 7 (top row, far left)
            (1.5, 3.5),    # Pin 8 (top row, left)
            (2.5, 3.5),    # Pin 9 (top row, right)
            (3.5, 3.5),    # Pin 10 (top row, far right)
        ]
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.0)
        ax.set_aspect('equal')
        
        # Draw pins - i represents the pin number (0-based index for pin 1-10)
        for i in range(10):
            pin_number = i + 1  # Actual pin number (1-10)
            x, y = visual_positions[i]
            true_val = true_pins[i]
            pred_val = pred_pins[i]
            
            # Determine color based on correctness
            if true_val == pred_val:
                if true_val == 1:
                    color = 'green'  # Correct standing
                    marker = 'o'
                else:
                    color = 'lightgray'  # Correct fallen
                    marker = 'x'
            else:
                if pred_val == 1:
                    color = 'red'  # Wrong: predicted standing
                    marker = 'o'
                else:
                    color = 'orange'  # Wrong: predicted fallen
                    marker = 'x'
            
            # Draw pin
            ax.scatter(x, y, c=color, s=200, marker=marker, edgecolors='black', linewidth=2)
            
            # Add pin number
            ax.text(x, y-0.3, str(pin_number), ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Correct Standing'),
            plt.Line2D([0], [0], marker='x', color='lightgray', 
                      markersize=10, label='Correct Fallen'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Wrong Standing'),
            plt.Line2D([0], [0], marker='x', color='orange', 
                      markersize=10, label='Wrong Fallen')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Title and labels
        true_count = sum(true_pins)
        pred_count = sum(pred_pins)
        ax.set_title(f'Pin State Comparison\nTrue: {true_count} standing | Pred: {pred_count} standing', 
                    fontsize=12, pad=10)
        ax.set_xlabel('Duckpin Layout (Front → Back)')
        ax.axis('off')

    def calculate_accuracy_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        if not self.test_results:
            return
        
        # Convert to arrays for easier calculation
        all_true = np.array([result['true_ending'] for result in self.test_results])
        all_pred = np.array([result['predicted_ending'] for result in self.test_results])
        
        # Overall metrics
        overall_accuracy = np.mean(all_true == all_pred)
        exact_match_accuracy = np.mean([result['exact_match'] for result in self.test_results])
        count_accuracy = np.mean([result['count_accuracy'] for result in self.test_results])
        
        # Pin-by-pin accuracy
        pin_accuracies = []
        for pin_idx in range(self.num_pins):
            pin_acc = np.mean(all_true[:, pin_idx] == all_pred[:, pin_idx])
            pin_accuracies.append(pin_acc)
        
        # Confidence statistics
        confidences = [result['confidence'] for result in self.test_results]
        
        # Count distribution analysis
        true_counts = [result['true_standing_count'] for result in self.test_results]
        pred_counts = [result['pred_standing_count'] for result in self.test_results]
        
        self.accuracy_metrics = {
            'overall_pin_accuracy': overall_accuracy,
            'exact_match_accuracy': exact_match_accuracy,
            'standing_count_accuracy': count_accuracy,
            'pin_by_pin_accuracy': pin_accuracies,
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'total_images_tested': len(self.test_results),
            'true_count_distribution': np.bincount(true_counts, minlength=11),
            'pred_count_distribution': np.bincount(pred_counts, minlength=11)
        }
    
    def generate_report(self, output_dir='test_reports'):
        """Generate comprehensive accuracy report"""
        if not self.test_results:
            print("❌ No test results available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate text report
        self._generate_text_report(output_dir)
        
        # Generate visualizations
        self._generate_visualizations(output_dir)
        
        # Generate detailed CSV
        self._generate_csv_report(output_dir)
        
        print(f"📊 Reports generated in {output_dir}/")
    
    def _generate_text_report(self, output_dir):
        """Generate text-based accuracy report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"accuracy_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎳 DUCKPIN AI MODEL ACCURACY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Total Images Tested: {self.accuracy_metrics['total_images_tested']}\n\n")
            
            f.write("📊 OVERALL ACCURACY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Pin Accuracy: {self.accuracy_metrics['overall_pin_accuracy']:.3f}\n")
            f.write(f"Exact Match Accuracy: {self.accuracy_metrics['exact_match_accuracy']:.3f}\n")
            f.write(f"Standing Count Accuracy: {self.accuracy_metrics['standing_count_accuracy']:.3f}\n")
            f.write(f"Average Confidence: {self.accuracy_metrics['average_confidence']:.3f}\n\n")
            
            f.write("📍 PIN-BY-PIN ACCURACY\n")
            f.write("-" * 25 + "\n")
            for i, acc in enumerate(self.accuracy_metrics['pin_by_pin_accuracy']):
                f.write(f"Pin {i+1:2d}: {acc:.3f}\n")
            
            f.write("\n📈 STANDING PIN COUNT DISTRIBUTION\n")
            f.write("-" * 35 + "\n")
            f.write("Count | True | Pred\n")
            f.write("------|------|-----\n")
            true_dist = self.accuracy_metrics['true_count_distribution']
            pred_dist = self.accuracy_metrics['pred_count_distribution']
            for i in range(11):
                f.write(f"{i:5d} | {true_dist[i]:4d} | {pred_dist[i]:4d}\n")
        
        print(f"📄 Text report saved: {report_file}")
    
    def _generate_visualizations(self, output_dir):
        """Generate visualization plots"""
        plt.style.use('default')
        
        # Pin-by-pin accuracy plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        pin_accs = self.accuracy_metrics['pin_by_pin_accuracy']
        plt.bar(range(1, 11), pin_accs)
        plt.title('Pin-by-Pin Accuracy')
        plt.xlabel('Pin Number')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, acc in enumerate(pin_accs):
            plt.text(i+1, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Count distribution comparison
        plt.subplot(2, 2, 2)
        true_dist = self.accuracy_metrics['true_count_distribution']
        pred_dist = self.accuracy_metrics['pred_count_distribution']
        x = np.arange(11)
        width = 0.35
        plt.bar(x - width/2, true_dist, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_dist, width, label='Predicted', alpha=0.8)
        plt.title('Standing Pin Count Distribution')
        plt.xlabel('Number of Standing Pins')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Confidence distribution
        plt.subplot(2, 2, 3)
        confidences = [result['confidence'] for result in self.test_results]
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        # Accuracy over time
        plt.subplot(2, 2, 4)
        timestamps = [result['timestamp'] for result in self.test_results]
        accuracies = [result['pin_accuracy'] for result in self.test_results]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, accuracies))
        sorted_times, sorted_accs = zip(*sorted_data)
        
        plt.plot(sorted_times, sorted_accs, 'o-', alpha=0.7)
        plt.title('Accuracy Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Pin Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f"accuracy_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved: {plot_file}")
    
    def _generate_csv_report(self, output_dir):
        """Generate detailed CSV report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        
        # Convert results to DataFrame
        df_data = []
        for result in self.test_results:
            row = {
                'filename': result['filename'],
                'timestamp': result['timestamp'],
                'true_standing_count': result['true_standing_count'],
                'pred_standing_count': result['pred_standing_count'],
                'confidence': result['confidence'],
                'pin_accuracy': result['pin_accuracy'],
                'exact_match': result['exact_match'],
                'count_accuracy': result['count_accuracy']
            }
            
            # Add individual pin predictions
            for i in range(self.num_pins):
                row[f'true_pin_{i+1}'] = result['true_ending'][i]
                row[f'pred_pin_{i+1}'] = result['predicted_ending'][i]
                row[f'prob_pin_{i+1}'] = result['probabilities'][i]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        
        print(f"📈 CSV report saved: {csv_file}")

def main():
    """Main function to run model testing"""
    print("🎳 Duckpin AI Model Testing Suite")
    print("=" * 40)
    
    # Initialize tester
    tester = DuckpinModelTester()
    
    # Define test parameters
    # Test images from the last 7 days
    test_after_date = datetime.now() - timedelta(days=7)
    
    print(f"🔍 Testing images created after: {test_after_date.strftime('%Y-%m-%d')}")
    
    # Run test suite
    results = tester.run_test_suite(
        after_date=test_after_date,
        limit=50  # Test on up to 50 recent images
    )
    
    if results:
        print("\n📊 TEST RESULTS SUMMARY:")
        print(f"Overall Pin Accuracy: {results['overall_pin_accuracy']:.3f}")
        print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
        print(f"Standing Count Accuracy: {results['standing_count_accuracy']:.3f}")
        print(f"Average Confidence: {results['average_confidence']:.3f}")
        print(f"Images Tested: {results['total_images_tested']}")
        
        # Generate comprehensive report
        tester.generate_report()
        
        # Show mismatched predictions with images
        print("\n" + "="*60)
        response = input("Would you like to view mismatched predictions with images? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            max_images = input("How many mismatches to show? (default: 5): ").strip()
            try:
                max_images = int(max_images) if max_images else 5
            except ValueError:
                max_images = 5
            tester.show_mismatched_predictions(max_images=max_images)
        
        print("\n✅ Testing completed successfully!")
        print("📁 Check the test_reports/ directory for detailed analysis")
    else:
        print("❌ Testing failed or no results available")

if __name__ == "__main__":
    main()
