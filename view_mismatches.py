
# View Mismatched Predictions Script
# This script specifically shows images where the AI model made incorrect predictions

import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
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

def get_custom_objects():
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
        try:
            from tensorflow.keras.layers import DepthwiseConv2D
        except ImportError:
            from keras.layers import DepthwiseConv2D
        
        class CompatibleDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)  # Remove incompatible parameter
                super().__init__(*args, **kwargs)
        
        custom_objects['DepthwiseConv2D'] = CompatibleDepthwiseConv2D
    except ImportError:
        pass
    
    return custom_objects

def load_model(model_path='models/duckpin_detector_compatible.h5'):
    """Load the trained model with compatibility handling"""
    try:
        # Strategy 1: Try SavedModel format
        if os.path.isdir(model_path):
            model = keras.models.load_model(model_path)
            print(f"📁 Model loaded (SavedModel)")
            return model
        
        # Strategy 2: Try H5 with custom objects
        try:
            with keras.utils.custom_object_scope(get_custom_objects()):
                model = keras.models.load_model(model_path)
            print(f"📁 Model loaded (H5 with custom objects)")
            return model
        except Exception as e1:
            print(f"⚠️  Custom objects failed: {str(e1)[:50]}...")
            
            # Strategy 3: Try without compilation
            try:
                model = keras.models.load_model(model_path, compile=False)
                # Recompile with current version
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', Precision(), Recall()]
                )
                print(f"📁 Model loaded (recompiled)")
                return model
            except Exception as e2:
                print(f"⚠️  Recompilation failed: {str(e2)[:50]}...")
        
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running fix_model_compatibility.py first")
        return None

def parse_filename(filename):
    """Parse filename to get pin configuration"""
    try:
        if "finalframe_pins_" in filename:
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
        
        return [0] * 10, [0] * 10, datetime.now()
    except:
        return [0] * 10, [0] * 10, datetime.now()

def predict_image(model, image_path):
    """Make prediction on image"""
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        image_batch = np.expand_dims(image, axis=0)
        prediction = model.predict(image_batch, verbose=0)[0]
        binary_prediction = (prediction > 0.5).astype(int)
        
        return binary_prediction, prediction
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None, None

def create_pin_diagram(ax, true_pins, pred_pins):
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

def show_mismatch(image_path, filename, true_pins, pred_pins, timestamp, probabilities):
    """Show a single mismatched prediction with image"""
    print(f"\n🖼️  ANALYZING MISMATCH")
    print("-" * 40)
    print(f"📅 Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 File: {filename}")
    print(f"🎯 True pins:  {true_pins} (standing: {sum(true_pins)})")
    print(f"🤖 Pred pins:  {pred_pins} (standing: {sum(pred_pins)})")
    print(f"📊 Avg Confidence: {np.mean(probabilities):.3f}")
    
    # Calculate differences
    differences = []
    for i, (true_val, pred_val) in enumerate(zip(true_pins, pred_pins)):
        if true_val != pred_val:
            status = "Wrong Standing" if pred_val == 1 else "Wrong Fallen"
            confidence = probabilities[i]
            differences.append(f"Pin {i+1}: {status} (conf: {confidence:.3f})")
    
    print(f"❌ Errors: {', '.join(differences)}")
    
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
        create_pin_diagram(ax2, true_pins, pred_pins)
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user input before continuing
        input("Press Enter to continue...")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error displaying image: {e}")

def view_mismatches():
    """Main function to view mismatched predictions"""
    print("🎯 Duckpin AI Mismatch Viewer")
    print("-" * 40)
    
    # Load model
    model = load_model()
    if not model:
        return
    
    # Setup Azure connection
    try:
        account_name = credentials.STORAGE_ACCOUNT_NAME
        account_key = credentials.STORAGE_ACCOUNT_KEY
        account_url = "https://duckpinjson.blob.core.windows.net"
        
        azure_credential = AzureNamedKeyCredential(account_name, account_key)
        blob_service_client = BlobServiceClient(account_url, credential=azure_credential)
        container_client = blob_service_client.get_container_client('jsoncontdp')
        
        print("✅ Azure connection established")
    except Exception as e:
        print(f"❌ Azure connection error: {e}")
        return
    
    # Get recent images
    test_dir = 'mismatch_images'
    os.makedirs(test_dir, exist_ok=True)
    
    blob_list = list(container_client.list_blobs(name_starts_with="Lane4Free/images/"))
    recent_blobs = sorted(blob_list, key=lambda x: x.last_modified, reverse=True)[:20]
    
    print(f"🔍 Checking {len(recent_blobs)} most recent images for mismatches...")
    
    mismatches_found = 0
    images_checked = 0
    
    for blob in recent_blobs:
        if blob.name.endswith(('.jpg', '.jpeg', '.png')):
            filename = os.path.basename(blob.name)
            local_path = os.path.join(test_dir, filename)
            
            # Download image
            try:
                with open(local_path, "wb") as download_file:
                    download_stream = container_client.download_blob(blob.name)
                    download_file.write(download_stream.readall())
            except:
                continue
            
            # Get ground truth
            _, true_pins, timestamp = parse_filename(filename)
            
            # Make prediction
            pred_pins, probabilities = predict_image(model, local_path)
            
            if pred_pins is not None:
                images_checked += 1
                exact_match = np.array_equal(true_pins, pred_pins)
                
                if not exact_match:
                    mismatches_found += 1
                    print(f"\n🔍 Found mismatch #{mismatches_found}")
                    show_mismatch(local_path, filename, true_pins, pred_pins, timestamp, probabilities)
                    
                    # Ask if user wants to continue
                    if mismatches_found >= 5:
                        response = input("\nContinue looking for more mismatches? (y/n): ").lower().strip()
                        if response not in ['y', 'yes']:
                            break
    
    print(f"\n📊 SUMMARY:")
    print(f"   Images checked: {images_checked}")
    print(f"   Mismatches found: {mismatches_found}")
    if images_checked > 0:
        accuracy = (images_checked - mismatches_found) / images_checked
        print(f"   Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    view_mismatches()