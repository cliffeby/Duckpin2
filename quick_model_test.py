# Quick Model Testing Script
# Simple version for testing the duckpin AI model with a few recent images

import os
import cv2
import numpy as np
from datetime import datetime, timedelta
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
            name_part = filename.replace("finalframe_pins_", "").replace(".jpg", "")
            parts = name_part.split("_")
            
            if len(parts) >= 3 and parts[1] == "to":
                ending_decimal = int(parts[2])
                ending_binary = [int(bit) for bit in format(ending_decimal, '010b')]
                return ending_binary
        return [0] * 10
    except:
        return [0] * 10

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

def quick_test():
    """Run a quick test on a few recent images"""
    print("🎳 Quick Duckpin AI Model Test")
    print("-" * 30)
    
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
    test_dir = 'quick_test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    blob_list = list(container_client.list_blobs(name_starts_with="Lane4Free/images/"))
    recent_blobs = sorted(blob_list, key=lambda x: x.last_modified, reverse=True)[:5]
    
    print(f"🔍 Testing on {len(recent_blobs)} most recent images...")
    
    correct_predictions = 0
    total_tests = 0
    
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
            true_pins = parse_filename(filename)
            
            # Make prediction
            pred_pins, probabilities = predict_image(model, local_path)
            
            if pred_pins is not None:
                # Calculate accuracy
                accuracy = np.mean(true_pins == pred_pins)
                exact_match = np.array_equal(true_pins, pred_pins)
                
                if exact_match:
                    correct_predictions += 1
                total_tests += 1
                
                print(f"\n📸 {filename}")
                print(f"   True:  {true_pins} (standing: {sum(true_pins)})")
                print(f"   Pred:  {pred_pins.tolist()} (standing: {sum(pred_pins)})")
                print(f"   Accuracy: {accuracy:.3f} | Exact Match: {'✅' if exact_match else '❌'}")
    
    if total_tests > 0:
        overall_accuracy = correct_predictions / total_tests
        print(f"\n📊 QUICK TEST RESULTS:")
        print(f"   Exact Match Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_tests})")
        print(f"   Model Status: {'✅ Good' if overall_accuracy > 0.7 else '⚠️  Needs Improvement'}")
    else:
        print("❌ No images could be tested")

if __name__ == "__main__":
    quick_test()