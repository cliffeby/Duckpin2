# Model Compatibility Checker and Fixer
# This script helps diagnose and fix model loading issues

import os
import sys

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

def check_model_compatibility(model_path='models/duckpin_detector.h5'):
    """Check model compatibility and suggest fixes"""
    print("🔍 Checking Model Compatibility")
    print("-" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📁 Model file found: {model_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Try different loading strategies
    strategies = [
        ("Direct loading", lambda: keras.models.load_model(model_path)),
        ("Without compilation", lambda: keras.models.load_model(model_path, compile=False)),
        ("With custom objects", lambda: load_with_custom_objects(model_path))
    ]
    
    for strategy_name, load_func in strategies:
        print(f"\n🔄 Trying: {strategy_name}")
        try:
            model = load_func()
            print(f"✅ Success with {strategy_name}")
            print(f"   Model input shape: {model.input_shape}")
            print(f"   Model output shape: {model.output_shape}")
            print(f"   Total parameters: {model.count_params():,}")
            return True
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")
    
    # If all strategies fail, try to inspect the model file
    print("\n🔧 Attempting model inspection...")
    try:
        inspect_model_file(model_path)
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
    
    return False

def load_with_custom_objects(model_path):
    """Load model with custom objects for compatibility"""
    custom_objects = {
        'RandomRotation': layers.RandomRotation,
        'RandomTranslation': layers.RandomTranslation,
        'RandomZoom': layers.RandomZoom,
        'RandomContrast': layers.RandomContrast,
        'Precision': Precision,
        'Recall': Recall
    }
    
    # Add DepthwiseConv2D compatibility
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
    
    with keras.utils.custom_object_scope(custom_objects):
        return keras.models.load_model(model_path)

def inspect_model_file(model_path):
    """Inspect the model file structure"""
    import h5py
    
    print("🔍 Inspecting HDF5 model structure...")
    
    with h5py.File(model_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 {name}/")
            else:
                print(f"📄 {name} - shape: {obj.shape if hasattr(obj, 'shape') else 'N/A'}")
        
        f.visititems(print_structure)

def convert_model_to_savedmodel(h5_path, savedmodel_path):
    """Convert H5 model to SavedModel format for better compatibility"""
    print(f"🔄 Converting {h5_path} to SavedModel format...")
    
    try:
        # Try to load the H5 model
        model = load_with_custom_objects(h5_path)
        
        # Save as SavedModel
        os.makedirs(savedmodel_path, exist_ok=True)
        model.save(savedmodel_path, save_format='tf')
        
        print(f"✅ Model converted and saved to {savedmodel_path}")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def fix_model_compatibility(model_path='models/duckpin_detector.h5'):
    """Attempt to fix model compatibility issues"""
    print("🔧 Attempting Model Compatibility Fixes")
    print("-" * 40)
    
    # Strategy 1: Convert to SavedModel format
    savedmodel_path = model_path.replace('.h5', '_savedmodel')
    if convert_model_to_savedmodel(model_path, savedmodel_path):
        print(f"💡 Use this path instead: {savedmodel_path}")
        return savedmodel_path
    
    # Strategy 2: Create a new compatible model and transfer weights
    print("\n🔄 Attempting weight transfer to new model...")
    try:
        # Load the problematic model without compilation
        old_model = keras.models.load_model(model_path, compile=False)
        
        # Create a new model with the same architecture
        new_model = create_compatible_model(old_model.input_shape[1:], old_model.output_shape[1])
        
        # Try to transfer weights
        new_model.set_weights(old_model.get_weights())
        
        # Save the new model
        new_path = model_path.replace('.h5', '_fixed.h5')
        new_model.save(new_path)
        
        print(f"✅ Fixed model saved to {new_path}")
        return new_path
        
    except Exception as e:
        print(f"❌ Weight transfer failed: {e}")
    
    return None

def create_compatible_model(input_shape, output_size):
    """Create a compatible model architecture"""
    from tensorflow.keras.applications import MobileNetV2
    
    # Create base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Add custom top layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(output_size, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    return model

def main():
    """Main function"""
    print("🎳 Duckpin Model Compatibility Checker")
    print("=" * 40)
    
    model_path = 'models/duckpin_detector.h5'
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("📁 Creating models directory...")
        os.makedirs('models', exist_ok=True)
    
    # Check compatibility
    if not check_model_compatibility(model_path):
        print("\n🔧 Model has compatibility issues. Attempting fixes...")
        
        fixed_path = fix_model_compatibility(model_path)
        
        if fixed_path:
            print(f"\n✅ Model fixed! Use this path: {fixed_path}")
        else:
            print("\n❌ Could not fix model automatically.")
            print("💡 Consider retraining the model with the current TensorFlow version.")
    else:
        print("\n✅ Model is compatible with current TensorFlow version!")

if __name__ == "__main__":
    main()