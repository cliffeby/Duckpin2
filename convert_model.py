# Advanced Model Converter for TensorFlow/Keras Compatibility
# This script specifically handles the DepthwiseConv2D 'groups' parameter issue

import os
import h5py
import json
import numpy as np

# Import TensorFlow with proper error handling
try:
    import tensorflow as tf
    try:
        import keras
        from keras import layers
        from keras.metrics import Precision, Recall
        from keras.applications import MobileNetV2
        print(f"✅ Using Keras {keras.__version__}")
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.metrics import Precision, Recall
        from tensorflow.keras.applications import MobileNetV2
        print(f"✅ Using TensorFlow.Keras {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    exit(1)

def create_new_compatible_model(input_shape=(224, 224, 3), num_classes=10):
    """Create a new compatible model with the same architecture"""
    print("🔧 Creating new compatible model...")
    
    # Create base model (MobileNetV2)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create the full model
    inputs = keras.Input(shape=input_shape)
    
    # Add data augmentation layers
    x = layers.RandomRotation(0.2)(inputs)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # Apply base model
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Add classification layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    return model

def extract_custom_weights(old_model_path):
    """Extract weights from the problematic model"""
    print("🔍 Extracting weights from old model...")
    
    weights_data = {}
    
    with h5py.File(old_model_path, 'r') as f:
        # Extract custom layer weights (not MobileNetV2)
        custom_layers = [
            'batch_normalization',
            'dense', 'dense_1', 'dense_2'
        ]
        
        for layer_name in custom_layers:
            layer_path = f"model_weights/{layer_name}"
            if layer_path in f:
                print(f"📦 Extracting {layer_name} weights...")
                layer_group = f[layer_path]
                
                weights_data[layer_name] = {}
                
                def extract_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weights_data[layer_name][name] = obj[:]
                
                layer_group.visititems(extract_weights)
    
    return weights_data

def transfer_compatible_weights(new_model, weights_data):
    """Transfer compatible weights to the new model"""
    print("🔄 Transferring compatible weights...")
    
    transferred_count = 0
    
    # Try to transfer dense layer weights
    for layer_name in ['dense', 'dense_1', 'dense_2']:
        if layer_name in weights_data:
            try:
                # Find the corresponding layer in the new model
                new_layer = None
                for layer in new_model.layers:
                    if layer.name == layer_name:
                        new_layer = layer
                        break
                
                if new_layer is None:
                    print(f"⚠️  Layer {layer_name} not found in new model")
                    continue
                
                # Get weights from extracted data
                layer_weights = weights_data[layer_name]
                
                # Look for kernel and bias in the nested structure
                kernel_data = None
                bias_data = None
                
                # Search through the nested structure
                for key, value in layer_weights.items():
                    if isinstance(value, dict):
                        if 'kernel:0' in value:
                            kernel_data = value['kernel:0']
                        if 'bias:0' in value:
                            bias_data = value['bias:0']
                    elif key.endswith('kernel:0'):
                        kernel_data = value
                    elif key.endswith('bias:0'):
                        bias_data = value
                
                if kernel_data is not None and bias_data is not None:
                    # Check if shapes match
                    expected_shapes = [w.shape for w in new_layer.get_weights()]
                    actual_shapes = [kernel_data.shape, bias_data.shape]
                    
                    if len(expected_shapes) == 2 and expected_shapes == actual_shapes:
                        new_layer.set_weights([kernel_data, bias_data])
                        print(f"✅ Transferred weights for {layer_name}")
                        transferred_count += 1
                    else:
                        print(f"⚠️  Shape mismatch for {layer_name}: expected {expected_shapes}, got {actual_shapes}")
                else:
                    print(f"⚠️  Incomplete weights for {layer_name}")
                        
            except Exception as e:
                print(f"⚠️  Failed to transfer {layer_name}: {e}")
                continue
    
    # Try to transfer batch normalization weights
    if 'batch_normalization' in weights_data:
        try:
            # Find batch normalization layers in new model
            bn_layers = [layer for layer in new_model.layers if 'batch_normalization' in layer.name]
            
            if bn_layers:
                bn_layer = bn_layers[0]  # Use the first one
                layer_weights = weights_data['batch_normalization']
                
                # Look for BN parameters
                gamma_data = None
                beta_data = None
                mean_data = None
                var_data = None
                
                for key, value in layer_weights.items():
                    if isinstance(value, dict):
                        for param_key, param_value in value.items():
                            if param_key.endswith('gamma:0'):
                                gamma_data = param_value
                            elif param_key.endswith('beta:0'):
                                beta_data = param_value
                            elif param_key.endswith('moving_mean:0'):
                                mean_data = param_value
                            elif param_key.endswith('moving_variance:0'):
                                var_data = param_value
                
                if all(x is not None for x in [gamma_data, beta_data, mean_data, var_data]):
                    expected_shapes = [w.shape for w in bn_layer.get_weights()]
                    actual_shapes = [gamma_data.shape, beta_data.shape, mean_data.shape, var_data.shape]
                    
                    if len(expected_shapes) == 4 and expected_shapes == actual_shapes:
                        bn_layer.set_weights([gamma_data, beta_data, mean_data, var_data])
                        print(f"✅ Transferred weights for batch_normalization")
                        transferred_count += 1
                    else:
                        print(f"⚠️  Shape mismatch for batch_normalization")
                        
        except Exception as e:
            print(f"⚠️  Failed to transfer batch_normalization: {e}")
    
    print(f"📊 Successfully transferred weights for {transferred_count} layers")
    return transferred_count > 0

def convert_model(old_model_path='models/duckpin_detector.h5', 
                 new_model_path='models/duckpin_detector_fixed.h5'):
    """Convert the problematic model to a compatible version"""
    print("🚀 Starting Model Conversion")
    print("-" * 40)
    
    if not os.path.exists(old_model_path):
        print(f"❌ Old model not found: {old_model_path}")
        return False
    
    # Create new compatible model
    new_model = create_new_compatible_model()
    
    print(f"📋 New model summary:")
    print(f"   Input shape: {new_model.input_shape}")
    print(f"   Output shape: {new_model.output_shape}")
    print(f"   Total parameters: {new_model.count_params():,}")
    
    # Extract weights from old model
    weights_data = extract_custom_weights(old_model_path)
    
    if not weights_data:
        print("⚠️  No compatible weights found to transfer")
        print("💡 Saving model with ImageNet weights only...")
    else:
        # Transfer compatible weights
        if not transfer_compatible_weights(new_model, weights_data):
            print("⚠️  Weight transfer failed, using ImageNet weights only")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    
    # Save the new model
    try:
        new_model.save(new_model_path)
        print(f"✅ New compatible model saved: {new_model_path}")
        
        # Also save as SavedModel format (use export instead)
        savedmodel_path = new_model_path.replace('.h5', '_savedmodel')
        try:
            tf.saved_model.save(new_model, savedmodel_path)
            print(f"✅ SavedModel format saved: {savedmodel_path}")
        except Exception as e:
            print(f"⚠️  SavedModel save failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def test_converted_model(model_path):
    """Test the converted model"""
    print("\n🧪 Testing Converted Model")
    print("-" * 30)
    
    try:
        # Try loading the model
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        
        # Test prediction
        dummy_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"📊 Prediction test:")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"   Sample prediction: {prediction[0][:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main conversion function"""
    print("🎳 Duckpin Model Compatibility Converter")
    print("=" * 45)
    
    old_model_path = 'models/duckpin_detector.h5'
    new_model_path = 'models/duckpin_detector_compatible.h5'
    
    # Convert the model
    if convert_model(old_model_path, new_model_path):
        # Test the converted model
        if test_converted_model(new_model_path):
            print("\n🎉 SUCCESS!")
            print(f"📁 Use this model path: {new_model_path}")
            print(f"📁 Or SavedModel: {new_model_path.replace('.h5', '_savedmodel')}")
            print("\n💡 Update your testing scripts to use the new model path.")
        else:
            print("\n⚠️  Model converted but testing failed")
    else:
        print("\n❌ Model conversion failed")

if __name__ == "__main__":
    main()