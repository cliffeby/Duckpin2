"""
Simple test script to verify TensorFlow and Keras imports are working correctly
"""

def test_tensorflow_imports():
    """Test TensorFlow imports and basic functionality"""
    print("🧪 Testing TensorFlow imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic TensorFlow operations
        test_tensor = tf.constant([1, 2, 3, 4])
        print(f"✅ Basic TensorFlow operations: {test_tensor}")
        
        # Try standalone Keras first
        try:
            import keras
            from keras import layers
            from keras.metrics import Precision, Recall
            print(f"✅ Standalone Keras version: {keras.__version__}")
            backend = "standalone"
        except ImportError:
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.metrics import Precision, Recall
            print(f"✅ TensorFlow.Keras available")
            backend = "tensorflow"
        
        # Test model creation
        print("🧪 Testing model creation...")
        
        # Simple test model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Test compilation with metrics
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        print(f"✅ Model created and compiled successfully")
        print(f"📊 Model parameters: {model.count_params():,}")
        
        # Test data augmentation layers
        print("🧪 Testing data augmentation...")
        
        augmentation = keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name='test_augmentation')
        
        print("✅ Data augmentation layers created successfully")
        
        # Test MobileNetV2
        print("🧪 Testing MobileNetV2...")
        
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        print("✅ MobileNetV2 loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_custom_objects():
    """Test custom object scope functionality"""
    print("\n🧪 Testing custom object scope...")
    
    try:
        import tensorflow as tf
        
        # Try standalone Keras first
        try:
            import keras
            from keras import layers
            from keras.metrics import Precision, Recall
        except ImportError:
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.metrics import Precision, Recall
        
        custom_objects = {
            'RandomRotation': layers.RandomRotation,
            'RandomTranslation': layers.RandomTranslation,
            'RandomZoom': layers.RandomZoom,
            'RandomContrast': layers.RandomContrast,
            'Precision': Precision,
            'Recall': Recall
        }
        
        # Test custom object scope
        with keras.utils.custom_object_scope(custom_objects):
            print("✅ Custom object scope created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom object scope error: {e}")
        return False

if __name__ == "__main__":
    print("🎳 TensorFlow Import Test")
    print("=" * 40)
    
    success1 = test_tensorflow_imports()
    success2 = test_custom_objects()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your environment is ready for the Duckpin AI Detector.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")