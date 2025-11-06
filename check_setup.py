"""
Duckpin AI Detector - Installation Check and Setup

Run this script first to verify your environment is ready for the AI detector.
"""

def check_all_dependencies():
    """Check all required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('tensorflow', 'pip install tensorflow'),
        ('opencv-python', 'pip install opencv-python'),
        ('numpy', 'pip install numpy'),
        ('matplotlib', 'pip install matplotlib'),
        ('scikit-learn', 'pip install scikit-learn'),
        ('azure-storage-blob', 'pip install azure-storage-blob'),
    ]
    
    missing_packages = []
    
    for package, install_cmd in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ OpenCV version: {cv2.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                print(f"✅ TensorFlow version: {tf.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✅ NumPy version: {np.__version__}")
            elif package == 'matplotlib':
                import matplotlib
                print(f"✅ Matplotlib version: {matplotlib.__version__}")
            elif package == 'scikit-learn':
                import sklearn
                print(f"✅ Scikit-learn version: {sklearn.__version__}")
            elif package == 'azure-storage-blob':
                from azure.storage.blob import BlobServiceClient
                print(f"✅ Azure Storage Blob client available")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append((package, install_cmd))
    
    if missing_packages:
        print("\n📦 Missing packages - install with these commands:")
        for package, install_cmd in missing_packages:
            print(f"  {install_cmd}")
        print("\nOr install all at once:")
        print("  pip install tensorflow opencv-python numpy matplotlib scikit-learn azure-storage-blob")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_credentials():
    """Check if credentials file exists"""
    try:
        import credentials
        if hasattr(credentials, 'STORAGE_ACCOUNT_NAME') and hasattr(credentials, 'STORAGE_ACCOUNT_KEY'):
            print("✅ Credentials file found and configured")
            return True
        else:
            print("❌ Credentials file exists but missing required variables")
            return False
    except ImportError:
        print("❌ credentials.py file not found")
        print("Create a credentials.py file with:")
        print("STORAGE_ACCOUNT_NAME = 'your_account_name'")
        print("STORAGE_ACCOUNT_KEY = 'your_account_key'")
        return False

if __name__ == "__main__":
    print("🎳 Duckpin AI Detector - Environment Check")
    print("=" * 50)
    
    deps_ok = check_all_dependencies()
    creds_ok = check_credentials()
    
    if deps_ok and creds_ok:
        print("\n🚀 Your environment is ready for the Duckpin AI Detector!")
        print("You can now run: python duckpin_ai_detector.py")
    else:
        print("\n⚠️  Please fix the issues above before proceeding.")