# setup script
import subprocess
import sys
import os

def install_requirements():
    # Install required packages
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("[YES] Requirements installed successfully")

def verify_structure():
    print("\nVerifying directory structure...")
    
    if not os.path.exists("models"):
        os.makedirs("models")
        print("[YES] Created models directory")
    
    if not os.path.exists("models/DenseNet121_Augmented_FINAL.keras"):
        print("[WARNING] Model file not found at models/DenseNet121_Augmented_FINAL.keras")
        print("  Please place your trained model in the models directory")
    else:
        print("[YES] Model file found")

def configure_api():
    # Help you, yes you, configure API key
    print("\n" + "="*50)
    print("API Configuration")
    print("="*50)
    print("\nYou need to add your OpenAI API key to medical_ai_gui.py")
    print("\n1. Open medical_ai_gui.py in a text editor")
    print("2. Find the line: OPENAI_API_KEY = \"your-api-key-here\"")
    print("3. Replace 'your-api-key-here' with your actual API key")
    print("\nExample: OPENAI_API_KEY = \"sk-proj-abc123...\"")

def main():
    print("="*50)
    print("Medical AI Assistant - Setup")
    print("="*50)
    
    try:
        install_requirements()
        verify_structure()
        configure_api()
        
        print("\n" + "="*50)
        print("Setup Complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Add your OpenAI API key to medical_ai_gui.py")
        print("2. Ensure your model is at: models/DenseNet121_Augmented_FINAL.keras")
        print("3. Run: python medical_ai_gui.py")
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()