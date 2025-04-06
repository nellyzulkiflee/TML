# File: main.py
import os
import sys
import time

def print_menu():
    """Print the main menu with Arduino camera options"""
    print("\n========= Arduino Camera Gesture Controller =========")
    print("1. Collect gesture data with Arduino camera")
    print("2. Process data and train model")
    print("3. Run real-time gesture detection")
    print("4. Exit")
    print("====================================================")
    return input("Select an option (1-4): ")

def main():
    """Main function to run the complete workflow"""
    print("Welcome to the Arduino Camera Gesture Controller!")
    print("This project uses OV5642 Mini-5MP-Plus camera for gesture recognition.")
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            print("\nStarting data collection with Arduino camera...")
            os.system("python data_collection.py")
            
        elif choice == '2':
            if os.path.exists('data/gesture_data_dirs.txt'):
                print("\nProcessing data and training model...")
                os.system("python model_training.py")
            else:
                print("\nNo gesture data found! Please collect data first (option 2).")
                input("Press Enter to continue...")
                
        elif choice == '3':
            if os.path.exists('models/gesture_simple_cnn_float.tflite'):
                print("\nRunning real-time gesture detection...")
                os.system("python inference.py")
            else:
                print("\nNo trained model found! Please train the model first (option 3).")
                input("Press Enter to continue...")
            
        elif choice == '4':
            print("\nExiting program.")
            sys.exit(0)
            
        else:
            print("\nInvalid option. Please select 1-4.")
        
        time.sleep(1)

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    base_dirs = ['data', 'models']
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    main()