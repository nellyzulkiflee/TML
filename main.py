# File: main.py
# Main entry point with Arduino camera support

import os
import sys
import time

def print_menu():
    """Print the main menu with Arduino camera options"""
    print("\n========= Arduino Camera Gesture Controller =========")
    print("1. Detect and configure Arduino camera")
    print("2. Collect gesture data with Arduino camera")
    print("3. Process data and train model")
    print("4. Visualize model performance")
    print("5. Run real-time gesture detection")
    print("6. Exit")
    print("====================================================")
    return input("Select an option (1-6): ")

def check_arduino_camera_configured():
    """Check if Arduino camera has been configured"""
    if not os.path.exists('arduino_camera_settings.txt'):
        print("\nWARNING: Arduino camera not configured!")
        print("Please run option 1 first to detect and configure your camera.")
        input("Press Enter to continue...")
        return False
    return True

def main():
    """Main function to run the complete workflow"""
    print("Welcome to the Arduino Camera Gesture Controller!")
    print("This project uses your OV5642 camera for gesture recognition.")
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            print("\nDetecting and configuring Arduino camera...")
            os.system("python camera_test.py")
            
        elif choice == '2':
            if check_arduino_camera_configured():
                print("\nStarting data collection with Arduino camera...")
                os.system("python data_collection.py")
            
        elif choice == '3':
            if os.path.exists('data/gesture_data_dirs.txt'):
                print("\nProcessing data and training model...")
                os.system("python model_training.py")
            else:
                print("\nNo gesture data found! Please collect data first (option 2).")
                input("Press Enter to continue...")
            
        elif choice == '4':
            if os.path.exists('models/gesture_model_best.h5'):
                print("\nVisualizing model performance...")
                os.system("python visualization.py")
            else:
                print("\nNo trained model found! Please train the model first (option 3).")
                input("Press Enter to continue...")
            
        elif choice == '5':
            if os.path.exists('models/gesture_model_best.h5'):
                if check_arduino_camera_configured():
                    print("\nRunning real-time gesture detection...")
                    os.system("python inference.py")
            else:
                print("\nNo trained model found! Please train the model first (option 3).")
                input("Press Enter to continue...")
            
        elif choice == '6':
            print("\nExiting program.")
            sys.exit(0)
            
        else:
            print("\nInvalid option. Please select 1-6.")
        
        time.sleep(1)

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    base_dirs = ['data', 'models', 'arduino_deploy']
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    main()