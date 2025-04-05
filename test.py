import pickle
try:
    with open('data/processed/simple_model_input/class_map.pkl', 'rb') as f:
        class_map = pickle.load(f)
    print("Loaded Class Map:", class_map)
    # Expected output might be like: {'neutral': 0, 'swipe_right': 1, 'swipe_left': 2}
    # OR {'neutral': 0, 'swipe_left': 1, 'swipe_right': 2}
except FileNotFoundError:
    print("ERROR: class_map.pkl not found in data/processed/simple_model_input/")
except Exception as e:
    print(f"Error loading class_map.pkl: {e}")