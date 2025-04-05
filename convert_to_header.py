import os
import sys

def convert_to_header(input_file, output_file):
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} does not exist")
        return False
    
    # Read the input file as binary
    with open(input_file, 'rb') as f:
        data = f.read()
    
    # Get base name of the file without extension
    base_name = os.path.basename(input_file)
    array_name = os.path.splitext(base_name)[0].replace('.', '_').replace('-', '_')
    
    # Create C array from binary data
    hex_bytes = [f"0x{b:02x}" for b in data]
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(f"// Generated from {input_file}\n")
        f.write(f"// Model size: {len(data)} bytes\n\n")
        f.write(f"unsigned char {array_name}[] = {{\n  ")
        
        # Write bytes in groups of 12 for readability
        for i, byte in enumerate(hex_bytes):
            f.write(byte)
            if i < len(hex_bytes) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
        
        f.write("\n};\n")
        f.write(f"unsigned int {array_name}_len = {len(data)};\n")
    
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Array name: {array_name}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_header.py <input_tflite_file> [output_header_file]")
        print("Example: python convert_to_header.py models/gesture_simple_model_quantized.tflite model_data.h")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # If output file not specified, use model_data.h
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = "model_data.h"
    
    convert_to_header(input_file, output_file)