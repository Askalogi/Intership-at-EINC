import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F 
import numpy as np 
from class_dataset_test import CustomSinDataset
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import torch

transform = transforms.ToTensor()

test_dataset = CustomSinDataset(root="../SNN with CUSTOM Dataset/custom_dataset",transform=transform)

def create_convolution_matrix(kernel, input_shape, stride=1, padding=0):
    """
    Create the convolution matrix for matrix multiplication equivalent of conv2d
    
    Args:
        kernel: 2D kernel tensor (k_h, k_w)
        input_shape: tuple (H, W) of input image
        stride: convolution stride
        padding: padding amount
    
    Returns:
        weights_matrix: matrix where each row represents one output position
    """
    k_h, k_w = kernel.shape
    in_h, in_w = input_shape
    
    # Calculate output dimensions
    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1
    
    print(f"Input shape: {in_h}x{in_w}")
    print(f"Kernel shape: {k_h}x{k_w}")
    print(f"Output shape: {out_h}x{out_w}")
    print(f"Matrix shape will be: ({out_h*out_w}, {in_h*in_w})")
    
    # Initialize the convolution matrix
    weights_matrix = torch.zeros((out_h * out_w, in_h * in_w))
    
    # Fill the matrix
    output_idx = 0
    
    for out_y in range(out_h):
        for out_x in range(out_w):
            # Calculate the top-left corner of the receptive field in input
            start_y = out_y * stride - padding
            start_x = out_x * stride - padding
            
            # Fill in the kernel weights for this output position
            for k_y in range(k_h):
                for k_x in range(k_w):
                    input_y = start_y + k_y
                    input_x = start_x + k_x
                    
                    # Check if we're within input bounds (handle padding)
                    if 0 <= input_y < in_h and 0 <= input_x < in_w:
                        # Convert 2D input coordinates to 1D index
                        input_idx = input_y * in_w + input_x
                        # Place kernel weight at the appropriate position
                        weights_matrix[output_idx, input_idx] = kernel[k_y, k_x]
            
            output_idx += 1
    
    return weights_matrix

# Example usage with your dimensions
input_h, input_w = 16, 16
kernel_size = 4
stride = 1

# Create a sample kernel (4x4)
kernel = torch.randn(kernel_size, kernel_size)
print("Sample kernel:")
print(kernel)

# Create the convolution matrix
conv_matrix = create_convolution_matrix(kernel, (input_h, input_w), stride=stride)

# Visualize the matrix
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(conv_matrix.numpy(), aspect='auto', cmap='viridis')
plt.title(f'Convolution Matrix\n({conv_matrix.shape[0]} output positions × {conv_matrix.shape[1]} input pixels)')
plt.xlabel('Input pixels (flattened)')
plt.ylabel('Output positions (flattened)')
plt.colorbar()

# Zoom in on a small section to see the pattern better
plt.subplot(1, 2, 2)
plt.imshow(conv_matrix[:50, :100].numpy(), aspect='auto', cmap='viridis')
plt.title('Zoomed view (first 50 outputs, first 100 inputs)')
plt.xlabel('Input pixels (flattened)')
plt.ylabel('Output positions (flattened)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Test that it works correctly
def test_convolution_matrix():
    """Test that our matrix multiplication gives same result as conv2d"""
    
    # Create test input - FIX: Handle dataset properly
    data_sample = test_dataset[100]
    
    # Check if dataset returns tuple (image, label) or just image
    if isinstance(data_sample, tuple):
        x = data_sample[0]  # Get the image part
        print(f"Dataset returns tuple, using image part with shape: {x.shape}")
    else:
        x = data_sample
        print(f"Dataset returns single tensor with shape: {x.shape}")
    
    # Ensure x has the right dimensions for Conv2d (batch_size, channels, height, width)
    if x.dim() == 2:  # If it's just (H, W)
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif x.dim() == 3:  # If it's (C, H, W)
        x = x.unsqueeze(0)  # Add batch dimension
    
    print(f"Input tensor shape for conv: {x.shape}")
    
    # Method 1: Standard convolution
    conv_layer = torch.nn.Conv2d(1, 1, kernel_size, stride=stride, bias=False)
    conv_layer.weight.data = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    result_conv = conv_layer(x)
    
    # Method 2: Matrix multiplication
    # Get the actual 2D image data for matrix multiplication
    if x.dim() == 4:  # (batch, channel, height, width)
        x_2d = x.squeeze(0).squeeze(0)  # Remove batch and channel dims
    elif x.dim() == 3:  # (channel, height, width)
        x_2d = x.squeeze(0)  # Remove channel dim
    else:
        x_2d = x
    
    x_flat = x_2d.view(-1)  # Flatten input
    result_matrix = torch.matmul(conv_matrix, x_flat)
    result_matrix = result_matrix.view(1, 1, result_conv.shape[2], result_conv.shape[3])
    
    # Compare results
    print(f"\nConv2d result shape: {result_conv.shape}")
    print(f"Matrix mult result shape: {result_matrix.shape}")
    print(f"Results are close: {torch.allclose(result_conv, result_matrix, atol=1e-6)}")
    print(f"Max difference: {torch.max(torch.abs(result_conv - result_matrix)).item()}")
    
    return result_conv, result_matrix

# Run the test
try:
    conv_result, matrix_result = test_convolution_matrix()
    print("Test completed successfully!")
except Exception as e:
    print(f"Error in test: {e}")
    print("Let's debug the dataset structure...")
    
    # Debug the dataset
    sample = test_dataset[100]
    print(f"Dataset sample type: {type(sample)}")
    if isinstance(sample, tuple):
        print(f"Tuple length: {len(sample)}")
        for i, item in enumerate(sample):
            print(f"Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
    else:
        print(f"Single item shape: {sample.shape}")

# Visualize the difference in patterns
def show_pattern_analysis():
    """Show why your original code created diagonal patterns"""
    
    print("\n" + "="*50)
    print("ANALYSIS OF THE DIAGONAL PATTERN ISSUE")
    print("="*50)
    
    print("\nYour original code had these issues:")
    print("1. You were iterating over 'row in weights_unrolled_1' but then using 'counter' to index")
    print("2. The inner loop 'for i in range(im_w - k_w + stride)' was wrong")
    print("3. You weren't properly handling the 2D sliding window")
    
    print(f"\nFor a {input_h}×{input_w} input with {kernel_size}×{kernel_size} kernel:")
    print(f"- Output should be {input_h-kernel_size+1}×{input_w-kernel_size+1} = {input_h-kernel_size+1}×{input_w-kernel_size+1}")
    print(f"- Matrix should be {(input_h-kernel_size+1)*(input_w-kernel_size+1)}×{input_h*input_w}")
    print(f"- Each row represents one output pixel's receptive field")
    
    # Show correct indexing pattern for first few output positions
    print(f"\nCorrect receptive field positions for first few outputs:")
    out_h = input_h - kernel_size + 1
    out_w = input_w - kernel_size + 1
    
    for out_idx in range(min(5, out_h * out_w)):
        out_y = out_idx // out_w
        out_x = out_idx % out_w
        print(f"Output[{out_y},{out_x}] (flattened idx {out_idx}) uses input pixels:")
        
        receptive_field = []
        for k_y in range(kernel_size):
            for k_x in range(kernel_size):
                input_y = out_y + k_y
                input_x = out_x + k_x
                input_idx = input_y * input_w + input_x
                receptive_field.append(f"[{input_y},{input_x}]→{input_idx}")
        
        print(f"  {receptive_field}")

show_pattern_analysis()

print(f"\nYour corrected convolution matrix shape: {conv_matrix.shape}")
print("This matrix should have a block-diagonal structure, not simple diagonals!")

