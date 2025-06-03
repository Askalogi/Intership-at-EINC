"""
This is a custom conv layer but it MOSTLY works for 2 output channels and it interleaves the kernels 
"""
import torch.nn.functional as F 
from class_dataset_test import CustomSinDataset
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import torch
import numpy as np

torch.set_printoptions(threshold=float('inf'))
transform = transforms.ToTensor()

test_dataset = CustomSinDataset(root="../SNN with CUSTOM Dataset/custom_dataset",transform=transform)
data_img, _ = test_dataset[10]

plt.figure(figsize=(12,6))
plt.title("Input Test Image")
plt.imshow(data_img[0])
plt.colorbar()
plt.tight_layout()
plt.show()

in_channel = 1 
out_channel = 2
kernel_width = 4
kernel_height = 4  
stride = 1 
groups = 1
padding = 0


def custom_conv2d(input_image,in_channel,out_channel,stride,padding,groups,k_w,k_h):
    
    im_w = input_image.shape[1]
    im_h = input_image.shape[2]

    weights = torch.empty(out_channel, (in_channel//groups), k_w,k_h).uniform_(-np.sqrt(groups/(in_channel*k_w*k_h)),np.sqrt(groups/(in_channel*k_w*k_h)))
    
    kernels = {}
    
    for i in range(out_channel):
        kernel = weights[i][0]
        kernels[f"kernel_{i+1}"] = kernel

    #Visualization of the kernels commented out for now 
    for name, kernel in kernels.items():
        plt.figure()
        plt.title(name)
        plt.imshow(kernel.detach().numpy(), cmap='grey')
        plt.colorbar()
        plt.show()

    out_h = (im_h + 2*padding - k_h)//stride + 1
    out_w = (im_w + 2*padding - k_w)//stride + 1

    #the weight matrix

    weights_unrolled= torch.zeros((out_channel*in_channel* out_h *out_w , (input_image.shape[0]*input_image.shape[1]*input_image.shape[2])))
    
    #contains the kernels for easier acces

    kernel_list = list(kernels.values())

    curr_k_i = 0

    current_kernel = kernel_list[curr_k_i]

    row_counter = 1
    counter = 0 
    line_idx = 0

    for row in weights_unrolled:

        row_counter +=1

        for i in range(im_w - k_w + stride):

            row[(k_w - 1)*line_idx + counter :(k_w - 1)*line_idx + counter + k_w ] = current_kernel[0, :k_w]
            row[(k_w - 1)*line_idx + counter + im_w :(k_w - 1)*line_idx + counter + im_w + k_w ] = current_kernel[1,:k_w]
            row[(k_w - 1)*line_idx + counter + 2*im_w :(k_w - 1)*line_idx + counter + 2*im_w + k_w ] = current_kernel[2,:k_w]
            row[(k_w - 1)*line_idx + counter + 3*im_w :(k_w - 1)*line_idx + counter + 3*im_w + k_w ] = current_kernel[3,:k_w]

        counter += 1

        if counter % out_h == 0 :

            line_idx +=1
        
        if row_counter % (weights_unrolled.shape[0]/out_channel + 1) == 0:
            
            curr_k_i = curr_k_i +1
            current_kernel = kernel_list[+1]
            counter = 0
            line_idx = 0
            row_counter = 0

    final = np.zeros_like(weights_unrolled)
    final[::2] = weights_unrolled[:out_h*out_w]
    final[1::2] = weights_unrolled[out_w*out_h:]

    weights_unrolled = final
    plt.figure(figsize=(12,6))
    plt.title("Weights Matrix")
    plt.imshow(weights_unrolled, interpolation="none")
    plt.colorbar()
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 1000
    plt.show()

    # lin = F.linear(input_image.flatten(), weights_unrolled) #make it linear 

    
    # conved_out = lin.view(out_channel,out_w,-1)
    
    # plt.figure(figsize=(12,6))
    # plt.title("Conved Outputs for Each Kernel")
    # for j in range(conved_out.shape[0]):
    #     plt.subplot(1,conved_out.shape[0],1+j)
    #     plt.title(f"Conved out images for kernel {j+1}")
    #     plt.imshow(conved_out[j])
        
    
    # plt.tight_layout()
    # plt.show()


    return final


custom_conv2d(input_image=data_img,in_channel=in_channel, out_channel=out_channel,stride=stride,padding=padding,groups=groups,k_w=kernel_width,k_h=kernel_height)

