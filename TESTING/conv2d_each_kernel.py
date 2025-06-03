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



in_channel = 1 
out_channel = 2
kernel_size = 5
stride = 1 
groups = 1
padding = 0

weights = torch.empty(out_channel, (in_channel//groups), kernel_size, kernel_size).uniform_(-np.sqrt(groups/(in_channel*kernel_size*kernel_size)),np.sqrt(groups/(in_channel*kernel_size*kernel_size)))

weights
kernel_1 = weights[0][0]
kernel_2 = weights[1][0]

def weights_unrolled(input_img, kernel,weights):

    im_w = input_img.shape[1]
    im_h = input_img.shape[2]

    k_w = kernel.shape[0]
    k_h = kernel.shape[1]

    out_h = (input_img.shape[1] + 2*padding - kernel.shape[0])//stride + 1
    out_w = (input_img.shape[2] + 2*padding - kernel.shape[1])//stride + 1

    weights_un = torch.zeros((out_h *out_w, (input_img.shape[0]*input_img.shape[1]*input_img.shape[2])))

    counter = 0
    line_idx = 0

    kernel_mask = np.zeros((im_w), dtype=bool)
    kernel_mask[:k_h] = True
    kernel_mask = np.tile(kernel_mask, k_h)
    print(kernel_mask)

    kernel_flattend = np.zeros((k_h * im_w))
    kernel_flattend[kernel_mask] = kernel.reshape(-1)
    print(kernel_flattend)

    #kernel_unrolled_col = np.zeros((im_w, k_h * im_w)) 
    #kernel_unrolled_col[row_idx, col_idx] = 

    offsets = np.repeat(np.arange(im_w - k_w + 1), im_h - k_h + 1) * (k_w - 1) + np.arange(12 * 12)
    print("Offset",  offsets)
    #print(np.tile(np.arange(im_w * im_h), im_h))
    #kernel_mask = np.zeros((im_w), dtype=bool)
    #kernel_mask[:k_h] = True
    ##kernel_mask = np.tile(kernel_mask, k_h)
    #kernel_flattend = np.arange((k_h * im_w))
    #kernel_flattend[~kernel_mask] = 0.

    kernel_flattend = (np.tile(np.arange(k_w), k_h) + np.arange(k_h).repeat(k_h) * im_w)

    col_idx = np.repeat(kernel_flattend[None, :], (im_w - k_w + 1)* (im_h - k_h + 1), axis=0)

    col_idx = col_idx + offsets[:, None]

    row_idx = np.repeat(np.arange((im_w - k_w + 1)* (im_h - k_h + 1)), k_h*k_w)

    print(row_idx[:25])
    print(col_idx[0])

    print(row_idx[25:50])
    print(col_idx[1])

    weight_mask = np.zeros(((im_w - k_w + 1)* (im_h - k_h + 1), im_h*im_w))
    weight_mask[row_idx, col_idx.reshape(-1)] = np.tile(kernel.reshape(-1), (im_w - k_w + 1)* (im_h - k_h + 1))


    weight_mask = torch.tensor(weight_mask)

    plt.figure(figsize=(12,6))
    plt.title("Weight mask")
    plt.imshow(weight_mask,cmap="grey")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    lin_1 = F.linear(input_img.flatten().float(), weight_mask.float())
    test = lin_1.view(12,-1)

    plt.figure(figsize=(12,6))
    plt.title("Conved Test")
    plt.imshow(test,cmap="grey")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    for row in weights_un:
        col_idx_in = []
        col_idx_out = []

        #idx_begin = np.zeros((im_w)).fill()

        for i in range(im_w):
            col_idx_in.append(3*line_idx + counter)
            col_idx_out.append(3*line_idx + counter + k_w)

            row[3*line_idx + counter :3*line_idx + counter + k_w ] = kernel[0, :k_w]
            row[3*line_idx + counter + im_w :3*line_idx + counter + im_w + k_w ] = kernel[1,:k_w]
            row[3*line_idx + counter + 2*im_w :3*line_idx + counter + 2*im_w + k_w ] = kernel[2,:k_w]
            row[3*line_idx + counter + 3*im_w :3*line_idx + counter + 3*im_w + k_w ] = kernel[3,:k_w]


        counter += 1

        if counter%out_h == 0 :
            line_idx +=1

    #Original Image
    plt.figure(figsize=(12,6))
    plt.title("Original Image")
    plt.imshow(input_img[0],cmap="grey")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    conved = F.conv2d(input_img.float(),weights.float())

    
    #Original Image
    plt.figure(figsize=(12,6))
    plt.title("Conved 1")
    plt.imshow(conved[0],cmap="grey")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.title("Conved 2")
    plt.imshow(conved[1],cmap="grey")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


    # #Kernel
    # plt.figure(figsize=(12,6))
    # plt.title("Original Image")
    # plt.imshow(kernel,cmap="grey")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    
    # # Unrolled Weights
    # plt.figure(figsize=(12,6))
    # plt.title("Unrolled Weights for this kernel")
    # plt.imshow(weights_un, cmap="grey")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


    return weights_un



weights_unrolled_1 = weights_unrolled(data_img,kernel_1,weights=weights)

weights_unrolled_2 = weights_unrolled(data_img,kernel_2,weights=weights)

lin_1 = F.linear(data_img.flatten(), weights_unrolled_1)
lin_2 = F.linear(data_img.flatten(), weights_unrolled_2)

final_1 = lin_1.view(16,-1)
final_2 = lin_2.view(16,-1)

#add them together 

conved = torch.stack((final_1,final_2),dim=0)

conved.shape 

input_t = torch.rand([10,20,1,16,16])


input_t[0][0].shape

out = []



for time_slice in input_t:
    
    time_out = []

    for batch_slice in time_slice:

        lin_test = F.linear(batch_slice.flatten(),weights_unrolled_1)
        
        time_out.append(lin_test.view(1,16,-1))

        print(time_out[0].shape)

    time_out = torch.stack(time_out)

    out.append(time_out)

result = torch.stack(out)

result.shape






