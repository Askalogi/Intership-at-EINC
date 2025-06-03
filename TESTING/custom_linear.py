"""
Custom linear layer that mimics the conv2d layer [TESTING]
"""
import torch.nn.functional as F 
import numpy as np 
from class_dataset_test import CustomSinDataset
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import torch
from jax.scipy import linalg

torch.set_printoptions(threshold=float('inf'))
transform = transforms.ToTensor()

test_dataset = CustomSinDataset(root="../SNN with CUSTOM Dataset/custom_dataset",transform=transform)
data_img, _ = test_dataset[100]

# data_img
# plt.figure(figsize=(10,6))
# plt.imshow(data_img.squeeze(0),cmap="gray")
# plt.title("This is the test image we are going to be using")
# plt.colorbar()
# plt.legend()


# data_img.shape
# #HYPERPARAMETERS OF THE CONV LAYER 
# in_channel = 1 
# out_channel = 2 
# kernel_size = 4 
# stride = 1 
# groups = 1
# padding = 0

# weights = torch.empty(out_channel, (in_channel//groups), kernel_size, kernel_size).uniform_(-np.sqrt(groups/(in_channel*kernel_size*kernel_size)),np.sqrt(groups/(in_channel*kernel_size*kernel_size)))
# weights.shape 

# x_conved = F.conv2d(data_img,weights)
# x_conved.shape

# #Plot the conved outputs
# plt.figure(figsize=(12, 6))

# # Plot first channel
# plt.subplot(1, 2, 1)
# plt.imshow(x_conved[0].detach().numpy(), cmap="RdGy")
# plt.title("Convolved Image - Channel 1 ")
# plt.colorbar()

# # Plot second channel
# plt.subplot(1, 2, 2)
# plt.imshow(x_conved[1].detach().numpy(), cmap="RdGy")
# plt.title("Convolved Image - Channel 2 ")
# plt.colorbar()
# plt.tight_layout()
# # plt.savefig("./after_conv")
# plt.show()

# #!---------------------TESTER---------------------------------
# #
# fold_inpu = F.unfold(data_img,kernel_size=kernel_size,stride=stride,padding=padding)
# fold_inpu.shape

# linear_weights = weights.view(out_channel,-1)
# linear_weights.shape

# try_input = fold_inpu.transpose(0,1)
# try_input.shape

# output=F.linear(try_input,linear_weights)
# output.shape

# finall = output.transpose(0,1).view(out_channel,x_conved.shape[1],x_conved.shape[2])
# finall.shape

# #Plot the conved outputs
# plt.figure(figsize=(12, 6))

# # Plot first channel
# plt.subplot(1, 2, 1)
# plt.imshow(finall[0].detach().numpy(), cmap="RdGy")
# plt.title("Conn using F.linear - Channel 1")
# plt.colorbar()

# # Plot second channel
# plt.subplot(1, 2, 2)
# plt.imshow(finall[1].detach().numpy(), cmap="RdGy")
# plt.title("Conv using F.linear - Channel 2 ")
# plt.colorbar()
# plt.tight_layout()
# # plt.savefig("./after_conv")
# plt.show()

# #!-------CORRECT METHODOLOGY-----------------------------
#HYPERPARAMETERS OF THE CONV LAYER 

in_channel = 1 
out_channel = 2 
kernel_size = 4 
stride = 1 
groups = 1
padding = 0

x , _ = test_dataset[10] 

weights = torch.empty(out_channel, (in_channel//groups), kernel_size, kernel_size).uniform_(-np.sqrt(groups/(in_channel*kernel_size*kernel_size)),np.sqrt(groups/(in_channel*kernel_size*kernel_size)))
weights.shape 

x.shape
x[0][14][15]
x_flat = torch.flatten(x)
x_flat.shape

x_flat[0] == x[0][0][0]

#script for mapping each pixel individually :
b = 0
c = 0
mapper = []

range(x.shape[2]-1)

for b in range(x.shape[1]):
    for c in range(x.shape[2]):
            mapper.append(x[0][b][c])


mapper[31].item() == x_flat[31].item() == x[0][1][15].item()#should be true 
len(mapper)

x_conv = F.conv2d(x , weights)
x_conv.shape
weights.shape

conv_flatten = torch.flatten(x_conv)
conv_flatten
conv_flatten.shape

x_conv[0][0][0].item() == conv_flatten[0].item()

d= 0 
f= 0
e =0 

conv_mapper=[]

for d in range(x_conv.shape[0]):
    for f in range(x_conv.shape[1]):
        for e in range(x_conv.shape[2]):
            conv_mapper.append(x[0][f][e])

len(conv_mapper)/(13*13*2)


x_conv[0].shape

plt.figure(figsize=(10,6))
plt.imshow(x[0])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(x_conv[0])
plt.subplot(1,2,2)
plt.imshow(x_conv[1])
plt.tight_layout()
plt.show()


kernel = weights

kernel_1 = kernel[0][0] #this gives the output of the 1st channel
kernel_2 = kernel[1][0] #this gives the output of the 2nd channel

#plot the kernels 

plt.figure(figsize=(13 ,6))
plt.subplot(1,2,1)
plt.imshow(kernel_1,cmap="grey")
plt.subplot(1,2,2)
plt.imshow(kernel_2, cmap="grey")
plt.colorbar()
plt.tight_layout()
plt.show()


#flatten the kernels 
kernel_1_f = torch.flatten(kernel_1)
kernel_2_f = torch.flatten(kernel_2)

kernel_2_f.view(4,4) == kernel_2
#test the kernels 
kernel_1_f[2] == kernel_1[2]

kernel_1.shape[0]
kernel_1.shape[1]
x.shape[1]

channel_1 = []
channel_2 = []

x.shape[0] *x.shape[1] *x.shape[2]
kernel_1[0,3] == kernel_1[0][3]
kernel_1.shape[0]

x.shape[2]

((x.shape[1]+2 * padding-1)/stride +1)

out_h = (x.shape[1] + 2*padding - kernel_1.shape[0])//stride + 1
out_w = (x.shape[2] + 2*padding - kernel_1.shape[1])//stride + 1

#!--------------------------------------------------------------------------------------------------------------------------

weights_unrolled= torch.zeros((out_channel*in_channel* out_h *out_w , (x.shape[0]*x.shape[1]*x.shape[2])))
weights_unrolled_1 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))
weights_unrolled.shape
weights_unrolled_1.shape
a=torch.tensor

#!--------------------------------------------------------------------------------------------------------------------------

#TEST
k_w = kernel_1.shape[0]
k_h = kernel_1.shape[1]
im_w = x.shape[1]
im_h = x.shape[2]

kernel_2 == kernel_1

weights_unrolled_1 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))
weights_unrolled_2 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))
weights_unrolled.shape
weights_unrolled_1
weights_unrolled_2 
weights_unrolled_test = torch.zeros(out_h*out_w)

counter = 0

# #! FOR TEST KERNEL 

# for row in weights_unrolled_test:

#     for i in range(im_w - k_w +stride):

#         weights_unrolled_test[counter :counter + k_w ] = kernel_1[0, :k_w]
#         weights_unrolled_test[ counter + im_w : counter + im_w + k_w ] = kernel_1[1,:k_w]
#         weights_unrolled_test[ counter + 2*im_w :counter + 2*im_w + k_w ] = kernel_1[2,:k_w]
#         weights_unrolled_test[ counter + 3*im_w : counter + 3*im_w + k_w ] = kernel_1[3,:k_w]

#     counter +=1

# plt.figure(figsize=(12,6))
# plt.imshow(weights_unrolled_1)
# plt.colorbar()
# plt.tight_layout()
# plt.show()

counter = 0
line_idx = 0
#! FOR KERNEL 1
weights_unrolled_1 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))

for row in weights_unrolled_1:

    for i in range(im_w - k_w +stride):

        row[3*line_idx + counter :3*line_idx + counter + k_w ] = kernel_1[0, :k_w]
        row[3*line_idx + counter + im_w :3*line_idx + counter + im_w + k_w ] = kernel_1[1,:k_w]
        row[3*line_idx + counter + 2*im_w :3*line_idx + counter + 2*im_w + k_w ] = kernel_1[2,:k_w]
        row[3*line_idx + counter + 3*im_w :3*line_idx + counter + 3*im_w + k_w ] = kernel_1[3,:k_w]

    counter += 1

    if counter%out_h == 0 :
        line_idx +=1



plt.figure(figsize=(12,6))
plt.imshow(weights_unrolled_1)
plt.colorbar()
plt.tight_layout()
plt.show()

lin_1 = F.linear(x_flat, weights_unrolled_1)
test_1 = lin_1.view(13,-1)

plt.figure(figsize=(12,6))
plt.imshow(test_1)
plt.colorbar()
plt.tight_layout()
plt.show()

#! FOR KERNEL 2

k_w = kernel_2.shape[0]
k_h = kernel_2.shape[1]

counter = 0
row_counter =0
line_idx = 0
#! FOR KERNEL 1

weights_unrolled_2 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))

# for j in range(out_channel):
    
#     counter = 0

counter = 0
row_counter =0
line_idx = 0
weights_unrolled= torch.zeros((out_channel*in_channel* out_h *out_w , (x.shape[0]*x.shape[1]*x.shape[2])))

for row in weights_unrolled:


    if row_counter % (weights_unrolled.shape[0]/out_channel) == 0:

        counter = 0 
        line_idx = 0


    for i in range(im_w - k_w +stride):

        row[3*line_idx + counter :3*line_idx + counter + k_w ] = kernel_2[0, :k_w]
        row[3*line_idx + counter + im_w :3*line_idx + counter + im_w + k_w ] = kernel_2[1,:k_w]
        row[3*line_idx + counter + 2*im_w :3*line_idx + counter + 2*im_w + k_w ] = kernel_2[2,:k_w]
        row[3*line_idx + counter + 3*im_w :3*line_idx + counter + 3*im_w + k_w ] = kernel_2[3,:k_w]

    counter += 1

    if counter%out_h == 0 :
        line_idx +=1
    row_counter +=1



plt.figure(figsize=(12,6))
plt.imshow(weights_unrolled)
plt.colorbar()
plt.tight_layout()
plt.show()

lin_f = F.linear(x_flat, weights_unrolled)
conved_out = lin_f.view(2,13,-1)
conved_out.shape[0]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(conved_out[0])
plt.subplot(1,2,2)
plt.imshow(conved_out[1])
plt.colorbar()
plt.tight_layout()
plt.show()
conved_out[0].shape
conved_out[1].shape
weights_unrolled_2.flatten()
weights_unrolled_1.shape

weights_unrolled_1 == weights_unrolled_2

x[0].shape


lin_1 = F.linear(x_flat,weights_unrolled_2)
x_flat.shape
weights_unrolled_1.shape
lin_1
lin_1.shape 

x_conv[0]


lin_2 = F.linear(x_flat,weights_unrolled_2)
test_2 = lin_2.view(13,-1)
plt.figure(figsize=(12,10))
plt.imshow(test_2)
plt.colorbar()
plt.tight_layout()
plt.show()


#! FOR BOTH KERNELS :

# #!--------------------------------------------------------------------------------------------------------------------------

counter = 0
row_counter =0
line_idx = 0
weights_unrolled= torch.zeros((out_channel* in_channel* out_h *out_w , (x.shape[0]*x.shape[1]*x.shape[2])))
weights_unrolled_2 = torch.zeros((out_h *out_w, (x.shape[0]*x.shape[1]*x.shape[2]) ))

weights_unrolled.shape
out_w

for j in range(start=0,stop=2,step=1):

    for row in weights_unrolled_2:

        row_counter +=1


        for i in range(im_w - k_w +stride):

            row[3*line_idx + counter :3*line_idx + counter + k_w ] = kernel_2[0, :k_w]
            row[3*line_idx + counter + im_w :3*line_idx + counter + im_w + k_w ] = kernel_2[1,:k_w]
            row[3*line_idx + counter + 2*im_w :3*line_idx + counter + 2*im_w + k_w ] = kernel_2[2,:k_w]
            row[3*line_idx + counter + 3*im_w :3*line_idx + counter + 3*im_w + k_w ] = kernel_2[3,:k_w]

        counter += 1

        
        if counter%out_h == 0 :

            line_idx +=1

        if row_counter % (out_h*out_w) == 0:

            counter = 3
            line_idx = 0
            row_counter = 0


plt.figure(figsize=(12,6))
plt.imshow(weights_unrolled_2)
plt.colorbar()
plt.tight_layout()
plt.show()

lin_f = F.linear(x_flat, weights_unrolled)
conved_out = lin_f.view(2,13,-1)
conved_out.shape[0]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(conved_out[0])
plt.subplot(1,2,2)
plt.imshow(conved_out[1])
plt.tight_layout()
plt.show()