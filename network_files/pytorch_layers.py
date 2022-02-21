import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from pickle5 import load, dump
from os import listdir
from random import sample
import cv2
import matplotlib.pyplot as plt
from code_tkinter.tkinter_objects import recurs_return_all

class Dataset(torch.utils.data.Dataset):
    """
    initialization of the labels 
    """
    def __init__(self, label_path, image_path, file_list):
        self.label_path = image_path
        self.image_path = label_path
        self.file_list = file_list

    """
    gets the length of the folder
    """
    def __len__(self):
        return len(self.file_list)

    """
    gets the item from the hard drive
    """
    def __getitem__(self, index):
        # Select sample
        file = self.file_list[index]

        # load the image
        with open(self.image_path + file, "rb") as fd:
            input_image = load(fd)
        # load the label
        with open(self.label_path + file, "rb") as fd:
            output_label = load(fd)
        return input_image, output_label

class Residual_CNN_BB_obj(nn.Module):
    def __init__(self):
        super(Residual_CNN_BB_obj,self).__init__()
        # 2x2 max pooling, but return the indece of the maximum for unpooling
        self.max = nn.MaxPool2d(2,2)

        # flattening the cnn to linear
        self.flatten = nn.Flatten()
        # linear layer for bottlenecking the autoencoder
        self.bottleneck = nn.Linear(128, 256)
        self.out = nn.Linear(256, 8)

        # deconvolution dictionary
        self.conv = {}
        self.conv_bn = {}

        # initialize deconvolution and batch normalization layers
        self.conv_layer_init(name='init_layer', in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.conv_layer_init(name='res1_layer1', in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res1_layer2', in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res1_layer3', in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res1_layer4', in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv_layer_init(name='res2_layer1', in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer2', in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer3', in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer4', in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_layer_init(name='res2_layer1', in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer2', in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer3', in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_layer_init(name='res2_layer4', in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # convert dictionaries to layer modules
        self.conv = nn.ModuleDict(self.conv)
        self.conv_bn = nn.ModuleDict(self.conv_bn)

    def conv_layer_init(self, name, in_channels, out_channels, kernel_size, stride, padding):
        self.conv[name] = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_bn[name] = nn.BatchNorm2d(out_channels)



    def conv_layer(self, name, input):
        # convolution layer, batch normalization layer and leaky relu
        output = self.conv[name](input)
        output = self.conv_bn[name](output)
        output = self.lrelu(output)
        return output       

    def forward(self, x):
        
        #### INIT
        init_layer = self.conv_layer(name='init_layer', input=x)

        down_sized_layer1 = self.max(init_layer) 

        #### RESIDUAL GROUPING 1
        ## residual block 1
        res1_layer1 = self.conv_layer(name='res1_layer1', input=down_sized_layer1)

        res1_layer2 = self.conv_layer(name='res1_layer2', input=res1_layer1)

        res1_added1 = torch.add(down_sized_layer1, res1_layer2)

        ## residual block 2
        res1_layer3 = self.conv_layer(name='res1_layer3', input=res1_added1)

        res1_layer4 = self.conv_layer(name='res1_layer4', input=res1_layer3)

        res1_added2 = torch.add(res1_added1, res1_layer4)


        down_sized_layer2 = self.max(res1_added2)
        
        #### RESIDUAL GROUPING 2
        ## residual block 3
        res2_layer1 = self.conv_layer(name='res2_layer1', input=down_sized_layer2)

        res2_layer2 = self.conv_layer(name='res2_layer2', input=res2_layer1)

        res2_added1 = torch.add(down_sized_layer2, res2_layer2)

        ## residual block 4
        res2_layer3 = self.conv_layer(name='res2_layer3', input=res2_added1)

        res2_layer4 = self.conv_layer(name='res2_layer4', input=res2_layer3)

        res2_added2 = torch.add(res2_added1, res2_layer4)


        down_sized_layer3 = self.max(res2_added2)

        #### RESIDUAL GROUPING 3
        ## residual block 5
        res3_layer1 = self.conv_layer(name='res3_layer1', input=down_sized_layer3)

        res3_layer2 = self.conv_layer(name='res3_layer2', input=res3_layer1)

        res3_added1 = torch.add(down_sized_layer3, res3_layer2)

        ## residual block 6
        res3_layer3 = self.conv_layer(name='res3_layer3', input=res3_added1)

        res3_layer4 = self.conv_layer(name='res3_layer4', input=res3_layer3)

        res2_added3 = torch.add(res3_added1, res3_layer4)

        down_sized_layer4 = self.max(res2_added3)

        flattened_conv = self.flatten(down_sized_layer4)
        
        output = self.bottleneck(flattened_conv)

        return output

class Network():
    def __init__(self, host, label_lookup):
        self.host = host
        # iterate through frames to find the ones we want
        self.host_children = {}
        # get a dict of all the objects that are a child to the setup train tab
        recurs_return_all(self.host, self.host_children)

        self.host_children['button_nntrain'].object.configure(command=self.train_and_eval)

        self.lookup = label_lookup


    def operate_nn(self, input_image_batch, output_label_batch, Residual_CNN_BB, loss_func):
        # convert image inputs to gpu
        input_image_batch = Variable(input_image_batch).cuda()
        output_label_batch = Variable(output_label_batch).cuda()
        # encode the image, get max pool indeces and skip connections
        nn_output_batch = Residual_CNN_BB(input_image_batch)
        # calculate loss using MSE
        loss = loss_func(nn_output_batch,output_label_batch)

        return loss, nn_output_batch

    def train_and_eval(self):

        learning_rate = self.host_children['entry_batch'].variable.get()
        epoch_count = self.host_children['entry_epoch'].variable.get()
        batch_size = self.host_children['entry_learning'].variable.get()
        train_val_split = self.host_children['entry_trainpercent'].variable.get()
        file_name = self.host_children['entry_filename'].variable.get()


        label_path = 'database/mutated_labels/'
        image_path = 'database/mutated_images/'

        all_files = listdir(image_path)
        # shuffle the file names for splitting
        file_array_shuffle = sample( all_files, len(all_files) )
        
        # split the validation and training files
        training_files=all_files[:int(len(file_array_shuffle)*train_val_split)]
        validation_files=all_files[int(len(file_array_shuffle)*train_val_split):]
        
        dataset_train = Dataset(label_path, image_path, training_files)
        dataset_val = Dataset(label_path, image_path, validation_files)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=True)


        Residual_CNN_BB = Residual_CNN_BB_obj().cuda()

        parameters = list(Residual_CNN_BB.parameters())
        # mean squared error loss calculation
        loss_func = nn.MSELoss()
        # adam optimizer, default beta 1 and beta 2, only learning rate set
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)



        # otherwise train new network
        # train the encoder and decoder
        # define training ground validation los
        min_val_loss = 1
        loss_list = {}
        loss_list['loss'] = []
        loss_list['val_loss'] = []
        loss_list['epoch'] = []
        for i in range(epoch_count):
            average_loss = 0
            average_val_loss = 0

            Residual_CNN_BB.train()
            for j, (input_image_batch, output_label_batch) in enumerate(dataloader_train):
                # soemtimes loads in batch size of 16
                if input_image_batch.size()[0] == batch_size:
                    optimizer.zero_grad()
                    loss, nn_output_batch = self.operate_nn(input_image_batch, output_label_batch[:, 2], Residual_CNN_BB, loss_func)
                    average_loss = average_loss + loss.item()
                    # backwards propogation based on loss
                    loss.backward()
                    optimizer.step()
                average_loss = average_loss
            average_loss = average_loss  / (j+1)

            #update_canvas_image(self.host_children['canvas_nninput'], input_image_batch[0])
            #update_canvas_image(self.host_children['canvas_nnoutput'], input_image_batch[0])
            #draw_points(self.host_children['canvas_nninput'], actions, nn_output_batch, self.lookup)



            # enact validation
            Residual_CNN_BB.eval()
            # no gradiant activation
            with torch.no_grad():
                # for each image batch in the test
                for j, (val_input_image_batch, val_output_label_batch) in enumerate(dataloader_val):
                    # soemtimes loads in batch size of 16
                    if val_input_image_batch.size()[0] == batch_size:
                        val_loss = self.operate_nn(val_input_image_batch, val_output_label_batch, Residual_CNN_BB, loss_func)
                        average_val_loss = average_val_loss + val_loss.item()
                # average all the validation losses from the testing batches 
                average_val_loss = average_val_loss / (j+1)
            
            #cv2.imwrite('output_per_epoch/network_input_and_output_at_epoch_' + str(i+1) + '.jpeg', display_info(image_valc[0], image_n_valc[0], output[0], validation=True)*255)
            # if this is the new lowest validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('new minimum, saved')
                # save the network to hard drive
                torch.save([Residual_CNN_BB],'./model/' + file_name + '.pkl')
            # print info to user
            print('epoch: ' + str(i+1) + '/' + str(epoch_count) + ' ||| loss: ' + str(loss.item()) + ' ||| val_loss: ' + str(val_loss))
            # append list for usage in printing graph
            loss_list['epoch'].append(i+1)
            loss_list['loss'].append(loss.item())
            loss_list['val_loss'].append(val_loss)
        plt.plot(loss_list['epoch'], loss_list['loss'], label='loss')
        plt.plot(loss_list['epoch'], loss_list['val_loss'], label='val_loss')
        plt.show()
        '''                
        # set evaluation mode
        Residual_CNN_BB.eval()

        # no gradiant descent
        with torch.no_grad():
            # for each image and label batch in the test dataset
            for image,label in test_loader:
                # if the batch size is 32
                if image.size()[0] == batch_size:
                    # generate some random noise
                    noise = torch.rand(batch_size,1,28,28)

                    # add the noise to the image to generate the input
                    image_n = torch.mul(image+0.5, 0.7 * noise)
                    # convert images to gpu 
                    image = Variable(image).cuda()
                    image_n = Variable(image_n).cuda()
                    # run through encoder to get encoded image, the max pool indexes and the skip connections
                    encoded, unmax_indeces, skip_con = encoder(image_n)
                    # then decode the image
                    decoded = decoder(encoded, unmax_indeces, skip_con)
                    # for each of the images in the image batch
                    for single_image, single_image_n, single_decoded in zip(image, image_n, decoded):
                        # display it for testing purposes
                        display_info(single_image, single_image_n, single_decoded)
        '''

