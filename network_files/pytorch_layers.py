from configparser import Interpolation
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable
from os import listdir, makedirs
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from shutil import rmtree
from torchviz import make_dot
from code_tkinter.tkinter_objects import recurs_return_all
from threading import Thread, Event as Event_Th
from PIL import Image, ImageTk


import cv2

class Dataset(torch.utils.data.Dataset):
    """
    initialization of the labels 
    """
    def __init__(self, data_path, file_list, maximum):
        self.data_path = data_path
        self.file_list = file_list
        self.maximum = maximum

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
        with open(self.data_path + file, "rb") as fd:
            input_data = pk.load(fd)/self.maximum

        # load the label
        output_label = F.one_hot(torch.tensor([int(file.split('_')[0])-1]), num_classes=2)
        return input_data[None, :, :], output_label[0]

class Residual_1DCNN_obj(nn.Module):
    def __init__(self):
        super(Residual_1DCNN_obj,self).__init__()
        output_size = 2

        # Leaky ReLU as opposed to ReLU
        self.lrelu = nn.LeakyReLU()

        # 2x2 max pooling, but return the indece of the maximum for unpooling
        self.max = nn.MaxPool2d((1,2))

        # pool together all values into 1x1 indece for each filter
        self.avg = nn.AvgPool2d((1,256))

        self.dropout = nn.Dropout(p=0.5)

        # flattening the cnn to linear
        self.flatten = nn.Flatten()
        # linear layer for bottlenecking the autoencoder
        self.bottleneck = nn.Linear(96, output_size)
        self.outlinear = nn.Linear(output_size, output_size)
        self.softmax = nn.Softmax(1)

        # deconvolution dictionary
        self.conv = {}
        self.conv_bn = {}

        # initialize deconvolution and batch normalization layers
        self.conv_layer_init(name='LoRa_1', in_channels=1, out_channels=16, kernel_size=(1, 5), stride=1, padding=(0, 2))
        
        self.conv_layer_init(name='LoRa_2', in_channels=16, out_channels=24, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='LoRa_3', in_channels=24, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='LoRa_4', in_channels=32, out_channels=48, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='LoRa_5', in_channels=48, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='distinct_LoRa_6', in_channels=64, out_channels=96, kernel_size=(2, 3), stride=1, padding=(0, 1))


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
        LoRa_1_layer = self.conv_layer(name='LoRa_1', input=x)

        # height out = (8192 + 2 + 2 - 5) / (1) + 1 = 8192
        # width out = (2 + 0 + 0 - 1) / (1) + 1 = 2

        down_sized_layer1 = self.max(LoRa_1_layer) 

        # height out = 8192//2 = 4096
        # width out = 2//1 = 2

        LoRa_2_layer = self.conv_layer(name='LoRa_2', input=down_sized_layer1)

        down_sized_layer2 = self.max(LoRa_2_layer) 
        
        # height out = 4096//2 = 2048
        # width out = 2//1 = 2

        LoRa_3_layer = self.conv_layer(name='LoRa_3', input=down_sized_layer2)

        down_sized_layer3 = self.max(LoRa_3_layer) 

        # height out = 2048//2 = 1024
        # width out = 2//1 = 2

        LoRa_4_layer = self.conv_layer(name='LoRa_4', input=down_sized_layer3)

        down_sized_layer4 = self.max(LoRa_4_layer) 

        # height out = 1024//2 = 512
        # width out = 2//1 = 2

        LoRa_5_layer = self.conv_layer(name='LoRa_5', input=down_sized_layer4)

        down_sized_layer5 = self.max(LoRa_5_layer) 

        # height out = 512//2 = 256
        # width out = 2//1 = 2

        distinct_LoRa_6 = self.conv_layer(name='distinct_LoRa_6', input=down_sized_layer5)

        down_sized_layer6 = self.avg(distinct_LoRa_6)


        flattened_conv = self.flatten(down_sized_layer6)
        
        bottlenecked_layer = self.bottleneck(flattened_conv)

        activated_bottleneck = self.lrelu(bottlenecked_layer)

        dropout_bttlnck = self.dropout(activated_bottleneck)

        linear_2 = self.outlinear(dropout_bttlnck)

        output = self.softmax(linear_2)



        return output

class Network():
    def __init__(self, host):
        self.host = host
        # iterate through frames to find the ones we want
        self.host_children = {}
        # get a dict of all the objects that are a child to the setup train tab
        recurs_return_all(self.host, self.host_children)

        # fairly thread safe way of telling main gui to display a change
        self.host.object.bind('<<ReadUpdate>>', lambda e: self.on_read_update(e))
        self.host.object.bind('<<EpochUpdate>>', lambda e: self.on_epoch_update(e))

        self.host_children['button_nntrain'].object.configure(command=self.begin_train_thread)

    def operate_nn(self, input_image_batch, output_label_batch, Residual_CNN, loss_func):
        # convert image inputs to gpu
        cinput_image_batch = Variable(input_image_batch).cuda()
        coutput_label_batch = Variable(output_label_batch).cuda()
        # encode the image, get max pool indeces and skip connections
        nn_output_batch = Residual_CNN(cinput_image_batch)
        # calculate loss using MSE
        loss = loss_func(nn_output_batch,coutput_label_batch)

        return loss, nn_output_batch
    
    def on_read_update(self, event):
        guessed_values = cv2.resize(self.guessed_values*255, (240, 240), interpolation= cv2.INTER_NEAREST)
        known_values = cv2.resize(self.known_values*255, (240, 240), interpolation= cv2.INTER_NEAREST)
        im = Image.fromarray(known_values)
        imgtk = ImageTk.PhotoImage(image=im) 
        
        self.host_children['label_nnknown'].object['image'] = self.host_children['label_nnknown'].object.img = imgtk
        im = Image.fromarray(guessed_values)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nnguess'].object['image'] = self.host_children['label_nnguess'].object.img = imgtk
        conversion = cv2.convertScaleAbs((255-abs(known_values-guessed_values)), alpha=1)
        depth_colormap = cv2.applyColorMap(conversion, 11)
        im = Image.fromarray(depth_colormap)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nndiff'].object['image'] = self.host_children['label_nndiff'].object.img = imgtk

        self.host_children['label_nnloss'].object.configure(text=str(self.loss.item()))
        if (self.val_loss != 0):
            self.host_children['label_nnvalloss'].object.configure(text=str(self.val_loss.item()))
        self.host_children['label_nnavgloss'].object.configure(text=str(self.average_loss))
        self.host_children['label_nnavgvloss'].object.configure(text=str(self.average_val_loss))
        self.host_children['label_nnacc'].object.configure(text=str(self.acc.item()))
        if (self.val_acc != 0):
            self.host_children['label_nnvalacc'].object.configure(text=str(self.val_acc.item()))
        self.host_children['label_nnavgacc'].object.configure(text=str(self.average_acc))
        self.host_children['label_nnavgvacc'].object.configure(text=str(self.average_val_acc))

        self.host_children['label_nnpercentage'].object.configure(text=str(self.percentage))
        self.host_children['label_nnvpercentage'].object.configure(text=str(self.val_percentage))
        self.host_children['label_nnepoch'].object.configure(text=str(self.epoch) + '/' + str(self.epoch_count))

    def update_plot(self, plot_host, parameter, val_parameter):
        ax = plot_host.fig.axes[0]
        ax.set_xlim( min(self.loss_list['epoch']),  max(self.loss_list['epoch']))
        ylim_max =  max(max(self.loss_list[parameter]), max( self.loss_list[val_parameter]))
        ylim_min =  min(min(self.loss_list[parameter]), min( self.loss_list[val_parameter]))
        ax.set_ylim(ylim_min, ylim_max)    
        plot_host.object.draw()
        plot_host.object.flush_events()

    def on_epoch_update(self, event):
        self.update_plot(self.host_children['figcan_loss'], 'loss', 'val_loss')
        self.update_plot(self.host_children['figcan_acc'], 'acc', 'val_acc')


    def begin_train_thread(self):
        start_training_th = Thread(target=self.train_and_eval)
        start_training_th.start()

    def train_and_eval(self):

        learning_rate = self.host_children['entry_learning'].variable.get()
        self.epoch_count = self.host_children['entry_epoch'].variable.get()
        batch_size = self.host_children['entry_batch'].variable.get()
        train_val_split = self.host_children['entry_trainpercent'].variable.get()
        file_name = self.host_children['entry_filename'].variable.get()
        between_iters = self.host_children['entry_timebetween'].variable.get()
        input_path = 'network_data_rest/'

        with open('normalize_maximum/maximum', "rb") as fd:
            max = pk.load(fd)
        # check if torch is running on a gpu
        avail = torch.cuda.is_available()
        if (avail):
            print(torch.cuda.get_device_name(0))

        all_files = listdir(input_path)
        # shuffle the file names for splitting
        file_array_shuffle = sample( all_files, len(all_files) )[0:10000]

        
        
        # split the validation and training files
        training_files=file_array_shuffle[:int(len(file_array_shuffle)*train_val_split)]
        validation_files=file_array_shuffle[int(len(file_array_shuffle)*train_val_split):]
        
        print(len(validation_files))
        print(len(training_files))
        
        dataset_train = Dataset(input_path, training_files, max)
        dataset_val = Dataset(input_path, validation_files, max)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)


        Residual_1DCNN = Residual_1DCNN_obj().cuda()

        parameters = list(Residual_1DCNN.parameters())
        # mean squared error loss calculation
        loss_func = nn.CrossEntropyLoss()
        # adam optimizer, default beta 1 and beta 2, only learning rate set
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)



        # otherwise train new network
        # train the encoder and decoder
        # define training ground validation los
        min_val_loss = 1
        self.loss_list = {}
        self.loss_list['loss'] = []
        self.loss_list['val_loss'] = []
        self.loss_list['acc'] = []
        self.loss_list['val_acc'] = []
        self.loss_list['epoch'] = []

        loss_train, = self.host_children['figcan_loss'].plot.plot(0, 0, label='loss', color='cyan')
        loss_val, = self.host_children['figcan_loss'].plot.plot(0, 0, label='val_loss', color='white')
        acc_train, = self.host_children['figcan_acc'].plot.plot(0, 0, label='acc', color='cyan')
        acc_val, = self.host_children['figcan_acc'].plot.plot(0, 0, label='val_acc', color='white')
        self.host_children['figcan_loss'].plot.legend(handles=[loss_train, loss_val])
        self.host_children['figcan_acc'].plot.legend(handles=[acc_train, acc_val])
        


        #test_batch, _ = next(iter(dataloader_train))
        #yhat = Residual_1DCNN(test_batch.float().cuda())
        #make_dot(yhat, params=dict(Residual_1DCNN.named_parameters())).render("rnn_torchviz", format="png")
        self.val_loss = 0
        self.val_acc = 0
        self.loss = 0
        self.acc = 0
        self.average_loss = 0
        self.average_acc = 0
        self.average_val_loss = 0
        self.average_val_acc = 0
        self.percentage = 0
        self.val_percentage = 0
        self.epoch = 0

        for i in range(self.epoch_count):
            self.epoch = i
            track_loss = 0
            track_acc = 0
            track_val_loss = 0
            track_val_acc = 0


            ####################
            # BEGIN TRAINING   #
            ####################
            # honestly val and training should be put into a function cause they basically do the same thing

            Residual_1DCNN.train()
            # run through each batch
            for j, (input_data_batch, output_label_batch) in enumerate(dataloader_train):
                # soemtimes loads in batch size of 16
                if input_data_batch.size()[0] == batch_size:
                    optimizer.zero_grad()
                    self.loss, nn_output_batch = self.operate_nn(input_data_batch.float(), output_label_batch.float(), Residual_1DCNN, loss_func)
                    # backwards propogation based on loss
                    self.loss.backward()
                    optimizer.step()
                    # throw data to the output in the gui
                    track_loss = track_loss + self.loss.item()
                    self.known_values = output_label_batch.cpu().detach().numpy()
                    self.guessed_values = nn_output_batch.cpu().detach().numpy()
                    guess_correct = (np.argmax(self.known_values, axis = 1) == np.argmax(self.guessed_values, axis = 1))
                    self.acc = np.sum(guess_correct) / batch_size
                    track_acc = track_acc + self.val_acc

                    if (j % between_iters == 0):
                        self.percentage = ((j * batch_size) / len(training_files)) * 100

                        self.host.object.event_generate('<<ReadUpdate>>')
            self.average_loss = track_loss  / (j+1)
            self.average_acc = track_acc / (j+1)


            ####################
            # BEGIN VALIDATION #
            ####################

            # enact validation
            Residual_1DCNN.eval()
            # no gradiant activation
            with torch.no_grad():
                # for each image batch in the test
                for j, (val_input_data_batch, val_output_label_batch) in enumerate(dataloader_val):
                    # soemtimes loads in batch size of 16
                    if val_input_data_batch.size()[0] == batch_size:
                        self.val_loss, val_nn_output_batch = self.operate_nn(val_input_data_batch.float(), val_output_label_batch.float(), Residual_1DCNN, loss_func)
                        track_val_loss = track_val_loss + self.val_loss.item()
                        self.known_values = val_output_label_batch.cpu().detach().numpy()
                        self.guessed_values = val_nn_output_batch.cpu().detach().numpy()
                        guess_correct = (np.argmax(self.known_values, axis = 1) == np.argmax(self.guessed_values, axis = 1))
                        self.val_acc = np.sum(guess_correct) / batch_size
                        track_val_acc = track_val_acc + self.val_acc

                    if (j % int(between_iters * (1-train_val_split)) == 0):
                        self.val_percentage = ((j * batch_size) / len(validation_files)) * 100
                        self.host.object.event_generate('<<ReadUpdate>>')
                # average all the validation losses from the testing batches 
                self.average_val_loss = track_val_loss / (j+1)
                self.average_val_acc = track_val_acc / (j+1)
            
            #cv2.imwrite('output_per_epoch/network_input_and_output_at_epoch_' + str(i+1) + '.jpeg', display_info(image_valc[0], image_n_valc[0], output[0], validation=True)*255)
            # if this is the new lowest validation loss
            if self.val_loss < min_val_loss:
                min_val_loss = self.val_loss
                print('new minimum, saved')
                # save the network to hard drive
                torch.save([Residual_1DCNN],'./model/' + file_name + '.pkl')
            # print info to user
            # append list for usage in printing graph
            self.loss_list['epoch'].append(i+1)
            self.loss_list['loss'].append(self.average_loss)
            self.loss_list['val_loss'].append(self.average_val_loss)
            self.loss_list['acc'].append(self.average_acc)
            self.loss_list['val_acc'].append(self.average_val_acc)
            loss_train.set_xdata( self.loss_list['epoch'])
            loss_val.set_xdata( self.loss_list['epoch'])
            acc_train.set_xdata( self.loss_list['epoch'])
            acc_val.set_xdata( self.loss_list['epoch'])
            loss_train.set_ydata( self.loss_list['loss'])
            loss_val.set_ydata( self.loss_list['val_loss'])
            acc_train.set_ydata( self.loss_list['acc'])
            acc_val.set_ydata( self.loss_list['val_acc'])
            self.host.object.event_generate('<<EpochUpdate>>')



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

def restructure():
    data_path = 'network_data/'
    all_files = listdir(data_path)
    # clear up the new path for newly formatted data and also the maximum used to normalize
    new_path = 'network_data_rest/'
    rmtree(new_path)
    makedirs(new_path)
    maximum_path = 'normalize_maximum/'
    rmtree(maximum_path)
    makedirs(maximum_path)
    # the previous classification
    prev = 'k'
    out = 0
    i=0
    maximum_all = 0
    for file in all_files:
        
        # check if it's the same classification, if so keep running on the number counter
        if prev == file.split('_')[0]:
            out = out + i + 1
        else:
            out = 0

        prev = file.split('_')[0]
        input_data = np.fromfile(data_path + file, dtype="float32")
        maximum_input = max(abs(input_data))
        maximum_all = max(maximum_input, maximum_all)
        I_data = input_data[::2]
        Q_data = input_data[1::2]
        input_data = np.vstack((I_data, Q_data))
        leftover_data = len(Q_data) - (len(Q_data) % (8192))
        input_data = np.moveaxis(np.reshape(input_data[:, :leftover_data], (2, 8192, -1), 'F'), -1, 0)
        if file.split('_')[0] == '01' or file.split('_')[0] == '02' or file.split('_')[0] == '03' or file.split('_')[0] == '04':
            for i, frame in enumerate(input_data):
                with open('network_data_rest/' + file.split('_')[0] + '_' + str(i+ out).zfill(8), "bx") as fd:
                    pk.dump(frame, fd)
    with open(maximum_path + 'maximum', 'bx') as fd:
        pk.dump(maximum_all, fd)



if __name__ == "__main__":
    #restructure()
    network = Network()
    network.train_and_eval()