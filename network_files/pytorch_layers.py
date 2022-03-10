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
from torchviz import make_dot
from network_files.model_architectures import Residual_1DCNN_obj, ResidualExp_1DCNN_obj, LoRa_1DCNN_obj
from code_tkinter.tkinter_objects import recurs_return_all
from threading import Thread, Event as Event_Th
from PIL import Image, ImageTk


import cv2

class Dataset(torch.utils.data.Dataset):
    """
    initialization of the labels 
    """
    def __init__(self, data_path, file_list, maximum, num_labels):
        self.data_path = data_path
        self.file_list = file_list
        self.maximum = maximum
        self.num_labels = num_labels

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
        output_label = F.one_hot(torch.tensor([int(file.split('_')[0])-1]), num_classes=self.num_labels)
        return input_data[None, :, :], output_label[0]

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

        #double_guess_array = self.guess_correct[:, None]*255
        color_array = np.zeros((self.batch_size, 1, 3), dtype=np.uint8)
        x=np.arange(32, dtype=int)
        y=np.zeros((32), dtype=int)
        z=self.guess_correct.astype('int')
        color_array[x, y, z] = 255
        im = Image.fromarray(cv2.resize(color_array, (240, 240), interpolation= cv2.INTER_NEAREST))
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nncorrect'].object['image'] = self.host_children['label_nncorrect'].object.img = imgtk

        # find the loss label and overide the current text with the current loss
        self.host_children['label_nnloss'].object.configure(text=str(self.loss.item()))
        # check if the validation has started
        if (self.val_loss != 0):
            # if so, overrite the validation loss text
            self.host_children['label_nnvalloss'].object.configure(text=str(self.val_loss.item()))
        # overrite the average loss text
        self.host_children['label_nnavgloss'].object.configure(text=str(self.average_loss))
        # overrite the average validation loss text
        self.host_children['label_nnavgvloss'].object.configure(text=str(self.average_val_loss))
        # overrite the accuracy text
        self.host_children['label_nnacc'].object.configure(text=str(self.acc.item()))
        # check if the validation has started
        if (self.val_acc != 0):
            # overrite the validation accuracy text
            self.host_children['label_nnvalacc'].object.configure(text=str(self.val_acc.item()))
        # overrite the average accuracy 
        self.host_children['label_nnavgacc'].object.configure(text=str(self.average_acc))
        # overrite the validation accuracy text
        self.host_children['label_nnavgvacc'].object.configure(text=str(self.average_val_acc))

        self.host_children['label_nnpercentage'].object.configure(text=str(self.percentage))
        self.host_children['label_nnvpercentage'].object.configure(text=str(self.val_percentage))
        self.host_children['label_nnepoch'].object.configure(text=str(self.epoch) + '/' + str(self.epoch_count))

    def update_plot(self, plot_host, parameter):
        ax = plot_host.fig.axes[0]
        ax.set_xlim( min(self.loss_list['epoch']),  max(self.loss_list['epoch']))
        all_items = [self.loss_list[k] for k in parameter]
        ylim_max =  np.max(all_items)
        ylim_min =  np.min(all_items)
        ax.set_ylim(ylim_min - 0.1*ylim_max, 1.1*ylim_max)    
        plot_host.object.draw()
        plot_host.object.flush_events()

    def on_epoch_update(self, event):
        self.update_plot(self.host_children['figcan_loss'], ['loss', 'val_loss'])
        self.update_plot(self.host_children['figcan_acc'], ['acc', 'val_acc'])
        parameter_indi_loss = []
        parameter_indi_acc = []
        for i in range(self.num_labels):
            parameter_indi_loss.append('val_loss' + str(i))
            parameter_indi_acc.append('val_acc' + str(i))
        self.update_plot(self.host_children['figcan_indiloss'], parameter_indi_loss)
        self.update_plot(self.host_children['figcan_indiacc'], parameter_indi_acc)


    def begin_train_thread(self):
        start_training_th = Thread(target=self.train_and_eval)
        start_training_th.start()

    def train_and_eval(self):
        with open('number_classes/number_classes', "rb") as fd:
            self.num_labels = pk.load(fd)

        model = Residual_1DCNN_obj(self.num_labels)

        learning_rate = self.host_children['entry_learning'].variable.get()
        self.epoch_count = self.host_children['entry_epoch'].variable.get()
        self.batch_size = self.host_children['entry_batch'].variable.get()
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
        file_array_shuffle = sample( all_files, len(all_files) )

        
        
        # split the validation and training files
        training_files=file_array_shuffle[:int(len(file_array_shuffle)*train_val_split)]
        validation_files=file_array_shuffle[int(len(file_array_shuffle)*train_val_split):]
        
        print(len(validation_files))
        print(len(training_files))
        
        dataset_train = Dataset(input_path, training_files, max, self.num_labels)
        dataset_val = Dataset(input_path, validation_files, max, self.num_labels)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True)


        model_c = model.cuda()

        parameters = list(model_c.parameters())
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
        for i in range(self.num_labels):
            self.loss_list['val_loss'+str(i)] = []
            self.loss_list['val_acc'+str(i)] = []

        loss_train, = self.host_children['figcan_loss'].plot.plot(0, 0, label='loss', color='cyan')
        loss_val, = self.host_children['figcan_loss'].plot.plot(0, 0, label='val_loss', color='white')
        acc_train, = self.host_children['figcan_acc'].plot.plot(0, 0, label='acc', color='cyan')
        acc_val, = self.host_children['figcan_acc'].plot.plot(0, 0, label='val_acc', color='white')
        self.host_children['figcan_loss'].plot.legend(handles=[loss_train, loss_val])
        self.host_children['figcan_acc'].plot.legend(handles=[acc_train, acc_val])


        loss_val_indi = []
        acc_val_indi = []
        for i in range(self.num_labels):
            k, = self.host_children['figcan_indiloss'].plot.plot(0, 0, label='vMSE_dev_' + str(i), color=((0.5+0.14*i)%1, (0.3+0.22*i)%1, (0.7+0.27*i)%1))
            loss_val_indi.append(k)
            k, = self.host_children['figcan_indiacc'].plot.plot(0, 0, label='vTPvsFN_dev_' + str(i), color=((0.5+0.14*i)%1, (0.3+0.22*i)%1, (0.7+0.27*i)%1))
            acc_val_indi.append(k)
        self.host_children['figcan_indiloss'].plot.legend(handles=loss_val_indi)
        self.host_children['figcan_indiacc'].plot.legend(handles=acc_val_indi)

        


        test_batch, _ = next(iter(dataloader_train))
        yhat = model_c(test_batch.float().cuda())
        make_dot(yhat, params=dict(model_c.named_parameters())).render("rnn_torchviz", format="png")
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
            track_MSE = np.zeros((self.num_labels))
            track_acc_indi = np.zeros((self.num_labels))
            check_guess = np.zeros((self.num_labels))
            check_known  = np.zeros((self.num_labels))



            ####################
            # BEGIN TRAINING   #
            ####################
            # honestly val and training should be put into a function cause they basically do the same thing

            model_c.train()
            # run through each batch
            for j, (input_data_batch, output_label_batch) in enumerate(dataloader_train):
                # soemtimes loads in batch size of 16
                if input_data_batch.size()[0] == self.batch_size:
                    optimizer.zero_grad()
                    self.loss, nn_output_batch = self.operate_nn(input_data_batch.float(), output_label_batch.float(), model_c, loss_func)
                    # backwards propogation based on loss
                    self.loss.backward()
                    optimizer.step()
                    # throw data to the output in the gui
                    track_loss = track_loss + self.loss.item()
                    self.known_values = output_label_batch.cpu().detach().numpy()
                    self.guessed_values = nn_output_batch.cpu().detach().numpy()
                    self.guess_correct = (np.argmax(self.known_values, axis = 1) == np.argmax(self.guessed_values, axis = 1))
                    self.acc = np.sum(self.guess_correct) / self.batch_size
                    track_acc = track_acc + self.acc

                    if (j % between_iters == 0):
                        self.percentage = ((j * self.batch_size) / len(training_files)) * 100

                        self.host.object.event_generate('<<ReadUpdate>>')
                        
            self.average_loss = track_loss  / (j+1)
            self.average_acc = track_acc / (j+1)


            ####################
            # BEGIN VALIDATION #
            ####################

            # enact validation
            model_c.eval()
            # no gradiant activation
            with torch.no_grad():
                # for each image batch in the test
                for j, (val_input_data_batch, val_output_label_batch) in enumerate(dataloader_val):
                    # soemtimes loads in batch size of 16
                    if val_input_data_batch.size()[0] == self.batch_size:
                        # track validation loss and the validation output batch
                        self.val_loss, val_nn_output_batch = self.operate_nn(val_input_data_batch.float(), val_output_label_batch.float(), model_c, loss_func)
                        # track the validation loss to average
                        track_val_loss = track_val_loss + self.val_loss.item()
                        # convert the known and guessed values back to cpu format for numpy files
                        self.known_values = val_output_label_batch.cpu().detach().numpy()
                        self.guessed_values = val_nn_output_batch.cpu().detach().numpy()
                        track_MSE = track_MSE + np.sum((self.known_values - self.guessed_values) ** 2, axis=0)
                        self.guess_correct = (np.argmax(self.known_values, axis = 1) == np.argmax(self.guessed_values, axis = 1))
                        unique_k, counts_known = np.unique(np.argmax(self.known_values, axis = 1), return_counts=True)
                        TP_and_FN = (np.argmax(self.guessed_values, axis = 1) + 1) * self.guess_correct - 1
                        unique_g, counts_guess = np.unique(np.hstack((TP_and_FN, -1)), return_counts=True)
                        check_guess[np.asarray(unique_g[1:])] = counts_guess[1:]
                        check_known[np.asarray(unique_k)] = counts_known
                        track_acc_indi = track_acc_indi + (check_guess/(check_known+1e-15))
                        self.val_acc = np.sum(self.guess_correct) / self.batch_size
                        track_val_acc = track_val_acc + self.val_acc

                    if (j % int(between_iters * (1-train_val_split)) == 0):
                        self.val_percentage = ((j * self.batch_size) / len(validation_files)) * 100
                        self.host.object.event_generate('<<ReadUpdate>>')
                # average all the validation losses from the testing batches 
                self.average_val_loss = track_val_loss / (j+1)
                self.average_val_acc = track_val_acc / (j+1)
                self.average_val_MSE = track_MSE / (j+1)
                self.average_val_accindi = track_acc_indi / (j+1)
            
            #cv2.imwrite('output_per_epoch/network_input_and_output_at_epoch_' + str(i+1) + '.jpeg', display_info(image_valc[0], image_n_valc[0], output[0], validation=True)*255)
            # if this is the new lowest validation loss
            if self.val_loss < min_val_loss:
                min_val_loss = self.val_loss
                print('new minimum, saved')
                # save the network to hard drive
                torch.save([model_c],'./model/' + file_name + '.pkl')
            # print info to user
            # append list for usage in printing graph
            self.loss_list['epoch'].append(i+1)
            self.loss_list['loss'].append(self.average_loss)
            self.loss_list['val_loss'].append(self.average_val_loss)
            self.loss_list['acc'].append(self.average_acc)
            self.loss_list['val_acc'].append(self.average_val_acc)
            for i in range(self.num_labels):
                self.loss_list['val_loss' + str(i)].append(self.average_val_MSE[i])
                self.loss_list['val_acc' + str(i)].append(self.average_val_accindi[i])
                loss_val_indi[i].set_xdata( self.loss_list['epoch'])  
                loss_val_indi[i].set_ydata( self.loss_list['val_loss' + str(i)])
                acc_val_indi[i].set_xdata( self.loss_list['epoch'])
                acc_val_indi[i].set_ydata( self.loss_list['val_acc' + str(i)])

            loss_train.set_xdata( self.loss_list['epoch'])
            loss_val.set_xdata( self.loss_list['epoch'])
            acc_train.set_xdata( self.loss_list['epoch'])
            acc_val.set_xdata( self.loss_list['epoch'])
            loss_train.set_ydata( self.loss_list['loss'])
            loss_val.set_ydata( self.loss_list['val_loss'])
            acc_train.set_ydata( self.loss_list['acc'])
            acc_val.set_ydata( self.loss_list['val_acc'])
            self.host.object.event_generate('<<EpochUpdate>>')
            
