from tkinter import filedialog
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import pickle as pk
from os import listdir
from .pytorch_layers import Dataset
from random import sample
from PIL import Image, ImageTk
from code_tkinter.tkinter_objects import recurs_return_all
import cv2
import os
from threading import Thread, Event as Event_Th
dir_path = os.path.dirname(os.path.realpath(__file__))

class Test_Network():
    def __init__(self, host):
        self.host = host
        # iterate through frames to find the ones we want
        self.host_children = {}
        # get a dict of all the objects that are a child to the setup train tab
        recurs_return_all(self.host, self.host_children)

        # fairly thread safe way of telling main gui to display a change
        self.host.object.bind('<<ReadUpdate>>', lambda e: self.on_read_update(e))

        self.host_children['button_nnloadmodel'].object.configure(command=self.load_network)
        self.host_children['button_nnloaddata'].object.configure(command=self.load_data_folder)
        self.host_children['button_nntest'].object.configure(command=self.begin_test_thread)

    def load_network(self):
        file_name = filedialog.askopenfilename(initialdir = dir_path,
                                              title = "Select a File",
                                              filetypes = (("Text files", "*.pkl*"),("all files", "*.*")))
    
        # Change label contents
        self.host_children['label_nnloadedfile'].object.configure(text="File Opened: "+file_name)

        self.model_c = torch.load(file_name)[0]


    def load_data_folder(self):
        folder_name = filedialog.askdirectory(initialdir = dir_path,
                                              title = "Select a Folder")
    
        # Change label contents
        self.host_children['label_nndatafolder'].object.configure(text="Data Folder: "+folder_name)

        self.input_path = folder_name + '/'


    def on_read_update(self, event):
        guessed_values = cv2.resize(self.guessed_values*255, (240, 240), interpolation= cv2.INTER_NEAREST)
        known_values = cv2.resize(self.known_values*255, (240, 240), interpolation= cv2.INTER_NEAREST)
        im = Image.fromarray(known_values)
        imgtk = ImageTk.PhotoImage(image=im) 
        
        self.host_children['label_nntknown'].object['image'] = self.host_children['label_nntknown'].object.img = imgtk
        im = Image.fromarray(guessed_values)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nntguess'].object['image'] = self.host_children['label_nntguess'].object.img = imgtk
        conversion = cv2.convertScaleAbs((255-abs(known_values-guessed_values)), alpha=1)
        depth_colormap = cv2.applyColorMap(conversion, 11)
        im = Image.fromarray(depth_colormap)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nntdiff'].object['image'] = self.host_children['label_nntdiff'].object.img = imgtk

        #double_guess_array = self.guess_correct[:, None]*255
        color_array = np.zeros((self.batch_size, 1, 3), dtype=np.uint8)
        x=np.arange(32, dtype=int)
        y=np.zeros((32), dtype=int)
        z=self.guess_correct.astype('int')
        color_array[x, y, z] = 255
        im = Image.fromarray(cv2.resize(color_array, (240, 240), interpolation= cv2.INTER_NEAREST))
        imgtk = ImageTk.PhotoImage(image=im) 
        self.host_children['label_nntcorrect'].object['image'] = self.host_children['label_nntcorrect'].object.img = imgtk

        self.host_children['label_nntavgloss'].object.configure(text=str(self.average_test_loss))

        self.host_children['label_nntavgacc'].object.configure(text=str(self.average_test_acc))
        self.host_children['label_nntpercentage'].object.configure(text=str(self.test_percentage))


    def operate_nn(self, input_image_batch, output_label_batch, Residual_CNN, loss_func):
        # convert image inputs to gpu
        cinput_image_batch = Variable(input_image_batch).cuda()
        coutput_label_batch = Variable(output_label_batch).cuda()
        # encode the image, get max pool indeces and skip connections
        nn_output_batch = Residual_CNN(cinput_image_batch)
        # calculate loss using MSE
        loss = loss_func(nn_output_batch,coutput_label_batch)

        return loss, nn_output_batch

    def begin_test_thread(self):
        start_testing_th = Thread(target=self.test_model)
        start_testing_th.start()


    def test_model(self):  

        self.batch_size = 32
        with open('number_classes/number_classes', "rb") as fd:
            self.num_labels = pk.load(fd)

        with open('normalize_maximum/maximum', "rb") as fd:
            max = pk.load(fd)
        # check if torch is running on a gpu
        avail = torch.cuda.is_available()
        if (avail):
            print(torch.cuda.get_device_name(0))

        all_files = listdir(self.input_path)
        # shuffle the file names for splitting
        file_array_shuffle = sample( all_files, 10000 )
        
        print(len(file_array_shuffle))
        
        dataset_test = Dataset(self.input_path, file_array_shuffle, max, self.num_labels)

        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)

        # mean squared error loss calculation
        loss_func = nn.CrossEntropyLoss()

        # enact validation
        self.model_c.eval()

        track_test_loss = 0
        track_test_acc = 0
        track_MSE = np.zeros((self.num_labels))
        track_acc_indi = np.zeros((self.num_labels))
        check_guess = np.zeros((self.num_labels))
        check_known  = np.zeros((self.num_labels))
        # no gradiant activation
        with torch.no_grad():
            # for each image batch in the test
            for j, (test_input_data_batch, test_output_label_batch) in enumerate(dataloader_test):
                # soemtimes loads in batch size of 16
                if test_input_data_batch.size()[0] == self.batch_size:
                    # track validation loss and the validation output batch
                    self.test_loss, test_nn_output_batch = self.operate_nn(test_input_data_batch.float(), test_output_label_batch.float(), self.model_c, loss_func)
                    # track the validation loss to average
                    track_test_loss = track_test_loss + self.test_loss.item()
                    # convert the known and guessed values back to cpu format for numpy files
                    self.known_values = test_output_label_batch.cpu().detach().numpy()
                    self.guessed_values = test_nn_output_batch.cpu().detach().numpy()
                    track_MSE = track_MSE + np.sum((self.known_values - self.guessed_values) ** 2, axis=0)
                    self.guess_correct = (np.argmax(self.known_values, axis = 1) == np.argmax(self.guessed_values, axis = 1))
                    unique_k, counts_known = np.unique(np.argmax(self.known_values, axis = 1), return_counts=True)
                    TP_and_FN = (np.argmax(self.guessed_values, axis = 1) + 1) * self.guess_correct - 1
                    unique_g, counts_guess = np.unique(np.hstack((TP_and_FN, -1)), return_counts=True)
                    check_guess[np.asarray(unique_g[1:])] = counts_guess[1:]
                    check_known[np.asarray(unique_k)] = counts_known
                    track_acc_indi = track_acc_indi + (check_guess/(check_known+1e-15))
                    self.test_acc = np.sum(self.guess_correct) / self.batch_size
                    track_test_acc = track_test_acc + self.test_acc

                if (j % 100 == 0):
                    self.test_percentage = ((j * self.batch_size) / len(file_array_shuffle)) * 100
                    # average all the validation losses from the testing batches 
                    self.average_test_loss = track_test_loss / (j+1)
                    self.average_test_acc = track_test_acc / (j+1)
                    self.average_test_MSE = track_MSE / (j+1)
                    self.average_test_accindi = track_acc_indi / (j+1)
                    self.host.object.event_generate('<<ReadUpdate>>')
            self.host.object.event_generate('<<ReadUpdate>>')
