import torch
import torch.nn as nn

class LoRa_1DCNN_obj(nn.Module):
    def __init__(self, num_labels):
        super(LoRa_1DCNN_obj,self).__init__()
        output_size = num_labels

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
        self.conv_layer_init(name='LoRa_1', in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
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


class ResidualExp_1DCNN_obj(nn.Module):
    def __init__(self, num_labels):
        super(ResidualExp_1DCNN_obj,self).__init__()
        output_size = num_labels

        # Leaky ReLU as opposed to ReLU
        self.lrelu = nn.LeakyReLU()

        # 2x2 max pooling, but return the indece of the maximum for unpooling
        self.max = nn.MaxPool2d((1,2))

        # pool together all values into 1x1 indece for each filter
        self.avg = nn.AvgPool2d((1,256))

        self.dropout = nn.Dropout(p=0.4)

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
        self.conv_layer_init(name='LoRa_1', in_channels=1, out_channels=16, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.conv_layer_init(name='LoRa_2', in_channels=16, out_channels=24, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='expansion_2', in_channels=16, out_channels=24, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv_layer_init(name='LoRa_3', in_channels=24, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='expansion_3', in_channels=24, out_channels=32, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv_layer_init(name='LoRa_4', in_channels=32, out_channels=48, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='expansion_4', in_channels=32, out_channels=48, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv_layer_init(name='LoRa_5', in_channels=48, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='expansion_5', in_channels=48, out_channels=64, kernel_size=(1, 1), stride=1, padding=(0, 0))
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
        expansion_2 = self.conv_layer(name='expansion_2', input=down_sized_layer1)

        LoRa_2_layer = self.conv_layer(name='LoRa_2', input=down_sized_layer1)

        skip_layer1 = torch.add(expansion_2, LoRa_2_layer)

        down_sized_layer2 = self.max(skip_layer1) 
        
        # height out = 4096//2 = 2048
        # width out = 2//1 = 2

        expansion_3 = self.conv_layer(name='expansion_3', input=down_sized_layer2)

        LoRa_3_layer = self.conv_layer(name='LoRa_3', input=down_sized_layer2)

        skip_layer2 = torch.add(expansion_3, LoRa_3_layer)

        down_sized_layer3 = self.max(skip_layer2) 

        # height out = 2048//2 = 1024
        # width out = 2//1 = 2
        expansion_4 = self.conv_layer(name='expansion_4', input=down_sized_layer3)

        LoRa_4_layer = self.conv_layer(name='LoRa_4', input=down_sized_layer3)

        skip_layer3 = torch.add(expansion_4, LoRa_4_layer)        

        down_sized_layer4 = self.max(skip_layer3) 

        # height out = 1024//2 = 512
        # width out = 2//1 = 2

        expansion_5 = self.conv_layer(name='expansion_5', input=down_sized_layer4)

        LoRa_5_layer = self.conv_layer(name='LoRa_5', input=down_sized_layer4)

        skip_layer4 = torch.add(expansion_5, LoRa_5_layer)        

        down_sized_layer5 = self.max(skip_layer4) 

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


class Residual_1DCNN_obj(nn.Module):
    def __init__(self, num_labels):
        super(Residual_1DCNN_obj,self).__init__()
        output_size = num_labels

        # Leaky ReLU as opposed to ReLU
        self.lrelu = nn.LeakyReLU()

        # 2x2 max pooling, but return the indece of the maximum for unpooling
        self.max = nn.MaxPool2d((1,2))

        # pool together all values into 1x1 indece for each filter
        self.avg = nn.AvgPool2d((1,256))

        self.dropout = nn.Dropout(p=0.4)

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
        self.conv_layer_init(name='entry_LoRa_1', in_channels=1, out_channels=4, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.conv_layer_init(name='entry_LoRa_2', in_channels=4, out_channels=8, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.conv_layer_init(name='entry_LoRa_3', in_channels=8, out_channels=12, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='res_LoRa_4exp', in_channels=12, out_channels=24, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.res_layer_init(name='res_LoRa_4', in_channels=12, out_channels=24, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.res_layer_init(name='res_LoRa_5', in_channels=24, out_channels=24, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='res_LoRa_6exp', in_channels=24, out_channels=48, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.res_layer_init(name='res_LoRa_6', in_channels=24, out_channels=48, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.res_layer_init(name='res_LoRa_7', in_channels=48, out_channels=48, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_layer_init(name='distinct_LoRa_8', in_channels=48, out_channels=96, kernel_size=(2, 3), stride=1, padding=(0, 1))


        # convert dictionaries to layer modules
        self.conv = nn.ModuleDict(self.conv)
        self.conv_bn = nn.ModuleDict(self.conv_bn)

    def res_layer_init(self, name, in_channels, out_channels, kernel_size, stride, padding):
        self.conv[name+'1'] = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_bn[name+'1'] = nn.BatchNorm2d(out_channels)
        self.conv[name+'2'] = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_bn[name+'2'] = nn.BatchNorm2d(out_channels)

    def conv_layer_init(self, name, in_channels, out_channels, kernel_size, stride, padding):
        self.conv[name] = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_bn[name] = nn.BatchNorm2d(out_channels)

    def conv_layer(self, name, input):
        # convolution layer, batch normalization layer and leaky relu
        output = self.conv[name](input)
        output = self.conv_bn[name](output)
        output = self.lrelu(output)
        return output       
    
    def residual_layer(self, name, input, expansion):
        residual = input
        conv_layer_1 = self.conv_layer(name+'1', residual)
        conv_layer_2 = self.conv_layer(name+'2', conv_layer_1)
        if expansion:
            residual = self.conv_layer(name+'exp', input)
        skip_layer1 = torch.add(residual, conv_layer_2)
        return skip_layer1


    def forward(self, x):
        
        #### INIT
        entry_LoRa_1 = self.conv_layer(name='entry_LoRa_1', input=x)

        down_sized_layer1 = self.max(entry_LoRa_1) 

        # 8192 // 2 = 4096

        entry_LoRa_2 = self.conv_layer(name='entry_LoRa_2', input=down_sized_layer1)

        down_sized_layer2 = self.max(entry_LoRa_2) 

        # 4096 // 2 = 2048

        entry_LoRa_3 = self.conv_layer(name='entry_LoRa_3', input=down_sized_layer2)

        down_sized_layer3 = self.max(entry_LoRa_3) 

        # 2048 // 2 = 1024

        res_LoRa_4 = self.residual_layer(name='res_LoRa_4', input=down_sized_layer3, expansion=True)

        res_LoRa_5 = self.residual_layer(name='res_LoRa_5', input=res_LoRa_4, expansion=False)

        down_sized_layer5 = self.max(res_LoRa_5) 

        # 1024 // 2 = 512

        res_LoRa_6 = self.residual_layer(name='res_LoRa_6', input=down_sized_layer5, expansion=True)

        res_LoRa_7 = self.residual_layer(name='res_LoRa_7', input=res_LoRa_6, expansion=False)

        down_sized_layer7 = self.max(res_LoRa_7) 

        # 512 // 2 = 256

        distinct_LoRa_8 = self.conv_layer(name='distinct_LoRa_8', input=down_sized_layer7)

        down_sized_layer8 = self.avg(distinct_LoRa_8)


        flattened_conv = self.flatten(down_sized_layer8)
        
        bottlenecked_layer = self.bottleneck(flattened_conv)

        activated_bottleneck = self.lrelu(bottlenecked_layer)

        dropout_bttlnck = self.dropout(activated_bottleneck)

        linear_2 = self.outlinear(dropout_bttlnck)

        output = self.softmax(linear_2)



        return output
