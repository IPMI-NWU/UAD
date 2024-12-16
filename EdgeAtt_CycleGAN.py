import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

# def init_conv(conv, glu=True):
#     init.xavier_uniform_(conv.weight)
#     if conv.bias is not None:
#         conv.bias.data.zero_()

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class SE_Attention(nn.Module):
    def __init__(self, in_channel):
        super(SE_Attention, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2),
                                    nn.BatchNorm2d(in_channel))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 8, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        b, c, _, _ = out1.size()
        y = self.avg_pool(out2).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Edge_Attention(nn.Module):
    def __init__(self, in_dim, activation=F.relu):
        super(Edge_Attention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        # init_conv(self.f)
        # init_conv(self.g)
        # init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height) # B * C * (W * H)
        
        attention = torch.bmm(f.permute(0, 2, 1), g) # B * (W * H) * (W * H)
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention) # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H
        
        out = self.gamma * self_attetion + x
        return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        encoder = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(3):
            encoder += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            encoder += [ResidualBlock(in_features)]


        # Upsampling
        out_features = in_features//2
        decoder_sub1 = [SE_Attention(in_features)]
        edge_pre_sub1 = [Edge_Attention(in_features)]
        for _ in range(2):
            decoder_sub1 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            edge_pre_sub1 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        
        decoder_sub2 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
        edge_pre_sub2 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]

        # Output layer
        decoder_sub2 += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        edge_pre_sub2 += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder_sub1 = nn.Sequential(*decoder_sub1)
        self.decoder_sub2 = nn.Sequential(*decoder_sub2)
        self.edge_pre_sub1 = nn.Sequential(*edge_pre_sub1)
        self.edge_pre_sub2 = nn.Sequential(*edge_pre_sub2)

    def forward(self, x):
        out1 = self.encoder(x)
        de_out = self.decoder_sub1(out1)
        edge_out = self.edge_pre_sub1(out1)
        de_out = torch.add(de_out, edge_out)
        image_out = self.decoder_sub2(de_out)
        edge_out = self.edge_pre_sub2(edge_out)
        return image_out, edge_out

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)