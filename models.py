import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------

def conv_bn(in_channels, out_channels):
    """Helper function to create a convolutional layer followed by batch normalization."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        # Convolution and batch normalization layers
        self.layers = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 64),
            conv_bn(64, 128),
            conv_bn(128, 1024)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Transpose to match the dimensionality for conv1d (B, C, N)
        points = points.transpose(1, 2)

        # Pass through the convolution and batch norm layers
        out = self.layers(points)

        # Global max pooling
        out = torch.max(out, dim=2)[0]

        # Pass through the fully connected layers
        out = self.fc(out)

        return out

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # Convolution and batch normalization layers
        self.conv_bn_1 = conv_bn(3, 64)
        self.conv_bn_2 = conv_bn(64, 64)
        self.conv_bn_3 = conv_bn(64, 128)
        self.conv_bn_4 = conv_bn(128, 1024)

        # Point-wise layers for segmentation
        self.point_layer = nn.Sequential(
            conv_bn(1088, 512),
            conv_bn(512, 256),
            conv_bn(256, 128),
            nn.Conv1d(128, num_seg_classes, 1)  # Final layer does not need BN and ReLU
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        N = points.shape[1]
        points = points.transpose(1, 2)

        local_out = self.conv_bn_1(points)
        local_out = self.conv_bn_2(local_out)

        global_out = self.conv_bn_3(local_out)
        global_out = self.conv_bn_4(global_out)
        global_out = torch.amax(global_out, dim=-1, keepdims=True).repeat(1, 1, N)

        out = torch.cat((local_out, global_out), dim=1)
        out = self.point_layer(out).transpose(1, 2)  # Transpose back for correct output shape

        return out

class SA_Layer(nn.Module):
    """
    Simplified version of a set abstraction layer
    It groups points (implicitly here) &
    applies a series of 1D convolutional transformations to extract features.
    """
    def __init__(self, in_channels, mlp_channels):
        super(SA_Layer, self).__init__()
        self.conv = nn.Sequential()
        current_channels = in_channels
        for out_channels in mlp_channels:
            self.conv.add_module('conv_bn_{}'.format(out_channels),
                                 conv_bn(current_channels, out_channels))
            current_channels = out_channels

    def forward(self, x):
        return self.conv(x)

class cls_ppp(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_ppp, self).__init__()
        
        self.conv0 = conv_bn(3, 64)
        # Set Abstraction layers
        self.sa1 = SA_Layer(3, [64, 64, 128])
        self.sa2 = SA_Layer(128, [128, 128, 256])
        self.sa3 = SA_Layer(256, [256, 512, 1024])

        self.sa = nn.Sequential(
            self.conv0,
            self.sa1,
            self.sa2,
            self.sa3
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        points = points.transpose(1, 2)
        B, _, _ = points.size()

        # Applying Set Abstraction layers
        sa_out = self.sa(points)
        # Global max pooling
        out = F.adaptive_max_pool1d(sa_out, 1).view(B, -1)

        # Fully connected layers
        out = self.fc(out)
        return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', action='store_true', help='Visualize the classification model')
    parser.add_argument('--seg', action='store_true', help='Visualize the segmentation model')
    parser.add_argument('--cls_ppp', action='store_true', help='Visualize the pointnet++ cls model')
    args = parser.parse_args()

    if args.cls:
        model = cls_model(num_classes=3)
        print(model)
    
    if args.seg:
        model = seg_model(num_seg_classes=6)
        print(model)
    
    if args.cls_ppp:
        model = cls_ppp(num_classes=3)
        # print(model)