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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', action='store_true', help='Visualize the classification model')
    parser.add_argument('--seg', action='store_true', help='Visualize the classification model')
    args = parser.parse_args()

    if args.cls:
        model = cls_model(num_classes=3)
        print(model)
    
    if args.seg:
        model = seg_model(num_seg_classes=6)
        print(model)