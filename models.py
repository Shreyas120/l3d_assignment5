import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class PointTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super(PointTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = v = src2
        src2 = self.self_attn(q, k, v)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(F.relu(self.dropout(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class cls_tra(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_tra, self).__init__()
        self.embedding = nn.Linear(3, 64)
        self.pos_encoder = PositionalEncoding(64)
        self.transformer = PointTransformerLayer(64)
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
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
        points = self.embedding(points)  # (B, N, 64)
        points = self.pos_encoder(points)
        points = points.transpose(0, 1)  # Transformer expects (N, B, D)
        points = self.transformer(points)
        points = points.transpose(0, 1)  # Back to (B, N, D)
        points = torch.max(points, dim=1)[0]  # Global max pooling
        points = self.fc(points)
        return points

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', action='store_true', help='Visualize the classification model')
    parser.add_argument('--seg', action='store_true', help='Visualize the segmentation model')
    parser.add_argument('--cls_ppp', action='store_true', help='Visualize the pointnet++ cls model')
    parser.add_argument('--cls_tra', action='store_true', help='Visualize the point transformer cls model')
    args = parser.parse_args()

    if args.cls:
        model = cls_model(num_classes=3)
        print(model)
    
    if args.seg:
        model = seg_model(num_seg_classes=6)
        print(model)
    
    if args.cls_ppp:
        model = cls_ppp(num_classes=3)
        print(model)

    if args.cls_tra:
        model = cls_tra(num_classes=3)
        print(model)