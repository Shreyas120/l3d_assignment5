import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import *


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=128, help='The number of images in a batch.')
    
    parser.add_argument('--viz', action='store_true', help='Visualize the results')
    parser.add_argument('--rot_range', action='store_true', help='Rotate theinput point cloud')
    parser.add_argument('--vary_pts', action='store_true', help='Vary the number of points in the input point cloud')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    if args.viz: 
        # Sample Points per Object
        ind = np.random.choice(10000,args.num_points, replace=False)
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

        # ------ TO DO: Make Prediction ------
        pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
        pred_label = torch.argmax(pred_label, dim=2).to('cpu')

        res =  pred_label.eq(test_label.data).cpu()
        # per object accuracy
        test_accuracy_objs = res.sum(dim=1)/args.num_points

        # overall accuracy
        test_accuracy = (test_accuracy_objs.sum()/res.shape[0]).item()
        print ("test accuracy: {}".format(test_accuracy))

        sorted_results = torch.argsort(test_accuracy_objs)
        
        for idx in torch.cat((sorted_results[:2], sorted_results[-3:])):
            idx = idx.item()
            viz_seg(test_data[idx], test_label[idx], "{}/gt_{}.gif".format(args.output_dir, idx), args.device)
            viz_seg(test_data[idx], pred_label[idx], "{}/pred_{}_acc{}.gif".format(args.output_dir, idx, int(test_accuracy_objs[idx]*100)), args.device)
   
    if args.rot_range:
        ind = np.random.choice(10000,args.num_points, replace=False)
        test_label = torch.from_numpy(np.load(args.test_label))
        # degs = list(range(0, 10, 1)) + list(range(10, 25, 5)) + list(range(30, 190, 20))
        degs = list(range(0, 360, 10))
        accuracies = []
        from tqdm import tqdm
        for d in tqdm(degs): 
            torch.cuda.empty_cache()
            # Sample Points per Object
            ind = np.random.choice(10000,args.num_points, replace=False)
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
            test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
            
            R = get_rotation_matrix_tensor(d)
            test_data = R @ test_data.permute(0,2,1)
            test_data = test_data.permute(0,2,1)
            pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
            pred_label = torch.argmax(pred_label, dim=2).to('cpu')

            res =  pred_label.eq(test_label.data).cpu()
            # per object accuracy
            test_accuracy_objs = res.sum(dim=1)/args.num_points

            # overall accuracy
            test_accuracy = (test_accuracy_objs.sum()/res.shape[0]).item()
            accuracies.append(test_accuracy*100)

        import matplotlib.pyplot as plt
        plt.plot(degs, accuracies, marker='x', linestyle='-', color='b', markersize=1)
        plt.xlabel('Rotation Magnitude (degrees)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Rotation Magnitude')
        plt.savefig("{}/accuracy_vs_rotation.png".format(args.output_dir))

    if args.vary_pts:
        npts = list(range(100, 1000, 200)) + list(range(1000, 5000, 500)) +list(range(5000, 10000, 1000))
        accuracies = []

        from tqdm import tqdm
        for npt in tqdm(npts): 
            torch.cuda.empty_cache()

            ind = np.random.choice(10000, npt, replace=False)
            test_label = torch.from_numpy(np.load(args.test_label))[:,ind]
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])

            pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
            pred_label = torch.argmax(pred_label, dim=2).to('cpu')

            test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
            accuracies.append(test_accuracy*100)

        import matplotlib.pyplot as plt
        plt.plot(npts, accuracies, marker='x', markersize=2)
        plt.xlabel('Number of points')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Points')
        plt.savefig("{}/accuracy_vs_npts.png".format(args.output_dir))
