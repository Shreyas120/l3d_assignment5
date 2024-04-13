import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


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


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data.to(args.device))
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
   
   
    