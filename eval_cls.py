import numpy as np
import argparse

import torch
from models import cls_model, cls_ppp, cls_tra
from utils import create_dir, viz_cls

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--model', type=str, default='cls', help='Random seed')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    if args.model == 'cls':
        model = cls_model().to(args.device)
    elif args.model == 'cls_ppp':
        model = cls_ppp().to(args.device)
    elif args.model == 'cls_tra':
        model = cls_tra().to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/{}/{}.pt'.format(args.model,args.load_checkpoint)
    print(model_path)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data.to(args.device))
    pred_label = torch.argmax(pred_label, dim=1).to('cpu')

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Save Visualization    
    viz_cls(test_data[args.i], "{}/{}_pred{}_gt{}.gif".format(args.output_dir, args.model, pred_label[args.i], int(test_label[args.i].item())), args.device)