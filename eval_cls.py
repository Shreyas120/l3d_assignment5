import numpy as np
import argparse

import torch
from models import cls_model, cls_ppp, cls_tra
from utils import *

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
    
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    parser.add_argument('--viz', action='store_true', help='Visualize the results')
    parser.add_argument('--rot_range', action='store_true', help='Rotate theinput point cloud')
    parser.add_argument('--vary_pts', action='store_true', help='Vary the number of points in the input point cloud')
    
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
    test_label = torch.from_numpy(np.load(args.test_label))
   
    # Save Visualization    
    if args.viz: 
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
        pred_label = torch.argmax(pred_label, dim=1).to('cpu')

        # Compute Accuracy
        res = pred_label.eq(test_label.data).cpu()
        test_accuracy =res.sum().item() / (test_label.size()[0])
        print ("test accuracy: {}".format(test_accuracy))

        viz_conf_mat(test_label, pred_label, [f"Class {i}" for i in range(args.num_cls_class)], "{}/{}/conf_mat.png".format(args.output_dir, args.model))

        tp_idxs = np.argwhere(res==True)[0]
        for idx in np.random.choice(tp_idxs, 3, replace=False):
            viz_cls(test_data[idx], "{}/{}/pred{}_gt{}_TP{}.gif".format(args.output_dir, args.model, pred_label[idx], int(test_label[idx].item()), idx), args.device)

        fp_idxs = np.argwhere(res==False)[0]
        cl = np.arange(0, args.num_cls_class)
        for idx in fp_idxs:
            if int(test_label[idx].item()) in cl:
                cl = np.delete(cl, np.where(cl == int(test_label[idx].item())))
                viz_cls(test_data[idx], "{}/{}/pred{}_gt{}_FP{}.gif".format(args.output_dir, args.model, pred_label[idx], int(test_label[idx].item()), idx), args.device)
    
    if args.rot_range:
        # degs = list(range(0, 10, 1)) + list(range(10, 25, 5)) + list(range(30, 190, 20))
        degs = list(range(0, 360, 3))
        accuracies = []
        from tqdm import tqdm
        for d in tqdm(degs): 
            torch.cuda.empty_cache()
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])

            R = get_rotation_matrix_tensor(d)
            test_data = R @ test_data.permute(0,2,1)
            test_data = test_data.permute(0,2,1)
            pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
            pred_label = torch.argmax(pred_label, dim=1).to('cpu')
            # Compute Accuracy
            res = pred_label.eq(test_label.data).cpu()
            test_accuracy =res.sum().item() / (test_label.size()[0])
            accuracies.append(test_accuracy*100)

        import matplotlib.pyplot as plt
        plt.plot(degs, accuracies, marker='x', linestyle='-', color='b', markersize=1)
        plt.xlabel('Rotation Magnitude (degrees)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Rotation Magnitude')
        plt.savefig("{}/{}/accuracy_vs_rotation.png".format(args.output_dir, args.model))

    if args.vary_pts:
        npts = list(range(100, 1000, 200)) + list(range(1000, 5000, 500)) +list(range(5000, 10000, 1000))
        accuracies = []

        from tqdm import tqdm
        for npt in tqdm(npts): 
            torch.cuda.empty_cache()

            ind = np.random.choice(10000, npt, replace=False)
            test_label = torch.from_numpy(np.load(args.test_label))
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])

            pred_label = eval_in_batches(test_data, model, args.device, args.batch_size)
            pred_label = torch.argmax(pred_label, dim=1).to('cpu')

            # Compute Accuracy
            res = pred_label.eq(test_label.data).cpu()
            test_accuracy =res.sum().item() / (test_label.size()[0])
            accuracies.append(test_accuracy*100)

        import matplotlib.pyplot as plt
        plt.plot(npts, accuracies, marker='x', linestyle='-', color='b', markersize=1)
        plt.xlabel('Number of points')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Points')
        plt.savefig("{}/{}/accuracy_vs_npts.png".format(args.output_dir, args.model))
