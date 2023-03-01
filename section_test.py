import argparse

import numpy
import os
import torch
import torchvision

import core.models
from core.loader.data_loader_F3 import *
from core.metrics import runningScore
from core.utils import np_to_tb, append_filter, detect_gabor_edges


def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # logging setup
    log_dir, model_name = os.path.split(args.model_path)
    
    # load model:
    model = torch.load(args.model_path, map_location=device)
    model = model.to(device)
    two_stream = type(model) is core.models.section_two_stream

    class_names = ['upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    if "both" in args.split: 
        splits = ["test1", "test2"]
    else:
        splits = [args.split]

    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = numpy.load(os.path.join('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape

        if args.inline:
            i_list = list(range(irange))
            i_list = ['i_'+str(inline) for inline in i_list]
        else:
            i_list = []

        if args.crossline:
            x_list = list(range(xrange))
            x_list = ['x_'+str(crossline) for crossline in x_list]
        else:
            x_list = []

        list_test = i_list + x_list

        file_object = open(os.path.join('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()

        test_set = section_dataset(channel_delta=args.channel_delta, split=split, is_transform=True, augmentations=None)
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      num_workers=4,
                                      shuffle=False)

        # print the results of this split:
        running_metrics_split = runningScore(n_classes)

        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for batch, (images, labels) in enumerate(test_loader):
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels
                if two_stream:
                    gabors = detect_gabor_edges(images, frequency=0.1)
                    images, gabors, labels = images.to(device), gabors.to(device), labels.to(device)
                    outputs = model(images, gabors)
                else:
                    if args.filter is not None:
                        images = append_filter(images, args.filter)
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                pred = outputs.detach().max(1)[1].cpu().numpy()
                gt = labels.detach().cpu().numpy()
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)

                numbers = [0, 99, 149, 399, 499]

                if batch in numbers:
                    tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)

                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(numpy.squeeze(labels_original), save_name=os.path.join(log_dir,f'plot_grount_truth_{split}_{str(batch)}.pdf'))
                    out = torch.nn.functional.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(numpy.squeeze(prediction), save_name=os.path.join(log_dir,f'plot_predictions_{split}_{str(batch)}.pdf'))

                    # uncomment if you want to visualize the different class heatmaps
                    unary = outputs.cpu().detach()
                    unary_max = torch.max(unary)
                    unary_min = torch.min(unary)
                    unary = unary.add((-1*unary_min))
                    unary = unary/(unary_max - unary_min)

                    for channel in range(0, len(class_names)):
                        decoded_channel = unary[0][channel]
                        tb_channel = torchvision.utils.make_grid(decoded_channel, normalize=True, scale_each=True)

        # get scores
        score, class_iou = running_metrics_split.get_scores()
        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    print('--------------- FINAL RESULTS -----------------')
    print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names):
        print(f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
    print('Confusion Matrix', score['confusion_matrix'])

    # Save confusion matrix: 
    numpy.savetxt(os.path.join(log_dir,'confusion.csv'), score['confusion_matrix'], delimiter=" ")
    numpy.save(os.path.join(log_dir,'score'), score, allow_pickle=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--model_path',    type=str,                             help='Path to the saved model')
    parser.add_argument('--channel_delta', type=int,             default=0,      help='# of variable input channels')
    parser.add_argument('--device',        type=str,             default='cpu',  help='Cuda device or cpu execution')
    parser.add_argument('--filter',        type=str,             default='None', help='Add filter as an extra channel/layer', choices=['None', 'gabor','hessian', 'sobel'])
    
    parser.add_argument('--split',         type=str,  nargs='?', default='both', help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline',     type=bool, nargs='?', default=True,   help='whether to test in crossline mode')
    parser.add_argument('--inline',        type=bool, nargs='?', default=True,   help='whether to test inline mode')
    
    custom_params = [
        '--model_path', 'runs/20230228_114334_section_deconvnet_aug_weighted_delta=0_filter=gabor/section_deconvnet_model.pkl',
        '--channel_delta', '0',
        '--device', 'cuda:0',
        '--filter', 'gabor',
        '--split', 'both',
    ]

    args = parser.parse_args(None)
    test(args)
