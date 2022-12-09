import argparse


import numpy
import torch
import torchvision

from os.path import join as pjoin
from tensorboardX import SummaryWriter

from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb


def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    log_dir, model_name = os.path.split(args.model_path)
    # load model:
    model = torch.load(args.model_path, map_location=device)
    model = model.to(device)  # Send to GPU if available
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ['upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    if "both" in args.split: 
        splits = ["test1", "test2"]
    else:
        splits = [args.split]

    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = numpy.load(pjoin('data', 'test_once', split + '_labels.npy'))
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

        file_object = open(pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
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
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                pred = outputs.detach().max(1)[1].cpu().numpy()
                gt = labels.detach().cpu().numpy()
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)

                numbers = [0, 99, 149, 399, 499]

                if batch in numbers:
                    tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)
                    writer.add_image('test/original_image', tb_original_image, batch)

                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(numpy.squeeze(labels_original), save_name=pjoin(log_dir,f'plot_grount_truth_{split}_{str(batch)}.pdf'))
                    writer.add_image('test/original_label', np_to_tb(correct_label_decoded), batch)
                    out = torch.nn.functional.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(numpy.squeeze(prediction), save_name=pjoin(log_dir,f'plot_predictions_{split}_{str(batch)}.pdf'))
                    writer.add_image('test/predicted', np_to_tb(decoded), batch)
                    writer.add_image('test/confidence', tb_confidence, batch)

                    # uncomment if you want to visualize the different class heatmaps
                    unary = outputs.cpu().detach()
                    unary_max = torch.max(unary)
                    unary_min = torch.min(unary)
                    unary = unary.add((-1*unary_min))
                    unary = unary/(unary_max - unary_min)

                    for channel in range(0, len(class_names)):
                        decoded_channel = unary[0][channel]
                        tb_channel = torchvision.utils.make_grid(decoded_channel, normalize=True, scale_each=True)
                        writer.add_image(f'test_classes/_{class_names[channel]}', tb_channel, batch)

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()

        # Add split results to TB:
        writer.add_text(f'test__{split}/', f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
        for cdx, class_name in enumerate(class_names):
            writer.add_text(
                f'test__{split}/', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)

        writer.add_text(f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
        writer.add_text(f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
        writer.add_text(f'test__{split}/', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)

        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    # Add split results to TB:
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
    for cdx, class_name in enumerate(class_names):
        writer.add_text('test_final', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)

    writer.add_text('test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
    writer.add_text('test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)

    print('--------------- FINAL RESULTS -----------------')
    print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names):
        print(f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
    print('Confusion Matrix', score['confusion_matrix'])

    # Save confusion matrix: 
    numpy.savetxt(pjoin(log_dir,'confusion.csv'), score['confusion_matrix'], delimiter=" ")
    numpy.save(pjoin(log_dir,'score'), score, allow_pickle=True)

    writer.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Cuda device or cpu execution')
    parser.add_argument('--channel_delta', type=int, default=0,
                        help='# of variable input channels')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Dec08_211808_section_deconvnet_delta=3/section_deconvnet_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    args = parser.parse_args()
    test(args)


# python section_test.py --channel_delta  0 --split both --model_path runs/Nov08_145036_section_deconvnet_delta=0/section_deconvnet_model.pkl  --device cuda:1 > runs/Nov08_145036_section_deconvnet_delta=0/output.txt
# python section_test.py --channel_delta  1 --split both --model_path runs/Nov15_215216_section_deconvnet_delta=1/section_deconvnet_model.pkl  --device cuda:1 > runs/Nov15_215216_section_deconvnet_delta=1/output.txt
# python section_test.py --channel_delta  3 --split both --model_path runs/Dec08_211808_section_deconvnet_delta=3/section_deconvnet_model.pkl  --device cuda:1 > runs/Dec08_211808_section_deconvnet_delta=3/output.txt
# python section_test.py --channel_delta  5 --split both --model_path runs/Dec08_211922_section_deconvnet_delta=5/section_deconvnet_model.pkl  --device cuda:1 > runs/Dec08_211922_section_deconvnet_delta=5/output.txt
# python section_test.py --channel_delta  7 --split both --model_path runs/Dec08_212624_section_deconvnet_delta=7/section_deconvnet_model.pkl  --device cuda:1 > runs/Dec08_212624_section_deconvnet_delta=7/output.txt
# python section_test.py --channel_delta 10 --split both --model_path runs/Dec09_005410_section_deconvnet_delta=10/section_deconvnet_model.pkl --device cuda:1 > runs/Dec09_005410_section_deconvnet_delta=10/output.txt

# python section_test.py --channel_delta  0 --split both --model_path runs/Dec09_015419_section_deconvnet_aug_delta=0/section_deconvnet_model.pkl  --device cuda:1 > runs/Dec09_015419_section_deconvnet_aug_delta=0/output.txt