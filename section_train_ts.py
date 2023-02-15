import argparse

import numpy
import os
import torch
import torchvision

from datetime import datetime
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
from torch.utils import data
from tqdm import tqdm

import core.loss
import core.models
from core.augmentations import Compose, RandomHorizontallyFlip, RandomVerticallyFlip, RandomRotate, AddNoise
from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb, detect_gabor_edges

# Fix the random seeds: 
numpy.random.seed(seed=2022)
torch.backends.cudnn.deterministic = True
torch.manual_seed(2022)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(2022)


def split_train_val(args, per_val=0.1):
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = numpy.load(pjoin('data', 'train', 'train_labels.npy'))
    i_list = list(range(labels.shape[0]))
    i_list = ['i_'+str(inline) for inline in i_list]

    x_list = list(range(labels.shape[1]))
    x_list = ['x_'+str(crossline) for crossline in x_list]

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True)

    # write to files to disK:
    file_object = open(pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Generate the train and validation sets for the model:
    split_train_val(args, per_val=args.per_val)

    # Setup log files 
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'{current_time}_{args.arch}{"_aug" if args.aug else ""}{"_weighted" if args.class_weights else ""}_delta={args.channel_delta}')
    os.mkdir(log_dir)
    
    # Setup augmentations
    if args.aug:
        print('Data Augmentation Enabled.')
        data_aug = Compose([RandomRotate(10), RandomVerticallyFlip(), AddNoise()])
    else:
        data_aug = None

    # Traning accepts augmentation, unlike validation:
    train_set = section_dataset(channel_delta=args.channel_delta, split='train', is_transform=True, augmentations=data_aug)
    valid_set = section_dataset(channel_delta=args.channel_delta, split='val', is_transform=True)

    n_classes = train_set.n_classes

    # Create sampler:
    shuffle = False  # must turn False if using a custom sampler
    with open(pjoin('data', 'splits', 'section_train.txt'), 'r') as file_buffer:
        train_list = file_buffer.read().splitlines()
    with open(pjoin('data', 'splits', 'section_val.txt'), 'r') as file_buffer:
        val_list = file_buffer.read().splitlines()

    class CustomSamplerTrain(torch.utils.data.Sampler):
        def __iter__(self):
            char = ['i' if numpy.random.randint(2) == 1 else 'x']
            self.indices = [idx for (idx, name) in enumerate(train_list) if char[0] in name]
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

    class CustomSamplerVal(torch.utils.data.Sampler):
        def __iter__(self):
            char = ['i' if numpy.random.randint(2) == 1 else 'x']
            self.indices = [idx for (idx, name) in enumerate(val_list) if char[0] in name]
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  sampler=CustomSamplerTrain(train_list),
                                  num_workers=0, shuffle=shuffle)
    val_loader = data.DataLoader(valid_set, batch_size=args.batch_size,
                                sampler=CustomSamplerVal(val_list), 
                                num_workers=0)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        # model = get_model(args.arch, args.pretrained, n_classes)
        n_channels = 1 if args.channel_delta == 0 else 3
        print(f'Creating Model {args.arch.upper()}')
        model = getattr(core.models, args.arch)(n_channels=n_channels, n_classes=n_classes)
        two_stream = type(model) is core.models.section_two_stream

    # Use as many GPUs as we can
    # model = torch.nn.DataParallel(model, device_ids=[5,7])
    model = model.to(device)  # Send to GPU

    # PYTORCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU
    # optimizer = torch.optim.Adadelta(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    loss_fn = core.loss.cross_entropy

    if args.class_weights:
        # weights are inversely proportional to the frequency of the classes in the training set
        print('Weighted Loss Enabled.')
        class_weights = torch.tensor([0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
    else:
        class_weights = None

    best_iou = -100.0
    class_names = ['upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']

    # training
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, total_iteration = 0, 0

        for batch, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            image_original, labels_original = images, labels
            if two_stream:
                gabors = detect_gabor_edges(images, frequency=0.1)
                images, gabors, labels = images.to(device), gabors.to(device), labels.to(device)
                outputs = model(images, gabors)
            else:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics.update(gt, pred)

            loss = loss_fn(input=outputs, target=labels, weight=class_weights)
            loss_train += loss.item()
            loss.backward()

            # gradient clipping
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_iteration = total_iteration + 1

            if (batch) % 20 == 0:
                print("Epoch [%d/%d] training Loss: %.4f" % (epoch + 1, args.n_epoch, loss.item()))

            numbers = [0]
            if batch in numbers:
                # number 0 image in the batch
                tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)

                labels_original = labels_original.numpy()[0]
                correct_label_decoded = train_set.decode_segmap(numpy.squeeze(labels_original))
                out = torch.nn.functional.softmax(outputs, dim=1)

                # this returns the max. channel number:
                prediction = out.max(1)[1].cpu().numpy()[0]
                # this returns the confidence:
                confidence = out.max(1)[0].cpu().detach()[0]
                tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                decoded = train_set.decode_segmap(numpy.squeeze(prediction))

                unary = outputs.cpu().detach()
                unary_max = torch.max(unary)
                unary_min = torch.min(unary)
                unary = unary.add((-1*unary_min))
                unary = unary/(unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel]
                    tb_channel = torchvision.utils.make_grid(decoded_channel, normalize=True, scale_each=True)

        # Average metrics
        loss_train /= total_iteration
        score, class_iou = running_metrics.get_scores()
        running_metrics.reset()

        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval()
                loss_val, total_iteration_val = 0, 0

                for batch, (images_val, labels_val) in tqdm(enumerate(val_loader)):
                    image_original, labels_original = images_val, labels_val
                    if two_stream:
                        gabors_val = detect_gabor_edges(images_val, frequency=0.1)
                        images_val, gabors_val, labels_val = images_val.to(device), gabors_val.to(device), labels_val.to(device)
                        outputs_val = model(images_val, gabors_val)
                    else:
                        images_val, labels_val = images_val.to(device), labels_val.to(device)
                        outputs_val = model(images_val)

                    pred = outputs_val.detach().max(1)[1].cpu().numpy()
                    gt = labels_val.detach().cpu().numpy()

                    running_metrics_val.update(gt, pred)

                    loss = loss_fn(input=outputs_val, target=labels_val)

                    total_iteration_val = total_iteration_val + 1

                    if (batch) % 20 == 0:
                        print("Epoch [%d/%d] validation Loss: %.4f" % (epoch + 1, args.n_epoch, loss.item()))

                    numbers = [0]
                    if batch in numbers:
                        # number 0 image in the batch
                        tb_original_image = torchvision.utils.make_grid(
                            image_original[0][0], normalize=True, scale_each=True)
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = train_set.decode_segmap(numpy.squeeze(labels_original))

                        out = torch.nn.functional.softmax(outputs_val, dim=1)

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0]
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                        decoded = train_set.decode_segmap(numpy.squeeze(prediction))

                        unary = outputs_val.cpu().detach()
                        unary_max, unary_min = torch.max(unary), torch.min(unary)
                        unary = unary.add((-1*unary_min))
                        unary = unary/(unary_max - unary_min)

                        for channel in range(0, len(class_names)):
                            tb_channel = torchvision.utils.make_grid(unary[0][channel], normalize=True, scale_each=True)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)

                running_metrics_val.reset()

                if score['Mean IoU: '] >= best_iou:
                    best_iou = score['Mean IoU: ']
                    model_dir = os.path.join(log_dir, f"{args.arch}_model.pkl")
                    torch.save(model, model_dir)

        else:  # validation is turned off:
            # just save the latest model:
            if (epoch+1) % 10 == 0:
                model_dir = os.path.join(log_dir, f"{args.arch}_ep{epoch+1}_model.pkl")
                torch.save(model, model_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='section_two_stream', choices=['section_deconvnet', 'section_two_stream'],
                        help='Architecture to use [\'section_two_stream\']')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Cuda device or cpu execution')
    parser.add_argument('--channel_delta', type=int, default=0,
                        help='# of variable input channels')
    parser.add_argument('--n_epoch', type=int, default=120,
                        help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val', type=float, default=0.1,
                        help='percentage of the training data for validation')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug', action='store_true',
                        help='Whether to use data augmentation.')
    parser.add_argument('--class_weights', action='store_true',
                        help='Whether to use class weights to reduce the effect of class imbalance')

    args = parser.parse_args()
    train(args)
