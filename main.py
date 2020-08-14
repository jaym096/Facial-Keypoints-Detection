import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

from models import Net

import utils


def train_net(n_epochs, net):
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":

    net = Net()
    # print(net)

    data_transform = transforms.Compose([Rescale(225), RandomCrop(224), Normalize(), ToTensor()])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    print('Number of images: ', len(transformed_dataset))

    # iterate through the transformed dataset and print some stats about the first few samples
    for i in range(4):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())

    # load training data in batches
    batch_size = 10
    train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # create the test dataset
    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv', root_dir='data/test/',
                                          transform=data_transform)

    # load test data in batches
    batch_size = 10
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_images, test_outputs, gt_pts = utils.net_sample_output(test_loader, net)

    # print out the dimensions of the data to see if they make sense
    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    # visualize data before training
    utils.visualize_output("before_train", test_images, test_outputs, gt_pts)

    net.load_state_dict(torch.load('keypoints_model_final.pt'))
    ## print out your net and prepare it for testing
    net.eval()

    # criterion = nn.SmoothL1Loss()
    #
    # optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    #
    # # train your network
    # n_epochs = 7
    # train_net(n_epochs, net)
    #
    # # after training, save your model parameters in the dir 'saved_models'
    # model_name = 'keypoints_model_final.pt'
    # torch.save(net.state_dict(), model_name)

    # get a sample of test data again
    test_images, test_outputs, gt_pts = utils.net_sample_output(test_loader, net)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    # visualize_output(test_images, test_outputs, gt_pts)
    utils.visualize_output("saved_images", test_images, test_outputs, gt_pts=None,)

    # after training, save your model parameters in the dir 'saved_models'
    # model_dir = 'saved_models/'
    # model_name = 'keypoints_model_final.pt'
    # torch.save(net.state_dict(), model_dir + model_name)

    print("model saved!")

