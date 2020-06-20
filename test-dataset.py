import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformationsV2 import ConvertKeypoints, Scale, Rotate, CropPad, CropPad3, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def test_dataset(prepared_train_labels, train_images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
          val_labels, val_images_folder, val_output_name, checkpoint_after, val_after):

    stride = 8
    sigma = 7
    path_thickness = 1
    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad3(pad=(128, 128, 128)),
                                   Flip()]))
    #dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
    #                           stride, sigma, path_thickness,
    #                           transform=transforms.Compose([
    #                               ConvertKeypoints(),
    #                               Scale(),
    #                               Rotate(pad=(128, 128, 128)),
    #                               CropPad(pad=(128, 128, 128),center_perterb_max=40, crop_x=1920, crop_y=1920),
    #                               Flip()]))
    #dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
    #                           stride, sigma, path_thickness,
    #                           transform=transforms.Compose([
    #                               ConvertKeypoints(),
    #                               CropPad2(pad=(128, 128, 128)),
    #                              Flip()]))

    batch_data = dataset.__getitem__(0)

    print('batch data: {}'.format(batch_data))

    images = batch_data['image']
    keypoint_masks = batch_data['keypoint_mask']
    paf_masks = batch_data['paf_mask']
    keypoint_maps = batch_data['keypoint_maps']
    paf_maps = batch_data['paf_maps']

    print('images shape: {}'.format(images.shape))

    images = np.moveaxis(images, [0, 2], [2, 0]) 
    #print('image shape: {}'.format(image.shape))
    cv2.imwrite("imgage_tmp.jpg", images * 255)
    #print('keypoint_masks: {}'.format(keypoint_masks.shape))
    #print('keypoint_maps: {}'.format(keypoint_maps.shape))
    #for j in range(0, 19):
    #    mask = keypoint_masks[0,j,:,:].cpu().numpy()
    #    cv2.imwrite('mask_tmp_'+str(j)+'.jpg', mask * 255) 
    #for j in range(0, 19):
    #    mask = keypoint_maps[0,j,:,:].cpu().numpy()
    #    cv2.imwrite('keypoint_maps_tmp_'+str(j)+'.jpg', mask * 255) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=80, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')

    parser.add_argument('--val-labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--val-images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--val-output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--checkpoint-after', type=int, default=5000,
                        help='number of iterations to save checkpoint')
    parser.add_argument('--val-after', type=int, default=5000,
                        help='number of iterations to run validation')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    test_dataset(args.prepared_train_labels, args.train_images_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after)
