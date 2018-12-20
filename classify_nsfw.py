#!/usr/bin/env python
"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
import argparse
import glob
import time
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
    
import caffe
import cv2
from class_activation_map import save_CAM_caffe

try:
    caffe_root = os.environ['CAFFE_ROOT'] + '/'
except KeyError:
    raise KeyError("Define CAFFE_ROOT in ~/.bashrc")

import visualize_result
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve


class_dict = {
    'notsexy': 0,
    'sexy': 1
}


def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the 
    caffe, since it was used to generate training dataset 
    :param str data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    img_data = data
    im = Image.open(StringIO(img_data))

    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())

def caffe_preprocess(caffe_net, image_data,
                     caffe_transformer=None):

    img_data_rs = resize_image(image_data, sz=(256, 256))
    image = caffe.io.load_image(StringIO(img_data_rs))

    H, W, _ = image.shape
    _, _, h, w = caffe_net.blobs['data'].data.shape
    h_off = int(max((H - h) / 2, 0))
    w_off = int(max((W - w) / 2, 0))
    crop = image[h_off:h_off + h, w_off:w_off + w, :]
    transformed_image = caffe_transformer.preprocess('data', crop)
    transformed_image.shape = (1,) + transformed_image.shape

    return image, transformed_image


def caffe_compute(transformed_image,
                  caffe_net=None, output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.

    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """

    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                    **{input_name: transformed_image})


        outputs = all_outputs[output_layers[0]][0].astype(float)

        return outputs
    else:
        return []


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "--input_file",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--input_label_file",
        help="Path to the input label file"
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--save_cam_path",
        help="Save class activation map flag"
    )
    parser.add_argument(
        "--save_to_folder_path",
        help="Classify images and store them to scores folder"
    )
    parser.add_argument(
        "--save_result_path",
        default='result',
        help="Directory where to save ROC curve, confusion matrix"
    )

    args = parser.parse_args()

    # Pre-load caffe model.
    nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
        args.pretrained_model, caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    # move image channels to outermost
    caffe_transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))
    # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_raw_scale('data', 255)
    # swap channels from RGB to BGR
    caffe_transformer.set_channel_swap('data', (2, 1, 0))

    # Preprocess and compute image
    # One image only
    if args.input_file is not None:
        with open(args.input_file, 'rb') as f:
            image_data = f.read()
            # Preprocessing
            original_image, transformed_image = caffe_preprocess(
                caffe_net=nsfw_net, image_data=image_data,
                caffe_transformer=caffe_transformer
            )
            # Calculating scores
            scores = caffe_compute(
                transformed_image=transformed_image, caffe_net=nsfw_net,
                output_layers=['prob']
            )
            # Calculating class activation map
            if args.save_cam_path is not None:
                if not os.path.isdir(args.save_cam_path):
                    os.mkdir(args.save_cam_path)
                out_layer = 'fc_nsfw'
                last_conv = 'conv_stage3_block2_branch2c'
                weights_LR = nsfw_net.params[out_layer][0].data
                activation_lastconv = nsfw_net.blobs[last_conv].data
                save_CAM_caffe(image_name=args.input_file,
                    image=original_image, fc_weights=weights_LR,
                    activation_lastconv=activation_lastconv,
                    class_dict=class_dict, class_name='sexy',
                    dest_folder='/home/daivuong/Desktop',
                    image_size=224
                )
        print("NSFW score: {}".format(scores[1]))
    # Input is a file of many images
    elif args.input_label_file is not None:
        scores = []
        df = pd.read_csv(
            args.input_label_file,
            header=None, delimiter=' ',
            names=['file_name', 'label']
        )
        for i in tqdm(range(len(df))):
            with open(df.iloc[i, 0], 'rb') as f:
                image_data = f.read()
                # Preprocessing
                try:
                    original_image, transformed_image = caffe_preprocess(
                        caffe_net=nsfw_net, image_data=image_data,
                        caffe_transformer=caffe_transformer
                    )
                except:
                    print("Cannot load images")
                    continue
                # Calculating scores
                sexy_score = caffe_compute(
                    transformed_image=transformed_image, caffe_net=nsfw_net,
                    output_layers=['prob']
                )[1]
                scores.append(sexy_score)


                # Caclulating class activation map
                # It will store predicted images into seperated
                # folders based on rounded scores (from 0.0 to 1.0)
                # and these two folders will be stored into ground
                # truth folder
                if args.save_cam_path is not None:
                    if not os.path.isdir(args.save_cam_path):
                        os.mkdir(args.save_cam_path)

                    # Ground truth folder
                    label_path = os.path.join(
                        args.save_cam_path,
                        str(df.iloc[i, 1])
                    )
                    if not os.path.isdir(label_path):
                        os.mkdir(label_path)

                    # Rounded scores folders
                    dest = os.path.join(
                        label_path, str(round(sexy_score, 1))
                    )
                    if not os.path.isdir(dest):
                        os.mkdir(dest)

                    # Calculate CAM
                    out_layer = 'fc_nsfw'
                    last_conv = 'conv_stage3_block2_branch2c'
                    weights_LR = nsfw_net.params[out_layer][0].data
                    activation_lastconv = nsfw_net.blobs[last_conv].data


                    save_CAM_caffe(image_name=df.iloc[i, 0],
                        image=original_image, fc_weights=weights_LR,
                        activation_lastconv=activation_lastconv,
                        class_dict=class_dict, class_name='sexy',
                        dest_folder=dest,
                        image_size=256
                    )
                if args.save_to_folder_path is not None:
                    if not os.path.isdir(args.save_to_folder_path):
                        os.mkdir(args.save_to_folder_path)

                    # Ground truth folder
                    label_path = os.path.join(
                        args.save_to_folder_path,
                        str(df.iloc[i, 1])
                    )
                    if not os.path.isdir(label_path):
                        os.mkdir(label_path)

                    # Rounded scores folders
                    dest = os.path.join(
                        label_path, str(round(sexy_score, 1))
                    )
                    if not os.path.isdir(dest):
                        os.mkdir(dest)
                    src = df.iloc[i, 0]
                    dst = os.path.join(dest, src.split('/')[-1])
                    os.rename(src, dst)
   


        df['scores'] = scores
        df['NSFW'] = (df['scores'] >= args.threshold)
        # From boolean to int
        df['NSFW'] = df['NSFW'] + 0
        y = df['label']
        y_pred = df['NSFW']

        # confusion matrix and classification report visualization
        target_names = ['nosexy', 'sexy']
        cnf_matrix = confusion_matrix(df['label'], df['NSFW'])
        report = classification_report(y, y_pred, target_names=target_names)
        file_name = args.pretrained_model.split('/')[-1].split('.')[0] + '_cnf_matrix.png'
        visualize_result.save_confusion_matrix_classification_report(cnf_matrix=cnf_matrix, 
                                                                     classification_report=report,
                                                                     class_names=target_names,
                                                                     file_name=file_name)
        
        # Accuracy
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy: {}".format(accuracy))

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y, df['scores'], pos_label=1)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='ROC curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        figname = args.pretrained_model.split('/')[-1].split('.')[0] + '_roc_curve.png'
        plt.savefig(figname)
        
        file_name = args.pretrained_model.split('/')[-1].split('.')[0] + '_result.txt'
        
        df[['file_name', 'label', 'scores', 'NSFW']].to_csv(
            file_name, sep=' ', header=None, index=None)

if __name__ == '__main__':
    main(sys.argv)
