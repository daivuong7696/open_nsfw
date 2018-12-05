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
#import visualize_result 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve    
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

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
    output_layers=None):
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

        img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = int(max((H - h) / 2, 0))
        w_off = int(max((W - w) / 2, 0))
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

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

    args = parser.parse_args()
    

    # Pre-load caffe model.
    nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
        args.pretrained_model, caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    if args.input_file is not None:
        with open(args.input_file, 'rb') as f:
            image_data = f.read()
        # Classify.
            scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])
        print("NSFW score:  " , scores)

    elif args.input_label_file is not None: 
        scores = []
        df = pd.read_csv(args.input_label_file, header=None, delimiter=' ', names=['file_name', 'label'])
        for i in tqdm(range(len(df))):
            with open(df.iloc[i, 0], 'rb') as f:
                image = f.read()
                scores.append(caffe_preprocess_and_compute(image, 
                                                           caffe_transformer=caffe_transformer, 
                                                           caffe_net=nsfw_net, 
                                                           output_layers=['prob'])[1])
        df['scores'] = scores
        df['NSFW'] = df['scores'] >= args.threshold
        # From boolean to int
        df['NSFW'] = df['NSFW'] + 0
        
        target_names = ['nosexy', 'sexy']
        cnf_matrix = confusion_matrix(df['label'], df['NSFW'])
        y = df['label']
        y_pred = df['NSFW']
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy: {}".format(accuracy))
        report = classification_report(y, y_pred, target_names=target_names)
        fpr, tpr, thresholds = roc_curve(y, df['scores'], pos_label=1)
        print("False positive rate: ", fpr)
        print("True positive rate: ", tpr)
        print('thresholds: ', thresholds)
        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Original caffemodel')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig('test.png')
#         visualize_result.save_confusion_matrix_classification_report(cnf_matrix=cnf_matrix, 
#                                                                      classification_report=report,
#                                                                      class_names=target_names,
#                                                                      file_name='cnf_matrix')

        df[['file_name', 'label', 'scores', 'NSFW']].to_csv(
            'result.txt', sep=' ', header=None, index=None)
        print("Test:", len(df))

    # Scores is the array containing SFW / NSFW image probabilities
    # scores[1] indicates the NSFW probability
    



if __name__ == '__main__':
    main(sys.argv)
