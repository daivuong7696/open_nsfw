import numpy as np
import cv2
import os


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap*255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)

def py_returnCAMmap(activation, weights_LR):
    if activation.shape[0] == 1: # only one image
        n_feat, w, h = activation[0].shape
        act_vec = np.reshape(activation[0], [n_feat, w*h])
        n_top = weights_LR.shape[0]
        out = np.zeros([w, h, n_top])

        for t in range(n_top):
            weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
            heatmap_vec = np.dot(weights_vec,act_vec)
            heatmap = np.reshape( np.squeeze(heatmap_vec) , [w, h])
            out[:,:,t] = heatmap
    else: # 10 images (over-sampling)
        raise Exception('Not implemented')
    return out

def save_CAM_caffe(image,
                   image_name,
                   fc_weights,
                   activation_lastconv,
                   class_dict,
                   class_name,
                   dest_folder='',
                   image_size=256,
                   save_to_folder=False):

    ## Class Activation Mapping
    IDX_category = [class_dict[class_name]]
    curCAMmapAll = py_returnCAMmap(activation_lastconv, fc_weights[IDX_category, :])

    # for one image
    curCAMmap_crops = curCAMmapAll[:, :]
    curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (image_size, image_size))
    curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (image_size, image_size))
    curHeatMap = im2double(curHeatMap)

    curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
    curHeatMap = im2double(image) * 0.5 + im2double(curHeatMap) * 0.4

    image_name_and_extension = image_name.split('/')[-1].split('.')
    cam_name = image_name_and_extension[0] + "_CAM." + image_name_and_extension[1]
    dest = os.path.join(dest_folder, cam_name)
    cv2.imwrite(dest, curHeatMap * 255)


