import concurrent
import csv
import logging
import os
import threading
from concurrent.futures import as_completed
from time import time

import cv2
import numpy as np
import pandas as pd
import pydegensac
import scipy
import scipy.io
import scipy.misc
import torch
from tqdm import tqdm

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image

print('Libraries have been imported')

logging.root.setLevel(logging.DEBUG)
logging.basicConfig(format="%(thread)d - %(asctime)s:\t%(message)s", datefmt='%d,%H:%M:%S.%f')

print('Libraries have been imported')
print('Loading parameters')

# PARAMETERS

BASE = "./"
OUTPUT = BASE + "output/"
INPUT = BASE + "input/"
MODULE_FILE = BASE + 'models/d2_tf.pth'

# D2NET METHODS
PREPROCESSING = 'caffe'
USE_RELU = True
OUTPUT_TYPE = 'npz'
MULTISCALE = True

# MAX EDGE SIZE (WIDTH OR HEIGHT)
MAX_EDGE = 1600

# MAX SUM OF EDGES (WIDTH + HEIGHT)
MAX_SUM_EDGES = 2800

# EXTRACTED FILE EXTENSION
OUTPUT_EXTENSION = '.d2-net'
# FUNCTION
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# PREDICTION INLIER THRESHOLD
THRESHOLD = 21


# TODO SUPPORT PNG
# def getImageUrls(folder):
#     # if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
#     return glob.glob(os.path.join(folder, '*/*.jpg'))

#
# def getD2Urls(folder):
#     return glob.glob(os.path.join(folder, '*/*.d2-net'))

def get_needle_haystack_from_csv(csvpath):
    data = pd.read_csv(csvpath, sep=',', quotechar='"', encoding='unicode_escape', header='infer')
    needle = data.iloc[:, 0]
    haystack = data.iloc[:, 1]
    return needle.to_numpy(), haystack.to_numpy()


needleImageUrls, haystackImageUrls = get_needle_haystack_from_csv(INPUT + 'dataset_list.csv')


def start_extracting():
    """
    This function extracts image keypoints, scores, and descriptors. It saves these into NumPy/MatLab interpretable
    files with an earlier specified output extension. This function needs only be ran only if the files
    do not already exist. The files are saved in the same folder as the corresponding image.
    """

    # Load the D2-NET model
    model = D2Net(model_file=MODULE_FILE, use_relu=USE_RELU, use_cuda=USE_CUDA)

    # Generate a list of unique image files
    imageURLs = np.concatenate((needleImageUrls, haystackImageUrls))
    imageURLs = np.unique(imageURLs)

    # For each image do the following (tqdm shows current progress)
    for path in tqdm(imageURLs, total=len(imageURLs)):

        # imread returns (height, width, num_channels)
        path = INPUT + path
        image = cv2.imread(path)

        # make file interpretable if necessary
        # if a None error occurs, it will most likely be the cv2.imread that is not working
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # preprocessing by some shape resizing specified by the parameters given
        resized_image = image
        if max(resized_image.shape) > MAX_EDGE:
            fraction = MAX_EDGE / max(resized_image.shape)
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        if sum(resized_image.shape[: 2]) > MAX_SUM_EDGES:
            fraction = MAX_SUM_EDGES / sum(resized_image.shape[: 2])
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        # D2NET image enhancement preprocessing
        input_image = preprocess_image(
            resized_image,
            preprocessing=PREPROCESSING
        )

        # Extraction core process
        with torch.no_grad():
            if MULTISCALE:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=DEVICE
                    ),
                    model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=DEVICE
                    ),
                    model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        # Save output as file
        if OUTPUT_TYPE == 'npz':
            with open(path + OUTPUT_EXTENSION, 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=keypoints,
                    scores=scores,
                    descriptors=descriptors
                )
        elif OUTPUT_TYPE == 'mat':
            with open(path + OUTPUT_EXTENSION, 'wb') as output_file:
                scipy.io.savemat(
                    output_file,
                    {
                        'keypoints': keypoints,
                        'scores': scores,
                        'descriptors': descriptors
                    }
                )
        else:
            raise ValueError('Unknown output type.')

    # make sure that the model is deleted and the CUDA cache is empty
    del model
    torch.cuda.empty_cache()


# MATCHING SECTION

# Load the files with the specified output extension
needleImageData = np.array([path + OUTPUT_EXTENSION for path in needleImageUrls])
haystackImageData = np.array([path + OUTPUT_EXTENSION for path in haystackImageUrls])

# Prepare and specify the output data file
db_fileName3 = OUTPUT + 'output_datalist.csv'
db_file3 = open(db_fileName3, 'w')
fieldnames = ['img1', 'img2', 'y_pred', 'match_time', 'matches', 'ransac_time', 'inliers']
print("Filepath is " + db_file3.name)

writer = csv.DictWriter(db_file3, delimiter=',', fieldnames=fieldnames)
writer.writeheader()
print("Header has been written")

# Thread lock and flush the file
write_lock = threading.Lock()
db_file3.flush()


def write_row(data):
    """
    This function writes thread safe to the output file
    :param data: the data to be written
    """
    print("Waiting for CSV lock ...")
    with write_lock:
        print('Writing csv')
        writer.writerow(data)
        print('Done writing csv')
        db_file3.flush()


def go_match_images(img1_url, img2_url, data_url1, data_url2, num):
    logging.debug("Analyzing...")
    if os.path.exists(data_url1):
        if os.path.exists(data_url2):
            feat1 = np.load(data_url1)
            feat2 = np.load(data_url2)

            t0 = time()
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(feat1['descriptors'], feat2['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            t1 = time()

            t_match = t1 - t0

            match1 = [m.queryIdx for m in matches]
            match2 = [m.trainIdx for m in matches]

            keypoints_left = feat1['keypoints'][match1, : 2]
            keypoints_right = feat2['keypoints'][match2, : 2]

            np.random.seed(0)
            t0 = time()

            H, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)

            t1 = time()
            t_ransac = t1 - t0

            n_inliers = np.sum(inliers)

            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

            pred = 0
            if n_inliers >= THRESHOLD:
                logging.debug("Match has been found!")
                # draw_params = dict(matchColor=(0, 255, 0),singlePointColor=(255, 0, 0), matchesMask = matchesMask, flags=0)
                image1 = cv2.imread(img1_url)
                image2 = cv2.imread(img2_url)
                image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right,
                                         placeholder_matches,
                                         None)
                path3 = OUTPUT + "/output_" + str(num) + ".jpg"

                cv2.imwrite(path3, image3)

                image1.close()
                image2.close()

                pred = 1
            else:
                logging.debug("Match not found!")

            feat1.close()
            feat2.close()

            new_row = {'img1': str(img1_url), 'img2': str(img2_url), 'y_pred': pred,
                       'match_time': t_match, 'matches': len(matches), 'ransac_time': t_ransac,
                       'inliers': n_inliers}
            write_row(new_row)
        else:
            print(data_url2 + " does not exists")
    else:
        print(data_url1 + " does not exists")


def start_matching():
    """
    This function starts the matching process using multithreading. We ran this with 8 cores (1.5 workers per core).
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        for i in range(len(needleImageUrls)):
            img1 = INPUT + needleImageUrls[i]
            img2 = INPUT + haystackImageUrls[i]
            data1 = INPUT + needleImageData[i]
            data2 = INPUT + haystackImageData[i]
            executor.submit(go_match_images, img1, img2, data1, data2, i)


print('Extracting...')
start_extracting()
print('Matching...')
start_matching()

# Close the output file
db_file3.close()
print('Matching completed')

print('Script completed')
