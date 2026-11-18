import argparse
import sys
import os
import logging
import cv2 as cv

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger()

def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Bag of visual words image classifier')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_train = subparsers.add_parser('train', help='Train vocabulary and classifier')
    p_train.add_argument('-d', '--data', required=True, type=str, help='Path to dataset root')
    p_train.add_argument('-s', '--split', required=True, type=str, help='Path to splet.json')
    p_train.add_argument('-m', '--model', required=True, type=str, help='Where to save the nodel file')
    p_train.add_argument('-k', default=500, type=int, help='Number of visual words')

    p_test = subparsers.add_parser('test', help='Evaluate the model on the test split')
    p_test.add_argument('-d', '--data', required=True, type=str)
    p_test.add_argument('-s', '--split', required=True, type=str)
    p_test.add_argument('-m', '--model', required=True, type=str, help='Path to saved model')
    p_test.add_argument('-o', '--output', type=str, default=None, help='Optional file to save classification results')

    p_predict = subparsers.add_parser('predict', help='Classify a single image')
    p_predict.add_argument('-m', '--model', required=True, type=str, help='Path to saved model')
    p_predict.add_argument('-i', '--image', required=True, type=str, help='Path to the image')
    p_predict.add_argument('-o', '--output', type=str, default=None, help='Optional path to save the BoW histogram plot (e.g. hist.png)')
    
    return parser.parse_args

def read_image(img_path):
    if not os.pathisfile(image_path):
        raise FileNotFoundError(f'File not found: {img_path}')
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Failed to read the image: {img_path}')
    return img

