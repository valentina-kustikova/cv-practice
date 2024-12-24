#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='path to input image',
        type=str,
        dest='input',
    )
    parser.add_argument(
        '-o', '--output',
        help='path to output image',
        type=str,
        dest='output',
        default=None,
    )
    parser.add_argument(
        '-m', '--mode',
        help='Mode (\'image\', \'gray\', \'resize\', \'sepia\', \'vignette\', \'pixelate\')',
        type=str,
        dest='mode',
        default='image',
    )
    parser.add_argument(
        '-v', '--value',
        help='input strength of filter',
        type=float,
        dest='value',
        default=None,
    )
    args = parser.parse_args()
    return args

