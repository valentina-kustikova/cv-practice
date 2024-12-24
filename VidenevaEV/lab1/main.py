#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import filters
import cv2
import os
from parser import parse_args

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['drawing'] = True
        param['start_point'] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if param['drawing']:
            param['drawing'] = False
            param['end_point'] = (x, y)
            img_copy = param['image'].copy()
            cv2.rectangle(img_copy, param['start_point'], param['end_point'], (0, 255, 0), 2)
            cv2.imshow('Select region', img_copy)
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing']:
            img_copy = param['image'].copy()
            cv2.rectangle(img_copy, param['start_point'], (x, y), (0, 255, 0), 2)
            cv2.imshow('Select region', img_copy)

def select_region(img):
    params = {'drawing': False, 'start_point': None, 'end_point': None, "image": img}
    cv2.imshow('Select region', img)
    cv2.setMouseCallback('Select region', mouse_callback, param=params)
    cv2.waitKey(0)
    cv2.destroyWindow('Select region')

    if params['start_point'] and params['end_point']:
        return (params['start_point'][0], params['start_point'][1],
                params['end_point'][0] - params['start_point'][0],
                params['end_point'][1] - params['start_point'][1])
    else:
        return None

def main():
    args = parse_args()

    if args.input == None:
        raise ValueError("Input path is None")
    img = cv2.imread(args.input)
    try:
        if args.mode == 'gray':
            result = filters.grayscale(img)
        elif args.mode == 'resize':
            if args.value is None:
                raise ValueError("Value is None")
            result = filters.resize(img, args.value)
        elif args.mode == 'sepia':
            result = filters.sepia(img)
        elif args.mode == 'vignette':
            if args.value is None:
                raise ValueError("Value is None")
            result = filters.vignette(img, int(args.value))
        elif args.mode == 'pixelate':
            if args.value is None:
                raise ValueError("Value is None")
            result = filters.pixelate(img, int(args.value), select_region(img))

        if args.output is not None:
            output_path = os.path.join(
                os.path.dirname(args.output),
                args.mode + "_" + os.path.basename(args.input)
            )
        else:
            output_path = os.path.join(
                os.path.dirname(args.input),
                args.mode + os.path.basename(args.input)
            )

        if result is not None:
            cv2.imwrite(output_path, result)

            cv2.imshow('Init image', img)
            cv2.imshow('Output image', result)
            cv2.waitKey(0)
        else:
            raise "variable result is None"
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()