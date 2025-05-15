import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='/home/tjl/INpainting/celeba_hq_256/test_1')
parser.add_argument('--output', type=str, help='test_images.flist')
args = parser.parse_args()

ext = {'.jpg', '.png', '.JPG'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
