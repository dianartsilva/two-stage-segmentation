import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

import numpy as np
import importlib
i = importlib.import_module('dataloaders.' + args.dataset)
ds = getattr(i, args.dataset.upper())('train')
ds = list(ds)

shapes = np.array([d[0].shape for d in ds])
heights = shapes[:, 0]
print(f'height: [{heights.min()}, {heights.max()}], avg: {int(np.round(heights.mean()))}')
widths = shapes[:, 1]
print(f'width:  [{widths.min()}, {widths.max()}], avg: {int(np.round(widths.mean()))}')

yy = np.concatenate([np.ravel(y) for _, y in ds]).astype(int)
counts = np.bincount(yy)
print('abs counts:', counts)
counts = np.array(counts) / counts.sum()
print('rel counts:', counts)

weights = 1 / counts
pos_weight = weights[1] / weights[0]
print('pos weight:', pos_weight)
