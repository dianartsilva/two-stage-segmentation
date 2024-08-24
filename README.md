# Two Stage Segmentation

This is the repository of [Two-stage semantic segmentation in neural networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12701/127010G/Two-stage-semantic-segmentation-in-neural-networks/10.1117/12.2679881.short) (ICMV 2022 - Fifteenth International Conference on Machine Vision)

Semantic segmentation consists of classifying each pixel according to a set of classes. This process is particularly slow for high-resolution images, which are present in many applications, ranging from biomedicine to the automotive industry. In this work, we propose an algorithm targeted to segment high-resolution images based on two stages. During stage 1, a lower-resolution interpolation of the image is the input of a first neural network, whose low-resolution output is resized to the original resolution. Then, in stage 2, the probabilities resulting from stage 1 are divided into contiguous patches, with less confident ones being collected and refined by a second neural network. The main novelty of this algorithm is the aggregation of the low-resolution result from stage 1 with the high-resolution patches from stage 2. We propose the U-Net architecture segmentation, evaluated in six databases. Our method shows similar results to the baseline regarding the Dice coefficient, with fewer arithmetic operations.

Structure:

* `train.py`: train the various models for the various datasets.
* `test-baseline.py` evaluates the performance of the baseline.
* `test-twoseg.py` evaluates the performance of the proposed stage1+stage2 pipeline.
