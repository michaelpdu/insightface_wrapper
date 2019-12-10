# InsightFace Tensorflow

This version of InsightFace use Tensorflow InsightFace model, which is converted by `mmdnn`.

Due to `insightface_tf.py` depands on other tools, it CANNOT execute directly. 

This repository is just as a part of `FaceGen`.

`WARN: Input of InsightFace model is a int8 images, not float images.`

## Convert MXNET Model to Tensorflow Model

How to convert MXNet model to TF model?

refer to [https://blog.csdn.net/dupei/article/details/103469407](https://blog.csdn.net/dupei/article/details/103469407)

## Download Released TF Model

download `r100.pb` from [release](https://github.com/michaelpdu/insightface_wrapper/releases/download/1.0/r100.pb), and place it in current directory.

## Test Command

```python
python insightface_tf.py img_path_1 img_path_2
```