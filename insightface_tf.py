import os
from os.path import dirname, realpath
import sys
root_dir = dirname(dirname(dirname(realpath(__file__))))
utility_dir = os.path.join(root_dir, 'utility')
#print(utility_dir)
sys.path.append(utility_dir)
from dnnlib.tflib import tfutil
from skimage_util import save_image
# from pilimage_util import array2img

import argparse
import tensorflow as tf
import numpy as np
import skimage.io as io
from insightface_img_util import load_image_for_insightface

def calc_dist(f1, f2):
    print('calculate distance in numpy:')
    dist = np.sqrt(np.sum(np.square(f1-f2)))
    print('dist:', dist)
    sim = np.dot(f1, f2.T)
    print('similarity:', sim)
    print('f1 norm:', np.linalg.norm(f1))
    print('f2 norm:', np.linalg.norm(f2))

class InsightFace(object):
    """
    Warning:
        Input image for InsightFace model must be int8 images!!!
    """
    def __init__(self, model_path):
        #
        tfutil.init_tf()
        #
        print('[MD] Load insight face model...')
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,input_map=None,return_elements=None,name="")

        self.image_input = tf.get_default_graph().get_tensor_by_name('data:0')
        embedding = tf.get_default_graph().get_tensor_by_name('output:0')
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        self.embedding = tf.div(embedding, embedding_norm, name='norm_embedding')

        self.target_emb = tf.placeholder(tf.float32,shape=[None,512],name='target_emb_input')
        self.cos_loss = tf.reduce_sum(tf.multiply(self.embedding, self.target_emb))
        self.l2_loss = tf.norm(self.embedding-self.target_emb)
        self.grads_op = tf.gradients(self.l2_loss, self.image_input)

        # self.fdict = {keep_prob:1.0, is_train:False}
        self.fdict = {}
        self.sess = tf.get_default_session()

    def calc_embedding(self, imgs):
        assert self.sess is not None, "Default session is None, please call tfutil.init_tf firstly."
        self.fdict[self.image_input] = imgs
        return self.sess.run(self.embedding, feed_dict=self.fdict)

    def set_target_embedding(self, e0):
        self.fdict[self.target_emb] = e0

    def compare(self, imgs):
        assert self.sess is not None, "Default session is None, please call tfutil.init_tf firstly."
        self.fdict[self.image_input] = imgs
        return self.sess.run([self.cos_loss, self.l2_loss], feed_dict=self.fdict)
    
    def optimize(self, imgs):
        eps = 1.
        for i in range(100):
            self.fdict[self.image_input] = imgs
            grads, cos_loss, l2_loss = self.sess.run( \
                [self.grads_op, self.cos_loss, self.l2_loss], feed_dict=self.fdict)
            grads = grads[0]
            grads = np.sign(grads)
            # print('grads.shape:', grads.shape)
            imgs = imgs-grads*eps
            imgs = np.clip(imgs,0,255)
            print('index:', i, ', cosine_similarity:', cos_loss, ', l2_dist:', l2_loss)
        imgs = np.squeeze(imgs)
        print('imgs.shape:', imgs.shape)
        save_image(imgs, 'test.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path_1', type=str, help='Path to first input image')
    parser.add_argument('img_path_2', type=str, help='Path to second input image')
    args = parser.parse_args()

    insight_face = InsightFace('r100.pb')
    img1 = load_image_for_insightface(args.img_path_1, align=False, hwc2chw=False)
    img2 = load_image_for_insightface(args.img_path_2, align=False, hwc2chw=False)
    e1 = insight_face.calc_embedding(img1)
    e2 = insight_face.calc_embedding(img2)
    calc_dist(e1, e2)

    # xinchi/xinchi_1 -> 0.8586558
    # xinchi/xinchi_2 -> 0.848077
    # xinchi_1/xinchi_2 -> 0.79819506

    insight_face.set_target_embedding(e1)
    print('calculate in tf[similarity,l2_dist]:', insight_face.compare(img2))

    insight_face.optimize(img2)
