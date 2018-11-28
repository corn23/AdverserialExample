from urllib.request import urlretrieve
import tarfile
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim
import saliency

def maybe_download_and_extract(dest_directory,data_url):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory,filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def classify(img, p, imagenet_labels, correct_class=None, target_class=None, is_cluster=False):
    if not is_cluster:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.sca(ax1)
        ax1.imshow(img)
        fig.sca(ax1)
        topk = list(p.argsort()[-10:][::-1])
        topprobs = p[topk]
        barlist = ax2.bar(range(10), topprobs)
        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk:
            barlist[topk.index(correct_class)].set_color('g')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10),
                   [imagenet_labels[i][:15] for i in topk],
                   rotation='vertical')
        fig.subplots_adjust(bottom=0.2)
        plt.show()


def preprocess_img(img,target):
    wide_is_bigger = img.width > img.height
    new_w = target if not wide_is_bigger else int(img.width * target / img.height)
    new_h = target if wide_is_bigger else int(img.height * target / img.width)
    new_img = img.resize((new_w, new_h)).crop((0, 0, target, target))
    new_img = (np.asarray(new_img) / 255.0).astype(np.float32)
    return new_img


def load_imagenet_label(file_path):
    with open(file_path) as f:
        imagenet_labels = json.load(f)
    return imagenet_labels


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
        plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(title)


def calculate_region_importance(greymap,center,radius=(10,10)):
    if greymap.ndim == 4:
        greymap = np.squeeze(greymap,axis=0)
    if greymap.ndim == 3:
        greymap = np.mean(abs(greymap),axis=2)
    length,width = greymap.shape
    up_bound = max(center[0]-radius[0],0)
    lower_bound = min(center[0]+radius[0],length)
    left_bound = max(center[1]-radius[1],0)
    right_bound = min(center[1]+radius[1],width)
    importance = np.sum(abs(greymap[up_bound:lower_bound,left_bound:right_bound]))
    return importance


def calculate_img_region_importance(map3D,center,radius=(10,10)):
    batch_num, length, width, channel = map3D.shape
    assert channel == 3
    assert batch_num == 1
    greymap = tf.squeeze(map3D,0)
    greymap = tf.reduce_sum(tf.abs(greymap),2)
    up_bound = max(center[0]-radius[0],0)
    lower_bound = min(center[0]+radius[0],length)
    left_bound = max(center[1]-radius[1],0)
    right_bound = min(center[1]+radius[1],width)
    importance = tf.reduce_sum(greymap[up_bound:lower_bound,left_bound:right_bound])
    return importance


def load_pretrain_model(model_name='vgg16'):
    sess = tf.InteractiveSession()
    if model_name == 'vgg16':
        img_size = 224
        images_v = tf.placeholder(dtype=tf.float32,shape=(1, img_size, img_size, 3))
        preprocessed = tf.multiply(tf.subtract(images_v, 0.5), 2.0)
        arg_scope = nets.vgg.vgg_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, _ = nets.vgg.vgg_16(
                preprocessed, num_classes=1000, is_training=False)

        # restore model
        data_url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
        dest_directory = './imagenet'
        maybe_download_and_extract(dest_directory, data_url)
        restore_vars = [
            var for var in tf.global_variables()
            if var.name.startswith('vgg_16/')
        ]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, os.path.join(dest_directory, 'vgg_16.ckpt'))
        graph = tf.get_default_graph()
    if model_name == "resnet_v1_50":
        img_size = 224
        images_v = tf.Variable(tf.zeros((1, img_size, img_size, 3)))
        preprocessed = tf.multiply(tf.subtract(images_v, 0.5), 2.0)
        arg_scope = nets.resnet_v1.resnet_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, _ = nets.resnet_v1.resnet_v1_50(
                preprocessed, num_classes=1000, is_training=False)
        logits = logits[0][0]
        # restore model
        data_url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
        dest_directory = './imagenet'
        maybe_download_and_extract(dest_directory, data_url)
        restore_vars = [
            var for var in tf.global_variables()
            if var.name.startswith('resnet_v1_50/')
        ]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, os.path.join(dest_directory, 'resnet_v1_50.ckpt'))
        graph = tf.get_default_graph()

    if model_name == "resnet_v1_152":
        img_size = 224
        images_v = tf.Variable(tf.zeros((1, img_size, img_size, 3)))
        preprocessed = tf.multiply(tf.subtract(images_v, 0.5), 2.0)
        arg_scope = nets.resnet_v1.resnet_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, _ = nets.resnet_v1.resnet_v1_152(
                preprocessed, num_classes=1000, is_training=False)
        logits = logits[0][0]
        # restore model
        data_url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
        dest_directory = './imagenet'
        maybe_download_and_extract(dest_directory, data_url)
        restore_vars = [
            var for var in tf.global_variables()
            if var.name.startswith('resnet_v1_152/')
        ]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, os.path.join(dest_directory, 'resnet_v1_152.ckpt'))
        graph = tf.get_default_graph()

    return sess, graph, img_size, images_v, logits


def show_gradient_map(graph, sess, y, x, img,
                      is_integrated=False, is_smooth=True, feed_dict=None, is_cluster=False):

    if not is_integrated and not is_smooth:
        gradient_saliency = saliency.GradientSaliency(graph, sess, y, x)
        vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict=feed_dict)
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
        if not is_cluster:
            ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient')
        return vanilla_mask_3d, vanilla_mask_grayscale

    if not is_integrated and is_smooth:
        gradient_saliency = saliency.GradientSaliency(graph, sess, y, x)
        smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict=feed_dict)
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
        if not is_cluster:
            ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad')
        return smoothgrad_mask_3d, smoothgrad_mask_grayscale

    if is_integrated and not is_smooth:
        baseline = np.zeros(img.shape)
        baseline.fill(-1)
        gradient_saliency = saliency.IntegratedGradients(graph, sess, y, x)
        vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict=feed_dict, x_steps=5, x_baseline=baseline)
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
        if not is_cluster:
            ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient')
        return vanilla_mask_3d, vanilla_mask_grayscale

    if is_integrated and is_smooth:
        baseline = np.zeros(img.shape)
        baseline.fill(-1)
        gradient_saliency = saliency.IntegratedGradients(graph, sess, y, x)
        smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict=feed_dict, x_steps=5, x_baseline=baseline)
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
        if not is_cluster:
            ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad')
        return smoothgrad_mask_3d, smoothgrad_mask_grayscale


def model_train(sess, optim_step, project_step, loss,
                feed_dict_optim, feed_dict_project, epoch=100):
    for i in range(epoch):
        _, _loss = sess.run([optim_step, loss],
                            feed_dict=feed_dict_optim)
        sess.run([project_step], feed_dict=feed_dict_project)
        print("epoch %d , loss=%g" % (i, _loss))


def mask_img_region(img,center,radius):
    new_img = np.array(img)
    new_img[center[0]-radius[0]:center[0]+radius[0],center[1]-radius[1]:center[1]+radius[1]] = 0
    plt.imshow(new_img)
    return new_img



