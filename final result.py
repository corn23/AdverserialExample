import numpy as np
import tensorflow as tf
from saliency_map import parse_arguments
from utils import load_pretrain_model,preprocess_img,load_imagenet_label
import matplotlib.pyplot as plt
import saliency
import PIL
from deepexplain.tensorflow import DeepExplain
from utils import plot
from collections import OrderedDict
import pickle

args = parse_arguments('')
model_name = args.model_name
img_path = args.img_path
img_label_path = 'imagenet.json'
true_class = args.true_label
adversarial_label = args.adv_label
label_num = args.label_num
lambda_up, lambda_down, lambda_label_loss = args.lambda_up, args.lambda_down, args.lambda_label_loss
sess, graph, img_size, images_pl, logits = load_pretrain_model(model_name, is_explain=True)
y_label = tf.placeholder(dtype=tf.int32, shape=())
img_label = load_imagenet_label(img_label_path)

img = PIL.Image.open(img_path)
img = preprocess_img(img, img_size)
#new_img = np.load('big_vgg16_30_0.0001_1000_0.001_0.03_3000.npy') # 258
new_img = np.load('vgg16_60_70_35_45_30_0.0001_800_0.0_0.0_9000.npy') # 208

batch_img = np.expand_dims(img,0)
new_batch_img = np.expand_dims(new_img,0)

true_class = 208
label_logits = logits[0, true_class]
gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl) # 1951/1874

attributions = OrderedDict()
with DeepExplain(session=sess) as de:
    ori_attributions = {
        # Gradient-based
        # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
        # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
        'Saliency maps': de.explain('saliency', label_logits, images_pl, batch_img),
        'Gradient * Input': de.explain('grad*input', label_logits, images_pl, batch_img),
        'Epsilon-LRP': de.explain('elrp', label_logits, images_pl, batch_img),
        'Integrated Gradients': de.explain('intgrad', label_logits, images_pl, new_batch_img,steps=25),
        'DeepLIFT (Rescale)': de.explain('deeplift', label_logits, images_pl, batch_img),
        # Perturbation-based (comment out to evaluate, but this will take a while!)
        # 'Occlusion [15x15]':    de.explain('occlusion', label_logits, images_pl, batch_img, window_shape=(15,15,3), step=4)
        'smoothgrad': np.expand_dims(gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:true_class}),axis=0)
    }  ####
    attack_attributions = {
        # Gradient-based
        # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
        # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
        'Saliency maps': de.explain('saliency', label_logits, images_pl, new_batch_img),
        'Gradient * Input': de.explain('grad*input', label_logits, images_pl, new_batch_img),
        'Integrated Gradients': de.explain('intgrad', label_logits, images_pl, new_batch_img,steps=25),
        'Epsilon-LRP': de.explain('elrp', label_logits, images_pl, new_batch_img),
        'DeepLIFT (Rescale)': de.explain('deeplift', label_logits, images_pl, new_batch_img),
        'smoothgrad':np.expand_dims(gradient_saliency.GetSmoothedMask(new_img, feed_dict={y_label:true_class}),axis=0)
        # Perturbation-based (comment out to evaluate, but this will take a while!)
        # 'Occlusion [15x15]':    de.explain('occlusion', label_logits, images_pl, batch_img, window_shape=(15,15,3), step=4)
    }  ####
    attributions['Saliency maps'] = \
        np.concatenate((ori_attributions['Saliency maps'], attack_attributions['Saliency maps']),axis=0)
    attributions['Gradient * Input'] = \
        np.concatenate((ori_attributions['Gradient * Input'], attack_attributions['Gradient * Input']), axis=0)
    attributions['Epsilon-LRP'] = \
        np.concatenate((ori_attributions['Epsilon-LRP'], attack_attributions['Epsilon-LRP']), axis=0)
    attributions['DeepLIFT (Rescale)'] = \
        np.concatenate((ori_attributions['DeepLIFT (Rescale)'], attack_attributions['DeepLIFT (Rescale)']), axis=0)
    attributions['smoothgrad'] = \
        np.concatenate((ori_attributions['smoothgrad'], attack_attributions['smoothgrad']),axis=0)
    attributions['Integrated Gradients'] = \
        np.concatenate((ori_attributions['Integrated Gradients'], attack_attributions['Integrated Gradients']), axis=0)


with open('result_208.pkl','wb') as f:
    a = pickle.dump((ori_attributions,attack_attributions),f)

n_cols = int(len(attributions)) + 1
n_rows = 2
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
all_img = [img, new_img]
img_name = ['Original','attack_208']
for i, xi in enumerate(all_img):
    # xi = (xi - np.min(xi))
    # xi /= np.max(xi)
    ax = axes.flatten()[i * n_cols]
    ax.imshow(xi)
    ax.set_title(img_name[i])
    ax.axis('off')
    for j, a in enumerate(attributions):
        axj = axes.flatten()[i * n_cols + j + 1]
        plot(attributions[a][i], xi=xi, axis=axj, dilation=.5, percentile=99, alpha=.2).set_title(a)


## show the saliency and smoothgradient map of img to class 208 and 258
label_logits = logits[0, y_label]
gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl) # 1951/1874

true_class = 208
vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:true_class}) # better
vanilla_mask_grayscale_208 = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
#
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:true_class}) # much clear, 2204/2192
smoothgrad_mask_grayscale_208 = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

true_class = 258
vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:true_class}) # better
vanilla_mask_grayscale_258 = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
#
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:true_class}) # much clear, 2204/2192
smoothgrad_mask_grayscale_258 = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3 * 2, 3 * 2))
axes[0,0].imshow(vanilla_mask_grayscale_208)
axes[0,0].set_title('raw gradient 208')
axes[0,0].axis('off')
axes[0,1].imshow(smoothgrad_mask_grayscale_208)
axes[0,1].set_title('smooth gradient 208')
axes[0,1].axis('off')
axes[1,0].imshow(vanilla_mask_grayscale_258)
axes[1,0].set_title('raw gradient 258')
axes[1,0].axis('off')
axes[1,1].imshow(smoothgrad_mask_grayscale_258)
axes[1,1].set_title('smooth gradient 258')
axes[1,1].axis('off')


# plot loss
#loss1 = np.load('loss_big_vgg16_30_0.0004_1000_0.001_0.03_0.99_0_1000_208.npy').reshape(500,1)
loss2 = np.load('loss_big_vgg16_30_0.0004_1000_0.001_0.03_0.99_0.9_1000_208.npy').reshape(1000,1)
loss3 = np.load('loss_big_vgg16_30_0.0004_1000_0.001_0.03_1_0_1000_208.npy').reshape(1000,1)
loss4 = np.load('loss_big_vgg16_30_0.0004_1000_0.001_0.03_1_0.9_1000_208.npy').reshape(1000,1)


final_loss = np.concatenate((loss2,loss3,loss4),axis=1)
plt.plot(final_loss)
plt.legend(['alpha=0.99,beta=0.9','alpha=1,beta=0','alpha=1,beta=0.9'])