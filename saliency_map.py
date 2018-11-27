from utils import load_pretrain_model,preprocess_img,calculate_region_importance,calculate_img_region_importance,load_imagenet_label
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import saliency

model_name = 'vgg16'
img_path = './picture/cat_and_dog.jpg'
img_label_path = 'imagenet.json'
sess, graph, img_size, images_pl, logits = load_pretrain_model(model_name)
y_label = tf.placeholder(dtype=tf.int32,shape=())
label_logits = logits[0,y_label]

img = PIL.Image.open(img_path)
img = preprocess_img(img, img_size)
img_old = np.array(img)
batch_img = np.expand_dims(img, 0)
imagenet_label = load_imagenet_label(img_label_path)
_logits = sess.run(logits, feed_dict={images_pl:batch_img})

# attribution method 1, d(logits[label])/d(img_pl)
grad_map_tensor = tf.gradients(label_logits,images_pl)[0]
grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:batch_img,y_label:285}) # very unclear

# gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl)
# vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:665}) # better
# vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
#
# smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:665}) # much clear
# smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
#
# print(calculate_region_importance(grad_map,(110, 100), 10))  # 0.475 or 26.05 (smooth)
# print(calculate_region_importance(grad_map,(70, 100), 10))  #1.04 or 172.83(smooth)

# construct to_inc_region and to_dec_region
to_inc_region = calculate_img_region_importance(grad_map_tensor,(110, 100), 10)
to_dec_region = calculate_img_region_importance(grad_map_tensor,(70,100), 10)

# finite difference
delta = 0.1
update_grad = np.zeros((img_size,img_size))
for i in range(img_size):
    for j in range(img_size):
        print(i,j)
        img_plus_value = min(img[i,j,0]+delta,1)
        img_minus_value = max(img[i,j,0]-delta,0)
        img_plus = np.array(img)
        img_plus[i,j,0] = img_plus_value
        img_minus = np.array(img)
        img_minus[i,j,0] = img_minus_value
        loss_plus = sess.run(to_dec_region,feed_dict={images_pl:np.expand_dims(img_plus,0),y_label:285})
        loss_minus = sess.run(to_dec_region,feed_dict={images_pl:np.expand_dims(img_minus,0),y_label:285})
        update_grad[i,j] = (loss_plus-loss_plus)/(2*(img_plus_value-img_minus_value))


# # test simple gradient
# w_value = np.random.rand(2,2,3,1)
# w_value = np.array([0.11,0.12,0.13,0.14,0.15,0.16,0.21,0.22,0.23,0.24,0.25,0.26]).reshape(2,2,3,1)
# w_value2 = np.array([0.11,0.12,0.13,0.14]).reshape(2,2,1,1)
# with tf.name_scope("test"):
#     x = tf.placeholder(dtype=tf.float64,shape=(1,7,7,3))
#     W = tf.constant(w_value)
#     unit1 = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID') # (1,6,6,1)
#     unit2 = tf.nn.relu(unit1)
#     unit3 = tf.nn.max_pool(unit2, ksize=(1,2,2,1), strides=[1,2,2,1], padding='VALID')# (1,3,3,1)
#     W2 = tf.constant(w_value2)
#     unit4 = tf.nn.conv2d(unit3,W2,strides=[1,1,1,1], padding='VALID')
#     unit5 = tf.reshape(unit4,[1,-1]) # (1)
#     final_out = tf.reduce_sum(unit4)
#     log_final_out = tf.log(final_out)
#
#
# test_grad = tf.gradients(final_out,x)[0]
# test_loss = tf.nn.l2_loss(test_grad)
# test_update_grad = tf.gradients(test_loss,x)[0]
#
