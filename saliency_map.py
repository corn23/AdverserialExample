from utils import load_pretrain_model,preprocess_img,calculate_region_importance,classify,calculate_img_region_importance,load_imagenet_label,mask_img_region
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
old_img = np.array(img)
batch_img = np.expand_dims(img, 0)
imagenet_label = load_imagenet_label(img_label_path)
prob = tf.nn.softmax(logits)
_prob = sess.run(prob, feed_dict={images_pl:batch_img})[0]
classify(img,_prob,imagenet_label,1,1)
# attribution method 1, logits[label])/d(img_pl)
grad_map_tensor = tf.gradients(label_logits,images_pl)[0]
grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:batch_img,y_label:285}) # very unclear
#new_grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:np.expand_dims(new_img,0),y_label:285})

gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl)
vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:285}) # better
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:285}) # much clear
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

to_dec_center = (135,130)
to_dec_radius = (60,20)
print(calculate_region_importance(grad_map, to_dec_center, to_dec_radius))  # 0.475 or 26.05 (smooth)
print(calculate_region_importance(grad_map, (10, 120), (10,10)))  # 1.04 or 172.83(smooth)

# construct to_inc_region and to_dec_region
to_dec_region = calculate_img_region_importance(grad_map_tensor, to_dec_center, to_dec_radius)
to_inc_region = calculate_img_region_importance(grad_map_tensor, (10, 120), (10,10))

# finite difference
# delta = 0.1
# update_grad = np.zeros((img_size,img_size,3))
# epoch = 3
# k = 0
# while epoch > 0:
#     for i in range(135,145):
#         for j in range(135,145):
#             print(i,j,k)
#             img_plus_value = min(img[i,j,k]+delta,1)
#             img_minus_value = max(img[i,j,k]-delta,0)
#             img_plus = np.array(img)
#             img_plus[i,j,k] = img_plus_value
#             img_minus = np.array(img)
#             img_minus[i,j,k] = img_minus_value
#             loss_plus = sess.run(to_dec_region,feed_dict={images_pl:np.expand_dims(img_plus,0),y_label:285})
#             loss_minus = sess.run(to_dec_region,feed_dict={images_pl:np.expand_dims(img_minus,0),y_label:285})
#             value = (loss_plus-loss_minus)/(2*(img_plus_value-img_minus_value))
#             print(value)
#             update_grad[i,j,k] = value
#     new_img = np.clip(img-0.1*update_grad,0,1)
#     new_loss,new_logits = sess.run([to_dec_region,logits],feed_dict={images_pl:np.expand_dims(new_img,0),y_label:285})
#     old_loss,old_logits = sess.run([to_dec_region,logits],feed_dict={images_pl:np.expand_dims(old_img,0),y_label:285})
#     print("new:{} ,{}".format(new_loss,np.argmax(new_logits)))
#     print("old:{}, {}".format(old_loss,np.argmax(old_logits)))
#     img=new_img
#     epoch -= 1

# try NES (Natural evolutionary strategies)
N = 80
sigma = 0.001
eta = 0.0001
epsilon = 0.05
epoch = 20
img = np.array(old_img)
old_loss, old_logits = sess.run([to_dec_region, logits],
                                feed_dict={images_pl: np.expand_dims(old_img, 0), y_label: 285})
while epoch > 0:
    delta = np.random.randn(int(N/2),img_size*img_size*3)
    delta = np.concatenate((delta,-delta),axis=0)
    estimate_grad = 0
    grad_sum = 0
    f_value_list = []
    for idelta in delta:
        img_plus = np.clip(img+sigma*idelta.reshape(img_size,img_size,3),0,1)
        f_value = sess.run(to_dec_region,feed_dict={images_pl:np.expand_dims(img_plus,0),y_label:285})
        f_value_list.append(f_value)
        grad_sum += f_value*idelta.reshape(img_size,img_size,3)
    grad_sum = grad_sum/(N*sigma)
    new_img = np.clip(img-eta*grad_sum,0,1)
    new_loss, new_logits = sess.run([to_dec_region, logits],
                                    feed_dict={images_pl: np.expand_dims(new_img, 0), y_label: 285})
    print("new:{}, {}".format(new_loss, np.argmax(new_logits)))
    print("old:{}, {}".format(old_loss, np.argmax(old_logits)))
    img = np.array(new_img)
    epoch -= 1

num_list = '_'.join([str(to_dec_center[0]),str(to_dec_center[1]),str(to_dec_radius[0]),str(to_dec_radius[1]),str(N),str(eta)])
np.save('new_img'+num_list,new_img)

# show the neighbour change
# yita = np.linspace(-0.2,0.2,40)
# new_loss_list = []
# for iyita in yita:
#     new_img = np.clip(old_img-iyita*grad_sum,0,1)
#     new_loss, new_logits = sess.run([to_dec_region, logits],
#                                 feed_dict={images_pl: np.expand_dims(new_img, 0), y_label: 285})
#     new_loss_list.append(new_loss)
# plt.plot(yita,(np.array(new_loss_list)-old_loss))
# plt.xlabel('eta')
# plt.ylabel('new_loss-old_loss')

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
