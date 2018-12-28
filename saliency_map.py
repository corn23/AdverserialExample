from utils import load_pretrain_model,preprocess_img,calculate_region_importance,classify,calculate_img_region_importance,load_imagenet_label,mask_img_region
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import saliency
import argparse
import sys
from utils import plot,calculate_deeplift_loss
from deepexplain.tensorflow import DeepExplain

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="specify the pretrained model name", required=False, default='vgg16')
    parser.add_argument("--img_path", help='specify the image path', required=False, default='./picture/dog_cat.jpg')
    parser.add_argument("--true_label", type=int, help='the groundtruth class for input image', required=False, default=208)
    parser.add_argument("--adv_label", type=int, help='specify the target adversarial label', default=100)
    parser.add_argument("--epoch", type=int, help="specify the training epochs", default=100)
    parser.add_argument("--eps", type=float, help="specify the degree of disturb", default=8/255)
    parser.add_argument("--lr", type=float, help="specify the learning rate", default=0.0001)
    parser.add_argument("--lambda_up", type=float, help="specify the lambda for increasing gradient in the given region", default=1)
    parser.add_argument("--lambda_down", type=float, help="specify the lambda for lowering gradient in the given region", default=0)
    parser.add_argument("--lambda_label_loss", type=float, help="specify the lambda for the crossentropy loss of classifying wrong label", default=10)
    parser.add_argument("--label_num", type=int, help="specify the number of final outputs in pretrained model", default=1000)
    parser.add_argument("--is_cluster", help="indicate if work on cluster so that graphic function is not available", action="store_true", required=False, default=False)
    parser.add_argument("--summary_path", help="specify the tensorflow summary file path", default='./summary')
    parser.add_argument("--write_summary", help="write summary or not", action="store_true", default=False)
    parser.add_argument("--image_interval", type=int, help="write image into summary every *image_interval*",default=5)
    parser.add_argument("-N", type=int,help="specify the number vectors in NES", default=30)
    parser.add_argument("--imp",help="specify a path for intermediate image",default='')
    parser.add_argument("--sigma", type=float,help="specify sigma vector in NES", default=0.001)

    pargs = parser.parse_args(argv)
    return pargs

def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

    model_name = args.model_name
    img_path = args.img_path
    img_label_path = 'imagenet.json'
    true_class = args.true_label
    adversarial_label = args.adv_label
    label_num = args.label_num
    lambda_up, lambda_down, lambda_label_loss = args.lambda_up, args.lambda_down,args.lambda_label_loss

    # model_name = 'inception_v3'
    # img_path = './picture/dog_cat.jpg'
    # img_label_path = 'imagenet.json'
    # true_class = 208
    sess, graph, img_size, images_pl, logits = load_pretrain_model(model_name,is_explain=True)
    y_label = tf.placeholder(dtype=tf.int32,shape=())
    label_logits = logits[0,y_label]

    if len(args.imp)>0:
        img = np.load(args.imp)
        init_epoch = int(args.imp[:-4].split('_')[-1])
        loss_list = list(np.load('loss_'+args.imp))
    else:
        img = PIL.Image.open(img_path)
        img = preprocess_img(img, img_size)
        init_epoch = 0
        loss_list = []

    old_img = np.array(img)
    batch_img = np.expand_dims(img, 0)

    #new_img = np.load('vgg16_30_0.0004_1000_0.001_0.03_4000.npy')
    #new_batch_img = np.concatenate((np.expand_dims(new_img,0),batch_img),axis=0)
    #new_batch_img = np.expand_dims(new_img,0)
    #all_img = np.concatenate((batch_img,new_batch_img))
    imagenet_label = load_imagenet_label(img_label_path)
    prob = tf.nn.softmax(logits)
    _prob = sess.run(prob, feed_dict={images_pl:batch_img})[0]
    #classify(img,_prob,imagenet_label,1,1)

    ####
    #deep explain
    # from deepexplain.tensorflow import DeepExplain
    # label_logits = logits[0,208]
    # with DeepExplain(session=sess) as de:
    #     attributions = {
    #         # Gradient-based
    #         # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
    #         # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
    #         'Saliency maps': de.explain('saliency', label_logits, images_pl, batch_img),
    #         'Gradient * Input': de.explain('grad*input', label_logits, images_pl, batch_img),
    #         # 'Integrated Gradients': de.explain('intgrad', label_logits, images_pl, new_batch_img),
    #         'Epsilon-LRP': de.explain('elrp', label_logits, images_pl, batch_img),
    #         'DeepLIFT (Rescale)': de.explain('deeplift', label_logits, images_pl, batch_img),
    #         # Perturbation-based (comment out to evaluate, but this will take a while!)
    #         #'Occlusion [15x15]':    de.explain('occlusion', label_logits, images_pl, batch_img, window_shape=(15,15,3), step=4)
    #     }    ####
    #     new_attributions = {
    #         # Gradient-based
    #         # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
    #         # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
    #         'Saliency maps': de.explain('saliency', label_logits, images_pl, new_batch_img),
    #         'Gradient * Input': de.explain('grad*input', label_logits, images_pl, new_batch_img),
    #         # 'Integrated Gradients': de.explain('intgrad', label_logits, images_pl, new_batch_img),
    #         'Epsilon-LRP': de.explain('elrp', label_logits, images_pl, new_batch_img),
    #         'DeepLIFT (Rescale)': de.explain('deeplift', label_logits, images_pl, new_batch_img),
    #         # Perturbation-based (comment out to evaluate, but this will take a while!)
    #         #'Occlusion [15x15]':    de.explain('occlusion', label_logits, images_pl, batch_img, window_shape=(15,15,3), step=4)
    #     }    ####
    #     attributions['Saliency maps'] = np.concatenate((attributions['Saliency maps'],new_attributions['Saliency maps']),axis=0)
    #     attributions['Gradient * Input'] = np.concatenate((attributions['Gradient * Input'],new_attributions['Gradient * Input']),axis=0)
    #     attributions['Epsilon-LRP'] = np.concatenate((attributions['Epsilon-LRP'],new_attributions['Epsilon-LRP']),axis=0)
    #     attributions['DeepLIFT (Rescale)'] = np.concatenate((attributions['DeepLIFT (Rescale)'],new_attributions['DeepLIFT (Rescale)']),axis=0)
    #
    # n_cols = int(len(attributions)) + 1
    # n_rows = 2
    # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    #
    # for i, xi in enumerate(all_img):
    #     # xi = (xi - np.min(xi))
    #     # xi /= np.max(xi)
    #     ax = axes.flatten()[i * n_cols]
    #     ax.imshow(xi)
    #     ax.set_title('Original')
    #     ax.axis('off')
    #     for j, a in enumerate(attributions):
    #         axj = axes.flatten()[i * n_cols + j + 1]
    #         plot(attributions[a][i], xi=xi, axis=axj, dilation=.5, percentile=99, alpha=.2).set_title(a)
    ######
    label_logits = logits[0,208]
    with DeepExplain(session=sess) as de:
        dlift = de.explain('deeplift', label_logits, images_pl, batch_img)

    grad_map_tensor = tf.gradients(label_logits,images_pl)[0]
    grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:np.expand_dims(img,0),y_label:true_class})

    gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl) # 1951/1874
    vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:true_class}) # better
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

    # smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:true_class}) # much clear, 2204/2192
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

    #
    # new_img = np.load('vgg16_60_70_35_45_30_0.0001_800_0.0_0.0_9000.npy')
    # new_grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:np.expand_dims(new_img,0),y_label:true_class})
    # new_vanilla_mask_3d = gradient_saliency.GetMask(new_img, feed_dict={y_label:true_class}) # better
    # new_vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(new_vanilla_mask_3d)
    # new_smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(new_img, feed_dict={y_label:true_class}) # much clear, 2204/2192
    # new_smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(new_smoothgrad_mask_3d)

    #to_dec_center = (60,70)
    to_dec_center = (100,65)
    #to_dec_radius = (35,45)
    to_dec_radius = (80,60)
    to_inc_center = (120,170)
    to_inc_radius = (40,30)
    _map = vanilla_mask_grayscale
    print(calculate_region_importance(_map, to_dec_center, to_dec_radius))
    print(calculate_region_importance(_map, to_inc_center, to_inc_radius))

    # construct to_inc_region and to_dec_region
    to_dec_region = calculate_img_region_importance(grad_map_tensor, to_dec_center, to_dec_radius)
    to_inc_region = calculate_img_region_importance(grad_map_tensor, to_inc_center, to_inc_radius)

    # try NES (Natural evolutionary strategies)
    N = args.N
    sigma = args.sigma
    epsilon = round(args.eps,2)
    epoch = args.epoch
    eta = args.lr
    #loss = to_dec_region/to_inc_region
    #old_loss = sess.run(loss,feed_dict={images_pl: np.expand_dims(img, 0), y_label: true_class})
    old_loss = calculate_deeplift_loss(dlift,to_dec_center, to_dec_radius, to_inc_center, to_inc_radius)
    num_list = '_'.join(['big',model_name, str(N), str(eta), str(epoch), str(sigma), str(epsilon)])
    print(num_list)
    for i in range(epoch):
        delta = np.random.randn(int(N/2),img_size*img_size*3)
        delta = np.concatenate((delta,-delta),axis=0)
        grad_sum = 0
        f_value_list = []
        for idelta in delta:
            img_plus = np.clip(img+sigma*idelta.reshape(img_size,img_size,3),0,1)
            #f_value = sess.run(loss,feed_dict={images_pl:np.expand_dims(img_plus,0),y_label:true_class})
            with DeepExplain(session=sess) as de:
                dlift = de.explain('deeplift', label_logits, images_pl, np.expand_dims(img_plus,0))
            f_value = calculate_deeplift_loss(dlift,to_dec_center, to_dec_radius, to_inc_center, to_inc_radius)
            f_value_list.append(f_value)
            grad_sum += f_value*idelta.reshape(img_size,img_size,3)
        grad_sum = grad_sum/(N*sigma)
        new_img = np.clip(np.clip(img-eta*grad_sum,old_img-epsilon,old_img+epsilon),0,1)
        #new_loss, new_logits = sess.run([loss, logits],
        #                                feed_dict={images_pl: np.expand_dims(new_img, 0), y_label: true_class})
        with DeepExplain(session=sess) as de:
            dlift = de.explain('deeplift', label_logits, images_pl, np.expand_dims(new_img, 0))
        new_loss = calculate_deeplift_loss(dlift, to_dec_center, to_dec_radius, to_inc_center, to_inc_radius)

        loss_list.append(new_loss)
        print("epoch:{} new:{}, old:{}, {}".format(i, new_loss,old_loss, np.argmax(_prob)))
        sys.stdout.flush()
        img = np.array(new_img)
        if i % args.image_interval ==0:
            temp_name = num_list+'_'+str(i+init_epoch)
            np.save(temp_name,new_img)
        if i % args.image_interval == 0:
            np.save('loss_'+temp_name,loss_list)
    np.save(num_list+'_'+str(epoch+init_epoch),new_img)
    np.save('loss_'+num_list+'_'+str(epoch+init_epoch),loss_list)


    # show the neighbour change
    # yita = np.linspace(-0.2,0.2,40)
    # new_loss_list = []
    # for iyita in yita:
    #     new_img = np.clip(old_img-iyita*sigma*grad_sum,0,1)
    #     new_loss, new_logits = sess.run([to_inc_region, logits],
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
if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)

# new_img = np.array(img)
# new_img[25:95,105:195,:] += sub_img
# new_img[25:95,25:115] =
# new_img = np.clip(new_img,0,1)
#
# # attack the smooth gradient map
# region_grad_value_list = []
# ration_list = []
# for ilabel in range(1000):
#     new_smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:ilabel}) # much clear, 2204/2192
#     new_smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(new_smoothgrad_mask_3d)
#     _map = new_smoothgrad_mask_grayscale
#     dec = calculate_region_importance(_map, to_dec_center, to_dec_radius)
#     inc = calculate_region_importance(_map, to_inc_center, to_inc_radius)
#     print("label:{},dec:{},inc:{},ration:{}".format(ilabel,dec,inc,dec/inc))
#     region_grad_value_list.append((dec,inc))
#     ration_list.append(dec/inc)


# def get_cam_map_overlay(cam_map,img_size):
#     cam_map_weight = np.mean(np.mean(cam_map, axis=0), axis=0)
#     cam_saliency_map = np.clip(np.mean(np.multiply(cam_map_weight, cam_map), axis=2), a_min=0, a_max=1)
#     cam_map_resized = cv.resize(cam_saliency_map, (224, 224))
#     cam_map_resized = cam_map_resized / cam_map_resized.max()
#
#     cam_map_overlay = np.zeros((img_size, img_size, 4))
#     cam_map_overlay[:, :, 0] = cam_map_resized
#     cam_map_overlay[:, :, 3] = cam_map_resized > 0.4 * cam_map_resized.max()
#     return cam_map_overlay

###############################
# grad-cam
# from skimage.transform import resize
# import cv2
# tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values()]
# maxpool = tf.get_default_graph().get_tensor_by_name('vgg_16/pool5/MaxPool:0')
# final_conv = tf.get_default_graph().get_tensor_by_name('vgg_16/conv5/conv5_3/BiasAdd:0')
# cam_map_tensor = tf.gradients(label_logits,final_conv)
#
# cam_map = sess.run(cam_map_tensor,feed_dict= {images_pl:np.expand_dims(new_img,0),y_label:true_class})[0][0]
#
# # calculate the weigths
# new_cam_map_overlay = get_cam_map_overlay(cam_map,img_size=img_size)
#
# plt.subplot(122)
# axe1 = plt.subplot(121)
# axe1.imshow(img)
# axe1.imshow(cam_map_overlay)
#
# axe2 = plt.subplot(122)
# axe2.imshow(new_img)
# axe2.imshow(new_cam_map_overlay)
#
# plt.subplot(122)
# plt.subplot(121)
# heatmap = cv2.applyColorMap(np.uint8(cam_map_overlay[:,:,0]*255),cv2.COLORMAP_JET)
# plt.imshow(heatmap)
# plt.subplot(122)
# new_heatmap = cv2.applyColorMap(np.uint8(new_cam_map_overlay[:,:,0]*255),cv2.COLORMAP_JET)
# plt.imshow(new_heatmap)

################################
# meaningful perturbation

##########
# deep explain
