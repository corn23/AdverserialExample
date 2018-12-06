from utils import load_pretrain_model,preprocess_img,calculate_region_importance,classify,calculate_img_region_importance,load_imagenet_label,mask_img_region
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import saliency
import argparse
import sys

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
    sess, graph, img_size, images_pl, logits = load_pretrain_model(model_name)
    y_label = tf.placeholder(dtype=tf.int32,shape=())
    label_logits = logits[0,y_label]

    if len(args.imp)>0:
        img = np.load(args.imp)
    else:
        img = PIL.Image.open(img_path)
        img = preprocess_img(img, img_size)
    old_img = np.array(img)
    batch_img = np.expand_dims(img, 0)
    imagenet_label = load_imagenet_label(img_label_path)
    prob = tf.nn.softmax(logits)
    _prob = sess.run(prob, feed_dict={images_pl:batch_img})[0]
    #classify(img,_prob,imagenet_label,1,1)

    # attribution method 1, logits[label])/d(img_pl)
    grad_map_tensor = tf.gradients(label_logits,images_pl)[0]
    grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:np.expand_dims(img,0),y_label:true_class})

    gradient_saliency = saliency.GradientSaliency(graph, sess, label_logits, images_pl) # 1951/1874
    vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict={y_label:true_class}) # better
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={y_label:true_class}) # much clear, 2204/2192
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    #
    # new_img = np.load('new_imgvgg16_60_70_35_45_30_0.0001_200_0.0_0.0.npy')
    # new_grad_map = sess.run(grad_map_tensor,feed_dict={images_pl:np.expand_dims(new_img,0),y_label:true_class})
    # new_vanilla_mask_3d = gradient_saliency.GetMask(new_img, feed_dict={y_label:true_class}) # better
    # new_vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(new_vanilla_mask_3d)
    # new_smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(new_img, feed_dict={y_label:true_class}) # much clear, 2204/2192
    # new_smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(new_smoothgrad_mask_3d)

    to_dec_center = (60,70)
    to_dec_radius = (35,45)
    to_inc_center = (120,170)
    to_inc_radius = (40,30)
    _map = grad_map
    print(calculate_region_importance(_map, to_dec_center, to_dec_radius))
    print(calculate_region_importance(_map, to_inc_center, to_inc_radius))

    # construct to_inc_region and to_dec_region
    to_dec_region = calculate_img_region_importance(grad_map_tensor, to_dec_center, to_dec_radius)
    to_inc_region = calculate_img_region_importance(grad_map_tensor, to_inc_center, to_inc_radius)

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
    N = args.N
    sigma = 0.001
    epsilon = args.eps
    epoch = args.epoch
    eta = args.lr
    #loss = -lambda_up*to_inc_region+lambda_down*to_dec_region
    loss = to_dec_region/to_inc_region
    old_loss = sess.run(loss,feed_dict={images_pl: np.expand_dims(img, 0), y_label: true_class})
    #eta = 0.01/abs(old_loss)
    num_list = '_'.join([model_name,str(to_dec_center[0]),str(to_dec_center[1]),str(to_dec_radius[0]),str(to_dec_radius[1]),
                         str(N),str(eta),str(epoch),str(lambda_down),str(lambda_up)])
    print(num_list)
    while epoch > 0:
        delta = np.random.randn(int(N/2),img_size*img_size*3)
        delta = np.concatenate((delta,-delta),axis=0)
        grad_sum = 0
        f_value_list = []
        for idelta in delta:
            img_plus = np.clip(img+sigma*idelta.reshape(img_size,img_size,3),0,1)
            f_value = sess.run(loss,feed_dict={images_pl:np.expand_dims(img_plus,0),y_label:true_class})
            f_value_list.append(f_value)
            grad_sum += f_value*idelta.reshape(img_size,img_size,3)
        grad_sum = grad_sum/(N*sigma)
        new_img = np.clip(np.clip(img-eta*grad_sum,old_img-epsilon,old_img+epsilon),0,1)
        new_loss, new_logits = sess.run([loss, logits],
                                        feed_dict={images_pl: np.expand_dims(new_img, 0), y_label: true_class})
        print("epoch:{} new:{}, {}, old:{}, {}".format(epoch, new_loss, np.argmax(new_logits),old_loss, np.argmax(_prob)))
        sys.stdout.flush()
        img = np.array(new_img)
        epoch -= 1
        if epoch % 200 ==0:
            temp_name = num_list+'_'+str(epoch)
            np.save(temp_name,new_img)
    np.save('new_img'+num_list,new_img)

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

