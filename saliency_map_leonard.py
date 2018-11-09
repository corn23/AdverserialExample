import argparse,sys,os

import tensorflow as tf
import numpy as np
import PIL

from utils import preprocess_img,classify,load_imagenet_label,calculate_img_region_importance,calculate_region_importance,load_pretrain_model,show_gradient_map


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="specify the pretrained model name", required=False, default='vgg16')
    parser.add_argument("--img_path", help='specify the image path', required=False, default='./picture/cat.jpg')
    parser.add_argument("--true_label", type=int, help='the groundtruth class for input image', required=False, default=281)
    parser.add_argument("--adv_label", type=int, help='specify the target adversarial label', default=100)
    parser.add_argument("--epoch", type=int, help="specify the training epochs", default=100)
    parser.add_argument("--eps", type=float, help="specify the degree of disturb", default=8/255)
    parser.add_argument("--lr", type=float, help="specify the learning rate", default=2)
    parser.add_argument("--lambda_up", type=float, help="specify the lambda for increasing gradient in the given region", default=10)
    parser.add_argument("--lambda_down", type=float, help="specify the lambda for lowering gradient in the given region", default=10)
    parser.add_argument("--lambda_label_loss", type=float, help="specify the lambda for the crossentropy loss of classifying wrong label", default=10)
    parser.add_argument("--label_num", type=int, help="specify the number of final outputs in pretrained model", default=1000)
    parser.add_argument("--is_cluster", help="indicate if work on cluster so that graphic function is not available", action="store_true", required=False, default=False)
    parser.add_argument("--summary_path", help="specify the tensorflow summary file path", default='./summary')
    parser.add_argument("--write_summary", help="write summary or not", action="store_true", default=False)
    parser.add_argument("--image_interval", type=int, help="write image into summary every *image_interval*",default=5)
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
    demo_epoch = args.epoch
    demo_eps = args.eps
    demo_lr = args.lr
    label_num = args.label_num
    lambda_up, lambda_down, lambda_label_loss = args.lambda_up, args.lambda_down,args.lambda_label_loss


    # load model
    sess, graph, img_size, images_v, logits = load_pretrain_model(model_name)
    probs = tf.nn.softmax(logits)
    print("sucessfully load model")

    if args.write_summary:
        unique_path_name = "up{}down{}epoch{}lr{}".format(args.lambda_up, args.lambda_down, args.epoch, args.lr)
        final_summary_path = os.path.join(args.summary_path, unique_path_name)
        if not os.path.exists(final_summary_path):
            os.makedirs(final_summary_path)
        summary_writer = tf.summary.FileWriter(final_summary_path, graph)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    step_init = tf.variables_initializer([global_step])

    y_hat = tf.placeholder(tf.int32, ())
    label_logits = tf.gather_nd(logits, [[0, y_hat]])

    img = PIL.Image.open(img_path)
    img = preprocess_img(img, img_size)
    batch_img = np.expand_dims(img, 0)
    imagenet_label = load_imagenet_label(img_label_path)

    # -------------------
    # Step 1: classify the image with original model
    p = sess.run(probs, feed_dict={images_v: batch_img})[0]
    predict_label = np.argmax(p)
    #classify(img, p, imagenet_label, correct_class=true_class, is_cluster=True)

    # -------------------
    # Step 2: Construct adversarial examples
    image_pl = tf.placeholder(tf.float32, (1, img_size, img_size, 3))
    assign_op = tf.assign(images_v, image_pl)
    learning_rate = tf.placeholder(tf.float32, ())
    var_eps = tf.placeholder(tf.float32,())
    labels = tf.one_hot(y_hat, label_num)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)[0]

    projected = tf.clip_by_value((tf.clip_by_value(images_v, image_pl - var_eps, image_pl + var_eps)), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(images_v, projected)

    # initialization step
    _ = sess.run([assign_op, step_init], feed_dict={image_pl: batch_img})

    # construct targeted attack
    # feed_dict_optim = {image_pl:batch_img,
    #                    y_hat:adversarial_label,
    #                    learning_rate:demo_lr}
    #
    # feed_dict_proj = {image_pl:batch_img,
    #                   var_eps:demo_eps}
    # optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[images_v])
    # model_train(sess=sess,
    #             optim_step=optim_step,
    #             project_step=project_step,
    #             loss=loss,
    #             feed_dict_optim=feed_dict_optim,
    #             feed_dict_project=feed_dict_proj,
    #             epoch=10)
    #
    # adv_img = np.squeeze(images_v.eval(),0)
    # adv_prob = sess.run(probs,feed_dict={images_v:np.expand_dims(adv_img,0)})
    # classify(adv_img, adv_prob[0],imagenet_label,correct_class=281,target_class=adversarial_label)
    #
    # # show the saliency map
    # feed_dict_gradient = {y_hat:true_class}
    # _ = show_gradient_map(graph=graph,
    #                   sess=sess,
    #                   y=label_logits,
    #                   x=images_v,
    #                   img=img,
    #                   is_integrated=False,
    #                   is_smooth=False,
    #                   feed_dict=feed_dict_gradient)
    #---------------
    # use gradient descent to control the saliency map

    # original gradient intensity
    map3D, map_grey = show_gradient_map(graph=graph,
                      sess=sess,
                      y=label_logits,
                      x=images_v,
                      img=img,
                      is_integrated=False,
                      is_smooth=True,
                      feed_dict={y_hat:true_class},
                      is_cluster=args.is_cluster)

    center_more, radius_more = (100, 110), 10
    center_less, radius_less = (100, 70), 10
    gradient_more = calculate_region_importance(map_grey, center_more, radius_more)
    gradient_less = calculate_region_importance(map_grey, center_less, radius_less)
    print("region 1 gradient intensity %.3f, region 2 gradient intensity %.3f" % (gradient_more, gradient_less))

    # construct new loss function
    grad_map = tf.gradients(label_logits, images_v)[0]
    to_down_gradient = calculate_img_region_importance(grad_map, center_more, radius_more)
    to_up_gradient = calculate_img_region_importance(grad_map, center_less, radius_less)
    grad_loss = -lambda_up*to_up_gradient + lambda_down*to_down_gradient
    final_loss = grad_loss+lambda_label_loss*loss
    if args.write_summary:
        up_gradient_summary = tf.summary.scalar("up_gradient", to_up_gradient)
        down_gradient_summary = tf.summary.scalar("down_gradient", to_down_gradient)
        loss_summary = tf.summary.scalar("loss", loss)
        train_summary_op = tf.summary.merge_all()
    change_grad_optim_step = tf.train.GradientDescentOptimizer(learning_rate=demo_lr).minimize(final_loss,var_list=[images_v],global_step=global_step)
    for i in range(demo_epoch):
        if args.write_summary:
            _,_loss,step,summary_str = sess.run([change_grad_optim_step, final_loss, global_step, train_summary_op],
                                                       feed_dict={image_pl:batch_img,y_hat:true_class,learning_rate:demo_lr})
            summary_writer.add_summary(summary_str,global_step=step)
        else:
            _,_loss,step = sess.run([change_grad_optim_step, final_loss, global_step],
                                                   feed_dict={image_pl:batch_img,y_hat:true_class,learning_rate:demo_lr})

        sess.run([project_step], feed_dict={image_pl:batch_img,var_eps:demo_eps})
        print("%d loss = %g" % (i,_loss))
        if i % args.image_interval == 0:
            adv_img = np.squeeze(images_v.eval(), 0)
            # check the prediction result
            p_adv = sess.run(probs,feed_dict={images_v: batch_img})[0]
            predict_label_adv = np.argmax(p_adv)
            #classify(adv_img, p_adv, imagenet_label, correct_class=true_class,is_cluster=args.is_cluster)

            # check the gradient map
            map3D_adv, map_grey_adv = show_gradient_map(graph=graph,
                              sess=sess,
                              y=label_logits,
                              x=images_v,
                              img=adv_img,
                              is_integrated=False,
                              is_smooth=False,
                              feed_dict={y_hat:true_class},
                              is_cluster=args.is_cluster)

            adv_gradient_more = calculate_region_importance(map_grey_adv, center_more, radius_more)
            adv_gradient_less = calculate_region_importance(map_grey_adv, center_less, radius_less)

            if args.write_summary:
                map_grey_adv = tf.expand_dims(tf.expand_dims(map_grey_adv, 0), 3)
                adv_map_sum = tf.summary.image('adv_map'+str(i), tf.convert_to_tensor(map_grey_adv))
                adv_str = sess.run(adv_map_sum)
                summary_writer.add_summary(adv_str)
            print(
                "Adversarial Case: predict label: %d, big region  gradient intensity: %.3f, small region gradient intensity: %.3f" % (
                predict_label_adv, adv_gradient_more, adv_gradient_less))
            print(
                "Normal Case: predict label: %d, big region gradient intensity: %.3f, small region gradient intensity: %.3f" % (
                predict_label, gradient_more, gradient_less))

    # write original map
    map_grey = tf.expand_dims(tf.expand_dims(map_grey, 0), 3)
    orig_map_sum = tf.summary.image('orig_map', tf.convert_to_tensor(map_grey))
    orig_str = sess.run(orig_map_sum)
    summary_writer.add_summary(orig_str)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
    # args = parse_arguments(['--model_name','resnet_v1_152'])