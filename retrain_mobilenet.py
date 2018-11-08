#! /usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

# 数据参数
MODEL_DIR = 'tmp/model/'  # inception-v3模型的文件夹
CACHE_DIR = 'tmp/bottleneck'  # 图像的特征向量保存地址
CHECKPOINT_NAME = 'tmp/model/checkpoint/retrain'
out_dir = 'tmp/'
INPUT_DATA = 'input_data/'  # 图片数据文件夹
frozen_graph = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'MobilenetV1/Logits/AvgPool_1a/AvgPool:0'  # 模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'input:0'  # 图像输入张量对应的名称
final_tensor_name = 'final_result'

FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

# 神经网络的训练参数
LEARNING_RATE = 0.03
STEPS = 50
BATCH = 64
NUM_CLASSES = 9
CHECKPOINT_EVERY = 10
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

def create_all_bottlenecks(sess, resize_data_tensor, bottleneck_tensor, jpeg_data_tensor, decoded_image_tensor):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录
    order = range(NUM_CLASSES)
    random.shuffle(order)

    is_root_dir = True  # 第一个目录为当前目录，需要忽
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []
    label_keys = {}

    # 分别对每个子目录进行操作
    i = 0
    for sub_dir in sub_dirs:
        print(sub_dir)
        if is_root_dir:
            is_root_dir = False
            continue
        label_index = order[i]
        i += 1

        # 获取当前目录下的所有有效图片
        extensions = {'jpg', 'jpeg', 'JPG', 'JPEG'}
        file_list = []  # 存储所有图像
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名字
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 将当前类别的图片随机分为训练数据集、测试数据集、验证数据集
        label_name = dir_name.lower()  # 通过目录名获取类别的名称
        label_keys.update({label_index: label_name})
        print(label_keys)

        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 获取该图片的名称
            bottleneck_path = get_or_create_bottleneck(sess, label_name, base_name, resize_data_tensor, bottleneck_tensor, jpeg_data_tensor, decoded_image_tensor)
            chance = np.random.randint(100)  # 随机产生100个数代表百分比
            if chance < VALIDATION_PERCENTAGE:
                valid_images.append(bottleneck_path)
                valid_labels.append(label_index)
            elif chance < (VALIDATION_PERCENTAGE + TEST_PERCENTAGE):
                test_images.append(bottleneck_path)
                test_labels.append(label_index)
            else:
                train_images.append(bottleneck_path)
                train_labels.append(label_index)

    image_list = {'training': (train_images, train_labels), 'validation': (valid_images, valid_labels),
                  'testing': (test_images, test_labels)}

    # 返回整理好的所有数据
    del train_images, train_labels, valid_images, valid_labels, test_images, test_labels
    return image_list, label_keys

# 获取一张图片经过inception-v3模型处理后的特征向量
def get_or_create_bottleneck(sess, label_name, base_name, resize_data_tensor, bottleneck_tensor, jpeg_data_tensor, decoded_image_tensor):
    # 获取一张图片对应的特征向量文件的路径
    sub_dir_path = os.path.join(CACHE_DIR, label_name)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = os.path.join(sub_dir_path, (base_name + '.txt'))

    # 如果该特征向量文件不存在，则通过inception-v3模型计算并保存
    if not os.path.exists(bottleneck_path):
        image_path = os.path.join(INPUT_DATA, label_name, base_name)  # 获取图片原始路径
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        resize_data = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = sess.run(bottleneck_tensor, {resize_data_tensor: resize_data})
        bottleneck_values = np.squeeze(bottleneck_values)


        # bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
        # bottleneck_values = np.squeeze(bottleneck_values) # 通过inception-v3计算特征向量

        # 将特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    return bottleneck_path

def get_bottleneck(bottleneck_path):
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # 返回得到的特征向量
    return bottleneck_values

# 随机获取一个batch图片作为训练数据
def get_random_cached_bottlenecks(image_list, category):
    images = image_list[category][0]
    labels = image_list[category][1]
    #print(len(images),len(labels))
    order = range(len(images))
    random.shuffle(order)
    bottlenecks = []
    ground_truths = []
    #for j in range(BATCH):
    #    bottleneck = get_bottleneck(images[order[j]])
    #    ground_truth = labels[order[j]]
    #    bottlenecks.append(bottleneck)
    #    ground_truths.append(ground_truth)
    #return bottlenecks, ground_truths
    for i in range(0, len(images)-BATCH, BATCH):
        for j in range(i, i+BATCH):
            bottleneck = get_bottleneck(images[order[j]])
         #    ground_truth = np.zeros(NUM_CLASSES, dtype=np.float32)
            ground_truth = labels[order[j]]
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        yield bottlenecks, ground_truths

# 获取全部的测试数据
def get_test_bottlenecks(image_list):
    images = image_list['testing'][0]
    labels = image_list['testing'][1]
    bottlenecks = []
    ground_truths = []
    # 枚举所有的类别和每个类别中的测试图片
    for ind, image in enumerate(images):
        bottleneck = get_bottleneck(image)
        # ground_truth = np.zeros(NUM_CLASSES, dtype=np.float32)
        # ground_truth[labels[ind]] = 1.0
        ground_truth = labels[ind]
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_final_retrain_ops(NUM_CLASSES, final_tensor_name, bottleneck_tensor, is_training=None):
    # batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    # assert batch_size is None, 'We want to work with arbitrary batch size.'
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    print('bottleneck_tensor_size', bottleneck_tensor_size)
    #assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        # creat input tensor for final layer
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        # create bias tensor
        ground_truth_input = tf.placeholder(
            tf.int64, [None], name='GroundTruthInput')

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, NUM_CLASSES], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor

def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction

def build_eval_session(module_spec):
    eval_graph, bottleneck_tensor, resize_data_tensor = (
        create_module_graph(frozen_graph))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        (_, _, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(
            NUM_CLASSES, final_tensor_name, bottleneck_tensor, is_training=False)
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    return eval_sess, resize_data_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction


def save_graph_to_file(graph_file_name, frozen_graph):
  sess, _, _, _, _, _ = build_eval_session(frozen_graph)
  graph = sess.graph

  output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [final_tensor_name])

  with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())

def create_module_graph(frozen_graph):
    with tf.Graph().as_default() as pre_graph:
        # load pre-trained model
        with gfile.FastGFile(frozen_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # get input tensor and bottleneck tensor
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
            # reshape bottleneck tensor
            bottleneck_tensor = tf.reshape(bottleneck_tensor, [-1, 1024])
    return pre_graph, bottleneck_tensor, jpeg_data_tensor

def add_jpeg_decoding():
    input_height = 224
    input_width = 224
    input_depth = 3
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
    return jpeg_data, resized_image


def main(_):
    # get graph bottleneck_tensor and data tensor from pretrained graph
    pre_graph, bottleneck_tensor, resize_data_tensor = create_module_graph(frozen_graph)

    with pre_graph.as_default():
        train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor = add_final_retrain_ops(
            NUM_CLASSES, final_tensor_name, bottleneck_tensor, is_training=True)

    # 训练过程
    with tf.Session(graph=pre_graph) as sess:
        # initialize
        init = tf.global_variables_initializer()
        sess.run(init)

        # get jpg input
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()

        print('start prepare image data........................')
        image_list, label_keys = create_all_bottlenecks(sess, resize_data_tensor, bottleneck_tensor, jpeg_data_tensor, decoded_image_tensor)
        print(len(image_list['training'][0]))

        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # 损失值和正确率的摘要
        loss_summary = tf.summary.scalar('loss', cross_entropy)
        acc_summary = tf.summary.scalar('accuracy', evaluation_step)

        # 训练摘要
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        print('start training..................................')
        train_saver = tf.train.Saver()
        for i in range(STEPS):
            print('step:', i)
            for train_bottlenecks, train_ground_truth in get_random_cached_bottlenecks(image_list, 'training'):
            #    print(len(train_bottlenecks))
                # 每次获取一个batch的训练数据
            # train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(image_list, 'training')
                _, train_summaries, train_accuracy, cross_entropy_value = sess.run([train_step, train_summary_op, evaluation_step, cross_entropy],
                                feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                # 保存每步的摘要
                train_summary_writer.add_summary(train_summaries, i)
            print('Step %d : Train accuracy is %f%%' % (i, train_accuracy * 100))
            print('Step %d : Cross entropy is %f' % (i, cross_entropy_value))

            # 测试正确率
            if i > 0 and (i % 5 == 0 or i + 1 == STEPS):
                valid_acc = 0
                for valid_bottlenecks, valid_ground_truth in get_random_cached_bottlenecks(image_list, 'validation'):
                    valid_accuracy, valid_summary = sess.run([evaluation_step, train_summary_op], feed_dict={
                        bottleneck_input: valid_bottlenecks, ground_truth_input: valid_ground_truth})
                    valid_acc += valid_accuracy
                valid_acc = valid_acc / (len(image_list['validation'][0])/BATCH)
                print('Step %d : Validation accuracy on validation is %f%%' % (i, valid_acc*100))

            # 每隔checkpoint_every保存一次模型和测试摘要
            if (i+1) % CHECKPOINT_EVERY == 0 or i + 1 == STEPS:
                path = train_saver.save(sess, CHECKPOINT_NAME)
                print('Saved model checkpoint to {}\n'.format(path))
                RETRAIN_MODEL = MODEL_DIR + 'trash_mobilenet_' + str(i+1) + '.pb'
                save_graph_to_file(RETRAIN_MODEL, frozen_graph)

            # 最后在测试集上测试正确率
            if i % CHECKPOINT_EVERY == 0 or i + 1 == STEPS:
                test_bottlenecks, test_ground_truth = get_test_bottlenecks(image_list)
                test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                     ground_truth_input: test_ground_truth})
                print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        #path = train_saver.save(sess, CHECKPOINT_NAME)
        #print('Saved model checkpoint to {}\n'.format(path))
        #save_graph_to_file(RETRAIN_MODEL, module_spec)
        # output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output/prob'])
        # with tf.gfile.FastGFile(RETRAIN_MODEL, mode='wb') as f:
        #     f.write(output_graph_def.SerializeToString())

        # 保存标签
        print(label_keys)
        output_labels = os.path.join(out_dir, 'labels.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            for i in range(NUM_CLASSES):
                f.write(label_keys[i]+'\n')

            # k = {}
            # for i in range(NUM_CLASSES):
            #     k[str(i)] = label_keys[i]
            # json.dump(k, f)


if __name__ == '__main__':
    tf.app.run()
