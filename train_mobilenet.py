#coding: utf-8
import json
import glob, cv2
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import mobilenet_v1 as mobilenet
from tensorflow.python.framework import graph_util

LEARNING_RATE = 0.01
STEPS = 10
BATCH_SIZE = 8
N_CLASSES = 8
IMAGE_SIZE = 224
IMAGE_PATH = '/home/xll/trash/captured_img/'
TRAIN_FILE = 'model/ckpt/'
RETRAIN_MODEL = 'model_tt/trash_mobilenet.pb'

#class_dict = {'cardboard':0,'metal':1,'glass':2,'plastic':3,'paper':4,'trash':5}
def train_test(datapath):

    order = range(N_CLASSES)
    random.shuffle(order)

    train_data = []
    valid_data = []
    train_labels = []
    valid_labels = []
    label_keys = {}

    i = 0
    for subdir in os.listdir(datapath):
        if os.path.isdir(datapath+subdir):
            print(datapath+subdir)
            label_index = order[i]
            i += 1
            label_name = subdir.lower()  # 通过目录名获取类别的名称
            label_keys.update({label_index: label_name})

            for image in os.listdir(datapath+subdir):
                image_path = datapath+subdir+'/'+image
                chance = np.random.randint(100)
                image_class = np.zeros(N_CLASSES, dtype=float)
                #print(image_class,class_dict[subdir])
                image_class[label_index] = 1
                #print(image_path,image_class)
                if chance <= 90:
                    #train_data.append(image_path)
                    train_data.append(cv2.resize(cv2.imread(image_path), (IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC))
                    train_labels.append(image_class)
                else:
                    #valid_data.append(image_path)
                    valid_data.append(cv2.resize(cv2.imread(image_path), (IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC))
                    valid_labels.append(image_class)
    print(label_keys)
    print(len(train_data), len(valid_data))
    return train_data, train_labels, valid_data, valid_labels, label_keys


def minibatches(inputs, targets, batch_size, shuffle):
    print(len(inputs), len(targets))
    targets = np.array(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        train_images = []
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        for ex in excerpt:
            image = inputs[ex]
            #train_images.append(cv2.resize(cv2.imread(image_path), (IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC))
            train_images.append(image)
        yield train_images, list(targets[excerpt])


def main():
    # 加载预处理好的数据。
    train_data, train_labels, valid_data, valid_labels, label_keys = train_test(IMAGE_PATH)
    print("%d training examples, %d validation examples." % (len(train_data), len(valid_data)))

    # 定义inception-v1的输入，images为输入图片，labels为每一张图片对应的标签。
    images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None, N_CLASSES], name='labels')

    with slim.arg_scope(mobilenet.mobilenet_v1_arg_scope()):
        logits, _ = mobilenet.mobilenet_v1(images, num_classes=N_CLASSES, is_training=True)
    predictions = tf.nn.softmax(logits, name='output/prob')

    # 定义损失函数和训练过程。
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    loss_summary = tf.summary.scalar('loss',cost)

    train_step = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    tf.add_to_collection("predict", predictions)

    with tf.Session() as sess:
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        # 初始化没有加载进来的变量。
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(STEPS):
            print('step:', i)
            for batch_xs, batch_ys in minibatches(train_data, train_labels, BATCH_SIZE, shuffle=True):
                # 训练数据
                _, summary_str = sess.run([train_step, merged_summary_op], feed_dict={images: batch_xs, labels: batch_ys})
                summary_writer.add_summary(summary_str, i)


            # 在验证集上测试正确率
            if (i+1) % 5 == 0 or i + 1 == STEPS:
                valid_accuracy = sess.run(evaluation_step, feed_dict={images: valid_data, labels: valid_labels})
                print('Step %d: Validation accuracy = %.1f%%' % (i, valid_accuracy * 100.0))

            # 保存ckpt
            #if (i+1) % 20 == 0 or i + 1 == STEPS:
            #    path = saver.save(sess, TRAIN_FILE, global_step=i)
            #    print('Saved model to {}\n'.format(path))

        # 保存标签
        output_labels = os.path.join('./', 'labels.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            k = {}
            for i in range(N_CLASSES):
                k[str(i)] = label_keys[i]
            json.dump(k, f)

        saver.save(sess, 'model/mobilenet.ckpt')
        tf.train.write_graph(sess.graph_def, 'model/','trash_mobilenet.pb', as_text=False)
        # 保存pb
        #converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [images], [predictions])
        #tflite_model = converter.convert()
        #open("convert_mobilenet.tflite", "wb").write(tflite_model)
        #output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output/prob'])
        #with tf.gfile.FastGFile(RETRAIN_MODEL, mode='wb') as f:
        #    f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    main()
