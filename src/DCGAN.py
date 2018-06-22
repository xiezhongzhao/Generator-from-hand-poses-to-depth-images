import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import skimage.io as io
import cv2
import random
import os

tf.reset_default_graph()
batch_size = 64
n_noise = 42
epoches = 50
sample_interval = 400

image_gen = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/DCGAN/data/NYU_Image.npy')
image_gen = image_gen.reshape([-1, 128, 128])

noise_gen = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/DCGAN/data/NYU_Label.npy')
noise_gen = noise_gen.reshape([-1,42])

print("original image", image_gen.shape)
print("train_label", noise_gen.shape)


'''
load depth images and joints of hand pose estimation
'''
X_in_gen = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='X_in_gen')
noise = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='noise')

# X_in_dis = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='X_in_dis')
# joint_dis = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='joint_dis')


'''
make leaky relu activation function
'''
def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

#calculate training time
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


'''
transform from hand pose estimation(21 joints) to depth images
'''
noise = tf.reshape(noise, [-1, 42])
fc1 = tf.layers.dense(noise, 1024)
fc1 = tf.layers.dropout(fc1,  0.3)
fc2 = tf.layers.dense(fc1, 1024)
fc2 = tf.layers.dropout(fc2,  0.3)
fc3 = tf.layers.dense(fc2, 2048)
fc3 = tf.layers.dropout(fc3,  0.3)
input_map = tf.layers.dense(fc3, units=8 * 8 * 32, activation=lrelu)

# noise(-1,63)->(-1,8*8*32)

x = tf.layers.batch_normalization(tf.layers.dense(input_map, units=8 * 8 * 32, activation=lrelu))
x_in = tf.reshape(x, shape=[-1, 8, 8, 32])


# image(-1,8,8,32)->(-1,16,16,32)
gen1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(x_in,
                                                                filters=32, kernel_size=5, strides=2,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))

# image(-1,16,16,32)->(-1,32,32,32)
gen2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen1,
                                                                filters=32, kernel_size=5, strides=2,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))

# image(-1,16,16,32)->(-1,32,32,32)
gen3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen2,
                                                                filters=32, kernel_size=1, strides=1,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))

# image(-1,32,32,32)->(1, 64, 64, 32)
gen4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen3,
                                                                filters=32, kernel_size=5, strides=2,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))

# image(1, 64, 64, 32)->(1, 128, 128, 32)
gen5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen4,
                                                                filters=32, kernel_size=5, strides=2,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))

# image(1, 128, 128, 32)->(-1,128,128,1)
gen6 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen5,
                                                                filters=1, kernel_size=1, strides=1,
                                                                use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(
                                                                    stddev=0.01),
                                                                padding='same', activation=lrelu))
print("gen4.shape",gen6.shape)


g_out = tf.reshape(gen6, [-1, 128, 128])

img_gan_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(g_out, X_in_gen), 1))

loss_g = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(img_gan_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/DCGAN/results/'

steps = []
g_loss = []
d_loss = []

start_time_sum = time.time()

for epoch in range(0, epoches):

    idx = np.random.randint(0, image_gen.shape[0], image_gen.shape[0])
    labels = noise_gen[idx]
    image = image_gen[idx]
    # image_depth.shape[0] // batch_size
    for i in range(0, image_gen.shape[0] // batch_size):

        batch = image[i * batch_size:(i + 1) * batch_size, ]
        n = labels[i * batch_size:(i + 1) * batch_size, ]

        # loss_dis = sess.run(img_dis_loss, feed_dict={X_in_dis: batch, joint_dis: n})
        # sess.run(loss_d, feed_dict={X_in_dis: batch, joint_dis: n})
        # print("epoch: %d, step: %d, loss_d: %f" % (epoch, i, loss_dis))

        start_time = time.time()

        loss_gen = sess.run(img_gan_loss, feed_dict={noise: n, X_in_gen: batch})
        sess.run(loss_g, feed_dict={noise: n, X_in_gen: batch})

        duration = time.time() - start_time

        print("epoch: %d, step: %d, loss_g: %f, Duration:%f sec" % (epoch, i, loss_gen, duration))

        steps.append(epoch * image_gen.shape[0] // batch_size + i)
        g_loss.append(loss_gen)
        # d_loss.append(loss_dis)

        # print("epoch: %d, step: %d, loss_g: %f, loss_d: %f"%(epoch, i, loss_gen, loss_dis))

        if i % sample_interval == 0:

            gen_imgs = sess.run(g_out, feed_dict={noise: n})

            r, c = 2, 2
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(path + 'gen_images/', '%d_%d.png' % (epoch, len(steps))))
            plt.close()

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(batch[cnt, :], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(path + 'raw_images/', '%d_%d.png' % (epoch, len(steps))))
            plt.close()

duration_time_sum = time.time() - start_time_sum

print("The total training time: ",elapsed(duration_time_sum))

'''
#plot the loss of generator
'''
fig1 = plt.figure(figsize = (8,5) )

plt.plot(steps, g_loss, label='the loss of generator')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('The loss of generator')
plt.legend()
plt.legend(loc = 'upper left')
plt.savefig(os.path.join(path, 'the loss of generator'))
plt.close()






