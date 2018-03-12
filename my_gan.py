import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt


def show_random_img():
    sample_image = mnist.train.next_batch(1)[0]
    print(sample_image.shape)

    sample_image = sample_image.reshape([28, 28])
    plt.imshow(sample_image, cmap='Greys')

    plt.show()


# todo modularize the code instead of these long functions
def new_convolutional_layer(input_x):
    print("not implmented")


# define the discrimnator. this is bascially a CNN from the TF mnist example
# our input image size will be 28x28
def descriminator(images, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    # first convolutional and pool layers.
    # finds 32 feature maps of size 5x5

    # tf.get_variable - Gets an existing variable with these parameters or create a new one.
    # args: name, shape, dtype, initializer
    # weights
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    # bias 1 dimensional 32 length. 0 const
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))

    # first convolutional layer.
    # convolution: bias. nonlinearity activation. avg pooling
    d_conv1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d_conv1 = d_conv1 + d_b1

    d_act1 = tf.nn.relu(d_conv1)
    # todo i forgot how the kernel size is computed. this is 2x2 window? but was is the 1,2,2, 1 mean?
    d1 = tf.nn.avg_pool(d_act1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First fully connected layer
    # todo why is this 7x7?
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])  # todo why do we have to reshape?
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    # /todo why do we have two FC layers?
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4

    return d4


# define the generator
# we can think of the generated as a reverse CNN
# we will take an input vector of d-dimensions of noise, and upsample it to become a 28x28 image
# ReLU and batch norm as used to stablize outputs

# z = random noise input. where do we define z_dim?
# #todo what is z_dim? where does 3136 come from?
def generator(z, batch_size, z_dim):
    # weight & bias
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.add(tf.matmul(z, g_w1), g_b1)
    # we are reshaping this random noise vector into a square?
    g1 = tf.reshape(g1, [-1, 56, 56, 1])  # todo what does the -1 mean?
    g1 = tf.layers.batch_normalization(g1, epsilon=1e-5)
    g1 = tf.nn.relu(g1)

    # generate 50 features. #todo why 50?
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56]) # todo why

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])


    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4








if __name__ == '__main__':
    print(tf.__version__)

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/")

    sample_image = mnist.train.next_batch(1)[0]
    print(sample_image.shape)

    sample_image = sample_image.reshape([28, 28])
    plt.imshow(sample_image, cmap='Greys')
    plt.show()



    z_dimensions = 100
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

    generated_image_output = generator(z_placeholder, 1, z_dimensions)
    z_batch = np.random.normal(0, 1, [1, z_dimensions])

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     generated_image = sess.run(generated_image_output,
    #                                feed_dict={z_placeholder: z_batch})
    #     generated_image = generated_image.reshape([28, 28])
    #     plt.imshow(generated_image, cmap='Greys')
    #     plt.show()



    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'pretrained-model/pretrained_gan.ckpt')
        z_batch = np.random.normal(0, 1, size=[10, z_dimensions])
        z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
        generated_images = generator(z_placeholder, 10, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        for i in range(10):
            plt.imshow(images[i].reshape([28, 28]), cmap='Greys')
            plt.show()