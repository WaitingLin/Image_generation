import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sys
import random
import os
from skimage import io
from skimage import transform
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parmeter
batch_size = 128
iteration = 10000
# training_data_num = 33430
training_data_num = 33430
z_dimensions = 100 # Generator input dim
pretrain_iter = 10 # Discrimanator pre-train iteration

# Global Variable
img_index = 0
Data = []

'''
maxi = 0
mini = 0
img = io.imread('./data_set/faces/0.jpg')
print(img[0][0])
img = transform.resize(img, (64,64,3))
print(img[0][0])
print(img.shape)
io.imsave('./test.jpg', img)
for i in img:
    for j in i:
        for k in j:
            if k>maxi: 
                maxi = k
            if k<mini: 
                mini = k
print(img[0][0])
print(maxi)
print(mini)
#img = (img /255.)*2. - 1.
#print(img[0][0])
#img = (img+1.) / 2. * 255.
#print(img[0][0])
#print(img.dtype)
#images = img.astype(np.uint8)
#io.imsave('./fuck.jpg', img)
'''

def load_all_img():
    global Data
    data_path = './data_set/faces/'
    print('Load training data ...')
    for i in tqdm(range(training_data_num+1)):
        img = io.imread(data_path+str(i)+'.jpg')
        img = transform.resize(img, (64,64,3))
        # img = (img / 255) * 2 - 1 # normalize to -1~1 
        Data.append(img)
def n_image_next_batch(batch_size):
    Img = []
    for i in range(batch_size):
        img_index = random.randint(0, training_data_num)  
        img = Data[img_index]
        Img.append(img)
    return np.array(Img)

# for debug
def image_next_batch(batch_size):
    data_path = './data_set/faces/'
    Img = []
    for i in range(batch_size):
        img_index = random.randint(0, training_data_num)
        img = io.imread(data_path+str(i)+'.jpg')
        img = transform.resize(img, (64, 64, 3))
        #img = (img / 255) * 2 - 1
        Img.append(img)
    return Img

def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        #noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=.02, dtype=tf.float32)
        #images = images + noise

        d_w1 = tf.get_variable('d_w1', [4, 4, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.leaky_relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w2 = tf.get_variable('d_w2', [4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.leaky_relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w3 = tf.get_variable('d_w3', [4, 4, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
        d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding='SAME')
        d3 = d3 + d_b3
        d3 = tf.nn.leaky_relu(d3)
        d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w4 = tf.get_variable('d_w4', [4, 4, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [256], initializer=tf.constant_initializer(0))
        d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 1, 1, 1], padding='SAME')
        d4 = d4 + d_b4
        d4 = tf.nn.leaky_relu(d4)
        d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
        d_w5 = tf.get_variable('d_w5', [4 * 4 * 256, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
        d5 = tf.reshape(d4, [-1, 4 * 4 * 256])
        d5 = tf.matmul(d5, d_w5) + d_b5
        #d5 = tf.sigmoid(d5)
        #d5 = tf.tanh(d5)
        return d5

def generator(z, batch_size, z_dim):

    g_w1 = tf.get_variable('g_w1', [z_dim, 8*8*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [8*8*1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.leaky_relu(g1)
    g1 = tf.reshape(g1, [-1, 8, 8, 1024])
    g1 = tf.image.resize_images(g1, (16,16), method=0) # Upsampling

    g_w2 = tf.get_variable('g_w2', [4, 4, 1024, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [512], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 1, 1, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.leaky_relu(g2)
    g2 = tf.image.resize_images(g2, (32,32), method=0) # Upsampling

    g_w3 = tf.get_variable('g_w3', [4, 4, 512, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [256], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 1, 1, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.leaky_relu(g3)
    g3 = tf.image.resize_images(g3, (64,64), method=0) # Upsampling

    g_w4 = tf.get_variable('g_w4', [4, 4, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 1, 1, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='bn4')
    g4 = tf.nn.leaky_relu(g4)

    g_w5 = tf.get_variable('g_w5', [4, 4, 128, 3], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b5 = tf.get_variable('g_b5', [3], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g5 = tf.nn.conv2d(g4, g_w5, strides=[1, 1, 1, 1], padding='SAME')
    g5 = g5 + g_b5
    g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='bn5')
    g5 = tf.sigmoid(g5)
    #g5 = tf.nn.tanh(g5)
    return g5

''' Load all image '''
#load_all_img() 

''' See the fake image we make '''
print('test img..')
# Define the plceholder and the graph
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
# For generator, one image for a batch
generated_image_output = generator(z_placeholder, 10, z_dimensions)
z_batch = np.random.normal(0, 1, [10, z_dimensions])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generated_image = sess.run(generated_image_output,
                                feed_dict={z_placeholder: z_batch})
    generated_image = generated_image[0].reshape([64, 64, 3])
    #generated_image = (generated_image + 1.) * 255. / 2.
    #print(generated_image[0][0])
    #generated_image = generated_image.astype(np.uint8)
    #generated_image = np.clip(generated_image, 0, 1)
    io.imsave("./img/test_img.jpg", generated_image)
    

''' Training GAN '''

tf.reset_default_graph()

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
x_placeholder = tf.placeholder(tf.float32, shape = [None,64,64,3], name='x_placeholder') 

Gz = generator(z_placeholder, batch_size, z_dimensions) 
Dx = discriminator(x_placeholder) # Real 
Dg = discriminator(Gz, reuse_variables=True) # Fake


# Loss function for generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))
# Two Loss Functions for discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
'''
g_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(Dg,1e-8,1.0)))
d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(Dx,1e-8,1.0)) + \
                            tf.log(tf.clip_by_value(1. - Dg,1e-8,1.0)))
'''

# Get the varaibles for different network
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])


# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0002).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0002).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0002).minimize(g_loss, var_list=g_vars)

'''
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0002)

g_trainer = optimizer_gen.minimize(g_loss, var_list=g_vars)
d_trainer = optimizer_disc.minimize(d_loss, var_list=d_vars)
'''
# ''' For setting TensorBoard '''

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

# tf.summary.scalar('Generator_loss', g_loss)
# tf.summary.scalar('Discriminator_loss_real', d_loss_real)
# tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

# images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
# tf.summary.image('Generated_images', images_for_tensorboard, 5)
# merged = tf.summary.merge_all()
# logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# writer = tf.summary.FileWriter(logdir, sess.graph)


''' Start Training Session '''

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# TimeLine
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Pre-train discriminator
print('Pre-train discriminator ...')
for i in range(pretrain_iter):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = image_next_batch(batch_size)
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})
    #disc_feed_dict = {x_placeholder: real_image_batch, z_placeholder: z_batch}
    #_, dl, dLoss = sess.run([d_trainer, d_loss], feed_dict = disc_feed_dict)

    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
print('Start training...')
for i in range(iteration):
    # Train discriminator 5 times
    for j in range(1):
        real_image_batch = image_next_batch(batch_size)
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                               {x_placeholder: real_image_batch, z_placeholder: z_batch}, options=options, run_metadata=run_metadata)
        #disc_feed_dict = {x_placeholder: real_image_batch, z_placeholder: z_batch}
        #_, dl, dLoss = sess.run([d_trainer, d_loss], feed_dict = disc_feed_dict)
    print("dLossReal:", dLossReal, "dLossFake:", dLossFake)
    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _, gLoss = sess.run([g_trainer, g_loss], feed_dict={z_placeholder: z_batch}, options=options, run_metadata=run_metadata)
    print("gLoss:", gLoss)
    # if i % 10 == 0:
    #     # Update TensorBoard with summary statistics
    #     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    #     summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
    #     writer.add_summary(summary, i)

    if i % 1000 == 0:
    # Save the model every 1000 iteration
        save_path = saver.save(sess, "./model/model{}.ckpt".format(i))
        print("Model saved in file: %s" % save_path)
    
    if i % 10 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[128, z_dimensions])
        generated_images = generator(z_placeholder, 128, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        #images = (images + 1.) / 2.* 255.
        #images = images.astype(np.uint8)
        #images = np.clip(images, 0, 1)
        io.imsave("./img/image{}.jpg".format(i), images[0].reshape([64,64,3]))
        # Show discriminator's estimate
        im = images[0].reshape([1, 64, 64, 3])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
        print("Estimate:", estimate)
