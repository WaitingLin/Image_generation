import tensorflow as tf
import numpy as np
import os, datetime, random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

flags = tf.app.flags
flags.DEFINE_boolean('debug', False, 'debug mode')
FLAGS = flags.FLAGS

# GPU Setting 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Parmeter
Parm = {
    'training_path': './data_set/faces/',
    'img_dir': './img/',
    'model_dir': './model/',
    'batch_size' : 64,
    'iteration' : 300,
    'training_data_num' : 430,
    'g_input_dim' : 100, # Generator input dim
    'pretrain_iter' : 300, # Discrimanator pre-train iteration
    'd_train_times' : 4, # Discrimanator training times per iteration
}
Parm['noise_index'] = Parm['iteration'] * Parm['d_train_times']

Data = []

def draw_image(row_data, id):
    row_data = np.array(((row_data+1.)/2.)*255., dtype=int)
    row_data = row_data.reshape((-1, 3))
    row_data = list(map(tuple, row_data))
    testImg = Image.new( "RGB", (64, 64), (0,0,0))
    testImg.putdata(row_data)
    testImg.save(Parm['img_dir']+ str(id) +'.jpg')

def load_all_img():
    global Data
    data_path = Parm['training_path']
    print('Load training data ...')
    for i in tqdm(range(Parm['training_data_num']+1)):
        img = Image.open(data_path+str(i)+'.jpg')
        img = img.resize((64, 64))
        img = np.array(img)
        img = (img / 255.) * 2. - 1. # normalize to -1~1 
        Data.append(img)

def image_next_batch():
    Img = []
    if FLAGS.debug == True:
        data_path = Parm['training_path']
        for i in range(Parm['batch_size']):
            img_index = random.randint(0, Parm['training_data_num'])
            img = Image.open(data_path+str(i)+'.jpg')
            img = img.resize((64, 64))
            img = np.array(img)
            img = (img / 255.) * 2. - 1. # normailize to -1~1
            Img.append(img)
    else:
        for i in range(Parm['batch_size']):
            img_index = random.randint(0, Parm['training_data_num'])  
            img = Data[img_index]
            Img.append(img)
    return np.array(Img)

D_var = {
    'd_w1' : tf.get_variable('d_w1', [4, 4, 3, 32], initializer=tf.random_normal_initializer(stddev=0.02)),
    'd_b1' : tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0)),
    'd_w2' : tf.get_variable('d_w2', [4, 4, 32, 64], initializer=tf.random_normal_initializer(stddev=0.02)),
    'd_b2' : tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0)),
    'd_w3' : tf.get_variable('d_w3', [4, 4, 64, 128], initializer=tf.random_normal_initializer(stddev=0.02)),
    'd_b3' : tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0)),
    'd_w4' : tf.get_variable('d_w4', [4, 4, 128, 256], initializer=tf.random_normal_initializer(stddev=0.02)),
    'd_b4' : tf.get_variable('d_b4', [256], initializer=tf.constant_initializer(0)),
    'd_w5' : tf.get_variable('d_w5', [4 * 4 * 256, 1], initializer=tf.random_normal_initializer(stddev=0.02)),
    'd_b5' : tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
}

G_var = {
    'g_w1' : tf.get_variable('g_w1', [Parm['g_input_dim'] , 8*8*512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'g_b1' : tf.get_variable('g_b1', [8*8*512], initializer=tf.constant_initializer(0)),
    'g_w2' : tf.get_variable('g_w2', [4, 4, 512, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'g_b2' : tf.get_variable('g_b2', [256], initializer=tf.constant_initializer(0)),
    'g_w3' : tf.get_variable('g_w3', [4, 4, 256, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'g_b3' : tf.get_variable('g_b3', [256], initializer=tf.constant_initializer(0)),
    'g_w4' : tf.get_variable('g_w4', [4, 4, 256, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'g_b4' : tf.get_variable('g_b4', [128], initializer=tf.constant_initializer(0)),
    'g_w5' : tf.get_variable('g_w5', [4, 4, 128, 3], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'g_b5' : tf.get_variable('g_b5', [3], initializer=tf.constant_initializer(0))
}

def discriminator(images):
    
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=.02, dtype=tf.float32)
    images = images + noise * Parm['noise_index'] / Parm['iteration']
    Parm['noise_index'] -= 1

    d1 = tf.nn.conv2d(input=images, filter=D_var['d_w1'], strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + D_var['d_b1']
    d1 = tf.nn.leaky_relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    d2 = tf.nn.conv2d(input=d1, filter=D_var['d_w2'], strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + D_var['d_b2']
    d2 = tf.nn.leaky_relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    d3 = tf.nn.conv2d(input=d2, filter=D_var['d_w3'], strides=[1, 1, 1, 1], padding='SAME')
    d3 = d3 + D_var['d_b3']
    d3 = tf.nn.leaky_relu(d3)
    d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    d4 = tf.nn.conv2d(input=d3, filter=D_var['d_w4'], strides=[1, 1, 1, 1], padding='SAME')
    d4 = d4 + D_var['d_b4']
    d4 = tf.nn.leaky_relu(d4)
    d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
    d5 = tf.reshape(d4, [-1, 4 * 4 * 256])
    d5 = tf.matmul(d5, D_var['d_w5']) + D_var['d_b5']
    d5 = tf.sigmoid(d5)
    
    return d5

def generator(z):

    g1 = tf.matmul(z, G_var['g_w1']) + G_var['g_b1']
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5)
    g1 = tf.nn.leaky_relu(g1)
    g1 = tf.reshape(g1, [-1, 8, 8, 512])
    g1 = tf.image.resize_images(g1, (16,16), method=0) # Upsampling

    g2 = tf.nn.conv2d(g1, G_var['g_w2'], strides=[1, 1, 1, 1], padding='SAME')
    g2 = g2 + G_var['g_b2']
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
    g2 = tf.nn.leaky_relu(g2)
    g2 = tf.image.resize_images(g2, (32,32), method=0) # Upsampling

    g3 = tf.nn.conv2d(g2, G_var['g_w3'], strides=[1, 1, 1, 1], padding='SAME')
    g3 = g3 + G_var['g_b3']
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5)
    g3 = tf.nn.leaky_relu(g3)
    g3 = tf.image.resize_images(g3, (64,64), method=0) # Upsampling

    g4 = tf.nn.conv2d(g3, G_var['g_w4'], strides=[1, 1, 1, 1], padding='SAME')
    g4 = g4 + G_var['g_b4']
    g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5)
    g4 = tf.nn.leaky_relu(g4)

    g5 = tf.nn.conv2d(g4, G_var['g_w5'], strides=[1, 1, 1, 1], padding='SAME')
    g5 = g5 + G_var['g_b5']
    g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5)
    g5 = tf.nn.tanh(g5)
    
    return g5


''' Load all image '''
if FLAGS.debug == False:
    load_all_img() 

''' Training GAN ''' 
g_input = tf.placeholder(tf.float32, [None, Parm['g_input_dim']], name='g_input') 
d_input = tf.placeholder(tf.float32, shape = [None,64,64,3], name='d_input') 

G = generator(g_input) 
Dr = discriminator(d_input) # Real 
Df = discriminator(G) # Fake

'''
# Loss function 
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Df, labels = tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Df, labels = tf.ones_like(Dr)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Df, labels = tf.zeros_like(Dg)))
'''
g_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(Df,1e-8,1.0)))
d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(Dr,1e-8,1.0)) + \
                            tf.log(tf.clip_by_value(1. - Df,1e-8,1.0)))

# Get the varaibles for different network
tvars = tf.trainable_variables()
g_vars = [var for var in tvars if 'g_' in var.name]
d_vars = [var for var in tvars if 'd_' in var.name]

g_trainer = tf.train.AdamOptimizer(0.0002).minimize(g_loss, var_list=g_vars)
#d_trainer_fake = tf.train.AdamOptimizer(0.0002).minimize(d_loss_fake, var_list=d_vars)
#d_trainer_real = tf.train.AdamOptimizer(0.0002).minimize(d_loss_real, var_list=d_vars)
d_trainer = tf.train.AdamOptimizer(0.0002).minimize(d_loss, var_list=d_vars)

'''
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0002)

g_trainer = optimizer_gen.minimize(g_loss, var_list=g_vars)
d_trainer = optimizer_disc.minimize(d_loss, var_list=d_vars)
'''

''' Start Training '''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
print('Pre-train discriminator ...')
for i in tqdm(range(Parm['pretrain_iter'])):
    g_input_batch = np.random.normal(0., 1., size=[Parm['batch_size'], Parm['g_input_dim']])
    real_image_batch = image_next_batch()
    #_, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
    #                                       {x_placeholder: real_image_batch, z_placeholder: z_batch})
    disc_feed_dict = {d_input: real_image_batch, g_input: g_input_batch}
    _, dLoss = sess.run([d_trainer, d_loss], feed_dict = disc_feed_dict)

print('Start training...')
for i in tqdm(range(Parm['iteration'])):
    # Train discriminator
    for j in range(Parm['d_train_times']):
        real_image_batch = image_next_batch()
        g_input_batch = np.random.normal(0., 1., size=[Parm['batch_size'], Parm['g_input_dim']])
        #_, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
        #                                       {x_placeholder: real_image_batch, z_placeholder: z_batch})
        disc_feed_dict = {d_input: real_image_batch, g_input: g_input_batch}
        _, dLoss = sess.run([d_trainer, d_loss], feed_dict = disc_feed_dict)

    # Train generator
    g_input_batch = np.random.normal(0., 1., size=[Parm['batch_size'], Parm['g_input_dim']])
    _, gLoss = sess.run([g_trainer, g_loss], feed_dict={g_input: g_input_batch})

    if i % 500 == 0 and i >= 20000:
        save_path = saver.save(sess, Parm['model_dir']+'model{}.ckpt'.format(i))
    if i % 1000 == 0 and i < 20000:
        g_input_batch = np.random.normal(0., 1., size=[Parm['batch_size'],  Parm['g_input_dim']])
        images = sess.run(G, {g_input: g_input_batch})
        draw_image(images[0],i)
    if i % 100 == 0 and i >= 20000:
        #print('Iteration:', i, 'at', datetime.datetime.now())
        #print('dLoss', dLoss, ', gLoss:', gLoss)
        g_input_batch = np.random.normal(0., 1., size=[Parm['batch_size'],  Parm['g_input_dim']])
        images = sess.run(G, {g_input: g_input_batch})
        draw_image(images[0],i)
print('Done.')
