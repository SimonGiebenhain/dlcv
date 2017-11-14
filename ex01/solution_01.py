import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


'''
Solutions of Jonas Probst, Gabriel Scheibler, Clemens Gutknecht and Simon Giebenhain.
I am so sorry for the mess, but there is no time left for a clean up. We will improve next time.
'''

number_of_pictures = 3
#Exploit that all pictures have same dimensions
w = 313
h = 500

#Preprocessing Data, load images and gather all pixel is big design matrix (3 columns for the different color channels)
x_images = np.empty((w * h * number_of_pictures, 3))
y_images = np.empty((w * h * number_of_pictures, 3))
for i in range(number_of_pictures):
    x_img = imread('cat_0%d.jpg' % (i + 1))
    y_img = imread('cat_0%d_vignetted.jpg' % (i + 1))
    #Rotate pictures 0 and 2 in order to match r and picture 1
    if i != 1:
        x_img = np.transpose(x_img, [1, 0, 2])
        y_img = np.transpose(y_img, [1, 0, 2])
    x_img = x_img.reshape((w * h, 3))
    y_img = y_img.reshape((w * h, 3))
    x_images[i * w * h:(i+1)*w*h, :] = x_img
    y_images[i * w * h:(i+1)*w*h, :] = y_img


#Construct vector of distances matching to the design matrix
wc = w / 2
hc = h / 2
xv, yv = np.meshgrid(np.arange(h) - hc, np.arange(w) - wc)
r_ = np.reshape(np.tile((np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)).reshape((w*h)), number_of_pictures), (w*h* number_of_pictures))


def cross_calidation_5():
    rand = np.random.permutation(number_of_pictures * w * h)
    rands = np.stack(np.split(rand, 5))
    number_of_degrees = 5
    step_size = 1
    errors = np.empty((number_of_degrees - 1, 2, 5))
    for i in range(5):
        print('Using segement %d as validation.' % (i+ 1))
        training_indices = np.reshape(np.delete(rands, i,0), int((4 * number_of_pictures * w * h) / 5))
        validation_indices = rands[i]
        training_x = np.empty((len(training_indices), 3))
        training_y = np.empty((len(training_indices),3))
        training_r = np.empty((len(training_indices)))
        validation_x = np.empty((len(validation_indices), 3))
        validation_y = np.empty((len(validation_indices), 3))
        validation_r = np.empty((len(validation_indices)))
        for j, random_index in enumerate(training_indices):
            training_x[j] = x_images[random_index]
            training_y[j] = y_images[random_index]
            training_r[j] = r_[random_index]
        for j, random_index in enumerate(validation_indices):
            validation_x[j] = x_images[random_index]
            validation_y[j] = y_images[random_index]
            validation_r[j] = r_[random_index]
        for n in range(1, number_of_degrees):
            params, training_error = train_params(n * step_size, training_x, training_y, training_r)
            errors[n - 1, 0, i] = training_error
            print('This was a polynomial degree: %d' % n * step_size)
            errors[n - 1, 1, i] = eval_params(params, validation_x, validation_y, validation_r)

    print(errors)
    mean_errors = np.sum(errors, axis=-1) / 5
    print(mean_errors)
    x_axis = np.arange(1,number_of_degrees) * step_size
    plt.plot(x_axis, mean_errors[:, 0], 'b--', x_axis, mean_errors[:, 1], 'g')
    plt.show()

def cross_calidation_5_regularization():
    rand = np.random.permutation(number_of_pictures * w * h)
    rands = np.stack(np.split(rand, 5))
    number_of_lambdas = 10
    errors = np.empty((number_of_lambdas - 1, 2, 5))
    for i in range(5):
        print('Using segement %d as validation.' % (i+ 1))
        training_indices = np.reshape(np.delete(rands, i,0), int((4 * number_of_pictures * w * h) / 5))
        validation_indices = rands[i]
        training_x = np.empty((len(training_indices), 3))
        training_y = np.empty((len(training_indices),3))
        training_r = np.empty((len(training_indices)))
        validation_x = np.empty((len(validation_indices), 3))
        validation_y = np.empty((len(validation_indices), 3))
        validation_r = np.empty((len(validation_indices)))
        for j, random_index in enumerate(training_indices):
            training_x[j] = x_images[random_index]
            training_y[j] = y_images[random_index]
            training_r[j] = r_[random_index]
        for j, random_index in enumerate(validation_indices):
            validation_x[j] = x_images[random_index]
            validation_y[j] = y_images[random_index]
            validation_r[j] = r_[random_index]
        for n in range(1,number_of_lambdas):
            params, training_error = train_params_regularized(8, training_x, training_y, training_r, 0.2 * n)
            errors[n - 1, 0, i] = training_error
            print('Polynomial degree: %d' % n)
            errors[n - 1, 1, i] = eval_params(params, validation_x, validation_y, validation_r)
    print(errors)
    mean_errors = np.sum(errors, axis=-1) / 5
    print(mean_errors)
    x_axis = np.arange(1, number_of_lambdas) * 0.2
    plt.plot(x_axis, mean_errors[:, 0], 'b--', label='training MSE')
    plt.plot(x_axis, mean_errors[:, 1], 'g', label='validation MSE')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def test_segemtation():
        r_ = np.arange(15)
        x_images = np.stack(np.split(np.arange(15 * 3), 15), axis=0)
        y_images = np.stack(np.split(np.arange(15 * 3), 15), axis=0)
        rand = np.random.permutation(15)
        rands = np.stack(np.split(rand,  5))
        print(rands.shape)
        for i in range(5):
            print('Using segement %d as validation.' % (i + 1))
            training_indices = np.reshape(np.delete(rands, i, 0), 12)
            print(training_indices.shape)
            validation_indices = rands[i]
            training_x = np.empty((len(training_indices), 3))
            training_y = np.empty((len(training_indices), 3))
            training_r = np.empty((len(training_indices)))
            validation_x = np.empty((len(validation_indices), 3))
            validation_y = np.empty((len(validation_indices), 3))
            validation_r = np.empty((len(validation_indices)))
            for j, random_index in enumerate(training_indices):
                training_x[j] = x_images[random_index]
                training_y[j] = y_images[random_index]
                training_r[j] = r_[random_index]
            for j, random_index in enumerate(validation_indices):
                validation_x[j] = x_images[random_index]
                validation_y[j] = y_images[random_index]
                validation_r[j] = r_[random_index]
            print('training_x: \n %s ' %training_x)
            print('training_y: \n %s ' %training_y)
            print('training_r: \n %s ' %training_r)
            print('validation_x: \n %s' % validation_x)
            print('validation_y: \n %s' % validation_y)
            print('validation_r: \n %s' % validation_r)

def eval_params(params, x_train, y_train, r_train):
    n = params.shape[0]
    # Model input and output
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 3])
    r = tf.placeholder(tf.float32, shape=[None])

    # loss
    r_vec = tf.stack([r ** i for i in range(n)])
    ar_vec = tf.multiply(r_vec, tf.reshape(params, (n, 1)))
    sums_vec = tf.reduce_sum(ar_vec, axis=0)
    pred = tf.stack([x[:, i] * sums_vec for i in range(3)], axis=-1)

    loss = tf.reduce_mean(tf.square(pred - y))  # sum of the squares

    sess = tf.Session()
    #print("Loss: %s" % sess.run(loss, {x: x_train, y: y_train, r: r_train}))
    return sess.run(loss, {x: x_train, y: y_train, r: r_train})


def train_params(n, x_train, y_train, r_train):
    # Model parameters
    W = tf.Variable(np.random.random((n+1)), dtype=tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 3])
    r = tf.placeholder(tf.float32, shape=[None])

    # loss
    r_vec = tf.stack([r ** i for i in range(n + 1)])
    ar_vec = tf.multiply(r_vec, tf.reshape(W, (n+1, 1)))
    sums_vec = tf.reduce_sum(ar_vec, axis=0)
    pred = tf.stack([x[:, i]* sums_vec for i in range(3)], axis=-1)

    loss = tf.reduce_mean(tf.square(pred - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train, r: r_train})

    # evaluate training accuracy
    curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train, r: r_train})
    print("W: %s loss: %s"%(curr_W, curr_loss))

    return curr_W, curr_loss

def train_params_regularized(n, x_train, y_train, r_train, lam):
    # Model parameters
    W = tf.Variable(np.random.random((n+1)), dtype=tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 3])
    r = tf.placeholder(tf.float32, shape=[None])

    # loss
    r_poly = tf.stack([r ** i for i in range(n + 1)])
    factor = tf.reduce_sum(tf.multiply(r_poly, tf.reshape(W, (n+1, 1))), axis=0)
    pred = tf.stack([x[:, i]* factor for i in range(3)], axis=-1)

    loss = tf.reduce_mean(tf.square(pred - y)) + (lam / 2) * tf.reduce_sum(W ** 2)
    # optimizer
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train, r: r_train})

    # evaluate training accuracy
    curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train, r: r_train})
    print("W: %s loss: %s"%(curr_W, curr_loss))

    return curr_W, curr_loss




def get_distances(I):
    w = I.shape[1]
    h = I.shape[0]
    wc = w / 2
    hc = h / 2
    xv, yv = np.meshgrid(np.arange(w) - wc, np.arange(h) - hc)
    return np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)

def de_vignett(J, W):
    r = get_distances(J)
    r_poly = np.stack([r ** i for i in range(len(W))])
    test = np.reshape(np.array(W), (len(W), 1, 1)) * r_poly
    I = np.stack([ np.clip(J[:, :, i] / np.sum(test, axis=0), 0, 255) for i in range(3)],axis=2)
    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(J)

    # Show the tinted image
    plt.subplot(1, 2, 2)

    # A slight gotcha with imshow is that it might give strange results
    # if presented with data that is not uint8. To work around this, we
    # explicitly cast the image to uint8 before displaying it.
    plt.imshow(np.uint8(I))
    plt.show()

#train_params(4, x_images, y_images, r_)
#cross_calidation_5_regularization()
#cross_calidation_5()
#test_segemtation()

#de_vignett(imread('cat_04_vignetted.jpg'), [1.00254107, -0.06471649, -0.22629637, -0.32916346, -0.25945607])
#de_vignett(imread('cat_05_vignetted.jpg'), [1.00254107, -0.06471649, -0.22629637, -0.32916346, -0.25945607])
#de_vignett(imread('cat_06_vignetted.jpg'), [1.00254107, -0.06471649, -0.22629637, -0.32916346, -0.25945607])
