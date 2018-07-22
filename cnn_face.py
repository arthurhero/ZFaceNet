import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import socket
from multiprocessing.dummy import Pool as ThreadPool
import sys 
import urllib2
import cv2 
import subprocess
import random
 
invalid_path="vgg_face_dataset/invalid/"
avg_path="vgg_face_dataset/avg/"
folder_path="vgg_face_dataset/files/"
validation_path="vgg_face_dataset/validation/"
test_path="vgg_face_dataset/test/"
model_path="models/model.ckpt"

tf.reset_default_graph()

#####################################CNN structure

# Convolutional Layer 1.
filter_size1 = 3
filter_dim1 = 3
num_filters1 = 64

# Convolutional Layer 2.
filter_size2 = 3
filter_dim2 = 64
num_filters2 = 64

# Convolutional Layer 3.
filter_size3 = 3
filter_dim3 = 64
num_filters3 = 128

# Convolutional Layer 4.
filter_size4 = 3
filter_dim4 = 128
num_filters4 = 128

# Convolutional Layer 5.
filter_size5 = 3
filter_dim5 = 128
num_filters5 = 256

# Convolutional Layer 6.
filter_size6 = 3
filter_dim6 = 256
num_filters6 = 256

# Convolutional Layer 7.
filter_size7 = 3
filter_dim7 = 256
num_filters7 = 256

# Convolutional Layer 8.
filter_size8 = 3
filter_dim8 = 256
num_filters8 = 512

# Convolutional Layer 9.
filter_size9 = 3
filter_dim9 = 512
num_filters9 = 512

# Convolutional Layer 10.
filter_size10 = 3
filter_dim10 = 512
num_filters10 = 512

# Convolutional Layer 11.
filter_size11 = 3
filter_dim11 = 512
num_filters11 = 512

# Convolutional Layer 12.
filter_size12 = 3
filter_dim12 = 512
num_filters12 = 512

# Convolutional Layer 13.
filter_size13 = 3
filter_dim13 = 512
num_filters13 = 512

# Convolutional Layer 14.
filter_size14 = 7
filter_dim14 = 512
num_filters14 = 4096

# Fully-connected layer.
fc_size = 4096

##################################Other params
orig_img_size=256
img_size=224
num_channels = 3
num_classes = 2622

#stochastic gradient descent
mini_batch_size  = 64
momentum_coeff = 0.9 

#regularization
weight_decay_coeff = 5e-4
dropout_rate = 0.5     #applied after 2 FC layers

learning_rate_init = 10e-2
decread_factor = 10.0    #when validation accuracy stop increasing

#weights initialization
weights_init_mean= 0.0 
weights_init_std = 10e-2

bias_init = 0.0

#data augmentation
flip_chance = 0.5 

#vali batch size
vali_batch_size = 256

#test batch size
test_batch_size = 256

#################################Helpers

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib2.urlopen(url,timeout=10)
    except urllib2.HTTPError, e:
        print "error"
        return np.array([])
    except urllib2.URLError, e:
        print "error"
        return np.array([])
    except socket.timeout as e:
        print "error"
        return np.array([])
    except socket.error as e:
        print "error"
        return np.array([])
    except Exception:
        print "error"
        return np.array([])
    if resp is None:
        print "error"
        return np.array([])
    if resp.getcode()!=200:
        print "error"
        return np.array([])

    try:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
    except urllib2.HTTPError, e:
        print "error"
        return np.array([])
    except urllib2.URLError, e:
        print "error"
        return np.array([])
    except socket.timeout as e:
        print "error"
        return np.array([])
    except socket.error as e:
        print "error"
        return np.array([])
    except Exception:
        print "error"
        return np.array([])
    if resp is None:
        print "error"
        return np.array([])
    if resp.getcode()!=200:
        print "error"
        return np.array([])

    if image.size == 0:
        print "error"
        return np.array([])
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        print "error"
        return np.array([])
    return image

def crop_and_scale(img,left,top,right,bottom):
    top=int(round(top))
    left=int(round(left))
    right=int(round(right))
    bottom=int(round(bottom))
    if img.shape[0]<bottom-top+1 or img.shape[1]<right-left+1 :
        print "error"
        return np.array([])
    crop_img = img[top:bottom+1, left:right+1]
    if crop_img.shape[0]<10 or crop_img.shape[1]<10:
        print "error"
        return np.array([])
    scale_img=cv2.resize(crop_img,(orig_img_size,orig_img_size))
    real_avg=cv2.imread(avg_path+"real_avg.png",cv2.IMREAD_UNCHANGED)
    return scale_img-real_avg

def under_prob(prob):
    x=random.randint(0,9999)
    return x<prob*10000

def random_proc(img):
    x_start=random.randint(0,orig_img_size-img_size)
    y_start=random.randint(0,orig_img_size-img_size)
    crop_img = img[x_start:x_start+img_size, y_start:y_start+img_size]
    if under_prob(flip_chance):
        flip_img = cv2.flip(crop_img, 1)
        return flip_img
    return crop_img

def get_mini_batch():
    imgs = list()
    labels = list()
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    count = 0
    while count < mini_batch_size:
        person_num = random.randint(0,2622-1)
        person = filenames[person_num][:-4]
        cmd = "cat "+folder_path+person+".txt"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        entry_total = len(entries)
        entry_num =  random.randint(0,entry_total-1)
        entry = entries[entry_num]
        e=entry.split()
        l=float(e[2])
        t=float(e[3])
        r=float(e[4])
        b=float(e[5])
        if l<=0 or t<=0 or r<=0 or b<=0:
            print "error"
            continue
        raw_img=url_to_image(e[1])
        if raw_img.shape==(0,):
            continue
        else:
            img=crop_and_scale(raw_img,l,t,r,b)
            if img.shape==(0,):
                continue
            else:
                imgs.append(random_proc(img))
                labels.append(person_num)
                count += 1
    return imgs, labels

def get_test_batch(vali=False):
    imgs = list()
    labels = list()
    path=""
    if (vali):
        path = validation_path
    else:
        path = test_path
    cmd = "ls "+path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    count = 0
    iter_num = 0
    if (vali):
        iter_num = vali_batch_size
    else:
        iter_num = test_batch_size 
    while count < iter_num:
        file_num=random.randint(0,len(filenames)-1)
        filename=filenames[file_num]
        cmd = "cat "+path+"/"+filename
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        entry_total = len(entries)
        entry_num =  random.randint(0,entry_total-1)
        entry = entries[entry_num]
        e=entry.split()
        person_num=int(e[0])
        l=float(e[4])
        t=float(e[5])
        r=float(e[6])
        b=float(e[7])
        if l<=0 or t<=0 or r<=0 or b<=0:
            print "error"
            continue
        raw_img=url_to_image(e[3])
        if raw_img.shape==(0,):
            continue
        else:
            img=crop_and_scale(raw_img,l,t,r,b)
            if img.shape==(0,):
                continue
            else:
                imgs.append(random_proc(img))
                labels.append(person_num)
                count += 1
    return imgs, labels

def num_to_name(num):
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    return filenames[num][:-4]

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(num_to_name(cls_true[i]))
        else:
            xlabel = "True: {0}, Pred: {1}".format(num_to_name(cls_true[i]), num_to_name(cls_pred[i]))

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

'''
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)
'''

#############################################Layer Constructors

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, mean=weights_init_mean, stddev=weights_init_std))

def new_biases(length):
    return tf.Variable(tf.constant(bias_init, shape=[length]))

def new_conv_layer(input,              # The previous layer.
        num_input_channels,            # Num. channels in prev. layer.
        filter_size,                   # Width and height of each filter.
        num_filters,                   # Number of filters.
        use_pooling=True):             # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
            filter=weights,
            strides=[1, 1, 1, 1],
            padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
        num_inputs,              # Num. inputs from prev. layer.
        num_outputs,             # Num. outputs.
        use_relu=True):          # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

############################################Placeholders

#img array
x_image = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x_image')

#label array
y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')

###########################################Constructing CNN
layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=False)

layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

layer_conv3, weights_conv3 = \
        new_conv_layer(input=layer_conv2,num_input_channels=num_filters2,filter_size=filter_size3,num_filters=num_filters3,use_pooling=False)

layer_conv4, weights_conv4 = \
        new_conv_layer(input=layer_conv3,num_input_channels=num_filters3,filter_size=filter_size4,num_filters=num_filters4,use_pooling=True)

layer_conv5, weights_conv5 = \
        new_conv_layer(input=layer_conv4,num_input_channels=num_filters4,filter_size=filter_size5,num_filters=num_filters5,use_pooling=False)

layer_conv6, weights_conv6 = \
        new_conv_layer(input=layer_conv5,num_input_channels=num_filters5,filter_size=filter_size6,num_filters=num_filters6,use_pooling=False)

layer_conv7, weights_conv7 = \
        new_conv_layer(input=layer_conv6,num_input_channels=num_filters6,filter_size=filter_size7,num_filters=num_filters7,use_pooling=True)

layer_conv8, weights_conv8 = \
        new_conv_layer(input=layer_conv7,num_input_channels=num_filters7,filter_size=filter_size8,num_filters=num_filters8,use_pooling=False)

layer_conv9, weights_conv9 = \
        new_conv_layer(input=layer_conv8,num_input_channels=num_filters8,filter_size=filter_size9,num_filters=num_filters9,use_pooling=False)

layer_conv10, weights_conv10 = \
        new_conv_layer(input=layer_conv9,num_input_channels=num_filters9,filter_size=filter_size10,num_filters=num_filters10,use_pooling=True)

layer_conv11, weights_conv11 = \
        new_conv_layer(input=layer_conv10,num_input_channels=num_filters10,filter_size=filter_size11,num_filters=num_filters11,use_pooling=False)

layer_conv12, weights_conv12 = \
        new_conv_layer(input=layer_conv11,num_input_channels=num_filters11,filter_size=filter_size12,num_filters=num_filters12,use_pooling=False)

layer_conv13, weights_conv13 = \
        new_conv_layer(input=layer_conv12,num_input_channels=num_filters12,filter_size=filter_size13,num_filters=num_filters13,use_pooling=True)

layer_conv14, weights_conv14 = \
        new_conv_layer(input=layer_conv13,num_input_channels=num_filters13,filter_size=filter_size14,num_filters=num_filters14,use_pooling=False)


layer_flat, num_features = flatten_layer(layer_conv14)


layer_fc1 = new_fc_layer(input=layer_flat,
        num_inputs=num_features,
        num_outputs=fc_size,
        use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
        num_inputs=fc_size,
        num_outputs=num_classes,
        use_relu=False)

#get prediction
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_pred_cls=tf.cast(y_pred_cls,tf.int32)

#get loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true_cls)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_init).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

# Counter for total number of iterations performed so far.
total_iterations = 0

# Saver to store model
saver = tf.train.Saver()

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    #Restore model
    try:
        saver.restore(session,model_path) 
        print("Model restored.")
    except ValueError:
        print("No stored model. Training from scratch.")
    except tf.errors.DataLossError:
        print("No stored model. Training from scratch.")

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
            total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = get_mini_batch() 

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x_image: x_batch,
                y_true_cls: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    # Save the trained model
    save_path = saver.save(session, model_path)
    print("Model saved in path: %s" % save_path)


#############################################Plotting methods

def plot_example_errors(test_imgs, test_labels, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = test_imgs[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = test_labels[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
            cls_true=cls_true[0:9],
            cls_pred=cls_pred[0:9])

def plot_confusion_matrix(test_labels, cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = test_labels

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
            y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Split the test-set into smaller batches of this size.

def print_test_accuracy(test_imgs, test_labels, show_example_errors=False,
        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(test_imgs)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Create a feed-dict with these images and labels.
    feed_dict = {x_image: test_imgs,
            y_true_cls: test_labels}

    # Calculate the predicted class using TensorFlow.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

    cls_true = test_labels

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(test_imgs, test_labels, cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(test_labels, cls_pred)


optimize(num_iterations=99)
test_imgs, test_labels = get_test_batch()
print_test_accuracy(test_imgs, test_labels, show_example_errors=True)

