#Step 0: Suppress all TF and DeepChem warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()
import DeepChem_Suppressor
dc = DeepChem_Suppressor.import_deepchem()

#Step 1: Load the Tox21 Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow.compat.v1 as tf
np.random.seed(456)
tf.disable_v2_behavior()
tf.set_random_seed(456)
train, valid, test = DeepChem_Suppressor.load_tox21_quiet()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w
#(CSU-Global, n.d.)

#Step 2: Remove extra datasets
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]
#(CSU-Global, n.d.)

#Step 3: Define placeholders that accept minibatches of different sizes
d = 1024
n_hidden = 50
learning_rate = .001
n_epochs = 10
batch_size = 100

#Generate TF graph
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None,))

#Step 4: Implement a hidden layer
with tf.name_scope("hidden-layer"):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
#(CSU-Global, n.d.)

#Step 5: Complete the fully connected architecture
with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_logit = tf.matmul(x_hidden, W) + b
    #the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    #Rounding P(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    #Compute the cross-entropy term for each datapoint
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
    #Sum all contributions
    l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()
#(CSU-Global, n.d.)

#Steps 6 and 7 are implemented together, modifying the architecture to include dropout

#Reset the graph to rebuild with dropout
tf.reset_default_graph()

#Rebuild the graph with dropout
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None,))
    keep_prob = tf.placeholder(tf.float32)  #dropout placeholder

with tf.name_scope("hidden-layer"):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
    x_hidden = tf.nn.dropout(x_hidden, keep_prob)  #Apply dropout

with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_logit = tf.matmul(x_hidden, W) + b
    #the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    #Rounding P(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    #Compute the cross-entropy term for each datapoint
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
    #Sum all contributions
    l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

#Step 8: Implement mini-batching training
train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21',
                                    tf.get_default_graph())

N = train_X.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x: batch_X, y: batch_y, keep_prob: 0.5}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)
            
            step += 1
            pos += batch_size

    #Make Predictions
    valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
#(CSU-Global, n.d.)

#Step 9: Evaluate the model using PyPlot, Sklearn, and TensorBoard
train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21', tf.get_default_graph())

N = train_X.shape[0]
loss_history = []
step_history = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0

    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x: batch_X, y: batch_y, keep_prob: 0.5}
            _, summary, loss_val = sess.run([train_op, merged, l], feed_dict=feed_dict)

            #TensorBoard
            train_writer.add_summary(summary, step)

            #For loss curve
            loss_history.append(loss_val)
            step_history.append(step)

            print("epoch %d, step %d, loss: %f" % (epoch, step, loss_val))
            step += 1
            pos += batch_size

    #Validation predictions (no dropout)
    valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0}).flatten()

    #Test predictions (no dropout)
    test_y_pred  = sess.run(y_pred,  feed_dict={x: test_X,  keep_prob: 1.0}).flatten()

    vmask = ~np.isnan(valid_y)
    tmask = ~np.isnan(test_y)

    v_acc  = accuracy_score(valid_y[vmask], valid_y_pred[vmask])
    v_prec = precision_score(valid_y[vmask], valid_y_pred[vmask])
    v_rec  = recall_score(valid_y[vmask], valid_y_pred[vmask])
    v_f1   = f1_score(valid_y[vmask], valid_y_pred[vmask])

    t_acc  = accuracy_score(test_y[tmask],  test_y_pred[tmask])
    t_prec = precision_score(test_y[tmask],  test_y_pred[tmask])
    t_rec  = recall_score(test_y[tmask],  test_y_pred[tmask])
    t_f1   = f1_score(test_y[tmask],  test_y_pred[tmask])
    t_cm   = confusion_matrix(test_y[tmask], test_y_pred[tmask])

    print("\nValidation — Acc: %.4f  Prec: %.4f  Rec: %.4f  F1: %.4f" % (v_acc, v_prec, v_rec, v_f1))
    print("Test       — Acc: %.4f  Prec: %.4f  Rec: %.4f  F1: %.4f" % (t_acc, t_prec, t_rec, t_f1))
    print("Test Confusion Matrix:\n", t_cm)
    #(Grimoire, 2025)

   #Loss Curve Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    plt.plot(step_history, loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Loss (sum of sigmoid CE per minibatch)")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("\nSaved loss curve to loss_curve.png")
    #(Grimoire, 2025)
train_writer.close()

print("\nTo view TensorBoard: tensorboard --logdir=/tmp/fcnet-tox21")