"""
W5CT Option 1: Improving the Accuracy of a Neural Network
This program improves the accuracy of the deep learning model from W4CT by implementing
hyperparameter tuning and comparing with a Random Forest baseline.
"""

# Suppress warnings
import warnings
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF Warnings Suppressor import
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()

# Step 1: Begin by loading the data
import numpy as np
np.random.seed(456)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(456)

# Temporarily suppress warnings during deepchem import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Redirect stderr to suppress DeepChem's import messages
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        import deepchem as dc
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

print("Loading Tox21 dataset...")
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

print("Dataset loaded successfully!")
print(f"Training set shape: {train_X.shape}")
print(f"Validation set shape: {valid_X.shape}")
print(f"Test set shape: {test_X.shape}")

# Step 2: Generate a TensorFlow graph using a random forest classifier
print("\n" + "="*60)
print("STEP 2: RANDOM FOREST BASELINE")
print("="*60)

# Generate sklearn model
sklearn_model = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=456)
print("About to fit Random Forest model on train set.")
sklearn_model.fit(train_X, train_y)

# Make predictions
train_y_pred = sklearn_model.predict(train_X)
valid_y_pred = sklearn_model.predict(valid_X)
test_y_pred = sklearn_model.predict(test_X)

# Calculate weighted accuracies
weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
print("Weighted train Classification Accuracy: %f" % weighted_score)
weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Weighted valid Classification Accuracy: %f" % weighted_score)
rf_baseline_score = weighted_score  # Store for comparison
weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
print("Weighted test Classification Accuracy: %f" % weighted_score)

# Step 3: Investigate mapping hyperparameters to different Tox21 fully connected networks
def eval_tox21_hyperparams(n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True, random_seed=456):
    """
    Evaluate Tox21 neural network with specified hyperparameters
    """
    print("---------------------------------------------")
    print("Model hyperparameters")
    print("n_hidden = %d" % n_hidden)
    print("n_layers = %d" % n_layers)
    print("learning_rate = %f" % learning_rate)
    print("n_epochs = %d" % n_epochs)
    print("batch_size = %d" % batch_size)
    print("weight_positives = %s" % str(weight_positives))
    print("dropout_prob = %f" % dropout_prob)
    print("random_seed = %d" % random_seed)
    print("---------------------------------------------")
    
    d = 1024
    graph = tf.Graph()
    with graph.as_default():
        # Set random seed for this specific model
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        
        # Load data within the graph context
        _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w
        
        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]
        
        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)
        
        # Build multiple layers
        layer_input = x
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                if layer == 0:
                    input_dim = d
                else:
                    input_dim = n_hidden
                    
                W = tf.Variable(tf.random_normal((input_dim, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(layer_input, W) + b)
                # Apply dropout
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)
                layer_input = x_hidden
        
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)
        
        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            # Multiply by weights
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            # Sum all contributions
            l = tf.reduce_sum(entropy)
        
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
        
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()
        
        hyperparam_str = "d-%d-hidden-%d-layers-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s-seed-%d" % (
            d, n_hidden, n_layers, learning_rate, n_epochs, batch_size, str(weight_positives), random_seed)
        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str,
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
                    batch_w = train_w[pos:pos+batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                    if step % 50 == 0:  # Print every 50 steps to reduce output
                        print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                    train_writer.add_summary(summary, step)
                    
                    step += 1
                    pos += batch_size
            
            # Make Predictions (set keep_prob to 1.0 for predictions)
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
            test_y_pred = sess.run(y_pred, feed_dict={x: test_X, keep_prob: 1.0})
        
        train_writer.close()
        
        # Handle NaN values
        valid_mask = ~np.isnan(valid_y)
        valid_y_clean = valid_y[valid_mask]
        valid_y_pred_clean = valid_y_pred[valid_mask].flatten()
        valid_w_clean = valid_w[valid_mask]
        
        weighted_score = accuracy_score(valid_y_clean, valid_y_pred_clean, sample_weight=valid_w_clean)
        print("Valid Weighted Classification Accuracy: %f" % weighted_score)
        
        # Also calculate test accuracy for final comparison
        test_mask = ~np.isnan(test_y)
        test_y_clean = test_y[test_mask]
        test_y_pred_clean = test_y_pred[test_mask].flatten()
        test_w_clean = test_w[test_mask]
        
        test_score = accuracy_score(test_y_clean, test_y_pred_clean, sample_weight=test_w_clean)
        
        return weighted_score, test_score

# Step 4: Hyperparameter grid search with multiple random seeds
print("\n" + "="*60)
print("STEP 4: HYPERPARAMETER GRID SEARCH")
print("="*60)

# Define hyperparameter ranges
hyperparams = {
    'n_hidden': [25, 50, 100],
    'n_layers': [1, 2, 3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_prob': [0.3, 0.5, 0.7],
    'n_epochs': [20, 30, 45],  # Reduced for faster testing
    'batch_size': [50, 100, 200],
    'weight_positives': [True, False]
}

# For demonstration, we'll use a subset to avoid extremely long runtime
# In practice, you might want to use full ranges
limited_hyperparams = {
    'n_hidden': [50, 100],
    'n_layers': [1, 2],
    'learning_rate': [0.001, 0.01],
    'dropout_prob': [0.5],
    'n_epochs': [20],  # Reduced for faster execution
    'batch_size': [100],
    'weight_positives': [True]
}

print("Running limited hyperparameter search for demonstration...")
print("Hyperparameter ranges:")
for param, values in limited_hyperparams.items():
    print(f"  {param}: {values}")

# Multiple seeds for variance reduction
seeds = [456, 789, 101112]
n_seeds = len(seeds)

best_score = 0
best_params = None
best_test_score = 0
results = []

# Grid search
import itertools
param_combinations = list(itertools.product(*limited_hyperparams.values()))
param_names = list(limited_hyperparams.keys())

print(f"\nTesting {len(param_combinations)} parameter combinations with {n_seeds} seeds each...")
print(f"Total evaluations: {len(param_combinations) * n_seeds}")

for i, param_values in enumerate(param_combinations):
    params = dict(zip(param_names, param_values))
    print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
    print("Parameters:", params)
    
    # Test with multiple seeds
    valid_scores = []
    test_scores = []
    
    for seed in seeds:
        print(f"Testing with seed {seed}...")
        valid_score, test_score = eval_tox21_hyperparams(
            n_hidden=params['n_hidden'],
            n_layers=params['n_layers'],
            learning_rate=params['learning_rate'],
            dropout_prob=params['dropout_prob'],
            n_epochs=params['n_epochs'],
            batch_size=params['batch_size'],
            weight_positives=params['weight_positives'],
            random_seed=seed
        )
        valid_scores.append(valid_score)
        test_scores.append(test_score)
    
    # Calculate averages
    avg_valid_score = np.mean(valid_scores)
    std_valid_score = np.std(valid_scores)
    avg_test_score = np.mean(test_scores)
    std_test_score = np.std(test_scores)
    
    print(f"Average Validation Accuracy: {avg_valid_score:.4f} +/- {std_valid_score:.4f}")
    print(f"Average Test Accuracy: {avg_test_score:.4f} +/- {std_test_score:.4f}")
    
    # Store results
    result = {
        'params': params.copy(),
        'valid_scores': valid_scores,
        'test_scores': test_scores,
        'avg_valid_score': avg_valid_score,
        'std_valid_score': std_valid_score,
        'avg_test_score': avg_test_score,
        'std_test_score': std_test_score
    }
    results.append(result)
    
    # Track best model
    if avg_valid_score > best_score:
        best_score = avg_valid_score
        best_params = params.copy()
        best_test_score = avg_test_score

print("\n" + "="*60)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*60)

# Sort results by validation score
results.sort(key=lambda x: x['avg_valid_score'], reverse=True)

print("Top 5 configurations:")
for i, result in enumerate(results[:5]):
    print(f"\n{i+1}. Validation Accuracy: {result['avg_valid_score']:.4f} +/- {result['std_valid_score']:.4f}")
    print(f"   Test Accuracy: {result['avg_test_score']:.4f} +/- {result['std_test_score']:.4f}")
    print(f"   Parameters: {result['params']}")

print(f"\nBest configuration:")
print(f"Parameters: {best_params}")
print(f"Validation Accuracy: {best_score:.4f}")
print(f"Test Accuracy: {best_test_score:.4f}")

print(f"\nComparison with Random Forest baseline:")
print(f"Random Forest Validation Accuracy: {rf_baseline_score:.4f}")
print(f"Best Neural Network Validation Accuracy: {best_score:.4f}")
print(f"Improvement: {((best_score - rf_baseline_score) / rf_baseline_score * 100):.2f}%")

# Create a summary report
print("\n" + "="*60)
print("FINAL ANALYSIS SUMMARY")
print("="*60)

print("\n1. BASELINE PERFORMANCE:")
print(f"   Random Forest: {rf_baseline_score:.4f}")

print("\n2. BEST NEURAL NETWORK PERFORMANCE:")
print(f"   Validation: {best_score:.4f}")
print(f"   Test: {best_test_score:.4f}")
print(f"   Best Parameters: {best_params}")

print("\n3. HYPERPARAMETER INSIGHTS:")
print("   Top performing configurations show that:")
for i, result in enumerate(results[:3]):
    params = result['params']
    print(f"   {i+1}. Hidden={params['n_hidden']}, Layers={params['n_layers']}, LR={params['learning_rate']}")

print(f"\n4. MODEL VARIANCE:")
best_result = results[0]
print(f"   Best model std deviation: +/- {best_result['std_valid_score']:.4f}")
print(f"   This indicates {'low' if best_result['std_valid_score'] < 0.01 else 'moderate' if best_result['std_valid_score'] < 0.02 else 'high'} variance across random seeds")

print("\n" + "="*60)
print("TENSORBOARD VISUALIZATION")
print("="*60)
print("To view training curves, run:")
print("tensorboard --logdir=/tmp/")
print("Then navigate to http://localhost:6006")

print("\nProgram completed successfully!")
print("="*60)