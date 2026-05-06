import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("action_model.h5")

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Predict
pred = model.predict(X)

# Convert to class labels
pred_labels = np.argmax(pred, axis=1)
true_labels = np.argmax(y, axis=1)

print("\nPredictions:", pred_labels)
print("Actual     :", true_labels)

# Accuracy
accuracy = np.mean(pred_labels == true_labels)
print("\nAccuracy:", accuracy)