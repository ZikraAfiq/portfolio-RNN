import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Synthetic sequence generator
def generate_sequence(length):
    return np.array([2*i + 5 for i in range(length)]) 

# NOTES - 2(i) + 5

# 1 - 2(1) + 5 = 7
# 2 - 2(2) + 5 = 9
# 3 - 2(3) + 5 = 11
# 4 - 2(4) + 5 = 13
# 5 - 2(5) + 5 = 15
# 6 - 2(6) + 5 = 17

SEQUENCE_LENGTH = 5
NUM_SAMPLES = 200

X = []
y = []
for _ in range(NUM_SAMPLES):
    start = np.random.randint(0, 50)
    seq = generate_sequence(SEQUENCE_LENGTH + 1) + start
    X.append(seq[:-1])
    y.append(seq[-1])

X = np.array(X).reshape((NUM_SAMPLES, SEQUENCE_LENGTH, 1))
y = np.array(y)

# Model
model = Sequential([
    SimpleRNN(32, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

model.fit(X, y, epochs=300, verbose=1)

# Save safely
model.save("timeseries_rnn.keras")  # new format
print("✅ Model saved as timeseries_rnn.keras")