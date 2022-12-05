import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

df = pd.read_csv('fake_news/news.csv')
print(df.columns)
df["label"].replace({"REAL": 1, "FAKE": 0}, inplace=True)
y = df.iloc[:, -1]
df["new_text"] = df["title"] + df["text"]
X = df.iloc[:, -1]
print(y.value_counts(normalize=True))
X = np.array(X)
y = np.array(y)
print(X)
print(y)


def length_dicsarding_space(text):
    return len(text) - text.count(' ')


max_features = 10000
#sequence_length = max(list(map(length_dicsarding_space, X)))
sequence_length = 25000
print(sequence_length)

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(X)

print(vectorize_layer(X[0]))

vectorized_text = vectorize_layer(X)
#print(vectorized_text)
vectorized_text = np.array(vectorized_text)
print(vectorized_text.shape)

X_train, X_test, y_train, y_test = train_test_split(vectorized_text, y, test_size=.33, random_state=4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.33, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs)

loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

