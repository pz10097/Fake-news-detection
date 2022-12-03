import pandas as pd
import numpy as np

from tensorflow.keras import layers

df = pd.read_csv('fake_news/news.csv')
print(df.columns)
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
sequence_length = 40000
print(sequence_length)

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(X)

print(vectorize_layer(X[0]))

vectorized_text = vectorize_layer(X)
print(vectorized_text)
