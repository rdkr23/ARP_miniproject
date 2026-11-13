import numpy as np

x = np.load("test/1_baidu_soccer_embeddings.npy")

print(x.shape)
print(x.dtype)
print(x[0, :10])

duration_seconds = 45 * 60
print(duration_seconds / len(x))
