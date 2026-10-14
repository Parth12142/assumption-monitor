import numpy as np

# 1. Creating arrays
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("array:", a)
print("shape:", a.shape, "dtype:", a.dtype)
print("zeros:", np.zeros(3), "arange:", np.arange(0, 10, 2))
print("-----------------------------------------------------")
# 2. Indexing and slicing
print("first:", a[0], "last:", a[-1])
print("slice 2:5 ->", a[2:5])
print("-----------------------------------------------------")

# 3. Vectorized math (no loops)
print("a * 2 ->", a * 2)
print("a ** 2 ->", a ** 2)
print("-----------------------------------------------------")

# 4. Statistics
print("mean:", a.mean(), "std:", a.std(), "max:", a.max(), "argmax:", a.argmax())
print("-----------------------------------------------------")

# 5. Boolean masks
mask = a > 3
print("mask:", mask)
print("values > 3:", a[mask])
print("count > 3:", mask.sum())
print("-----------------------------------------------------")

# 6. cumsum (the core of CUSUM later)
print("cumsum:", a.cumsum())
print("-----------------------------------------------------")

# 7. 2D arrays
m = np.arange(12).reshape(3, 4)
print(m)
print("column means:", m.mean(axis=0))
print("row sums:", m.sum(axis=1))
print("-----------------------------------------------------")

# 8. Random walk, the shape of a price series
rng = np.random.default_rng(42)
steps = rng.normal(0, 1, 200)
walk = steps.cumsum()
print("walk start/end:", walk[0], walk[-1])
print("-----------------------------------------------------")

# 9. Mini CUSUM preview: cumulative deviation from the mean
dev = walk - walk.mean()
print("max deviation at index:", np.abs(dev).argmax())


print(a[a % 2 == 0])