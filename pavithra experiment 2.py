import numpy as np
import pandas as pd
heights = np.array([160,165,170,175,180,155,168,172,169,158])
s = pd.Series(heights)
print("Mean:", s.mean())
print("Median:", s.median())
print("Std dev:", round(s.std(),3))
print("Variance:", round(s.var(),3))
print("P(height > 170):", (s>170).mean())
sample = np.random.normal(loc=s.mean(), scale=s.std(), size=10)
print("Random sample (10) from normal around mean:", np.round(sample,2))
