import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
import torch
from rpy2.robjects import pandas2ri
pandas2ri.activate()

gglasso = importr('gglasso')
robjects.r('data(bardet)')
bardet = robjects.r['bardet']
X = np.array(bardet.rx2('x'))
y = np.array(bardet.rx2('y'))
X_df = pd.DataFrame(X)
y_series = pd.Series(y)
X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
y_tensor = torch.tensor(y_series.values, dtype=torch.float32)

# Print the shapes to verify
print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)
