import pandas as pd
import numpy as np
import matplotlib
import lightgbm as lgb
import sklearn
from lightgbm import LGBMRegressor

model = LGBMRegressor()
print("All imports OK")
print(f"  pandas     {pd.__version__}")
print(f"  numpy      {np.__version__}")
print(f"  lightgbm   {lgb.__version__}")
print(f"  sklearn    {sklearn.__version__}")