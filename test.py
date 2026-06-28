import pandas as pd
from lightgbm import LGBMRegressor

X = pd.DataFrame({
    'x1': [1,2,3,4,5],
    'x2': [5,4,3,2,1]
})

y = [10,20,30,40,50]

model = LGBMRegressor()
model.fit(X, y)

print(model.predict([[6,0]]))