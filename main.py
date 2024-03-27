import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': [1,1,1,2,2,np.nan],
    'b': [5,5,6,7,7,7]
})

def get_categorical_summary(myser:pd.Series):
    return myser.value_counts(dropna=False).to_dict()

get_categorical_summary(df['a'])