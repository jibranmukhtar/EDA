'''This module contains functions related to cleaning data.'''
import pandas as pd
import numpy as np

import summary

def make_df_numerical(
    df:pd.DataFrame,
    null_replacements:dict = dict(),
):
    '''Transform the dataframe to be all numerical and non-missing
    
    Arguments:
        df:
            The dataframe to be transformed.
        null_replacements:
            A dictionary with the value we want to use to replace nulls.
            By default it will use the mean of the sample.

    Returns:
        num_df:
            A dataframe that has repplaced categorical variables with
            one-hots and explicitly modeled missing values so that all
            entries are non-missing
        notes:
            A dictionary with info about how the transformation was done
    '''
    
    # This holds info about each of the variables in df
    og_info = dict()
    # This holds info about each of the variables that would be in the
    # final transformed version of df
    new_info = dict()
    # Instantiate a list that will hold all the columns of num_df (to be
    # concatenated into a df at the end)
    num_df_cols_list = []
    # Get info on type for each variable in df
    for var in df.columns:
        og_var_info = dict()
        # Make a series version for easier access
        og_ser = df[var]
        # Some values that will be usefull no matter what.
        has_nulls = og_ser.isnull().any()
        unique_vals = og_ser.unique()
        var_type = summary.get_variable_type(
            unique_array=unique_vals,
            is_numerical=pd.api.types.is_numeric_dtype(og_ser)
        )
        og_var_info['Type'] = var_type
        
        # Assemble the new numerical-only data frame column by column
        if var_type == 'Constant':
            val = og_ser[1]
            new_col = pd.Series(
                np.where(
                    og_ser == True,
                    1,
                    1
                ),
                name=f'{var}: {val}'
            )
            num_df_cols_list.append(new_col)
        if var_type == 'Constant With Nulls':
            null_col = pd.Series(
                np.where(
                    og_ser.isnull(),
                    1,
                    0
                ),
                name=f'{var}: Null'
            )
            num_df_cols_list.append(null_col)
            val = og_ser.dropna().iloc[0]
            nonnull_col = pd.Series(
                np.where(
                    og_ser.isnull(),
                    0,
                    1
                ),
                name=f'{var}: {val}'
            )
            num_df_cols_list.append(nonnull_col)
        if var_type == 'Binary Bool':
            nullfill = null_replacements.get(var,og_ser.mean())
            og_var_info['Null Fill'] = nullfill
            bool_col = pd.Series(
                np.select(
                    [
                        og_ser == True,
                        og_ser == False
                    ],
                    [
                        1,
                        0
                    ],
                    nullfill
                ),
                name=var
            )
            num_df_cols_list.append(bool_col)
            if has_nulls:
                null_col = pd.Series(
                    np.where(
                        og_ser.isnull(),
                        1,
                        0
                    ),
                    name=f'{var}: Null'
                )
                num_df_cols_list.append(null_col)
        if var_type == 'Binary Numerical':
            nullfill = null_replacements.get(var,og_ser.mean())
            og_var_info['Null Fill'] = nullfill
            bool_col = pd.Series(
                np.select(
                    [
                        og_ser == 1,
                        og_ser == 0
                    ],
                    [
                        1,
                        0
                    ],
                    nullfill
                ),
                name=var
            )
            num_df_cols_list.append(bool_col)
            if has_nulls:
                null_col = pd.Series(
                    np.where(
                        og_ser.isnull(),
                        1,
                        0
                    ),
                    name=f'{var}: Null'
                )
                num_df_cols_list.append(null_col)
        if var_type in ['Binary Categorical', 'Categorical']:
            for val in og_ser.unique():
                col = pd.Series(
                    np.where(
                        og_ser == val,
                        1,
                        0
                    ),
                    name=f'{var}: {val}'
                )
                num_df_cols_list.append(col)
            if has_nulls:
                if has_nulls:
                    null_col = pd.Series(
                        np.where(
                            og_ser.isnull(),
                            1,
                            0
                        ),
                        name=f'{var}: Null'
                    )
                    num_df_cols_list.append(null_col)
        if var_type == 'Numerical':
            nullfill = null_replacements.get(var,og_ser.mean())
            col = pd.Series(
                np.where(
                    og_ser.isnull(),
                    nullfill,
                    og_ser
                ),
                name=var
            )
            num_df_cols_list.append(col)
            if has_nulls:
                null_col = pd.Series(
                    np.where(
                        og_ser.isnull(),
                        1,
                        0
                    ),
                    name=f'{var}: Null'
                )
                num_df_cols_list.append(null_col)
        
        og_info[var] = og_var_info
        
    notes = {
        'Original Vars': og_info,
        'Transformed Vars': new_info
    }
    
    num_df = pd.concat(num_df_cols_list, axis=1)
    
    return num_df, notes