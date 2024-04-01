import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportions_chisquare

df = pd.DataFrame({
    'a': [np.nan] + [True]*99 + [np.nan]*99 + [False],
    'mycat': ['a']*100 + ['b']*100,
    'mysort': ['c','d']*100
})

def evaluate_data(df:pd.DataFrame,comparison_groups:list[list]):
    
    # Define a dictionary where all info will go
    info_dict = dict()

    # First get summary info for all variables:
    for var in df.columns:
        ser = df[var]
        # Dict to hold all info about this variable
        var_dict = dict()
        var_dict['Null Info'] = null_info(ser)
        var_dict['Unique'] = ser.unique()
        
        # Record the type of the variable
        var_dict['Type'] = get_variable_type(
            var_dict['Unique'],
            pd.api.types.is_numeric_dtype(ser)
        )
        
        # Get General Stats on the thing
        var_dict['Stats'] = get_stats(
            var_type=var_dict['Type'],
            ser=ser,
            var=var
        )
        
        # Comparisons Across Groups
        var_dict['Comparisons'] = dict()
        for comparison_group in comparison_groups:
            comparison_group_dict = do_comparison(
                comparison_group=comparison_group,
                var=var,
                has_nulls=var_dict['Null Info']['Count'] > 0,
                var_df = df[comparison_group + [var]],
                type=var_dict['Type'],
                var_stats = var_dict['Stats']
            )
            
            var_dict['Comparisons'][str(comparison_group)] = comparison_group_dict
            
        
        # TODO: For numerical vars, check to see if there are any values
        # that have an unusual amount of repeats (like default values or
        # placeholder values). ALSO for any such values, do a proportions
        # test across groups.
        
        
        info_dict[var] = var_dict
    return info_dict

def do_comparison(
    comparison_group:list[str],
    var:str,
    has_nulls:bool,
    var_df:pd.DataFrame,
    type:str,
    var_stats:dict
):
    # Instantiate Comparison
    comparison_dict = dict()
    
    # Compare nulls
    comparison_dict['Nulls'] = compare_nulls_across_groups(
        comparison_group=comparison_group,
        var=var,
        has_nulls = has_nulls,
        var_df = var_df
    )
    
    # Now we do some other comparisons based on the variable type
    if type in ['Constant','Constant With Nulls']:
        pass
    elif type in ['Binary Bool','Binary Numerical']:
        comparison_dict['Proportion True'] = compare_proportion_true(
            var_df=var_df,
            var=var,
            comparison_group=comparison_group
        )
    elif type == 'Numerical':
        comparison_dict['Frequency Outliers'] = compare_all_frequency_outliers(
            var_stats=var_stats,
            var_df=var_df,
            var=var,
            comparison_group=comparison_group
        )
        # TODO comparison_dict['Mean'] = compare_mean()
        
    elif type in ['Categorical', 'Binary Categorical']:
        # TODO comparison_dict['Value Shares'] = compare_all_value_shares()
        # TODO comparison_dict['Unique'] = compare_unique_values()
        comparison_dict['Still Need to edit this'] = 'really'
    
    return comparison_dict
    
def compare_all_frequency_outliers(
    var_stats:dict,
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    if var_stats['Frequency Outlier Count'] <= 0:
        return 'No Frequency Outliers'
    else:
        freq_outliers_dict = dict()
        for i, row in var_stats['Frequency Outliers'].iterrows():
            value = row['Value']
            freq_outliers_dict[value] = compare_value_share(
                value=value,
                var_df=var_df,
                var=var,
                comparison_group=comparison_group
            )
        

def compare_value_share(
    value,
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    var_df['Is Value'] = np.select(
        [
            var_df[var].isnull(),
            var_df[var] == value
        ],
        [
            np.nan,
            1
        ],
        0
    )
    group_object = (
        var_df[comparison_group+['Is Value']]
        .groupby(by=comparison_group)
    )
    group_df = (
        group_object
        .sum()
        .reset_index()
        .rename(columns={'Is Value':'Count Of Value'})
    )
    group_df = group_df.merge(
        how='left',
        on=comparison_group,
        right=(
            group_object
            .count()
            .reset_index()
            .rename(columns={'Is Value':'Total Obs'})
        )
    )
    return equal_proportions_test_many_samples(
        n_list=list(group_df['Total Obs']),
        hits_list=list(group_df['Count of Value'])
    )
    
def compare_proportion_true(
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    var_df['Binary As Float'] = var_df[var].astype(float)
    group_object = (
        var_df[comparison_group + ['Binary As Float']]
        .groupby(by=comparison_group)
    )
    group_df = (
        group_object
        .sum()
        .reset_index()
        .rename(columns={'Binary As Float':'Number True'})
    )
    group_df = group_df.merge(
        how='left',
        on=comparison_group,
        right= (
            group_object
            .count()
            .reset_index()
            .rename(columns={'Binary As Float':'Total Obs'})
        )
    )
    return equal_proportions_test_many_samples(
        n_list=list(group_df['Total Obs']),
        hits_list=list(group_df['Number True'])
    )
  

def get_stats(var_type:str, ser:pd.Series, var:str):
    '''get summary statistics on the variable over the whole table.
    
    Arguments:
        var_type:
            The description of the type of variable
        ser:
            The column of the dataframe for the variable being analyzed
        var:
            The variable name

    Returns:
        A dictionary of information which will populate var_dict['Stats']
    '''
    if var_type == 'Constant':
        return {'Constant Value': ser.iloc[0]}
    elif var_type == 'Constant With Nulls':
        return {'Constant Value': ser.dropna().iloc[0]}
    elif var_type in ['Binary Bool','Binary Numerical']:
        return get_binary_stats(ser)
    elif var_type == 'Numerical':
        return get_numerical_stats(ser,var)
    elif var_type in ['Categorical', 'Binary Categorical']:
        return get_categorical_stats(ser,var)

def get_categorical_stats(ser:pd.Series, var:str):
    my_dict = dict()
    my_dict['Count Non Missing'] = len(ser.dropna())
    my_df = (
        pd.DataFrame(ser.value_counts())
        .reset_index()
        .rename(columns={var:'Value','count':'Count'})
    )
    my_df['Proportion'] = my_df['Count']/my_dict['Count Non Missing']
    my_dict['Counts And Proportions'] = my_df
    return my_dict

def get_numerical_stats(ser:pd.Series,var:str):
    my_dict = dict()
    my_dict['Mean'] = ser.mean()
    my_dict['Standard Dev'] = ser.std()
    my_dict['Median'] = ser.median()
    my_dict['Quartile 1'] = ser.quantile(0.25)
    my_dict['Quartile 3'] = ser.quantile(0.75)
    my_dict['IQR'] = my_dict['Quartile 3'] - my_dict['Quartile 1']
    my_dict['Max'] = ser.max()
    my_dict['Min'] = ser.min()
    upper_lim = my_dict['Quartile 3'] + 1.5*my_dict['IQR']
    lower_lim = my_dict['Quartile 1'] - 1.5*my_dict['IQR']
    # Outliers
    my_df = pd.DataFrame(ser)
    my_df['Is Upper Outlier'] = np.where(
        my_df[var] > upper_lim,
        1,
        0
    )
    my_df['Is Lower Outlier'] = np.where(
        my_df[var] < lower_lim,
        1,
        0
    )
    my_dict['Upper Outlier Count'] = my_df['Is Upper Outlier'].sum()
    my_dict['Lower Outlier Count'] = my_df['Is Lower Outlier'].sum()
    # TODO Now identify values that pop up unusually frequently
    freq_df = (
        pd.DataFrame(ser.value_counts())
        .reset_index()
        .rename(columns={var:'Value'})
    )
    freq_avg = freq_df['count'].mean()
    freq_std = freq_df['count'].std()
    freq_df['Is Outlier'] = np.where(
        freq_df['count'] > freq_avg + 3*freq_std,
        1,
        0
    )
    my_dict['Frequency Outlier Count'] = freq_df['Is Outlier'].sum()
    freq_outliers = []
    for i,row in freq_df[freq_df['Is Outlier']==1].iterrows():
        freq_outliers.append(
            [row['Value'],row['count']]
        )
    my_dict['Frequency Outliers'] = (
        freq_df[freq_df['Is Outlier']==1]
        [['Value','count']]
        .rename(columns={'count':'Count'})
    )
    return my_dict

def get_binary_stats(ser:pd.Series):
    my_dict = dict()
    my_dict['Not Null Count'] = len(ser.dropna())
    my_dict['True Count'] = ser.sum()
    my_dict['False Count'] = np.logical_not(ser).sum()
    my_dict['True Proportion'] = (
        my_dict['True Count']/my_dict['Not Null Count']
    )
    my_dict['False Proportion'] = (
        my_dict['False Count']/my_dict['Not Null Count']
    )
    return my_dict

def compare_nulls_across_groups(
    comparison_group:list,
    var:str,
    has_nulls:bool,
    var_df:pd.DataFrame
):
    if not has_nulls:
        return 'No Nulls'
    else:
        var_df['Is Null'] = np.where(
            var_df[var].isnull(),
            1,
            0
        )
        group_object = (
            var_df[comparison_group + ['Is Null']]
            .groupby(by=comparison_group)
        )
        group_df = (
            group_object
            .sum()
            .reset_index()
            .rename(columns={'Is Null':'Number Null'})
        )
        group_df = group_df.merge(
            how='left',
            on=comparison_group,
            right=(
                group_object
                .count()
                .reset_index()
                .rename(columns={'Is Null': 'Total Obs'})
            )
        )
        return (
            equal_proportions_test_many_samples(
                n_list=list(group_df['Total Obs']),
                hits_list=list(group_df['Number Null'])
            )
        )

def get_variable_type(unique_array:np.ndarray, is_numerical:bool):
    if len(unique_array) == 1:
        return 'Constant'
    elif (
        (len(unique_array) == 2)
        and {np.nan}.issubset(set(unique_array))
    ):
        return 'Constant With Nulls'
    elif set(unique_array).issubset({True, False, np.nan}):
        return 'Binary Bool'
    elif set(unique_array).issubset({1,0,np.nan}):
        return 'Binary Numerical'
    elif is_numerical:
        return 'Numerical'
    else:
        if len(unique_array) == 2:
            return 'Binary Categorical'
        else:
            return 'Categorical'
    
def null_info(ser:pd.Series):
    '''Get info on the nulls of a series.'''
    if len(ser) == 0:
        return 'Zero Length'
    null_info = dict()
    null_info['Count'] = ser.isna().sum()
    null_info['Proportion'] = null_info['Count']/len(ser)
    return null_info

def equal_proportions_test_two_samples(n_a:int, p_hat_a:float, n_b:int, p_hat_b:float):
    p_hat = (
        (n_a*p_hat_a + n_b*p_hat_b)
        /(n_a + n_b)
    )
    z = (
        (p_hat_a - p_hat_b)
        /np.sqrt(
            p_hat*(1-p_hat)*((1/n_a)+(1/n_b))
        )
    )
    test_info = dict()
    test_info['Z-score'] = z
    test_info['P-value'] = scipy.stats.norm.sf(abs(z))*2
    return test_info

def equal_proportions_test_many_samples(n_list:list, hits_list:list):
    test = proportions_chisquare(
        count = hits_list,
        nobs = n_list
    )
    test_info = dict()
    test_info['Chi_2_stat'] = test[0]
    test_info['P-value'] = test[1]
    return test_info


evaluate_data(df,[['mycat'],['mysort'],['mycat','mysort']])

