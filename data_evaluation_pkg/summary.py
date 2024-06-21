'''Functions related to getting summary stats and data-cleanliness info'''
import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import itertools
import warnings
from tqdm import tqdm
import logging
import time
import os
import datetime


def evaluate_data(df:pd.DataFrame, comparison_groups:list[list], alpha: float = 0.05, path: str = 'export.csv'):
    '''Variable summaries with tests for consistency across groups.
    
    Arguments:
        df:
            A dataframe to be analyzed.
        comparison_groups:
            A list of lists of variables that defined groups across which
            we want to compare variables.
        alpha:
            Preferred Alpha threshhold across various tests done with the report
    
    Returns:
        info_dict:
            A dictionary containing 3 strings: First with comprehensive report, 
            second with summary, third with only flagged instances of the report.
    '''
    # Filter out all warnings
    warnings.filterwarnings("ignore")

    # Configure logging. Uncomment the second line to disable logging.
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    #logging.disable(logging.CRITICAL)

    # Create a progress bar object
    #pbar = tqdm(total=len(df.columns), desc='Column Evaluation Progress', bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}    ')


    # Define a dictionary where all info will go
    info_dict = dict()

    # First get summary info for all variables:
    for idx, var in enumerate(df.columns, start=1):
        #Note start time of the loop
        start_time = time.time()
        logging.info(f'\n{"": <0}Evaluating: {var:<{50}}')

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
        if (any(var in sublist for sublist in comparison_groups)):
            var_dict['Comparisons'] = 'This column is part of comparison group'
        else:
            for comparison_group in comparison_groups:
                comparison_group_dict = do_comparison(
                    comparison_group=comparison_group,
                    var=var,
                    has_nulls=var_dict['Null Info']['Count'] > 0,
                    var_df = df[list(set(comparison_group + [var]))],
                    type=var_dict['Type'],
                    var_stats = var_dict['Stats'],
                    alpha = alpha
                )
                
                var_dict['Comparisons'][str(comparison_group)] = comparison_group_dict
            
        
        # TODO: For numerical vars, check to see if there are any values
        # that have an unusual amount of repeats (like default values or
        # placeholder values). ALSO for any such values, do a proportions
        # test across groups.
        
        
        info_dict[var] = var_dict

        end_time = time.time()  # Record end time
        duration = end_time - start_time  # Calculate duration
        logging.info(f'\n{"": <0}Evaluation Successful ({idx}/{len(df.columns)}): {var:<{50}} (Duration: {duration:.2f} seconds)')
        #pbar.update(1)  # Update progress bar for each sub-task

    #pbar.close()

    output = {
        'detailed_report': build_tree_string(info_dict),
        'summary_report': build_tree_string(create_summary(info_dict, alpha)),
        'flag_report': build_tree_string(filter_concern_flags(create_summary(info_dict, alpha)))
    }

    export_report(output, comparison_groups, path)

    # Restore default warning settings
    warnings.filterwarnings("default")

    return output
        
    

    

def do_comparison(
    comparison_group:list[str],
    var:str,
    has_nulls:bool,
    var_df:pd.DataFrame,
    type:str,
    var_stats:dict,
    alpha: float
):
    '''Compare a variable's values across subsets of the sample.
    
    Arguments:
        comparison_group:
            A list of variable names that define the distinct groups to
            be compared. E.G. if it's ['a','b'], then each unique
            combination of values for 'a' and 'b' will be treated as
            a group.
        var:
            The name of the variable to be analyzed.
        has_nulls:
            A bool indicating if var has null values in the sample.
        var_df:
            A dataframe including var and the comparison group variables
        type:
            A string describing the "type" of var. This is NOT the python
            -type. It will be from the following list defined of options:
                [Constant, Constant With Nulls, Binary Bool,
                Binary Numerical, Binary Categorical, Numerical,
                Categorical]   
            See function "get_variable_type()" for more info.
        var_stats:
            A dictionary of summary stats about the variable.

    Returns:
        comparison_dict:
            A dictionary with comparison results for var, across
            comparison_group.
    '''
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
        comparison_dict['Mean'] = compare_mean(
            var_df=var_df,
            comparison_group=comparison_group,
            var=var,
            alpha = alpha
        )
        
    elif type in ['Categorical', 'Binary Categorical']:
        comparison_dict['Value Shares'] = compare_all_value_shares(
            var_df=var_df,
            comparison_group=comparison_group,
            var=var
        )
        comparison_dict['Unique Values'] = compare_unique_values(
            var_df=var_df,
            comparison_group=comparison_group,
            var=var
        )
    
    return comparison_dict
    
def compare_unique_values(
    var_df:pd.DataFrame,
    comparison_group:list[str],
    var:str,
):
    '''Find inconsistencies in the values exhibited by categorical var.
    
    Arguments:
        var_df:
            A dataframe with var and the comparison_group variables.
        comparison_group:
            The group of variables whose unique combination of values
            defines a group.
        var:
            The name of the variable to be analyzed.
    
    Returns:
        string 'All Groups Share Common Support'
            It will return this string if var has the same support for
            all groups in comparison_group.
        vals_of_interest_df:
            A dataframe which contains the values of var which show up in
            some but not all groups. for each of these values, you can
            see the list of groups in which it does show up and the list
            of groups in which it does not show up.
    '''
    
    unique_ser = (
        var_df
        .groupby(by=comparison_group)
        .apply(lambda group_df: group_df[var].unique())
    )
    unique_df = unique_ser.reset_index().rename(columns={0:'Unique Set'})
    all_sets = [set(s) for s in unique_ser]
    intersection = set.intersection(*all_sets)
    vals_of_interest = list(set(var_df[var].unique()) - intersection)
    if len(vals_of_interest) <= 0:
        return 'All Groups Share Common Support'
    else:
        groups_including_col = []
        groups_excluding_col = []
        for val in vals_of_interest:
            groups_with_val = []
            groups_without_val = []
            for _, row in unique_df.iterrows():
                if val in row['Unique Set']:
                    groups_with_val.append(list(row[comparison_group]))
                else:
                    groups_without_val.append(list(row[comparison_group]))
            groups_including_col.append(groups_with_val)
            groups_excluding_col.append(groups_without_val)
        vals_of_interest_df = pd.DataFrame({
            'Value': vals_of_interest,
            'Groups Including Value': groups_including_col,
            'Groups Excluding Value': groups_excluding_col
        }).set_index('Value')
    return vals_of_interest_df

def compare_all_value_shares(
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    '''Do a proportion test for each val in a categorical variable.
    
    Arguments:
        var_df:
            A dataframe containing var and the comparison group variables
        var:
            The name of the variable to be analyzed.
        comparison_group:
            The list of variables for which a unique combination of
            values defines a group.
    Returns:
        value_shares_dict:
            a dictionary where the dictionary-keys are values of var, and
            the dictionary-values contain test results for if the share
            of the sample with that value is the same across groups.
    '''
    value_shares_dict = dict()
    for val in var_df[var].unique():
        value_shares_dict[val] = compare_value_share(
            value=val,
            var_df=var_df,
            var=var,
            comparison_group=comparison_group
        )
    return value_shares_dict

 
def compare_all_frequency_outliers(
    var_stats:dict,
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    '''For unusually frequent vals of a numerical var, is the proportion
    of the sample taking that val consistent across groups.
    
    Arguments:
        var_stats:
            The var_stats dictionary containing information on whether
            there are actually outliers.
        var_df:
            A dataframe containing the values of the var being analyzed
            and the variables in the comparison_group.
        var:
            The name of the analysis variable.
        comparison_group:
            The variables whose unique combination of values defines a 
            group.
    Returns:
        string 'No Frequency Outliers'
            This string is returned if there are no frequency outliers.
        freq_outliers_dict:
            A dictionary with info for each value about the test of
            whether the share of the sample taking that value is
            consistent across groups.
    '''
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

def compare_mean(
    var_df: pd.DataFrame,
    comparison_group: list[str],
    var: str,
    alpha: float
):

    '''Compare means of a numerical variable across groups defined by 
    combinations of categorical variables and perform ANOVA and Tukey's HSD test.

    Arguments:
        var_df:
            A dataframe containing the values of the variable being analyzed
            and the variables in the comparison_group.
        comparison_group:
            A list of column names whose unique combination of values defines a 
            group.
        var:
            The name of the numeric variable for which the means are to be compared.
        alpha:
            The significance level for the statistical tests (default is 0.05).
            
    Returns:
        mean_dict:
            A dictionary containing the means of the numeric variable for each 
            unique combination of values from the comparison_group.
        anova_dict:
            A dictionary containing the results of the ANOVA test, including the
            F-statistic, p-value, and a boolean indicating whether the null hypothesis
            is rejected.
        tukey_dict:
            A dictionary containing the results of Tukey's HSD test, with each key being 
            a tuple of group pairs and the value being another dictionary containing the
            mean difference, adjusted p-value, confidence interval, and the reject boolean.
    '''
    
    # Get the unique values for each column in the comparison group
    unique_values = [var_df[col].unique() for col in comparison_group]
    
    # Get the cartesian product of these unique values
    cartesian_product = list(itertools.product(*unique_values))
    
    # Initialize the dictionary to hold the means and collect group data
    mean_dict = {}
    group_labels = []
    group_data = []
    
    # Calculate the mean for each group in the cartesian product
    for group in cartesian_product:
        # Create a boolean mask for the group
        mask = pd.Series([True] * len(var_df))
        for col, val in zip(comparison_group, group):
            mask &= (var_df[col] == val)
        
        # Extract the data for the current group
        group_data_subset = var_df.loc[mask, var]
        mean_value = group_data_subset.mean()
        
        # Add the mean to the dictionary
        mean_dict[group] = mean_value
        
        # Collect data and labels for Tukey's HSD test
        group_labels.extend([str(group)] * len(group_data_subset))
        group_data.append(group_data_subset)
    
    # Flatten the group_data list and create a corresponding list of labels
    flat_group_data = [item for sublist in group_data for item in sublist]
    flat_group_labels = [group_labels[i] for i in range(len(group_labels))]
    
    # Perform ANOVA test
    anova_result = f_oneway(*group_data)
    anova_dict = {
        'F-statistic': anova_result.statistic,
        'p-value': anova_result.pvalue,
        'reject': anova_result.pvalue < alpha
    }
    
    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=flat_group_data, groups=flat_group_labels, alpha=alpha)
    
    # Extract the Tukey HSD results into a dictionary
    tukey_dict = {}
    for i in range(len(tukey_result.reject)):
        group_pair = (tukey_result.groupsunique[tukey_result._multicomp.pairindices[0][i]], 
                      tukey_result.groupsunique[tukey_result._multicomp.pairindices[1][i]])
        tukey_dict[group_pair] = {
            'meandiff': tukey_result.meandiffs[i],
            'p-adj': tukey_result.pvalues[i],
            'lower': tukey_result.confint[i][0],
            'upper': tukey_result.confint[i][1],
            'reject': tukey_result.reject[i]
        }
    
    mean_results = {
    'Average': mean_dict,
    'ANOVA': anova_dict,
    'Tukey HSD': tukey_dict
    }
    
    return mean_results
        
def compare_value_share(
    value,
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    '''Test if the share of sample with a value is cross-group consistent.
    
    Arguments:
        value:
            The value whose frequency is being tested.
        var_df:
            A dataframe containing the values of the var being analyzed
            and the variables in the comparison_group.
        var:
            The name of the analysis variable.
        comparison_group:
            The variables whose unique combination of values defines a 
            group.
        
    Returns:
        A dictionary with the chi-squared test statistic and p-value for
        the null hypothesis that the proportion of the sample with this
        value for var is the same across groups.
    '''
    var_df = var_df.copy()
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
        hits_list=list(group_df['Count Of Value'])
    )
   
    
def compare_proportion_true(
    var_df:pd.DataFrame,
    var:str,
    comparison_group:list[str]
):
    '''Compare the distribution of true values across groups.
    
    Arguments:
        var_df:
            A dataframe containing the values of the var being analyzed
            and the variables in the comparison_group.
        var:
            The name of the analysis variable.
        comparison_group:
            The variables whose unique combination of values defines a 
            group.
    Returns:
        A dictionary with the chi-squared test statistic and p-value for
        the null hypothesis that the proportion of the sample with this
        value for var is the same across groups.
        '''
    var_df = var_df.copy()
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
        A var-type -dependent dictionary of information on general stats
        for the variable var.
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
    '''Get summary stats for a categorical variable.
    
    Arguments:
        ser:
            A series object with the sample data for the variable.
        var:
            The variable name.
    
    Returns:
        A dictionariy with summary stats.
    '''

    # Drop NAs
    ser_clean = ser.dropna()

    # Get value counts and proportions
    value_counts = ser_clean.value_counts()
    proportions = ser_clean.value_counts(normalize=True)

    # Create the dictionary for counts and proportions
    counts_and_proportions = {
        value: {'count': count, 'proportion': proportion}
        for value, count, proportion in zip(value_counts.index, value_counts, proportions)
    }

    # Create the final dictionary with the required hierarchy
    result_dict = {
        "Counts And Proportions": counts_and_proportions,
        "Count Non Missing": len(ser_clean)
    }

    return result_dict


def get_numerical_stats(ser:pd.Series,var:str):
    '''Get summary statistics for a numerical variable.

    Arguments:
        ser:
            A series object with the sample data for the variable.
        var:
            The variable name.
    
    Returns:
        A dictionary with summary stats.

    '''
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
        freq_df['count'] > freq_avg + 2*freq_std,
        1,
        0
    )
    my_dict['Frequency Outlier Count'] = freq_df['Is Outlier'].sum()
    my_dict['Frequency Outliers'] = (
        freq_df[freq_df['Is Outlier']==1]
        [['Value','count']]
        .rename(columns={'count':'Count'})
    )
    return my_dict


def get_binary_stats(ser:pd.Series):
    '''Get summary stats for a binary variable.
    
    Arguments:
        ser:
            A series object with sample data for the variable.
    Returns:
        A dictionary with summary stats for the binary variable.   
    '''
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
    '''Test if the proportion of nulls is consistent across groups.

    Arguments:
        var_df:
            A dataframe containing the values of the var being analyzed
            and the variables in the comparison_group.
        var:
            The name of the analysis variable.
        comparison_group:
            The variables whose unique combination of values defines a 
            group.
        has_nulls:
            A bool indicating if the variable has null values.
    Returns:
        A dictionary with chi-squre test stat and p-value for testing
        whether all groups have the same proportion of nulls.
    '''
    if not has_nulls:
        return 'No Nulls'
    else:
        var_df = var_df.copy()
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
    '''Get the general type (not python datatype) of a variable.
    
    For data cleaning and exploratory analysis purposes, it's useful to
    know which of the following categories data falls into:
    [
        Constant
        Constant With Nulls
        Binary Bool
        Binary Numerical
        Binary Categorical
        Numerical
        Categorical
    ]
    
    Arguments:
        unique_array:
            The list of unique values in the variable's sample.
        is_numerical:
            A bool which indicates whether the pandas datatype of the
            variable is numerical.
        
    Returns:
        A string with one of the above listed types
    '''

    def is_subset(s1, s2):
    
        '''Determines if set s1 is a subset of set s2, correctly handling np.nan values.
        
        Arguments:
        s1 (set): The first set to be checked as a subset.
        s2 (set): The second set to be checked as a superset.
        
        Returns:
        bool: True if s1 is a subset of s2, accounting for np.nan values, otherwise False.'''
        
        def has_nan(s):
            return any(isinstance(x, float) and np.isnan(x) for x in s)
        
        # Remove np.nan and check for subset ignoring np.nan
        s1_no_nan = {x for x in s1 if not (isinstance(x, float) and np.isnan(x))}
        s2_no_nan = {x for x in s2 if not (isinstance(x, float) and np.isnan(x))}
        
        # Check if both sets have np.nan consistently
        has_nan_s1 = has_nan(s1)
        has_nan_s2 = has_nan(s2)
        
        # Ensure boolean and numeric types are handled separately
        def type_safe_issubset(small, large):
            small_bools = {x for x in small if isinstance(x, bool)}
            large_bools = {x for x in large if isinstance(x, bool)}
            small_non_bools = {x for x in small if not isinstance(x, bool)}
            large_non_bools = {x for x in large if not isinstance(x, bool)}
            return small_bools.issubset(large_bools) and small_non_bools.issubset(large_non_bools)
    
        return type_safe_issubset(s1_no_nan, s2_no_nan) and has_nan_s1 <= has_nan_s2 

    if len(unique_array) == 1:
        return 'Constant'
    elif (len(unique_array) == 2) and is_subset({np.nan}, set(unique_array)):
        return 'Constant With Nulls'
    elif is_subset(set(unique_array), {True, False, np.nan}):
        return 'Binary Bool'
    elif is_subset(set(unique_array), {1, 0, np.nan}):
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


def equal_proportions_test_two_samples(
    n_a:int,
    p_hat_a:float,
    n_b:int,
    p_hat_b:float
):
    '''Test whether two samples have equal proportions of a variable.
    
    Arguments:
        n_a:
            The number of observations in group a
        p_hat_a:
            The proportion of hits in group a
        n_b:
            The number of observations in group b
        p_hat_b:
            The proportion of hits in group b
    
    Returns
        A dict with the z-score and p-value.
    '''
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
    '''Test whether the probability of a hit is the same across samples.
    
    Arguments:
        n_list:
            A list of the number of observations in each sample.
        hits_list:
            A list of the number of hits in each sample.
    
    Returns:
        A dictionary with the chi-squared stat and P-Value of the test
    '''
    test = proportions_chisquare(
        count = hits_list,
        nobs = n_list
    )
    test_info = dict()
    test_info['Chi_2_stat'] = round(test[0],3)
    test_info['P-value'] = round(test[1],3)
    return test_info

def create_summary(data_dict, alpha = 0.05):
    '''Creates a summary from the detailed report. This function includes 
    all the necessary thesholds that booleanizes continuous metrics for flagging.
    
    Arguments:
        data_dict:
            Dictionary containing the comprehensive report.
        alpha:
            Optional alpha threshold for statastical tests.
    Returns:
        summary:
            Summary dictionary. 
    '''

    summary = {}

    for column_name, column_info in data_dict.items():
        col_summary = {}

        if (column_info.get('Type', {}) == 'Constant'):
            col_summary['Determined Type'] = column_info.get('Type', {})
            col_summary['Unique Value'] = list(column_info.get('Unique', {}))

        elif (column_info.get('Type', {}) == 'Constant With Nulls'):
            col_summary['Determined Type'] = column_info.get('Type', {})
            col_summary['Unique Value'] = list(column_info.get('Unique', {}))

            col_summary['Null Info'] = {}
            col_summary['Null Info']['Concern Flag'] = bool(column_info.get('Null Info', {}).get('Proportion', {}) > 0.20)

            col_summary['Comparison'] = {}
            if isinstance(column_info.get('Comparisons', {}), dict):
                for comp_group, comp_group_value in column_info.get('Comparisons', {}).items():
                    col_summary['Comparison'][comp_group] = {}
                    col_summary['Comparison'][comp_group]["Chi Sq. Concern Flag: "] = bool(comp_group_value.get('Nulls', {}).get('P-value', {}) < alpha)
            else:
                pass

        elif (column_info.get('Type', {}) == 'Binary Numerical'):
            col_summary['Determined Type'] = column_info.get('Type', {})
            col_summary['Unique Value'] = list(column_info.get('Unique', {}))

            col_summary['Null Info'] = {}
            col_summary['Null Info']['Concern Flag'] = bool(column_info['Null Info'].get('Proportion', {}) > 0.20)

            col_summary['Stats'] = {}
            col_summary['Stats']['Concern Flag, Proportion out-of-bound'] = bool(column_info['Stats'].get('True Proportion', {}) > 0.9 or column_info['Stats'].get('True Proportion', {}) < 0.1)

            col_summary['Comparison'] = {}
            if isinstance(column_info.get('Comparisons', {}), dict):
                for comp_group, comp_group_value in column_info.get('Comparisons', {}).items():
                    col_summary['Comparison'][comp_group] = {}
                    comp_values_nulls = column_info['Comparisons'].get(comp_group, {}).get('Nulls', {})
                    col_summary['Comparison'][comp_group]['Concern Flag, Chi-Sq. test - Nulls'] = bool(isinstance(comp_values_nulls, dict) and comp_values_nulls.get('P-value', {}) < alpha)


                    comp_values_trues = column_info['Comparisons'].get(comp_group, {}).get('Proportion True', {})
                    col_summary['Comparison'][comp_group]['Concern Flag, Chi-Sq. test - True Proportion'] = bool(comp_values_trues.get('P-value', {}) < alpha)
            else:
                pass

        elif (column_info.get('Type', {}) == 'Numerical'):
            col_summary['Determined Type'] = column_info.get('Type', {})

            if(len(column_info.get('Unique', {})) > 10):
                col_summary['Unique Values'] = 'Continuous Values...'
            else:
                col_summary['Unique Values'] = column_info.get('Unique', {})

            col_summary['Null Info'] = {}
            col_summary['Null Info']['Concern Flag'] = bool(column_info.get('Null Info', {}).get('Proportion', {}) > 0.20)

            col_summary['Stats'] = {}
            col_summary['Stats']['Concern Flag, Upper Outlier Crossed'] = bool(column_info['Stats'].get('Upper Outlier Count', {}) > 0)
            col_summary['Stats']['Concern Flag, Lower Outlier Crossed'] = bool(column_info['Stats'].get('Lower Outlier Count', {}) > 0)
            col_summary['Stats']['Concern Flag, Frequency Outlier Crossed'] = bool(column_info['Stats'].get('Frequency Outlier Count', {}) > 0)

            col_summary['Comparison'] = {}
            if isinstance(column_info.get('Comparisons', {}), dict):
                for comp_group, comp_group_value in column_info.get('Comparisons', {}).items():
                    col_summary['Comparison'][comp_group] = {}
                    comp_values = column_info['Comparisons'].get(comp_group, {}).get('Mean', {})
                    col_summary['Comparison'][comp_group]['Concern Flag, ANOVA'] = bool(comp_values.get('ANOVA', {}).get('reject', {}))
            else:
                pass  

        elif (column_info.get('Type', {}) == 'Binary Categorical'):
            col_summary['Determined Type'] = column_info.get('Type', {})

            if(len(column_info.get('Unique', {})) > 10):
                col_summary['Unique Values'] = 'Continuous Values...'
            else:
                col_summary['Unique Values'] = list(column_info.get('Unique', {}))

            col_summary['Null Info'] = {}
            col_summary['Null Info']['Concern Flag'] = bool(column_info.get('Null Info', {}).get('Proportion', {}) > 0.20)

            col_summary['Stats'] = {}
            for group, group_value in column_info.get('Stats', {}).get('Counts And Proportions',{}).items():
                col_summary['Stats']['Concern Flag, Proportion - ' + str(group)] = bool(group_value.get('proportion', {}) < 0.1 or group_value.get('proportion', {}) > 0.9)

            col_summary['Comparison'] = {}
            if isinstance(column_info.get('Comparisons', {}), dict):
                for comp_group, comp_group_value in column_info.get('Comparisons', {}).items():
                    for group, group_value in column_info.get('Comparisons', {}).get(comp_group,{}).get('Value Shares', {}).items():
                        col_summary['Comparison'][comp_group] = {}
                        col_summary['Comparison'][comp_group]["Concern Flag, Chi-Sq. test - " + str(group)] = bool(group_value.get('P-value', {}) < alpha)
            else:
                pass


        elif (column_info.get('Type', {}) == 'Categorical'):
            col_summary['Determined Type'] = column_info.get('Type', {})

            if(len(column_info.get('Unique', {})) > 10):
                col_summary['Unique Values'] = 'Many unique values'
            else:
                col_summary['Unique Values'] = list(column_info.get('Unique', {}))

            col_summary['Null Info'] = {}
            col_summary['Null Info']['Concern Flag'] = bool(column_info.get('Null Info', {}).get('Proportion', {}) > 0.20)

            col_summary['Stats'] = {}
            counts = []
            for group, group_value in column_info.get('Stats', {}).get('Counts And Proportions',{}).items():
                counts.append(group_value.get('count', {}))

            # Calculate the mean and standard deviation
            mean = np.mean(counts)
            std = np.std(counts)

            # Calculate the upper and lower bounds 3 standard deviations away from the mean
            upper_bound = mean + 3*std
            lower_bound = mean - 3*std

            # Check if any number is outside the bounds (one True in flags is enough)
            has_outlier = any((x < lower_bound) | (x > upper_bound) for x in counts)
            col_summary['Stats']['Concern Flag, Category Dispropotionate'] = has_outlier

            col_summary['Comparison'] = {}
            if isinstance(column_info.get('Comparisons', {}), dict):
                for comp_group, comp_group_value in column_info.get('Comparisons', {}).items():
                    for group, group_value in column_info.get('Comparisons', {}).get(comp_group,{}).get('Value Shares', {}).items():
                        col_summary['Comparison'][comp_group] = {}
                        col_summary['Comparison'][comp_group]["Concern Flag, Chi-Sq. test - " + str(group)] = bool(group_value.get('P-value', {}) < alpha)
            else:
                pass

        summary[column_name] = col_summary

    return summary

def filter_concern_flags(d):
    '''Filter only those dictionary instances, where the value is True.
    Arguments:
        d:
            Summary Dictionary
    Returns:
        result:
            Dictionary with concern flags
    '''
    def recursive_filter(d):
        if isinstance(d, dict):
            filtered = {}
            for key, value in d.items():
                try:
                    if isinstance(value, dict):
                        filtered_branch = recursive_filter(value)
                        if filtered_branch:
                            filtered[key] = filtered_branch
                    elif isinstance(value, bool) and value == True:
                        filtered[key] = value
                    else: 
                        #print("Is not Bool or is False", value, type(value))
                        #filtered[key] = value
                        pass
                except Exception as e:
                    print(e)
                    #print("ERROR in ", key, value)
                    #filtered[key] = "ERROR in " + str(key) + str(value)
            return filtered if filtered else None
        return None

    result = recursive_filter(d)
    return result if result else {}


def build_tree_string(d, prefix=''):
    """
    Recursively builds a string representation of a nested dictionary as a tree with branches.

    Args:
        d (dict): The nested dictionary.
        prefix (str): The current prefix.

    Returns:
        str: The string representation of the tree.
    """
    tree_string = ""
    keys = list(d.keys())
    for i, key in enumerate(keys):
        is_last = (i == len(keys) - 1)
        if isinstance(d[key], dict):
            tree_string += prefix + ('└── ' if is_last else '├── ') + str(key) + "\n"
            new_prefix = prefix + ('    ' if is_last else '│   ')
            tree_string += build_tree_string(d[key], new_prefix)
        else:
            tree_string += prefix + ('└── ' if is_last else '├── ') + str(key) + "\n"
            tree_string += prefix + ('    ' if is_last else '│   ') + '└── ' + str(d[key]) + "\n"
    return tree_string

def export_report(output, comparison_groups, path):
        file_name = os.path.splitext(os.path.basename(path))[0]
        file_name_wex = os.path.basename(path)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%I:%M:%S%p")
        dynamic_file_name = f"{file_name}-{current_time}.txt"

        # Open the file in write mode with the dynamic file name
        with open(dynamic_file_name, 'w') as file:

            file.write(f"Dataset: {file_name}\n")
            file.write(f"Comparison Groups: {comparison_groups}")
            file.write("\n\n")  # Add some space between sections

            file.write("================================================ FLAG REPORT ================================================\n")
            file.write(output['flag_report'])
            file.write("\n\n")  # Add some space between sections

            file.write("================================================ SUMMARY REPORT================================================\n")
            file.write(output['summary_report'])
            file.write("\n\n")  # Add some space between sections

            file.write("================================================ DETAILED REPORT ================================================\n")
            file.write(output['detailed_report'])
            file.write("\n")  # No need for extra space at the end

        logging.info(f'\nReport has been written to {dynamic_file_name} and stored at {os.getcwd()}')

        #print(f"Report has been written to {dynamic_file_name}.")