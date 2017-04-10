import numpy as np
import pandas as pd


def one_hot_encode_categorical_columns(df, max_n_categories):
    categorical_column_names = df.columns[df.dtypes == np.dtype('object')]
    for column_name in categorical_column_names:
        n_categories = len(set(df[column_name]))
        encoded = pd.get_dummies(df[column_name])
        del df[column_name]
        if n_categories > max_n_categories:
            print("%s: %d categories: dropping" % (column_name, n_categories))
            continue
        else:
            df = pd.concat([df, encoded], axis=1)
            print("%s: %d categories" % (column_name, n_categories))

    return df


def remove_rows_with_nulls_in_mostly_non_null_columns(df, max_nulls):
    column_null_counts = df.isnull().apply(sum, axis=0)
    columns_with_few_nulls = column_null_counts[column_null_counts <= max_nulls].index
    rows_with_nulls = (df.ix[:, columns_with_few_nulls].isnull()
                       .apply(sum, axis=1) > 0)
    return df.ix[~rows_with_nulls, :]


def get_X_y(df, y_column):
    if y_column in df:
        y = np.array(df[y_column])
        df = df.drop(y_column, axis=1)
        X = np.array(df)
        n, d = df.shape
        y = y.reshape((n, 1))
    else:
        X = np.array(df)
        y = None

    return X, y
