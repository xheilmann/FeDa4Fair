def drop_data(df, percentage, column1, value1, label_column,column2=None, value2=None):
    # Validate percentage
    if not (0 <= percentage <= 1):
        raise ValueError("Fraction must be between 0 and 1")

    # Filter matching rows based on one or two conditions
    condition = (df[column1] == value1)
    if column2 is not None and value2 is not None:
        condition &= (df[column2] == value2)

    matching_rows = df[condition & (df[label_column] == True)]


    # Determine how many rows to drop
    num_to_drop = int(len(matching_rows) * percentage)

    # Randomly sample rows to drop
    rows_to_drop = matching_rows.sample(n=num_to_drop, random_state=42).index

    # Drop them from the original DataFrame
    df_dropped = df.drop(index=rows_to_drop)

    return df_dropped


def flip_data(df, percentage, column1, value1, label_column, column2=None, value2=None):
    if not (0 <= percentage <= 1):
        raise ValueError("Fraction must be between 0 and 1")

        # Build filter condition
    condition = (df[column1] == value1)
    if column2 is not None and value2 is not None:
        condition &= (df[column2] == value2)

    # Filter rows where label is 1
    matching_rows = df[condition & (df[label_column] == True)]

    # Determine number of rows to flip
    num_to_flip = int(len(matching_rows) * percentage )

    # Sanity check
    if num_to_flip == 0:
        print("No labels flipped: fraction too low or no matching rows.")
        return df

    # Randomly choose rows to flip
    rows_to_flip = matching_rows.sample(n=num_to_flip, random_state=42).index

    # Flip the labels
    df.loc[rows_to_flip, label_column] = False

    return df
