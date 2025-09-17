import pandas as pd
import re
import os

def print_aligned_rows(rows, columns, col_widths):
    spacing = 4  # Number of spaces between columns
    header = (" " * spacing).join([col.ljust(col_widths[col]) for col in columns])
    print(header)
    print("-" * len(header))
    for row in rows:
        row_str = (" " * spacing).join([str(row[col]).ljust(col_widths[col]) for col in columns])
        print(row_str)

if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), '../../dataset/all_toxin_data_fixed.csv')
    df = pd.read_csv(dataset)
    columns = list(df.columns)
    col_widths = {col: max(len(col), 10) for col in columns}
    # Print the first 5 rows as a reference point
    first_five_rows = [{col: str(df.iloc[i][col])[:10] for col in columns} for i in range(min(5, len(df)))]
    print("Reference rows (first 5 rows):")
    print_aligned_rows(first_five_rows, columns, col_widths)
    print()  # Blank line after reference rows

    print("First row with invalid first, second, or third column values (including previous and next rows):")
    for idx in range(len(df)):
        row = df.iloc[idx]
        first_col = str(row[columns[0]])
        second_col = str(row[columns[1]])
        third_col = str(row[columns[2]])
        valid_first = first_col.isdigit()
        valid_second = second_col.isdigit()
        valid_third = bool(re.match(r'^T3D\d{4}$', third_col))
        if not (valid_first and valid_second and valid_third):
            rows_to_print = []
            if idx > 0:
                prev_row = {col: str(df.iloc[idx-1][col])[:10] for col in columns}
                rows_to_print.append(prev_row)
            curr_row = {col: str(row[col])[:10] for col in columns}
            rows_to_print.append(curr_row)
            if idx < len(df) - 1:
                next_row = {col: str(df.iloc[idx+1][col])[:10] for col in columns}
                rows_to_print.append(next_row)
            print_aligned_rows(rows_to_print, columns, col_widths)
            break
