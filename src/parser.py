import pandas as pd
import pdb

# expected 1-line file w/ comma-separated strings
def parse_header(filename):
    with open(filename, 'r') as f:
        line = f.readline().strip('\n')
        return line.split(',')

def parse_data(filename, headers=None):
    if headers:
        data = pd.read_csv(filename, names=headers)
    else:
        data = pd.read_csv(filename, header=None)
    restrs = data.values[0]
    include_cols = [data.columns[c] for c in range(0, len(restrs)) if restrs[c] == 1]
    return data.filter(items=include_cols).drop([0])

