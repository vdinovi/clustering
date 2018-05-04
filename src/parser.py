from pandas import read_csv
import re
import pdb


def parse_header(filename):
    with open(filename, 'r') as f:
        headers = []
        for line in f.readlines():
            # discard
            if not line or line[0] == "#":
                continue
        pdb.set_trace()

def parse_data(filename, headers):
    pass



import sys
if __name__ == "__main__":
    header_file = sys.argv[1]
    headers = parse_header(header_file)
    #parse_data(data_file, headers)

