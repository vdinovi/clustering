from parser import parse_header, parse_data
from argparse import ArgumentParser
from os import path
from datetime import datetime
import json
import pdb

def generate(data):
    return {
        'type': 'root',
        'height': 5.0,
        'nodes': [
            {
                'type': 'leaf',
                'height': 0,
                'data': 1.0
            },
            {
                'type': 'leaf',
                'height': 0,
                'data': 3.0
            }
        ]
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="A heirarchical clustering program using the agglomeration method.")
    parser.add_argument("filename", help="Name of input file in csv form. Note that the first line of this file must be a restrictions vector.")
    parser.add_argument("--threshold", help="Specify a threshold value for which to stop agglomeration. By default, will produce the full cluster heirachy")
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: [SINGLE, COMPLETE, AVERAGE, WARD]. By default SINGLE.")
    parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
    args = parser.parse_args()
    headers = None
    if args.headers:
        headers = parse_header(args.headers)
    data = parse_data(args.filename, headers)

    root = generate(data)

    timestamp = "_" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
    tree_filename = path.basename(args.filename).split('.')[0] + timestamp + ".json"
    with open(tree_filename, 'w') as file:
        file.write(json.dumps(root, sort_keys=True, indent=4, separators=(',', ': ')))


