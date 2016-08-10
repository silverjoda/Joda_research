
import json
import os
import pprint

# Path to the seed data provided by rito. In total 1k matches spread out over
# 10 json files
SEED_DATA_PATH = '/home/shagas/Data/SW/API/rito/static_seed_data/'

def load_json(filename):

    with open(filename) as data_file:
        data = json.load(data_file)

        return byteify(data)


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


def main():
    # Complete filename of first file
    filename = os.path.join(SEED_DATA_PATH, 'matches1.json')

    # Load file
    test_data = load_json(filename)

    pprint.pprint(test_data['matches'][0])


if __name__ == "__main__":
    main()


