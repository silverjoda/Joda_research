"""
Description: Script which serves batches of the dataset
--input_path : Path to sqlite3 database of archived data
"""

import argparse
import sqlite3
import unicodedata

import numpy as np

import matplotlib.pyplot as plt


def main():

    # Make input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, type=str,
                        help='string: path to sqlite3 database contained '
                             'archived data')

    # Parse arguments
    args = parser.parse_args()

    # Open database
    conn = sqlite3.connect(args.input_path)

    # Test database
    #testdb(conn)

    #get_data_by_daterange(conn, 0, 0)


def testdb(conn):
    """
    Test the database to a)see how big the gaps of missing data are
    b)see if the weather is being logged by delta time intervals or by absolute
    time intervals

    Parameters
    ----------
    conn: obj, database conn
    -------

    """
    cur = conn.cursor()
    cur.execute('select datetime as date from archive order by dateTime asc')


    # Get data from database
    data = cur.fetchall()
    data = [d[0] for d in data]

    # Get time deltas between each two points
    deltas = []

    # Initial time value
    begginning_of_time = 1273728600
    current_time = begginning_of_time

    # Go over the whole list
    for d in data:
        # Get time delta
        deltas.append(d - current_time)
        current_time = d

    # Plot histogram
    plt.hist(np.array(deltas), bins=np.arange(1,10000,10), log=True)
    plt.title("Histogram of deltas between unix time values")
    plt.xlabel('Time(ms)')
    plt.ylabel('Log frequency')
    plt.show()


    # Look at end of date
    cur = conn.cursor()
    cur.execute("select datetime(dateTime, 'unixepoch') as date from archive "
                "order by dateTime asc")

    # Get data from database
    data = cur.fetchall()

    # Transform from unicode to string
    data = [d[0].encode('ascii', 'ignore') for d in data]

    digits = [int(d[15]) for d in data]

    # Plot histogram
    plt.hist(np.array(digits), bins=np.arange(1, 10), log=True)
    plt.title("Histogram of last digit values")
    plt.xlabel('Digit value')
    plt.ylabel('Log frequency')
    plt.show()

def fix_missing_values(data, n):
    pass

def get_data_by_daterange(conn, datefrom, hours):
    cur = conn.cursor()
    cur.execute('select * from archive')

    data = cur.fetchmany(1000000)

    print len(data)

    return data

if __name__ == '__main__':
    main()