"""
Description: Script which serves batches of the dataset
--input_path : Path to sqlite3 database of archived data
"""

import argparse
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

import time

# Sampletime in miliseconds
SAMPLETIME_MS = 300

# Unix time of first sample
BEGINNING_OF_TIME = 1273728900

# Unix time of last sample
END_OF_TIME = 1469607600

# Amount of expected samples in the database after adjustment
EXPECTED_SAMPLES = 652929

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

    # Fix missing values
    fix_missing_values(conn)

    #get_data_by_daterange(conn, BEGINNING_OF_TIME, 24)


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
    cur.execute("select datetime(dateTime, 'unixepoch') as date from archive "
                "order by dateTime asc")

    # Get data from database
    data = cur.fetchall()

    # Transform from unicode to string
    data = [d[0].encode('ascii', 'ignore') for d in data]

    digits = [int(d[15]) for d in data]

    # Plot histogram
    plt.hist(np.array(digits), bins=np.arange(0, 10), log=True)
    plt.title("Histogram of last digit values")
    plt.xlabel('Digit value')
    plt.ylabel('Log frequency')
    plt.show()

def fix_missing_values(conn):
    """
    Fix missing values from database.
    Logs start at the BEGINNING_OF_TIME and are logged every 5 minutes.
    Delete Any logged value that has a time that does not end in 0 or 5 (
    there are about 0.01% of them )
    Interpolate values so that there is a value every 5 minutes from the
    beginning of time until the end of time.

    Parameters
    ----------
    conn: obj, database conn
    -------

    """

    cur = conn.cursor()
    cur.execute("select datetime(dateTime, 'unixepoch') as date from archive "
                "order by dateTime asc")

    # Fetch data in list form
    data = cur.fetchall()

    # Transform from unicode to string
    unwanted_dates = []

    # Find all dates that do not end with 0 or 5 and add them to list
    for d in data:
        date = d[0].encode('ascii', 'ignore')
        last_digit = int(date[15])
        if (last_digit != 0 and last_digit != 5):
            unwanted_dates.append(date)

    # Create all missing values with first order hold interpolation


    # Remove unwanted entries from the database

    # Add the missing values to the database

    # Check that the count of elements from beginning to end of time is
    # correct
    cur.execute("select count(*) from archive")

    # Get the actual count of elements
    data = cur.fetchall()

    print 'Amount of elements in the database: {}'.format(data)
    print 'Theoretical amount of elements that should be in the ' \
          'database: {}'.format(EXPECTED_SAMPLES)

    assert EXPECTED_SAMPLES == data


def get_data_by_daterange(conn, datefrom, hours):
    """
    Get values for a time period given in hours from a starting date given by
    unix time
    Parameters
    ----------
    conn: obj, dataset conn
    datefrom: int, date in UNIX time
    hours: int, hours of data that is required (1 hour = 12 samples)

    Returns: list of data entry tuples
    -------

    """

    cur = conn.cursor()
    cur.execute('select * from archive where datetime between {} and {} '
                'order by dateTime asc'.format(datefrom,
                                               datefrom +
                                               hours * 3600 -
                                               SAMPLETIME_MS))

    # Fetch data in list form
    data = cur.fetchall()

    # Use only first 20 values, the rest are Nones in this archive
    data = [d[:20] for d in data]

    return data

if __name__ == '__main__':
    main()