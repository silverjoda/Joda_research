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

PERFORM_DATABASE_CONTENT_ANALYSIS = False
FIX_MISSING_VALUES = True

def main():
    """
    Main function
    -------

    """

    # Make input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, type=str,
                        help='string: path to sqlite3 database contained '
                             'archived data')

    # Parse arguments
    args = parser.parse_args()

    # Open database
    conn = sqlite3.connect(args.input_path)

    # Perform some initial tasks (delete everything before beginning of time)
    _init_checks_on_db(conn)

    # Test database
    if PERFORM_DATABASE_CONTENT_ANALYSIS:
        _testdb(conn)

    # Fix missing values
    if FIX_MISSING_VALUES:
        _fix_missing_values(conn)

    #get_data_by_daterange(conn, BEGINNING_OF_TIME, 24)

def _init_checks_on_db(conn):
    cur = conn.cursor()
    cur.execute('delete from archive where datetime < {}'.format(
        BEGINNING_OF_TIME))

    cur.execute('select datetime from archive where datetime = {}'.format(
        BEGINNING_OF_TIME))

    data = cur.fetchall()

    assert data[0] == BEGINNING_OF_TIME, 'No element in database which ' \
                                         'corresponds to the beginning of time'



def _testdb(conn):
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

def _is_valid_sample_date(date):
    """
    Check if sample date is valid

    Parameters
    ----------
    date: int, UNIX time
    Returns: bool, validity of date
    -------

    """

    # Get the time difference
    delta = date - BEGINNING_OF_TIME

    # Sanity check
    assert delta >= 0

    return delta % SAMPLETIME_MS == 0

def _get_database_size(conn):
    cur = conn.cursor()
    cur.execute("select count(*) from archive")

    # Fetch data in list form
    return cur.fetchall()

def _fix_missing_values(conn):
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
    cur.execute("select * from archive order by datetime asc")

    # Fetch data in list form
    data = cur.fetchall()

    # Transform from unicode to string
    unwanted_entries = []

    # Find all dates that do not end with 0 or 5 and add them to list
    for d in data:

        # Date is the first entry
        date = d[0]

        # Check validity
        if not _is_valid_sample_date(date):
            unwanted_entries.append(d)

    # Remove unwanted dates from data list
    for d in unwanted_entries:
        data.remove(d)

    print 'Removing {} unwanted elements from the set'.format(
        len(unwanted_entries))

    # Values that will be added to database
    interpolated_values = []

    # Create all missing values with first order hold interpolation
    for dp,dc in zip(data[:-1],data[1:]):

        # Calculate time difference between curremt sample and previous
        delta = dp[0] - dc[0]

        # If this does not pass that means that the dataset might contain
        # values which were logged at invalid times
        assert delta % SAMPLETIME_MS == 0

        # If value delta is greater than 300 then find how many values fit in
        if delta > SAMPLETIME_MS:
            n_missing_samples = (delta / SAMPLETIME_MS) - 1

            for n in range(n_missing_samples):

                # Copy value
                new_entry = list(dp[0])

                # Add interpolated values
                for i in range(len(new_entry)):
                    value_increment =  (i + 1) * ((dc[i] - dp[i]) / (
                        n_missing_samples + 1))
                    new_entry[i] += value_increment

                # First 3 values are integers so cast them back
                for j in range(3):
                    new_entry[j] = int(new_entry[j])

                # Append the value
                interpolated_values.append(new_entry)


    # Remove unwanted entries from the database
    database_size_before_deletion  = _get_database_size(conn)

    for u in unwanted_entries:
        cur.execute("delete from archive where datetime = {}".format(
            unwanted_entries[0]))

    # Check that the elements have been deleted
    assert _get_database_size(conn) == database_size_before_deletion - \
                                       len(unwanted_entries)

    # Add the missing values to the database
    database_size_before_insertion = _get_database_size(conn)
    for m in interpolated_values:
        cur.execute("insert into archive values {}".format(
            unwanted_entries[0]))

    # Check that all elements have been inserted
    assert _get_database_size(conn) == database_size_before_insertion + len(
        interpolated_values)

    # Check that the count of elements from beginning to end of time is
    # correct
    cur.execute("select count(*) from archive")

    # Get the actual count of elements
    data = cur.fetchall()

    print 'Amount of elements in the database: {}'.format(data)
    print 'Theoretical amount of elements that should be in the ' \
          'database: {}'.format(EXPECTED_SAMPLES)

    assert EXPECTED_SAMPLES == len(data)


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