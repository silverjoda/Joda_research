"""
Description: Script which serves batches of the dataset
--input_path : Path to sqlite3 database of archived data
"""

import argparse
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import random
import pprint

# Sampletime in miliseconds
SAMPLETIME_MS = 300

# Unix time of first sample
BEGINNING_OF_TIME = 1273728900

# Unix time of last sample
END_OF_TIME = 1469607600

# Amount of expected samples in the database after adjustment
EXPECTED_SAMPLES = 1 + (END_OF_TIME - BEGINNING_OF_TIME) / SAMPLETIME_MS

# Visualize some database info
VISU_DATABASE = False
PRINT_MOMENTS = False

# Training to validation ratio split
T_V_RATIO = 0.85

# Training to validation split date
TRAINING_VALIDATION_SPLIT_DATE = int(BEGINNING_OF_TIME +
                                    (END_OF_TIME - BEGINNING_OF_TIME) *
                                     T_V_RATIO)

# Amount of seconds in one day
MS_IN_DAY = 24 * 3600

# Amount of hours of each element
BATCH_LENGTH_HOURS = 24


# Table moments:
MEAN = [1371556352.000, 1.000, 5.000, 30.025, 28.871, 30.023, 75.938, 50.707,
        25.130, 75.224, 4.224, 176.058, 8.413, 178.360, 0.003, 0.000, 42.206,
        49.454, 50.756, 0.000]

STD = [56542020.000, 0.000, 0.000, 0.242, 0.384, 0.339, 5.447, 15.278, 5.190,
       16.635, 2.793, 105.779, 4.978, 105.065, 0.072, 0.002, 11.989, 16.581,
       15.361, 0.002]

# Indeces of valid data elements from the sql table
RELEVANT_TABLE_COLUMNS = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

RELEVANT_MEAN = np.array([MEAN[i] for i in RELEVANT_TABLE_COLUMNS])
RELEVANT_STD = np.array([MEAN[i] for i in RELEVANT_TABLE_COLUMNS])

class DatasetServer:

    def __init__(self):

        # Make input argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', required=True, type=str,
                            help='string: path to sqlite3 database contained '
                                 'archived data')

        # Parse arguments
        args = parser.parse_args()

        # Open database
        self.conn = sqlite3.connect(args.input_path)

    def _init_checks_on_db(self):
        """
        Perform initial checks, delete all elements that date before the defined
        beginning of time

        Parameters
        ----------
        conn: obj, database connection
        -------

        """
        cur = self.conn.cursor()
        cur.execute('delete from archive where datetime < {}'.format(
            BEGINNING_OF_TIME))

        cur.execute('select datetime from archive where datetime = {}'.format(
            BEGINNING_OF_TIME))

        data = cur.fetchall()

        assert data[0][0] == BEGINNING_OF_TIME, 'No element in database which ' \
                                             'corresponds to the beginning of time'

    def _database_info(self):
        """
        Test the database to a)see how big the gaps of missing data are
        b)see if the weather is being logged by delta time intervals or by absolute
        time intervals

        Parameters
        ----------
        conn: obj, database conn
        -------

        """
        cur = self.conn.cursor()
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

    def _is_valid_sample_date(self, date):
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

    def _get_database_size(self):
        """
        Return amount of entries in the database
        Parameters
        ----------
        conn: obj, database connection

        Returns int, size of database
        -------

        """
        cur = self.conn.cursor()
        cur.execute("select count(*) from archive")

        # Fetch data in list form
        return cur.fetchall()[0][0]

    def _fix_missing_values(self):
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

        cur = self.conn.cursor()
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
            if not self._is_valid_sample_date(date):
                unwanted_entries.append(d)

        print 'Removing {} unwanted elements from the set'.format(
            len(unwanted_entries))

        # Remove unwanted dates from data list
        for d in unwanted_entries:
            data.remove(d)

        # Values that will be added to database
        interpolated_values = []

        # Create all missing values with first order hold interpolation
        for dp,dc in zip(data[:-1],data[1:]):

            # Calculate time difference between curremt sample and previous
            delta = dc[0] - dp[0]

            # If this does not pass that means that the dataset might contain
            # values which were logged at invalid times
            assert delta % SAMPLETIME_MS == 0

            # If value delta is greater than 300 then find how many values fit in
            if delta > SAMPLETIME_MS:
                n_missing_samples = (delta / SAMPLETIME_MS) - 1

                for n in range(n_missing_samples):

                    # Copy value
                    new_entry = list(dp)

                    # Add interpolated values
                    for i in range(len(new_entry)):
                        if dc[i] is None or dp[i] is None:
                            continue
                        value_increment = (n + 1) * ((dc[i] - dp[i]) / (
                            n_missing_samples + 1))
                        new_entry[i] += value_increment

                    # First 3 values are integers so cast them back
                    for j in range(3):
                        new_entry[j] = int(new_entry[j])

                    # Append the value
                    interpolated_values.append(new_entry)


        # Remove unwanted entries from the database
        database_size_before_deletion  = self._get_database_size()

        for u in unwanted_entries:
            cur.execute("delete from archive where datetime = {}".format(
                u[0]))

        # Check that the elements have been deleted
        assert self._get_database_size() == \
               database_size_before_deletion\
                                                - \
                                           len(unwanted_entries), 'Invalid ' \
                                                                  'elements have ' \
                                                                  'not been ' \
                                                                  'successfuly ' \
                                                                  'deleted from ' \
                                                                  'database'

        print '{} invalid values have been deleted'. \
            format(len(unwanted_entries))

        print 'Inserting {} interpolated values'. \
            format(len(interpolated_values))

        database_size_before_insertion = self._get_database_size()
        for m in interpolated_values:
            cur.execute('insert into archive values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'
                        '?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'
                        '?,?,?,?,?,?,?,?)', m)

        # Check that all elements have been inserted
        assert self._get_database_size() == \
               database_size_before_insertion + len(
            interpolated_values), 'Interpolated values have not been successfuly ' \
                                  'inserted'

        print '{} interpolated values have been inserted'.\
            format(len(interpolated_values))

        # Check that the count of elements from beginning to end of time is
        # correct
        assert EXPECTED_SAMPLES == self._get_database_size(), 'Expected amount of elements in ' \
                                              'database does not match true ' \
                                              'value: Expected: {}, true: {}'.\
            format(EXPECTED_SAMPLES, self._get_database_size())

        # Save (commit) the changes to the database
        self.conn.commit()

    def _print_avg_and_std_of_dataset(self):
        """
        print dataset statistics for normalization
        -------

        """

        cur = self.conn.cursor()

        # Get all data
        cur.execute('select * from archive')

        # Fetch data in list form
        data = cur.fetchall()

        # Get only first 20 values
        data = [d[:20] for d in data]

        # Turn to np.array
        data = np.array(data, dtype=np.float32)

        # Get moments
        mean = np.nanmean(data, axis=(0))
        std = np.nanstd(data, axis=(0))

        for m in mean:
            print "%.3f" % m

        for s in std:
            print "%.3f" % s

    def _process_raw_sql_data(self, data):
        # Get only relevant data
        relevant_data = np.array([data[i] for i in RELEVANT_TABLE_COLUMNS])

        # Normalize data
        normalized_data = (relevant_data - RELEVANT_MEAN) / RELEVANT_STD

        # Turn nans into zero
        for i in range(len(normalized_data)):
            if np.isnan(normalized_data[i]):
                normalized_data[i] = 0

        return normalized_data

    def get_data_by_daterange(self, datefrom, hours):
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

        cur = self.conn.cursor()
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

    def get_training_batch(self, hour, batchsize):
        """

        Parameters
        ----------
        hour: int, hour offset
        size: int, batch size

        Returns np array batch of 24 hour sequences
        -------

        """
        # Amount of offset in seconds
        offset = hour * 3600

        # Vector of available values
        available_values = range(BEGINNING_OF_TIME + offset,
                                 TRAINING_VALIDATION_SPLIT_DATE,
                                 MS_IN_DAY)

        # Select random value
        random_vals = random.sample(range(0,len(available_values)), batchsize)

        # List which will hold the batch
        batch = []

        # Get values
        for val in random_vals:
            # Get values from database
            raw_values = self.get_data_by_daterange(val, BATCH_LENGTH_HOURS)

            # Process values (use only neccessary values, normalize, etc)
            processed_values = self._process_raw_sql_data(raw_values)

            # Append to our batch
            batch.append(processed_values)

        assert len(batch[0]) == BATCH_LENGTH_HOURS * (3600 / SAMPLETIME_MS)

        return np.array(batch, dtype=np.float32)






def main():
    """
    Main function
    -------

    """

    # Make database object
    server = DatasetServer()

    # Test database
    if VISU_DATABASE:
        server._database_info()

    # Print dataset statistics
    if PRINT_MOMENTS:
        server._print_avg_and_std_of_dataset()

    # Perform some initial tasks (delete everything before beginning of time)
    server._init_checks_on_db()

    # Fix missing values
    server._fix_missing_values()

if __name__ == '__main__':
    main()