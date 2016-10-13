"""
Description: Script which serves batches of the dataset
--input_path : Path to sqlite3 database of archived data
"""

import argparse
import os
import random
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

# Sampletime in seconds
SAMPLETIME_S = 300

# Amount of seconds in one day
S_IN_DAY = 24 * 3600

# Unix time of first sample, Friday, 14 may 2010
BEGINNING_OF_TIME = 1273795200

# Unix time of last sample
END_OF_TIME = 1469577600

END_OF_TRAINING = 1469577600 - (3600 * 24 * 64)

# Amount of expected samples in the database after adjustment
EXPECTED_SAMPLES = 1 + (END_OF_TIME - BEGINNING_OF_TIME) / SAMPLETIME_S

# Visualize some database info
VISU_DATABASE = False
PRINT_MOMENTS = False

# Training to validation ratio split
T_V_RATIO = 0.9

N_DAYS_AVAILABLE = (END_OF_TIME - BEGINNING_OF_TIME) / S_IN_DAY

# Training to validation split date
TRAINING_VALIDATION_SPLIT_DATE = int(BEGINNING_OF_TIME + int(N_DAYS_AVAILABLE *
                                     T_V_RATIO) * S_IN_DAY)


# Table moments:
MEANS = [50.687202, 30.042913, 28.863577, 75.225151,
                4.2221103, 173.68834, 8.4116125, 175.86951, 0.0027571502,
                0.00016730967, 42.188217, 49.430134, 50.735664]

STDS = [15.308222, 0.24120753, 0.38397667, 16.645607,
               2.793514, 107.05489, 4.978723, 106.66589, 0.072300963,
               0.0023934443, 12.016179, 16.617676, 15.394153]

TOTAL_COLUMNS = 'outTemp,barometer,pressure,outHumidity,windSpeed,' \
               'windDir,windGust,windGustDir,rainRate,rain,dewpoint, ' \
               'windchill,heatindex'

USED_COLUMNS = 'outTemp,barometer,pressure'#,outHumidity,windSpeed'


DATASET_MEAN = [50.687202, 30.042913, 28.863577]#, 75.225151, 4.2221103]

DATASET_STD = [15.308222, 0.24120753, 0.38397667]#, 16.645607, 2.793514]

# Amount of input variables
INPUT_DIM = len(USED_COLUMNS.split(','))

class DatasetServer:
    """
    Class which acts as a data provider
    """

    def __init__(self, database_path):

        # Open database
        self.conn = sqlite3.connect(
            os.path.join(database_path, 'archive.db'))

        self.batch_counter = 0

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

        # Replace all empty entries with zeros
        for c in USED_COLUMNS.split(','):
            cur.execute("update archive set {} = IFNULL({}, '0')".format(c, c))

        # Check that first element begins at beginning of time
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

        return delta % SAMPLETIME_S == 0

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

        # Amount of nans
        n_nans = 0

        # Create all missing values with first order hold interpolation
        for j,(dp,dc) in enumerate(zip(data[:-1],data[1:])):

            # Calculate time difference between current sample and previous
            delta = dc[0] - dp[0]
            # If this does not pass that means that the dataset might contain
            # values which were logged at invalid times
            assert delta % SAMPLETIME_S == 0

            # If value delta is greater than 300 then find how many values fit in
            if delta > SAMPLETIME_S:
                n_missing_samples = (delta / SAMPLETIME_S) - 1

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

        print '{} nans have been replaced'.format(n_nans)

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

        print '{} invalid values have been deleted from the database'. \
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

        print '{} interpolated values have been inserted into the database'.\
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

        # Declare lists for average and std
        mean_list = []
        std_list = []

        for c in USED_COLUMNS.split(','):

            # Get all data
            cur.execute('select {} from archive'.format(c))

            # Fetch data in list form
            data = cur.fetchall()

            # Turn to np.array
            data = np.array(data, dtype=np.float32)

            assert np.sum(np.isnan(data)) == 0, 'Nans found in data'

            # Get moments
            mean_list.append(np.mean(data, axis=(0))[0])
            std_list.append(np.std(data, axis=(0))[0])

        print mean_list
        print std_list
        exit()

    def get_data_by_daterange(self, datefrom, sample_length):
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
        cur.execute('select {} from archive where datetime between {} and {} '
                    'order by dateTime asc'.format(USED_COLUMNS,
                                                   datefrom,
                                                   datefrom +
                                                   sample_length * SAMPLETIME_S -
                                                   SAMPLETIME_S))

        # Fetch data in list form
        data = cur.fetchall()

        return data

    def get_training_sequence_batch(self, offset_hours, sample_length, batchsize):

        # Amount of offset in seconds
        offset = offset_hours * 3600

        # List which will hold the batch
        batch = []

        # Get values from database
        raw_values = self.get_data_by_daterange(self.batch_counter *
                                                sample_length * SAMPLETIME_S,
                                                sample_length)

        if self.batch_counter < (END_OF_TIME - sample_length * SAMPLETIME_S):
            # Increment batch counter
            self.batch_counter += 1
        else:
            self.batch_counter = 0



        # Process values (normalize, etc)
        normalized_data = [(np.array(d) - np.array(DATASET_MEAN)) /
                           np.array(DATASET_STD) for d in raw_values]

        # Append to our batch
        batch.append(normalized_data)

        assert len(batch[0]) == sample_length * SAMPLETIME_S

        return np.array(batch, dtype=np.float32)

    def get_training_batch(self, offset_hours, sample_length, batchsize):
        """

        Parameters
        ----------
        offset_hours: int, hour offset from midnight
        size: int, batch size

        Returns np array batch of 24 hour sequences
        -------

        """
        # Amount of offset in seconds
        if offset_hours == -1:
            offset = np.random.randint(0,24) * 3600
        else:
            offset = offset_hours * 3600

        # Vector of available values
        available_values = range(BEGINNING_OF_TIME + offset,
                                 TRAINING_VALIDATION_SPLIT_DATE,
                                 S_IN_DAY)

        # Select batchsize random days
        random_vals = random.sample(available_values, batchsize)

        # List which will hold the batch
        batch = []

        # Get values
        for val in random_vals:

            # Get values from database
            raw_values = self.get_data_by_daterange(val, sample_length)

            # Process values (normalize, etc)
            normalized_data = [(np.array(d) - np.array(DATASET_MEAN)) /
                               np.array(DATASET_STD) for d in raw_values]

            # Append to our batch
            batch.append(normalized_data)

        assert len(batch[0]) == sample_length

        return np.array(batch, dtype=np.float32)

    def get_validation_batch(self, hour, sample_length, batchsize):
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
        available_values = range(TRAINING_VALIDATION_SPLIT_DATE + offset,
                                 END_OF_TIME - S_IN_DAY * 2,
                                 S_IN_DAY)

        # Select batchsize random days
        random_vals = random.sample(available_values, batchsize)

        # List which will hold the batch
        batch = []

        # Get values
        for val in random_vals:
            # Get values from database
            raw_values = self.get_data_by_daterange(val, sample_length)

            # Process values (normalize, etc)
            normalized_data = [(np.array(d) - np.array(DATASET_MEAN)) /
                               np.array(DATASET_STD) for d in raw_values]

            # Append to our batch
            batch.append(normalized_data)


        assert len(batch[0]) == sample_length

        return np.array(batch, dtype=np.float32)

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

    # Make database object
    server = DatasetServer(args.input_path)

    # Print dataset statistics
    if PRINT_MOMENTS:
        server._print_avg_and_std_of_dataset()

    # Test database
    if VISU_DATABASE:
        server._database_info()

    # Perform some initial tasks (delete everything before beginning of time)
    server._init_checks_on_db()

    # Fix missing values
    server._fix_missing_values()




if __name__ == '__main__':
    main()