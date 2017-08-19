import pandas as pd
import numpy as np

from batch_reader import BatchReader

class RepslyData(BatchReader):
    def __init__(self):
        pass

    def _prepare_data(self, **params):
        # the only param is input file name so it is passed directly as a string
        file_name = params['file_name']

        # load the dataset, create data frame from csv with all columns
        df = pd.read_csv(file_name)

        # check that input file is formatted as expected
        expected_columns = ['UserID', 'Purchased', 'Edition', 'TrialStarted', 'TrialDate', 'ActiveReps',
                            'ActivitiesPerRep', 'MessagesCnt', 'AuditsCnt', 'ClientNotesCnt', 'FormsCnt',
                            'NewPlaceCnt', 'OrdersCnt', 'PhotosCnt', 'StatusChangedCnt', 'WorkdayStartCnt',
                            'ScheduleCnt', 'ScheduledRepsCnt', 'ScheduledPlacesCnt', 'ImportCnt']
        assert np.array_equal(df.columns, expected_columns)

        # create index for columns that need to be casted to 15 trial days
        col_index = [0] + list(range(4, len(df.columns)))

        # cast the data columns into new X_all data frame
        trial_data = df.iloc[:, col_index].pivot(index='UserID', columns='TrialDate')
        # flatten the columns index because of the concat operation bellow
        trial_data.columns = range(len(trial_data.columns))

        # all other columns are grouped by UserID
        group = df.iloc[:, :4].groupby('UserID')

        # extract TrialStarted columns and convert it to datetime
        trial_started = group.TrialStarted.unique().str[0]
        trial_started = pd.to_datetime(trial_started)
        # subtract TrialStarted from first date
        first_date = pd.datetime(2016, 1, 1)
        trial_started = (trial_started - first_date).dt.days

        # extract purchased from the group
        purchased = group.Purchased.unique().str[0]

        # concatenate everything into one table indexed by UserID and case all data into floats
        data = pd.concat([trial_started, trial_data, purchased], axis=1).astype(float)

        # finally, split into X and y and convert into numpy array
        all_X, all_y = data.iloc[:, :-1].values, data.iloc[:, -1].values

        return all_X, all_y
