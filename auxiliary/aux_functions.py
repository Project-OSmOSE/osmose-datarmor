import matplotlib.pyplot as plt

import pandas as pd
import glob
import os
import numpy as np
import sys
import numpy as np
import os
import glob
import pickle
import shutil
import datetime

def custom_date_range(st,ed,freq,closed_opt=None):

#     print(st)
#     print(ed)
    time_periods = pd.date_range(start=st, end=ed, freq=freq,closed = closed_opt)

    if time_periods[-1] > pd.to_datetime(ed, utc=True):
        ed_new = pd.to_datetime(ed, utc=True) + datetime.timedelta(seconds=(time_periods[1] - time_periods[0]).total_seconds())
        time_periods = pd.date_range(start=st, end=ed_new, freq=freq)

    # time_periods = time_periods - (time_periods[0] - pd.to_datetime(st, utc=True))

    return time_periods
