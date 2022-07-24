from pandas_datareader import DataReader
from datetime import datetime, timedelta
import pandas as pd


def stock_data(stock_name_list,
               start_date = '2021-01-01',
               end_date = datetime.strftime(datetime.now() - timedelta(1),
                                            '%Y-%m-%d')
              ):

    vals = {}
    for stock_name in stock_name_list:
        try:
            stock_val = DataReader(stock_name, 
                                   data_source= 'yahoo', 
                                   start = start_date, 
                                   end = end_date)['Close'].values
            vals[stock_name] = stock_val
        except:
            print(f'{stock_name} not found!')
    df = pd.DataFrame(vals)

    return df