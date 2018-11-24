__author__ = 'David T. Pocock'


from scipy.io import arff
import pandas as pd


class Parser:

    def __init__(self, path):
        self.path = path

    def parse_data(self):
        data, meta = arff.loadarff(self.path)
        df = pd.DataFrame(data)
        return df
