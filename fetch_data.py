from data_processing import datafetcher
from datetime import datetime

YEAR_NOW = 2026
INSTRUMENT = "EUR_USD"
GRANULARITY = "M5"

datafetcher.getDataLoop(datetime(2010, 1, 1), datetime(YEAR_NOW, 5, 1), INSTRUMENT, GRANULARITY)