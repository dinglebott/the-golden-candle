from data_processing import datafetcher
from datetime import datetime

YEAR_NOW = 2026
INSTRUMENT = "EUR_USD"
GRANULARITY = "H1"

datafetcher.getDataLoop(datetime(YEAR_NOW - 21, 1, 1), datetime(YEAR_NOW, 4, 1), INSTRUMENT, GRANULARITY)