from data_processing import datafetcher
from datetime import datetime
import json

# PHASE 1: FETCH HISTORICAL DATA
with open("env.json", "r") as file:
    env = json.load(file)
yearNow = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]

datafetcher.getDataLoop(datetime(yearNow - 21, 1, 1), datetime(yearNow, 4, 1), instrument, granularity)