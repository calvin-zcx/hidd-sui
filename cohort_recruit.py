from collections import defaultdict
from datetime import datetime
import pickle
from utils import str_to_datetime
import pandas as pd


def exclude(prescription_taken_by_patient, patient_dates, eligibility_criteria):
    # # interval, followup, baseline, min_prescription):  # patient_1stDX_date, patient_start_date
    # # Input: drug --> patient --> [(date, supply day),]     patient_dates: patient --> [birth_date, other dates]
    # # Output: save_prescription: drug --> patients --> [date1, date2, ...] sorted
    # #         save_patient: patient --> drugs --> [date1, date2, ...] sorted
    # # print('exclude... interval:{}, followup:{}, baseline:{}, min_prescription:{}'.format(
    # #     interval, followup, baseline, min_prescription))
    # print('eligibility_criteria:\n', eligibility_criteria)
    # prescription_taken_by_patient_exclude = defaultdict(dict)
    # # patient_take_prescription_exclude = defaultdict(dict)
    # patient_take_prescription = defaultdict(dict)
    #
    # for drug, taken_by_patient in prescription_taken_by_patient.items():
    #     for patient, take_times in taken_by_patient.items():
    #         # no need for both if date and days, days are not '', they have int value, confused with value 0
    #         # actually no need for date, date is not '' according to the empirical data, but add this is OK, more rigid
    #         dates = [str_to_datetime(date) for (date, days) in take_times if date != '']  # and days datetime.strptime(date, '%m/%d/%Y')
    #         dates = sorted(dates)
    #         dates_days = {str_to_datetime(date): int(days) for (date,
    pass