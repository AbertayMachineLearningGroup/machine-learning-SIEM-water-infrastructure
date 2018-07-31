import sys
import csv
import os
from datetime import datetime

class DataSetFile:
    is_anomaly = 'yes'
    affected_component = 'None'
    situation = 'None'
    combined_affected_component = ''
    combined_situation = ''
    operational_scenario = ''
    fileName = ''

    def __init__(self, fileName, isAormaly, affectedComponent, situation, operationalScenario, combinedAffectedComponent = '', combinedSituation = ''):
        self.fileName = fileName

        self.is_anomaly = isAormaly
        self.affected_component = affectedComponent
        self.situation = situation
        self.operational_scenario = operationalScenario
        
        if combinedAffectedComponent != '':
            self.combined_affected_component = combinedAffectedComponent
        else: 
            self.combined_affected_component = affectedComponent
            
        if combinedSituation != '':
            self.combined_situation = combinedSituation
        else:
            self.combined_situation = situation
            
def main(logsPath):
    print('Start Pre-processing')
    print('Reading data from: ', logsPath)
    
    # Add dataset files
    records = []
    records.append(DataSetFile('normal_short.csv', 
                               'No',
                               'None', 
                               'Normal',
                               'Normal'))
    
    records.append(DataSetFile('plastic_bag.csv',
                               'Yes', 
                               'Untrasounder Sensor', 
                               'Plastic bag',
                               'Accident/Sabotage'))
    
    records.append(DataSetFile('blocked_1.csv',
                               'Yes',
                               'Untrasounder Sensor', 
                               'Blocked measure 1',
                               'Breakdown/Sabotage',
                               '',
                               'Blocked measure'))
    
    records.append(DataSetFile('blocked_2.csv', 
                               'Yes', 
                               'Untrasounder Sensor', 
                               'Blocked measure 2',
                               'Breakdown/Sabotage',
                               '', 
                               'Blocked measure'))
    
    records.append(DataSetFile('poly_2.csv',
                               'Yes',
                               'Untrasounder Sensor', 
                               '2 Floating objects', 
                               'Accident/Sabotage',
                               '', 
                               'Floating objects'))
    
    records.append(DataSetFile('poly_7.csv', 
                               'Yes', 
                               'Untrasounder Sensor',
                               '7 Floating objects', 
                               'Accident/Sabotage',
                               '',
                               'Floating objects'))
    
    records.append(DataSetFile('wet_sensor.csv', 
                               'Yes',
                               'Untrasounder Sensor', 
                               'Humidity',
                               'Breakdown'))

    records.append(DataSetFile('high_blocked.csv',
                               'Yes',
                               'Discrete Sensor 1',
                               'Sensor Failure', 
                               'Breakdown',
                               'Discrete Sensor', 
                               'Sensor Failure'))

    records.append(DataSetFile('second_blocked.csv', 
                               'Yes',
                               'Discrete Sensor 2',
                               'Sensor Failure', 
                               'Breakdown',
                               'Discrete Sensor',
                               'Sensor Failure'))

    records.append(DataSetFile('DoS_attack.csv',
                               'Yes',
                               'Network', 
                               'DoS',
                               'Cyber-attack'))
    
    records.append(DataSetFile('spoofing.csv',
                               'Yes', 
                               'Network', 
                               'Spoofing',
                               'Cyber-attack'))
    
    records.append(DataSetFile('bad_conection.csv', 
                               'Yes',
                               'Network', 
                               'Wrong connection',
                               'Breakdown/Sabotage'))

    records.append(DataSetFile('hits_1.csv', 
                               'Yes', 
                               'Whole',
                               'Person htting low intensity',
                               'Sabotage',
                               '', 
                               'Person hitting'))
    
    records.append(DataSetFile('hits_2.csv',
                               'Yes',
                               'Whole', 
                               'Person htting med intensity', 
                               'Sabotage',
                               '', 
                               'Person hitting'))
    
    records.append(DataSetFile('hits_3.csv',
                               'Yes',
                               'Whole', 
                               'Person htting high intensity',
                               'Sabotage',
                               '', 
                               'Person hitting'))

    global filewriterCSV
    
    datasetCominedFileCSV = open('dataset_processed.csv', 'w')
   
    filewriterCSV = csv.writer(datasetCominedFileCSV, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    filewriterCSV.writerow(
       ['IN3', 'IN2', 'IN1', 'IN0', 'PP', 'PG', 'VP', 'VG', 'R4', 'Slope', 'Anomaly', 'AffectedComponent', 'Situation', 'OperationalScenario', 'CombinedAffectedComponent', 'CombinedSituation'])

    for x in records:
        read_file_and_write_rows(logsPath + '/' + x.fileName, x.is_anomaly, x.affected_component, x.situation, x.operational_scenario, x.combined_affected_component, x.combined_situation)

    datasetCominedFileCSV.close()

    print('End PreProcessing')

def read_file_and_write_rows(fileFullPath, isAnomaly, affectedComponent, situation, operationalScenario, combinedAffectedComponent, combinedSituation):
    threshold_of_records = 2000  #-1 for no threshold
    
    entries_count = 0
    with open(fileFullPath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        cnt = 0
        dates_filled = 0
        
        dates = [''] * 10
        reg_4 = [0] * 10
        
        formattedRow = [''] * 10
        formattedRow.extend([isAnomaly, affectedComponent, situation, operationalScenario, combinedAffectedComponent, combinedSituation])
        
        for row in reader:                
            if row[1] == "2":
                value = format(int(row[2]), '08b')
                formattedRow[0] = value[4]
                formattedRow[1] = value[5]
                formattedRow[2] = value[6]
                formattedRow[3] = value[7]
                
            elif row[1] == "3":
                value = format(int(row[2]), '08b')
                formattedRow[4] = value[0]
                formattedRow[5] = value[1]
                formattedRow[6] = value[5]
                formattedRow[7] = value[4]
                
            elif row[1] == "4":
                formattedRow[8] = int(row[2])
                if dates_filled == 0:
                    dates[cnt] = row[0]
                    reg_4[cnt] = int(row[2])
                    cnt += 1
                    if cnt == 10:
                        cnt = 0
                        dates_filled = 1
                else:
                    date1 = datetime.strptime(dates[cnt],'%m/%d/%Y %H:%M:%S.%f ')
                    date2 = datetime.strptime(row[0],'%m/%d/%Y %H:%M:%S.%f ')
                    diff = (date2-date1).microseconds / 1000
                    formattedRow[9] = (int(row[2]) - reg_4[cnt])/diff
                    dates[cnt] = row[0]
                    reg_4[cnt] = int(row[2])
                
                    cnt += 1
                    if cnt == 10:
                        cnt = 0
                
                    filewriterCSV.writerow(formattedRow)
                    formattedRow = [''] * 10
                    formattedRow.extend([isAnomaly, affectedComponent, situation, operationalScenario, combinedAffectedComponent, combinedSituation])
                    entries_count += 1
                    if threshold_of_records != -1 and entries_count > threshold_of_records:
                        break
                
        print(fileFullPath)
        print(entries_count)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd() + '/logs'
        
    main(path)
