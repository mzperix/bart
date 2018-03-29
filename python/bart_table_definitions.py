### BART datajoint tables

import datajoint as dj


IMPORT_DATA_TABLE_PATH = "../_data/" 
FORMAT_EPRIME_TXT = 'eprime_txt'
FORMAT_PRESENTATION = 'presentation'
FORMAT_CSV = 'csv'

FORMAT_EXTENSION = dict(csv = '.csv',
                        eprime_txt = '.txt')

def Query_column_name(df, colname):
    print('Which column contains:  ',colname,'  ?')
    col_chosen = input()
    column_names = list(df.columns.values)
    while (col_chosen not in column_names):
        print('Incorrect column name, please try again.')
        col_chosen = input()
    logging.info('Column chosen by user for: '+colname+'  value: '+col_chosen)
    return(col_chosen)


def GetDataJointConfig():
    host = input('Database host address (e.g. local, tauri, tauri local): ')
    if (host == 'tauri'):
        dj.config['database.host'] = 'tauri:3306'
    elif(host == 'tauri local'):
        dj.config['database.host'] = '172.17.0.2:3306'
    elif(host == 'local'):
        dj.config['database.host'] = '127.0.0.1:3306'
    else:
        dj.config['database.host'] = host
    dj.config['database.user'] = input('Datajoint username: ')
    dj.config['database.password'] = getpass.getpass('Password: ')

# CONNECT TO DB
if (dj.config['database.user'] == None):    
    connection_ok = False
    while connection_ok != True:
        try:
            GetDataJointConfig()
            schema = dj.schema('bart', locals())
            connection_ok = True
            print('Connection OK.')
        except:
            print('Connection cannot be made.')
            connection_ok = False


class FileParser:
    def ReadCols(filename, file_format, cols = None):
        print('Reading file: '+filename)
        logging.info('Reading file: '+filename)
        if (file_format == FORMAT_PRESENTATION):
            return(FileParser._ReadCols_Presentation(filename, cols))
        if (file_format == FORMAT_CSV):
            return(FileParser._ReadCols_csv(filename, cols))

    def _ReadCols_Presentation(filename, cols = None):
        if (cols is None):
            return(pd.read_csv(filename, sep = '\t', encoding = "utf-8-sig"))
        else:
            return(pd.read_csv(filename, usecols = cols, sep = '\t', encoding = "utf-8-sig"))

    def _ReadCols_csv(filename, cols = None):
        if (cols is None):
            return(pd.read_csv(filename, sep = ','))
        else:
            return(pd.read_csv(filename, usecols = cols, sep = ','))

    def GenerateConfig(table, path = ""):
        _table = pd.DataFrame(table().fetch(limit = 1))
        cols = {c: c for c in _table.columns}
        filename = path+str(table.__name__)+'_config.json'
        f = open(filename, 'w')
        json.dump(cols, filename)
        f.close()

    def LoadConfig(table, path = ""):
        filename = path+str(table.__name__)+'_config.json'
        f = open(filename, 'r')
        config_dict = json.load(f)
        f.close()
        return(config_dict)

    def SaveConfig(table, path, config_dict):
        filename = path+str(table.__name__)+'_config.json'
        f = open(filename, 'w')
        json.dump(config_dict, f)
        f.close()
        logging.info('Saved config for '+str(table.__name__)+' at '+filename)


@schema
class Experiment(dj.Manual):
    definition = """
        experiment_id: varchar(12)
        ---
        experiment_name: varchar(64) # unique experiment id
        project_start: date
        data_path: varchar(256) # path to data files
        experiment_files_path: varchar(256) # path to the experiment files
        bart_format: enum('eprime_txt', 'presentation', 'csv', 'synthetic') # type of the data, for parsing purposes
        protocol_path: varchar(256)
        #special_params: varchar(256) # should be later changed to composite
        abstract_path: varchar(256) # path to short summary of experiment
        """


@schema
class Participant(dj.Imported):
    definition = """
        participant_id: varchar(32) # unique participant id
        ---
        """
    def _import(self):
        print('-- Importing participants --')
        logging.info('-- Importing participants --')
        experiment = Experiment() & 'data_format != "synthetic"'
        for row in experiment.fetch():
            data_dir = IMPORT_DATA_TABLE_PATH+row['data_path']
            files = listdir(data_dir)
            for f in files:
                if f.endswith(FORMAT_EXTENSION[row['data_format']]):
                    df = FileParser.ReadCols(filename = data_dir+f, file_format = row['data_format'], cols = ['Subject'])
                    subjects = []
                    for s in df['Subject'].unique():
                        subjects.append(dict(participant_id=row['experiment_id']+'_'+str(s)))
                    self.insert(subjects, skip_duplicates = True)
        print('Finished importing participants.')
        logging.info('Finished importing participants.')


@schema
class Session(dj.Imported):
    definition = """
        # Bart session (one data recording run)
        ->Experiment
        ->Participant
        session_name: varchar(64)
        session_id: int

        #session_date: date # date of starting the session
        #session_time: time # time of starting the session
        ---
        ->Condition
        #->Experimenter
        #->Computer
        #->Monitor
        #->Input_device 
        """

    class BartData(dj.Part):
        definition = """
            ->Session
            block: int # number of the block within the session
            trial: int # number of trial within the block
            ---
            rt:            int unsigned # RT of first input, in ms
            event:         varchar(32)   # 
            nr_pumps:      tinyint unsigned
            response:      enum("pump","cashout")
            balloon_earn:  int unsigned
         """


@schema
class Condition(dj.Manual):
    definition = """
        ->Experiment
        condition_id: int
        ---
        condition_name: varchar(16) # short name of the condition
        isi_rsi: enum("isi", "rsi") # interstimulus interval or response-to-stimulus interval is fixed
        trial_interval: int unsigned # value of the isi/rsi in msec
        rt_max: int unsigned # maximal rt of recording in msec
        description: varchar(256) # short description of the experimental condition
        special_params: longblob # parameter settings
        stimulus: varchar(32) # type of stimuli, e.g. dalmata
        """

    def _import(self):
        self._import_conditions_from_csv(IMPORT_DATA_TABLE_PATH)

    def _import_conditions_from_csv(self, path):
        print('-- Importing conditions --')
        logging.info('-- Importing conditions --')
        filename = path+'bart_condition.csv'
        df = FileParser.ReadCols(filename, file_format = 'csv')
        for col in df.columns:
            if 'datetime' in df.ftypes[col]:
                df[col] = [str(d) for d in df[col]]
        self.insert((r.to_dict() for _, r in df.iterrows()))
        print('Finished importing conditions.')
        logging.info('Finished importing conditions.')
