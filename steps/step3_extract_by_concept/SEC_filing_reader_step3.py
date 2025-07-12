"""
Roni Hogri, July 2025

This demo program utilizes large language models (LLMs) to extract Balance Sheet data from quarterly and yearly filings to the SEC (10-Q and 10-K, respectively).
Specifically, this demo version focuses on companies included in the S&P 500 on March 5th, 2024, and looks at filing data during the prior 5 years.
For additional details see github repo: https://github.com/ronihogri/financial-doc-reader

Note: 
To run this program, you must obtain your own OpenAI API key! 
This key should be set as an environment variable in your OS, or inserted as the str value of MY_API_KEY (see "user definable variables" below).
"""


"""User-definable variables; modify as necessary:"""

BATCH_SIZE = None #how many filings to process in each run, set to None if you want to process all of them at once
SKIP_EXISTING = True #set to False if you want existing data to be overwritten 
FIRST_ROW_TO_OVERWRITE = 1 #only relevant if SKIP_EXISTING set to False, lets you choose where in the DB to start overwriting
RETRY_LIST = [] #populate list with IDs of forms you want (list of ints) to retry (will process only them, and ignore BATCH_SIZE and SKIP_EXISTING)

MAX_MINI_VOTES = 5 #max number of votes for mini model
MAX_SUPERVISOR_VOTES = 1 #max number of votes for large model when acting as supervisor

REPORT_DB_FN = "filings_demo_step3.sqlite" #SQL file name 

MY_API_KEY = "insert_your_OpenAI_API_key_here_if_you_don't_want_to_set_it_as_an_environment_var"

"""End of user-defined variables"""

"""***************************************************************************************************************************************"""


"""Globals"""

# import required libraries for global use:
import os
import openai
from openai import OpenAI
import numpy as np
import time
from collections import Counter
import sqlite3
import re
import json
import sys
from datetime import datetime
import random
import ast

#paths, etc.
curdir = os.path.dirname(os.path.abspath(__file__)) #path of this script
filings_db_path = os.path.join(curdir, REPORT_DB_FN) #path to SQL file
NEW_TASKS = ['ValueColumn', 'CCP', 'LTD'] #tasks to be updated by this program in the SQL DB's Tasks table
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) or OpenAI(api_key=MY_API_KEY) #openai client

#third party interactions
PLAY_NICE = 1.0 #time (s) to wait before making a request to third party
GPT_ATTEMPTS = 3 #number of attempts to reach openai in case of failure 

#models:
MINI = "gpt-4o-mini-2024-07-18"
GPT_4O = "gpt-4o-2024-08-06"
SUPERVISOR = f"Supervisor ({GPT_4O})" #alias for when large model used to supervise previous responses (for convenience)

"""*********************************************************************************************************************************"""

"""Functions"""


def check_user_vars(): 
    """Validate that user-defined vars are correctly defined.

    Returns:
        None

    Raises:
        ValueError: If BATCH_SIZE is defined but not a positive integer, 
                    or if FIRST_ROW_TO_OVERWRITE is not a positive integer when SKIP_EXISTING is set to False.
        TypeError: If SKIP_EXISTING is not a boolean.
        Exception: If the path specified in filings_db_path does not exist.

    Globals:
        BATCH_SIZE (int or None): Size of the batch for processing, must be a positive integer or None (= all filings will be processed in one run).
        SKIP_EXISTING (bool): If set to False, completed filings will be re-processed and data re-written.
        FIRST_ROW_TO_OVERWRITE (int): ID of first form to overwrite if SKIP_EXISTING set to False.
        filings_db_path (str): Path to SQL DB holding the Forms table.
        RETRY_LIST (list): List of form IDs that user chose to process.
    """

    if ((BATCH_SIZE is not None) and (not isinstance(BATCH_SIZE, int))) or ((isinstance(BATCH_SIZE, int)) and (BATCH_SIZE < 1)):
        raise ValueError("**** BATCH_SIZE incorrectly defined, must be a positive int or None ****\n\n")
    
    if not isinstance(SKIP_EXISTING, bool):
        raise TypeError("**** SKIP_EXISTING incorrectly defined, must be True/False ****\n\n")
    
    if (not SKIP_EXISTING) and ((not isinstance(FIRST_ROW_TO_OVERWRITE, int)) or (FIRST_ROW_TO_OVERWRITE < 1)):
        raise ValueError("**** FIRST_ROW_TO_OVERWRITE incorrectly defined, must be a positive int when SKIP_EXISTING set to False ****\n\n")
    
    if RETRY_LIST and (
        not isinstance(RETRY_LIST, list)
        or any(not isinstance(item, int) for item in RETRY_LIST)
    ):
        raise TypeError("**** RETRY_LIST incorrectly defined, must be a list of form IDs (integers) - see Forms.id in SQL DB ****\n\n")
    
    if not os.path.exists(filings_db_path):
        raise Exception(f"**** Path to SQL DB incorrectly defined, no such path exists: ****\n{filings_db_path}\n\n")
    
   
def get_forms_info():
    """Gets the next filing to work on from the SQL DB's "Forms" table.     

    Returns:
        list: A list of tuples; each tuple contains the following elements extracted from the Forms table:
            i. int: id
            ii. str: FormName

    Globals:
        SKIP_EXISTING (bool): If set to False, completed filings will be re-processed and data re-written. 
        BATCH_SIZE (int or None): If int, the number of filings to process in each run; 
            if set to None, the program will run through all incomplete filings remaining in the Forms tables.
        filings_db_path (str): Path to SQL DB holding the Forms table.   
    """

    global BATCH_SIZE

    #connect to SQL DB and get identifiers for next filing to be processed
    with sqlite3.connect(filings_db_path) as conn:
        cur = conn.cursor() 

        if RETRY_LIST: #if list is populated, will only work on this list
            BATCH_SIZE = len(RETRY_LIST)
            placeholders = ",".join("?" for _ in range(len(RETRY_LIST)))
            cur.execute(f"SELECT id, FormName FROM Forms WHERE id IN ({placeholders})", RETRY_LIST)
            return cur.fetchall()        
   
        #get all form ids
        cur.execute("SELECT id FROM Forms ORDER BY id")
        form_ids = [item[0] for item in cur.fetchall()]

        if not SKIP_EXISTING and FIRST_ROW_TO_OVERWRITE > len(form_ids):
            raise ValueError(f"**** FIRST_ROW_TO_OVERWRITE out of range: must be between 1 and {len(form_ids)} ****\n\n")
                
        #get completed
        cur.execute(f"SELECT Form_id FROM Tasks WHERE {NEW_TASKS[-1]} NOT NULL")
        existing = [item[0] for item in cur.fetchall()]

        remaining_count = len(form_ids) - len(existing) #number of remaining forms to process if not overwriting

        if BATCH_SIZE is None: #if user chooses to go through all data at once
            if SKIP_EXISTING: #don't overwrite
                BATCH_SIZE = remaining_count
            else: #overwrite: do not skip existing
                BATCH_SIZE = len(form_ids) - FIRST_ROW_TO_OVERWRITE + 1
        else: #BATCH_SIZE is int
            if SKIP_EXISTING: #don't overwrite
                if BATCH_SIZE > remaining_count: #set batch size is larger than remaining forms
                    BATCH_SIZE = remaining_count
            else: #overwrite
                if (FIRST_ROW_TO_OVERWRITE + BATCH_SIZE) > len(form_ids): #set batch size is larger than remaining forms
                    BATCH_SIZE = len(form_ids) - FIRST_ROW_TO_OVERWRITE + 1

        if not SKIP_EXISTING: #if completed filings should be overwritten, overwrite batch starting at first row to overwrite
            cur.execute("SELECT id, FormName FROM Forms WHERE id >= ? AND id < ?", (FIRST_ROW_TO_OVERWRITE, FIRST_ROW_TO_OVERWRITE + BATCH_SIZE))
            return cur.fetchall()

        #make list of incomplete with len batch_size
        ids_to_get = [form for form in form_ids if form not in existing][:BATCH_SIZE]

        #get form info based on ids_to_get
        cur.execute("CREATE TEMPORARY TABLE temp_ids (id INTEGER PRIMARY KEY)")
        cur.executemany("INSERT INTO temp_ids VALUES (?)", [(id, ) for id in ids_to_get])
        cur.execute("SELECT f.id, f.FormName FROM Forms f JOIN temp_ids t ON f.id = t.id ORDER BY f.id")
        return cur.fetchall()
    

def check_overwrite():
    """Check if the user wants to proceed with overwriting existing data. Called when RETRY_LIST or SKIP_EXISTING lead to overwriting.
    If the user chooses to terminate overwriting, the function exits the program.

    Returns:
        None 

    Globals:
        RETRY_LIST (list): List of form IDs that may be overwritten if user confirms.
    """

    if RETRY_LIST: #RETRY_LIST overrides SKIP_EXISTING
        overwrite = input(f"\n** Warning: Data may be overwritten for the following forms (ids):\n{RETRY_LIST}\nAre you sure you wish to proceed? [y/N]    ")
    else:
        overwrite = input("\n** Warning: Existing data may be overwritten (SKIP_EXISTING set to False), are you sure you wish to proceed? [y/N]    ")

    if overwrite.strip().lower() != 'y':
        print("\n**** Program terminated by user ****\n\n")
        sys.exit()
    else:
        return
 

def check_previous_tasks(forms_info):
    """Checks if all tasks from previous step(s) have been completed, and stores forms with incomplete tasks in list to be skipped.

    Args:
        forms_info (list): A list of tuples; each tuple contains the following elements extracted from the Forms table:
            i. int: id
            ii. str: FormName (not actually used by this function)       

    Returns:
        previous_tasks_incomplete (list): List of form ids for which the Tasks table does not contain data regarding previous steps. 

    Globals:
        filings_db_path (str): Path to SQL DB holding the Forms table.   
    """

    with sqlite3.connect(filings_db_path) as conn:
        cur = conn.cursor()
        #get info re Tasks table - which tasks should have been completed before running this script?
        cur.execute("PRAGMA table_info(Tasks)") 
        columns = [result[1] for result in cur.fetchall()]
        cutoff_index = columns.index(NEW_TASKS[0])
        previous_tasks = ", ".join(columns[1:cutoff_index]) #all column names that are not Form_id or new tasks

        #for each form, check that previous tasks were successfully completed
        cur.execute("CREATE TEMP TABLE TempForms (Form_id INTEGER PRIMARY KEY)")
        cur.executemany("INSERT INTO TempForms (Form_id) VALUES (?)", ((form_id,) for form_id, _ in forms_info))
        cur.execute(f"""
            SELECT T.Form_id, {previous_tasks}
            FROM Tasks T
            JOIN TempForms TF ON T.Form_id = TF.Form_id
        """)
        results = {form_id: tuple(tasks) for form_id, *tasks in cur.fetchall()}
        previous_tasks_incomplete = [
            form_id for form_id, _ in forms_info
            if (form_id not in results) or 
            any(task is None for task in results[form_id]) or
            any((isinstance(task, int) and task < 0) for task in results[form_id])
            ]

        cur.execute("DROP TABLE TempForms")
    
        return previous_tasks_incomplete


def check_majority(votes, trials): 
    """Check if there a majority decision was already reached based on the current votes.

    Args:
        votes (dict): keys: vote IDs, values: GPT output per vote.
        trials (int): The maximum number of trials expected for this voting process.

    Returns:
        bool: True if a majority exists that cannot be overturned by the remaining trials, False otherwise.
    """

    vote_counter = Counter(str(v) for v in votes.values())

    if vote_counter.most_common(1)[0][1] == np.ceil(trials / 2): #a majority exists that can't be overturned by the remaining trials
        return True
    

def convert_model_output(model_output, type_):
    """Convert the model's output to the specified type if possible.

	Args:
		model_output (str): The model's decision (e.g., majority vote). 
        type_ (str): Name of the expected variable type (e.g., int, float)

	Returns:
		target (object of type type_) or None: The converted model's input, or None if conversion failed.
	"""
    
    try:        
        type_func = getattr(__builtins__, type_, None) #get type function from built-ins
        if callable(type_func): #use type function to convert model output to type type_
            target = type_func(ast.literal_eval(model_output))  
            return target 
    except: 
        return None
    

def gpt_completion(model, system_content, user_content, response_type='text', output_dtype='str', trials=1, trial_counter=0, set_seed=False): 
    """General function for querying GPT (completions mode).

    Args:
        model (str): The model to be used for generating completions.
        system_content (str): The content that sets the behavior of the assistant.
        user_content (str): The input content from the user for which a completion is requested.
        response_type (str): Expected response format from the model; default is 'text'
        output_dtype (str or type): Desired Python datatype for model output (e.g., 'str', 'int', 'list', 'dict'); 
            if not 'str', the output will be cast using convert_model_output().
        trials (int): The number of trials for querying the model, must be a positive integer; default is 1.
        trial_counter (int): Index of first trial upon function call; default value = 0.
        set_seed (bool): If True, actively sets the model seed to reduce output similarity across calls; default is False.

    Returns:
        votes (dict): A dictionary containing GPT outputs indexed by trial number.

    Globals:
        PLAY_NICE (float): Sleep time between API calls (seconds).
        GPT_ATTEMPTS (int): Number of attempts to connect to OpenAI API before failing.
    """

    fail_counter = 0
    votes = {}

    while True: #loop until broken by failed attempts or successful trials

        time.sleep(PLAY_NICE) #wait between API calls 
        seed = random.randint(0, 10**7) if set_seed else None #if required, actively set seed (to avoid similar random state in consecutive calls)

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
                ], 
                response_format={"type": response_type},
                seed=seed
                )
            
            gpt_output = completion.choices[0].message.content            
            votes[trial_counter] = gpt_output.replace("`", "").strip() #vote for this trial is the trimmed GPT output
            if output_dtype != 'str':
                conversion_result = convert_model_output(votes[trial_counter], output_dtype)
                if conversion_result is not None:
                    votes[trial_counter] = conversion_result

            if trials == 1: #no voting process, decision based on a single output
                break            
            
            trial_counter += 1

            if trial_counter >= np.ceil(trials / 2): #after half the votes, check if theoretical majority reached after each vote (if yes, stop voting)
                majority = check_majority(votes, trials)
                if majority:
                    break

            if trial_counter >= trials: #reached maximal number of votes
                break

        except openai.RateLimitError as e: 
            raise openai.RateLimitError(message='**** OpenAI API quota exceeded ****\n\n', response=e.response, body=e.body) from None

        except openai.OpenAIError as e:
            fail_counter += 1
            if fail_counter == GPT_ATTEMPTS:
                raise Exception(
                    f"Could not reach OpenAI server, error encountered: {e}\nResponse: {getattr(e, 'response', 'N/A')}\nBody: {getattr(e, 'body', 'N/A')}"
                    ) from None
            
    return votes


def count_votes(votes):
    """Gets the majority vote from the provided dictionary of votes.

    Args:
        votes (dict): keys: vote IDs (trial numbers), values: GPT output per vote.

    Returns:
        same type as vote values or None: The value that received an absolute majority (>= 50%) of the votes, or None if no value received this majority.
    """

    vote_pairs = [(v, str(v)) for v in votes.values()] #strs used for counter, but original (majority) value is returned
    vote_counter = Counter(pair[1] for pair in vote_pairs) 
    if vote_counter.most_common(1)[0][1] < np.ceil(len(votes)/2): #majority vote does not have 50% or higher - undecided
        return None 
    else:
        majority_str = vote_counter.most_common(1)[0][0]  
        return next(pair[0] for pair in vote_pairs if pair[1] == majority_str) #return first matching original value    


def read_from_json(file_path, key_path=()):
    """Read data from a JSON file and optionally retrieve nested values based on a given key path.

	Args:
		file_path (str): Path to the JSON file to read.
		key_path (tuple, optional): A sequence of keys to navigate through nested dictionaries. Default is an empty tuple, which returns the entire JSON object.

	Returns:
		json_dict (dict): The retrieved JSON structure, or a part of it as specified by key_path.

	Raises:
		KeyError: If any key in the key_path is not found in the JSON structure.
	"""    

    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    
    for key in key_path:
        try:
            json_dict = json_dict[key]
        except (KeyError, TypeError) as e:
            raise KeyError(f"Could not read JSON info from file:\n{file_path}\nKey path {key_path} is invalid at {key}:\n{e}")

    return json_dict  


def update_json(file_path, dict_path_list, value_list):
    """Update a JSON file at the specified path (for updating existing dicts).

	This function modifies the JSON structure located at file_path by following the paths specified in dict_path_list. 
    With the exception of model info, each dict path culminates in a dictionary that holds a key called 'data', where the corresponding value from value_list will be stored.     
    The function also records a timestamp of when the data was updated.

	Args:
		file_path (str): The path to the JSON file to be updated.
		dict_path_list (list): A list of paths (each path a tuple) in the JSON structure where each value from value_list should be inserted.
		value_list (list): A list of values to be stored in the corresponding paths specified by dict_path_list.
		model (str, optional): An optional model identifier to be included in the update. Defaults to None.

	Returns:
		None

	Raises:
		TypeError: If either dict_path_list or value_list is not a list.
		Exception: If the file at file_path does not exist or has not been initialized.
	"""

    if not isinstance(dict_path_list, list) or not isinstance(value_list, list):
        raise TypeError(f"Both dict_path_list and value_list must be lists! They are currently, respectively: {type(dict_path_list)}, {type(value_list)}")
    
    if os.path.exists(file_path): 
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise Exception(f"File not yet initialized:\n{file_path}\n\n")

    for dict_path, value in zip(dict_path_list, value_list):

        sub_dict = data

        for key in dict_path: #add dict paths as required (values always stored under 'data'!)
            if key not in sub_dict or not isinstance(sub_dict[key], dict):
                sub_dict[key] = {}  #add new sub dict as specified by user
            if key == 'model': #models don't have a 'data' or 'timestamp' key (model info is nested within parent data info)
                sub_dict[key] = value
            else: 
                sub_dict = sub_dict[key]

        if key == 'model': continue #model info complete, move to next item

        if key == 'problems': #problems should be treated as a list that could potentially contain more than one value
            if not sub_dict['data']:
                sub_dict['data'] = []
            if isinstance(value, list):
                sub_dict['data'].extend(value)
            else:
                sub_dict['data'].append(value)
            sub_dict['data'] = list(dict.fromkeys(sub_dict['data'])) #don't store the same problem twice (maintain problem logging order)

        else:
            sub_dict['data'] = value

        sub_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')        
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def insert_into_json(file_path, new_dict, dict_name):
    """Inserts a new dict into a JSON data file.
    
    Args:
        file_path (str): Full path of the file into which new data should be added. 
        new_dict (dict): Dictionary to be added to JSON file.
        dict_name (str): Key of new_dict.

    Returns:
        None    
    """
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data[dict_name] = new_dict 

    #problems should always be at the end of the log
    problems = data.pop('problems')
    data['problems'] = problems

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_json_path(form_id, form_name, file_type): 
    """Get/set the path for the specified JSON file.

	Args:
		form_id (int): Form ID, as appears in the Forms table.
		form_name (str): FormName, as appears in the Forms table.
        file_type (str): Type of JSON data; possible values: 'text', 'log', 'table'

	Returns:
		path (str) or None: The full path to the JSON file where specified data is stored, or None if path is expected to exist (created by step1) but does not.
	"""

    fn = f'{form_id}_{form_name}.json' #file name

    if file_type == 'text':
        path = os.path.join(curdir, 'extracted', 'text_blocks', fn)

    elif file_type == 'log':
        path = os.path.join(curdir, 'extracted', 'logs', 'balance', fn)

    elif file_type == 'table':
        path = os.path.join(curdir, 'extracted', 'tables', 'balance', fn)
        os.makedirs(os.path.dirname(path), exist_ok=True) #table jsons were not created in step 1
        return path

    else:
        raise ValueError(f"\n**** File type incorrectly specified for get_json_path(): ****'{file_type}'; should be 'text', 'log', or 'table'.\n\n") from None
    
    if not os.path.exists(path):
        print(f"** Skipping form - JSON file containing {file_type} data not found in expected location: {path} **")
        return None        

    return path


def reset_problems(log_path):
    """If overwriting, reset problems list in the log file.
    
    Args:
		log_path (str): Path to log JSON file for the Balance Sheet table.
    
    Returns:
        None
    """

    with open(log_path, 'r') as f:
        data = json.load(f)

    data['problems']['data'] = None
    
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=4)


def report_problems(form_name, path, problems): 
    """Print problems encountered during a specific process (i.e., not necessarily all logged problems).

	Args:
		form_name (str): FormName, as appears in the Forms table.
		path (str): The path of the JSON log file where the problems were detected.
		problems (list): List of problems encountered, each represented as a string.

	Returns:
		None
	"""

    if not problems:
        return 
    print(f"**** Problems detected in form {form_name}:")   
    for problem in problems:
        print(problem)
    print(f"File path: {path}")
    print("***************************************************")


def init_new_log_entries(log_path):
    """Initialize new log dicts to be used by this program in JSON log file.
    
    Args:
		log_path (str): Path to log JSON file for the Balance Sheet table.
    
    Returns:
        None
    """

    model_dict_template = {'data': None, 'model': None, 'timestamp': None}
    dict_titles = ['value_date_column', 'current_cash_position', 'long_term_debt']

    for title in dict_titles:
        insert_into_json(log_path, model_dict_template, title)


def find_key(table_json, search_term):
    """Search for a top-level key in a JSON dictionary that contains the given search term.

    Args:
        table_json (dict): Dictionary representing the structured table data.
        search_term (str): Substring to search for (case-insensitive) within top-level keys.

    Returns:
        str or None: The first matching key, if found; otherwise None.
    """
    
    for key in table_json:
        if search_term in key.lower():
            return key
        

def check_dict_paths(dict_paths, input_dict):
    """Check whether all dictionary paths exist in the given JSON object.

    Args:
        dict_paths (dict): Dictionary where each value is a list of keys representing a path in the JSON structure.
        input_dict (dict): The JSON object (e.g., Balance Sheet table) in which the paths should be validated.

    Returns:
        list: List of keys from dict_paths corresponding to invalid paths (i.e., paths that do not exist in input_dict).
    """

    invalid_paths = []

    for i, path in dict_paths.items():
        temp_dict = input_dict
        for key in path: 
            if not key in temp_dict:
                invalid_paths.append(i) #log problem and move on
                break
            temp_dict = temp_dict[key]
    
    return invalid_paths


def get_dict_path_value(dict_path, input_dict, column):
    """Retrieve the first integer value from the list at a specific dictionary path in a nested JSON object.

    Args:
        dict_path (list): Sequence of keys representing a path through the nested dictionary.
        input_dict (dict): The JSON object to traverse.
        column (int): The column in which the relevant value is stored

    Returns:
        int or None: The first element of the list at the target location, cast to int if possible; otherwise None.
    """

    for key in dict_path: 
        input_dict = input_dict[key]

    if not isinstance(input_dict, list): #value pointed to is not a list
        return None    
    try:
        return int(input_dict[column]) #date value - expected int
    except:
        return None    
    

def get_sums_per_key_paths(dict_paths, table_json, log_path):
    """Extract and return numeric values from specified key paths in a JSON table.

    Args:
        dict_paths (dict): Dictionary where each value is a list of keys representing a path in the JSON structure.
        table_json (dict): The JSON object from which values are to be extracted.
        log_path (str): Path to the JSON log file containing metadata (e.g., the relevant value date column name).

    Returns:
        dict: Keys are path identifiers from dict_paths; values are the corresponding integer values (or None if invalid).
    """

    column = read_from_json(log_path, ("value_date_column", "data", "value_date_column"))
    path_sums = {}

    for i, path in dict_paths.items():
        path_sums[i] = get_dict_path_value(path, table_json, column)

    return path_sums  
        

"""Functions for identifying the table column holding values for the report's value date (nested within get_vd_column())"""    

def ask_column_dates(table_comments, model=MINI, trials=1, output_dtype='list', set_seed=True, response_type='text'):
    """Ask GPT to extract dates from the Balance Sheet table header text.

    Args:
        table_comments (str): The text preceding the first row of Balance Sheet table (including column headers).
        model (str, optional): Model used for querying; defaults to MINI.
        trials (int, optional): Number of times to query the model (used for voting); defaults to 1.
        output_dtype (str, optional): Desired datatype for model output; defaults to 'list'.
        set_seed (bool, optional): Whether to set a random seed to encourage output diversity; defaults to True.
        response_type (str, optional): Format requested from the model response; defaults to "text".

    Returns:
        dict: Keys are trial numbers; values are the model's outputs per trial.
    """

    print(f"...Asking the '{model}' model to extract column dates....")

    get_column_dates_sys = """# Task

    You're an intern at a mutual fund whose only job is to scan the 'Consolidated Balance Sheets' comments in a single 10-Q or 10-K and extract every date-like occurrence.

    ## Date Extraction Rules
    - The text may split dates across tabs or lines, and can be separated by other text (e.g. "December 30,\t2023", "March\t26,\t2022", "December 31\tMillions of Dollars\t2023\t2022") or even another date (e.g., "September 26,\tDecember 28,\t2020\t2019").

    - A date can appear in any of these forms:
    1. MonthName Day, Year (e.g. December 31, 2023)  
    2. MonthName Day (e.g. December 31)  
    3. A shared MonthName and Day followed by two or more years (e.g. December 31,\t2017\t2018) - in this case, the same month and day apply to both years, and should be expanded into full dates like ['2017-12-31', '2018-12-31']
    4. MonthName Day [another MonthName Day] Year [another Year]

    - Normalize each found date to the format YYYY-MM-DD, use zeros as placeholders for missing values:
    • If the year is missing, use 0000 as a placeholder  
    • If the month is missing, use 00 as a placeholder 
    • If the day is missing, use 00 as a placeholder 
    • **Exception**: In case of a structure like shown in Rule #3, **do not use placeholders** - instead, apply the shared MM-DD to all the relevant years.

    - Scan left to right - append each normalized date string to a Python list in the order encountered.

    ## Output Format
    Return exactly one Python list literal - no extra text.  

    Examples:
    - Input: ...December 31, 2023...September 30, 2023...  
    Output: ['2023-12-31', '2023-09-30']  
    - Input: ...December 31...March 15...  
    Output: ['0000-12-31', '0000-03-15']  
    - Input: ...2021    2022...  
    Output: ['2021-00-00', '2022-00-00']  
    - Input: ...December 31,    2017    2018...  
    Output: ['2017-12-31', '2018-12-31']
    - Input: December 31\t...\t2023\t2022
    Output: ['2023-12-31', '2022-12-31']
    - Input: ...September 26,\tDecember 28,\t2020\t2019...
    Output: ['2020-09-26', '2019-12-28']    

    ## Constraints
    - Don't use any outside context or explanations.  
    - If you find no dates, return [].  
    """


    get_column_dates_user = f"""
    Return a single list of dates found in this text snippet - per the rules in the system prompt:

    '{table_comments}'
    """

    return gpt_completion(model, get_column_dates_sys, get_column_dates_user, response_type=response_type, output_dtype=output_dtype, trials=trials, set_seed=set_seed)


def collect_list_lengths(data):
    """Traverse a nested JSON object and collect the lengths of all list values.

    Args:
        data (dict): Nested dictionary representing a JSON object.

    Returns:
        list: List of integers representing the lengths of all list-type values found in the structure.
    """

    list_lens = []
    def helper(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    list_lens.append(len(value))
                elif isinstance(value, dict):
                    helper(value)
        # anything other than dicts and lists is ignored
    helper(data)

    return list_lens


def get_num_columns(table_path):
    """Estimate the number of columns in the table by computing the median length of lists contained in the table JSON.

    Args:
        table_path (str): Path to the JSON file containing the structured table.

    Returns:
        int: Estimated number of columns.
    """

    with open(table_path, 'r') as f:
        data = json.load(f)
    
    list_lens = collect_list_lengths(data)

    return round(np.median(list_lens))


def get_vd_column(log_path, table_path, form_name):
    """Main function for GPT-assisted identification of the value date column in a given Balance Sheet table.

    Args:
        log_path (str): Path to a JSON file where results and issues should be logged.
        table_path (str): Path to a JSON file containing the Balance Sheet table.
        form_name (str): Identifier for the specific form (e.g., 10-Q or 10-K) being processed.

    Returns:
        None
    """
    
    table_comments = read_from_json(log_path, ("table_comments", "data"))
    model_dict = {MINI: {'votes': None, 'decision': None}, GPT_4O: {'votes': None, 'decision': None}}
    column_dict = {'num_columns': None, 'value_date_column': None}
    column_dict['num_columns'] = get_num_columns(table_path)

    for i in range(2): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per model)

        model, trials, set_seed = (MINI, MAX_MINI_VOTES, True) if i == 0 else (GPT_4O, 1, False)

        #ask GPT model to identify the value date column based on the table comments 
        #votes = ask_vd_index(table_comments, model=model, trials=trials, set_seed=set_seed)
        votes = ask_column_dates(table_comments, model=model, trials=trials, set_seed=set_seed)
        model_dict[model]['votes'] = votes
        model_dict[model]['decision'] = count_votes(votes)

        if not isinstance(model_dict[model]['decision'], list): #problem with model output
            problems_list.append('value column: dates not returned as list')
            continue #try with larger model
        
        if not problems_list: #valid result obtained
            if model_dict[model]['decision']:
                column_dict['value_date_column'] = model_dict[model]['decision'].index(max(model_dict[model]['decision'][:column_dict['num_columns']])) #get maximum value, this is most recent date
            else:
                column_dict['value_date_column'] = 0  #default index of value date column (also if [] is returned) - it's usually the first one
            break #don't run again

    update_json(
        log_path, 
        [('value_date_column',), ('value_date_column', 'model'), ('problems',)],
        [column_dict, model_dict, problems_list]        
        )

    if problems_list: #after both models, problems remain        
        report_problems(form_name, log_path, problems_list) #print out detected problems  


"""Functions for extracting current cash position (CCP) from the Balance Sheet JSON (nested within get_ccp())."""

def ask_ccp_dict_paths(assets, model=MINI, trials=1, output_dtype='dict', set_seed=True, response_type='json_object'):
    """Ask GPT to extract dictionary paths corresponding to current cash position (CCP) items from a Balance Sheet.

    Args:
        assets (dict): JSON-formatted subsection of the Balance Sheet, typically under "Assets".
        model (str, optional): Model used for querying; defaults to MINI.
        trials (int, optional): Number of times to query the model (used for voting); defaults to 1.
        output_dtype (str, optional): Desired datatype for model output; defaults to 'dict'.
        set_seed (bool, optional): Whether to set a random seed to encourage output diversity; defaults to True.
        response_type (str, optional): Format requested from the model response; defaults to "json_object".

    Returns:
        dict: Keys are trial numbers; values are the model's outputs per trial.
	"""

    print(f"...Asking the '{model}' model to extract dictionary paths containing current cash position (CCP)-related data....")

    get_ccp_dict_paths_sys = """# Task  

You are an intern at a mutual fund. Your task is to analyze the "Assets" section of Balance Sheet tables, provided in JSON format.  
Your objective is to extract and return the paths in these JSON dictionaries that correspond to items included in a company's **current cash position**, in the context of assessing the company's health (see Peter Lynch).  

## Context and Definition: Current Cash Position

A company's **current cash position** consists of **Cash and Cash Equivalents** and **current (liquid) Marketable Securities** that are **immediately accessible** for operational needs, debt repayment, or investment opportunities.

### **Inclusions:**
- **Cash and Cash Equivalents**, such as:
  - Cash on hand  
  - Bank deposits  
  - Short-term investments like Treasury bills  
  - Other instruments that can be **converted to cash within 90 days** without significant loss of value  
- **Current (liquid) Marketable Securities**, meaning:
  - Readily tradable short-term investments  
  - Securities that can be sold for cash **without restriction**  

### **Exclusions:**
The current cash position **does NOT include**:
- **Restricted cash**, such as:
  - Cash in escrow  
  - Collateralized deposits  
- **Illiquid assets**, such as:
  - Inventories  
  - Prepaid expenses  
  - Receivables (even if due soon)  
  - Any other assets **not immediately convertible to cash**  

Current cash position-related items are typically found under **Current Assets** in the Balance Sheet, but only those that meet the strict liquidity criteria should be considered part of the **current cash position**.
Use your judgment and knowledge of balance sheets and 10-Q/10-K forms to determine if an item is relevant for the current cash position.   
Nevertheless, **do NOT include** non-specific entries like "Other current assets", or any assets which cannot be readily converted into cash. 
Also, **do NOT include** any entries that MAY OR MAY NOT relate to the current cash position (i.e., entries where additional parts of the document would be required for you to reach a conclusion).

## Requirements  

- **Path Extraction**: From the provided JSON data, return a structured subset that contains paths to items relevant to the **current cash position**.  
  - The output must be formatted as a dictionary where **each relevant path is assigned a numbered key** (e.g., "1", "2", etc.).  
  - Each numbered key must map to an **array of strings representing the exact sequence of keys** leading to a non-dictionary value in the original JSON.  
  - **Do not include non-key values** (such as numerical values, `null`, or placeholders).  

- **Key Accuracy**: Every key in the extracted paths must be an **exact match** to its corresponding key in the original JSON data.  
  - **Do not correct typos, formatting issues, or extra spaces**.  
  - Any deviation in key structure will be considered an error.  

- **Exclusion of Total Values**: Do not extract key paths that are related to total values that already include other values belonging to current cash position. 
For example, if there is a total like "TOTAL CASH, CASH EQUIVALENTS AND SHORT-TERM INVESTMENTS" that includes other current cash position items (e.g., "Cash and cash equivalents" and "Short-term investments"), that total should **not** be extracted.

### **Output Format**  

The expected output is a JSON object structured as follows:  
- Each key is a **numbered string ("1", "2", "3", etc.)** corresponding to a unique path leading to a non-dict value.  
- Each numbered key maps to an **array of strings**, representing the exact JSON key sequence for that path.  
- **Paths must be isolated**, meaning each numbered entry corresponds to a single, unbroken key sequence.  

### **Examples**  

#### **Example 1**

**Input JSON (original balance sheet data):**  
{
  "Current assets": {
    "Cash and cash equivalents": [12074, 13118],
    "Premium and trade receivables": [13272, 12238],
    "Short-term investments": [2321, 1539],
    "Other current assets": [2461, 1602],
    "Total current assets": [30128, 28497]
  },
  "Long-term investments": [14684, 14043],
  "Restricted deposits": [1217, 1068],
  "Property, software and equipment, net": [2432, 3391],
  "Goodwill": [18812, 19771],
  "Intangible assets, net": [6911, 7824],
  "Other long-term assets": [2686, 3781],
  "Total assets": [76870, 78375]
}

**Expected Output JSON:**  
{
  "1": ["Current assets", "Cash and cash equivalents"],
  "2": ["Current assets", "Short-term investments"]
}


#### **Example 2**

**Input JSON (original balance sheet data):** 
{"Current Assets": {"Cash and Cash Equivalents": [651, 1979, 2475], "Accounts Receivable, Net": [167, 240, 110], "Inventories": [820, 709, 636], "Other": [114, 81, 94], "Current Assets of Discontinued Operations": [0, 0, 1297], "Total Current Assets": [1752, 3009, 4612]}, "Property and Equipment, Net": [1059, 1009, 994], "Operating Lease Assets": [1058, 1021, 993], "Goodwill": [628, 628, 628], "Trade Names": [165, 165, 165], "Deferred Income Taxes": [44, 45, 61], "Other Assets": [154, 149, 146], "Other Assets of Discontinued Operations": [0, 0, 2947], "Total Assets": [4860, 6026, 10546]}

**Expected Output JSON:**  
{
  "1": ["Current assets", "Cash and cash equivalents"]
}


### **Important Note**  

This task requires **precise attention to detail**. I will verify that your output strictly follows the JSON structure and contains only the relevant key paths.  
Any inaccuracies or alterations will have serious consequences, including the risk of losing your internship. Proceed with caution.
"""  

    get_ccp_dict_paths_user = f"""Here is the relevant part of the JSON table, extract the JSON object containing the lists of current cash position dictionary path keys as instructed: {assets}"""

    return gpt_completion(model, get_ccp_dict_paths_sys, get_ccp_dict_paths_user, response_type=response_type, output_dtype=output_dtype, trials=trials, set_seed=set_seed) 


def ask_ccp_supervisor(dict_paths, model=GPT_4O, output_dtype='list', trials=1):
    """Ask a supervisor model to verify which entries in a proposed current cash position are incorrect.

    Args:
        dict_paths (dict): Dictionary where each key is a string (e.g., "1", "2"), and each value is a list of strings representing a key path in a financial statement.
        model (str, optional): Model used for verification; defaults to GPT_4O.
        output_dtype (str, optional): Desired datatype for model output; defaults to 'list'.
        trials (int, optional): Number of times to query the model (used for voting); defaults to 1.

    Returns:
        dict: Keys are trial numbers; values are the model's outputs per trial.
    """
    
    print(f"...Current cash position issues suspected. Asking '{model}' to double-check....")

    get_supervisor_call_sys = f"""Task: You are supervising interns at a mutual fund. Your interns provide you with structured JSON data containing references to various financial assets which they consider to be part of a company's current cash position. 
    Some of your interns' entries have been flagged by an automated system, suggesting that they may have **wrongly labeled some items as being related to the current cash position**. Your objective is to extract and return a Python list of numbered keys (e.g., ["1", "2", "3"]) corresponding to items that are **definitely not** part of the current cash position, as defined in terms of assessing a company's financial health (see Peter Lynch).

## **Context**  

A company's **current cash position** is the total sum of its **Cash and Cash Equivalents** and **current (liquid) Marketable Securities**. This includes assets such as cash on hand, bank deposits, and short-term investments like Treasury bills, which can be readily converted to cash.

## **Requirements**  

### **Input**: 
A dictionary where:  
  - Each key is a **string representing a number** (e.g., "1", "2", "3").  
  - Each value is a **list of strings** representing the key sequence in a financial statement, arranged in an order reflecting the hierarchy of the "assets" section of a balance sheet table (e.g., in a 10-Q or 10-K form). The earlier strings in the list correspond to higher levels of hierarchy, such as broader categories (e.g., "Current Assets"), and the later strings correspond to more specific items within that category (e.g., "Cash and Cash Equivalents").  

### **Output Format**  
A **Python list** of keys (e.g., ["1", "2", "3"]) corresponding to items that do **not** belong to the current cash position under the provided definition.
- The length of the list should be in the range of 0-{len(dict_paths)}.  
- **DO NOT include anything other than this list** — no explanations, extra text, numbers, or symbols.   

- **Exclusion Criteria**:  
  - Exclude items **unrelated to cash or highly liquid marketable securities**, such as:  
    - Accounts receivable  
    - Inventories  
    - Prepaid expenses  
    - Property, plant, and equipment  
    - Deferred tax assets  
    - Other non-cash assets  
  - When uncertain, ask: *Does this item represent current cash or a liquid marketable security in the context of company health (as per Peter Lynch)?*  
    - **If NO → Exclude it (include its key in the output list).**  
    - **If YES → Do NOT exclude it (do not include its key in the output list).**  

## **Examples**  

### **Example 1**

**Input:**  
{{
  "1": ["Current Assets", "Cash and Cash Equivalents"],
  "2": ["Current Assets", "Marketable Securities"],
  "3": ["Current Assets", "Accounts Receivable"],
  "4": ["Current Assets", "Inventories"]
}}

**Expected Output:**
["3", "4"]

### **Example 2**

**Input:**
{{'1': ['Cash and cash equivalents'], '2': ['Short-term investments']}}

**Expected Output:**
[]


## **Important Notes**

- Items such as "Cash and Cash Equivalents" and "Marketable Securities" should **not** be excluded, as they form part of the current cash position → **DO NOT** include them in the output list.
- Items such as "Accounts Receivable" and "Inventories" should **be** excluded because they are not cash or highly liquid marketable securities → **Include them** in the output list.

## **Strict Compliance Warning**

This task requires **precise attention to detail**. I will verify that:
- Your response contains only a Python list of keys (each key a **string representing a number**), or an empty list.
- The response strictly adheres to the exclusion criteria.
Failure to meet these requirements will result in serious consequences, including termination. Proceed with caution.
"""  

    get_supervisor_call_user = f"""Here is structured JSON data containing dictionary paths that represent rows in the **Assets** section of a **Balance Sheet table**. The entries provided are **suspected** to include some items that are **not** part of the company's **current cash position**.
Your task is to extract and return a Python list of numbered keys corresponding to items that **definitely do not** belong to the **current cash position**, as defined. Follow the instructions carefully and exclude only those items that fail to meet the liquidity criteria: {dict_paths}"""

    return gpt_completion(model, get_supervisor_call_sys, get_supervisor_call_user, output_dtype=output_dtype, trials=trials)


def suspect_ccp_terms(dict_paths, black_list=["escrow", "inventor", "receivable", "tax", "total"], required=["current"]):
    """Detect whether any key paths contain terms that suggest misclassification in current cash position labeling.

    Args:
        dict_paths (dict): Dictionary where each value is a list of strings representing a key path in a financial statement.
        black_list (list, optional): List of suspect substrings; if any appear in the final key of a path, it is flagged. Defaults to common non-CCP terms.
        required (list, optional): List of substrings that must appear somewhere in each path; defaults to ["current"].

    Returns:
        bool: True if any path is flagged as suspect based on blacklist or missing required terms; otherwise False.
    """
    
    if any(
        any(suspect in path[-1].lower().replace("-", " ") for suspect in black_list) # blacklist only applies to the last key
        or not any(req in key.lower().replace("-", " ") for key in path for req in required) #required should be somewhere in the path 
        for path in dict_paths.values()
    ):
        return True


def get_ccp(log_path, table_path, form_name):
    """Main function for extracting current cash position (CCP) information from a Balance Sheet using GPT-based assistance.
    
    Args:
        log_path (str): Path to a JSON file where results and issues should be logged.
        table_path (str): Path to a JSON file containing the Balance Sheet table.
        form_name (str): Identifier for the specific form (e.g., 10-Q or 10-K) being processed.

    Returns:
        None
    """
    
    table_json = read_from_json(table_path)
    assets_key = find_key(table_json, "asset")
    if not assets_key:
        update_json(log_path, [('problems',)], ['CCP: assets not found in json table'])
        report_problems(form_name, log_path, ['CCP: assets not found in json table'])
        return None
    
    assets = table_json[assets_key]

    ccp_dict = {"key_paths": None, "path_sums" : None, "total_sum": None}
    model_dict = {MINI: {'votes': None, 'decision': None}, GPT_4O: {'votes': None, 'decision': None}, SUPERVISOR: {}}

    for i in range(2): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per model)

        model, trials, set_seed = (MINI, MAX_MINI_VOTES, True) if i == 0 else (GPT_4O, 1, False)

        #ask GPT model to identify CCP entries
        votes = ask_ccp_dict_paths(assets, model=model, trials=trials, set_seed=set_seed)
        model_dict[model]['votes'] = votes
        model_dict[model]['decision'] = count_votes(votes)

        if ( 
            model_dict[model]['decision'] == 'null' 
            or model_dict[model]['decision'] == 'None' 
            or model_dict[model]['decision'] is None
        ):
            problems_list.append('CCP: key paths not found')
            continue #try with larger model

        if not isinstance(model_dict[model]['decision'], dict):
            problems_list.append('CCP: dict paths not in dict format')
            continue #try with larger model

        #check if dict paths are valid - if not, specify which are not
        invalid_dict_paths = check_dict_paths(model_dict[model]['decision'], assets)

        if invalid_dict_paths:
            for idx in invalid_dict_paths:
                problems_list.append(f'CCP: problematic dict path: index = {idx}')
            continue #try with larger model

        #non-fatal problems (log problems but accept large model's decision to be checked later as needed):

        dict_paths = model_dict[model]['decision']

        #if not all paths contain "current", this might indicate a problem - see if the supervisor can detect and fix it
        if suspect_ccp_terms(dict_paths):
            model_dict[SUPERVISOR][model] = {'votes': None, 'decision': None}
            supervisor_votes = ask_ccp_supervisor(dict_paths, trials=MAX_SUPERVISOR_VOTES) 
            model_dict[SUPERVISOR][model]['votes'] = supervisor_votes
            model_dict[SUPERVISOR][model]['decision'] = count_votes(supervisor_votes)

            if model_dict[SUPERVISOR][model]['decision'] and isinstance(model_dict[SUPERVISOR][model]['decision'], list):
                dict_paths = {
                    key: value for key, value in dict_paths.items() 
                    if key not in model_dict[SUPERVISOR][model]['decision']
                    }
            else: #supervisor found no issues (or issue with supervisor output), flag to check manually
                problems_list.append("CCP: suspicious key path(s) detected")

        path_sums = get_sums_per_key_paths(dict_paths, assets, log_path)
        for idx, path_sum in path_sums.items():
            if path_sum is None:
                problems_list.append(f'CCP: missing sum(s) detected: index = {idx}')

        ccp_dict['key_paths'] = dict_paths
        ccp_dict['path_sums'] = path_sums
        ccp_dict['total_sum'] = sum(v for v in path_sums.values() if v is not None)
        
        if not problems_list: #valid result obtained 
            print("- Extracting CCP-related sums....")
            break #don't run again

    update_json(
        log_path, 
        [('current_cash_position',), ('current_cash_position', 'model'), ('problems',)],
        [ccp_dict, model_dict, problems_list]        
        )

    if problems_list: #after both models, problems remain        
        report_problems(form_name, log_path, problems_list) #print out detected problems


"""Functions for extracting long-term debt (LTD) from the Balance Sheet JSON (nested within get_ltd())."""

def ask_ltd_dict_paths(liabilities, model=MINI, trials=1, set_seed=True, response_type='json_object', output_dtype='dict'):
    """Ask GPT to identify dictionary paths corresponding to long-term debt (LTD) in the Liabilities section of a Balance Sheet.

    Args:
        liabilities (dict): Subsection of the Balance Sheet JSON corresponding to liabilities.
        model (str, optional): The model to be used for extraction; defaults to MINI.
        trials (int, optional): Number of times to query GPT (maximum number of votes); defaults to 1.
        set_seed (bool, optional): Whether to set a random seed for response reproducibility; defaults to True.
        response_type (str, optional): Expected response format from the model; defaults to 'json_object'.
        output_dtype (str, optional): Expected data type to cast model output into; defaults to 'dict'.

    Returns:
        dict: Keys are trial numbers; values are GPT outputs (dictionary of LTD-related key paths) per trial.
    """

    print(f"...Asking the '{model}' model to extract dictionary paths containing long-term debt (LTD)-related data....")

    get_ltd_dict_paths_sys = """# Task  

You are an intern at a mutual fund. Your task is to analyze the "Liabilities" section of Balance Sheet tables, provided in JSON format.  
Your objective is to extract and return the paths in these JSON dictionaries that correspond to items included in a company's **long-term debt**.  

## Context  

A company's **long-term debt** consists of financial obligations that are due beyond one year. 
This includes instruments such as **Convertible Senior Notes, Term Loans, Bonds Payable, Debentures, Mortgage Payable, Capital Lease Obligations, Finance Lease Obligations, Notes Payable, Subordinated Debt**.  
These items are typically listed under **Non-Current Liabilities** in the Balance Sheet. However, some aspects of long-term debt may appear in the **Current Liabilities** section (e.g., "Term Debt" or "Current portion of long-term debt").
Use your judgment and knowledge of balance sheets and 10-Q/10-K forms to determine if an item is relevant for the long-term debt.   
Nevertheless, **do NOT include** non-specific entries like "Other non-current liabilities/obligations" or "Other long-term liabilities/obligations", or any instruments that do not **explicitly** represent borrowings, loans, or debt financing (e.g., "Operating Lease Liabilities").  
Also, **do NOT include** any entries that MAY OR MAY NOT relate to long-term debt (i.e., entries where additional parts of the document would be required for you to reach a conclusion, e.g., "Long-term lease liabilities").  

## Requirements  

- **Path Extraction**: From the provided JSON data, return a structured subset that contains paths to items relevant to **long-term debt**.  
  - The output must be formatted as a dictionary where **each relevant path is assigned a numbered key** (e.g., "1", "2", etc.).  
  - Each numbered key must map to an **array of strings representing the exact sequence of keys** leading to a non-dictionary value in the original JSON.  
  - **Do not include non-key values** (such as numerical values, `null`, or placeholders).  

- **Key Accuracy**: Every key in the extracted paths must be an **exact match** to its corresponding key in the original JSON data.  
  - **Do not correct typos, formatting issues, or extra spaces**.  
  - Any deviation in key structure will be considered an error.  

- **Exclusion of Total Values**: Do not extract key paths that are related to total values that already include other values belonging to long-term debt. For example, if there is a total like "Total non-current liabilities" that includes other long-term debt items (e.g., "Term debt" and "Other non-current liabilities"), that total should not be extracted.

### **Output Format**  

The expected output is a JSON object structured as follows:  
- Each key is a **numbered string ("1", "2", "3", etc.)** corresponding to a unique path leading to a non-dict value.  
- Each numbered key maps to an **array of strings**, representing the exact JSON key sequence for that path.  
- **Paths must be isolated**, meaning each numbered entry corresponds to a single, unbroken key sequence.  

#### **Examples**  

##### Example 1:
**Input JSON (original balance sheet data):**  
{
  "Current liabilities": {
    "Accounts payable": [32421, 46236],
    "Other current liabilities": [37324, 37720],
    "Deferred revenue": [5928, 5522],
    "Commercial paper and repurchase agreement": [10029, 5980],
    "Term debt": [10392, 10260],
    "Total current liabilities": [96094, 105718]
  },
  "Non-current liabilities": {
    "Term debt": [89086, 91807],
    "Other non-current liabilities": [56795, 50503],
    "Total non-current liabilities": [145881, 142310]
  },
  "Total liabilities": [241975, 248028]
}

**Expected Output JSON:**  
{
  "1": ["Current liabilities", "Term debt"], 
  "2": ["Non-current liabilities", "Term debt"]
}

##### Example 2:
**Input JSON (original balance sheet data):**  
{"Current Liabilities": 
    {"Accounts payable": [6305, 6455],
    "Accrued group welfare and retirement plan contributions": [934, 927],
    "Accrued wages and withholdings": [3701, 3569],
    "Current maturities of long-term debt, commercial paper and finance leases": [1811, 2623],
    "Current maturities of operating leases": [548, 560],
    "Liabilities to be disposed of": [296, 347],
    "Other current liabilities": [1608, 1450],
    "Self-insurance reserves": [1103, 1085],
    "Total Current Liabilities": [16306, 17016]},
 "Deferred Income Tax Liabilities": [1997, 488],
 "Long-Term Debt and Finance Leases": [21916, 22031],
 "Non-Current Operating Leases": [2524, 2540],
 "Other Non-Current Liabilities": [3816, 3847],
 "Pension and Postretirement Benefit Obligations": [9594, 15817]}

**Expected Output JSON:**  
{
  "1": ["Current liabilities", "Current maturities of long-term debt, commercial paper and finance leases"], 
  "2": ["Long-Term Debt and Finance Leases"]
}

##### Example 3:
**Input JSON (original balance sheet data):**  
{'Commitments and contingencies (Note 11)': None,
 'Current liabilities': 
    {'Accounts payable': [617, 790],
    'Accrued government and other rebates': [3585, 3928],
    'Current portion of long-term debt and other obligations, net': [1999, 2748],
    'Other accrued liabilities': [2760, 3139],
    'Total current liabilities': [8961, 10605]},
 'Long-term debt, net': [24084, 24574],
 'Long-term income taxes payable': [5837, 5922],
 'Other long-term obligations': [1577, 1040]}

**Expected Output JSON:**  
{
  "1": ["Current liabilities", "Current portion of long-term debt and other obligations, net"], 
  "2": ["Long-term debt, net"]
}


### **Important Note**  

This task requires **precise attention to detail**. I will verify that your output strictly follows the JSON structure and contains only the relevant key paths.  
Any inaccuracies or alterations will have serious consequences, including the risk of losing your internship. Proceed with caution.
"""  

    get_ltd_dict_paths_user = f"""Here is the relevant part of the JSON table, extract the JSON object containing the lists of long-term debt dictionary path keys as instructed: {liabilities}"""

    return gpt_completion(model, get_ltd_dict_paths_sys, get_ltd_dict_paths_user, response_type=response_type, output_dtype=output_dtype, trials=trials, set_seed=set_seed) 


def suspect_ltd_terms(dict_paths, gray_list=["current", "short term"], white_list=["non current", "long term", "term debt"], black_list=["tax", "total"]):
    """Detect whether any dictionary paths likely contain misclassified long-term debt (LTD) entries.

    Args:
        dict_paths (dict): Dictionary where each value is a list of strings representing a key path in the liabilities section of a Balance Sheet.
        gray_list (list, optional): Terms that suggest ambiguity (e.g., potentially short-term); requires a redeeming term to pass.
        white_list (list, optional): Terms that explicitly affirm LTD relevance (e.g., "long term", "term debt").
        black_list (list, optional): Terms that directly disqualify the entry if found in the final key of the path.

    Returns:
        bool: True if any path is flagged as suspect due to ambiguity or blacklist violations; otherwise False.
    """

    if any(
        (any(suspect in key.lower().replace("-", " ")  
            for key in path
            for suspect in gray_list)
        and not any(redeemer in key.lower().replace("-", " ") 
                    for key in path 
                    for redeemer in white_list))
        for path in dict_paths.values()
    ):
        return True
    
    if any(
        any(suspect in path[-1].lower().replace("-", " ") for suspect in black_list)
        for path in dict_paths.values()
    ):
        return True
 
    
def ask_ltd_supervisor(dict_paths, model=GPT_4O, output_dtype='list', trials=1):
    """Asks a supervisor model to verify the correctness of long-term debt (LTD) classifications.

    Args:
        dict_paths (dict): A dictionary where each key is a string representing a number,
            and each value is a list of strings representing the key sequence in a financial statement.
        model (str, optional): The GPT model used for verification; defaults to GPT_4O.
        output_dtype (str, optional): The format of the GPT model's response; defaults to 'list'.
        trials (int, optional): Number of times to query the model (votes); defaults to 1.

    Returns:
        dict: Keys are vote IDs (trial numbers), values are the GPT outputs per vote.
    """

    print(f"...LTD issues suspected. Asking '{model}' to double-check....")

    get_supervisor_call_sys = f"""# Task: You are supervising interns at a mutual fund. Your interns provide you with structured JSON data containing references to various financial liabilities which they consider to be part of a company's long-term debt. 
    Some of your interns' entries have been flagged by an automated system, suggesting that they may have **wrongly labeled some items as being related to the company's long-term debt**. Your objective is to extract and return a Python list of numbered keys (e.g., ["1", "2", "3"]) corresponding to items that are **definitely not** part of long-term debt (long-term debt), as defined in terms of assessing a company's financial health (see Peter Lynch).

## **Context**  

'Long-term debt' is a key measure of a company's financial stability. For this task, long-term debt includes both:  
1. **Obligations that extend beyond one year**, and  
2. **The current portion of long-term debt** (e.g., "Current maturities of long-term debt"), since it remains part of the company's long-term borrowing burden.  

## **Requirements**  

### **Input**: 
A dictionary where:  
  - Each key is a **string representing a number** (e.g., "1", "2", "3").  
  - Each value is a **list of strings** representing the key sequence in a financial statement, arranged in an order reflecting the hierarchy of the "liablilities" section of a balance sheet table (e.g., in a 10-Q or 10-K form). The earlier strings in the list correspond to higher levels of hierarchy, such as broader categories (e.g., "Current Liabilities"), and the later strings correspond to more specific items within that category (e.g., "Loans and notes payable").  

### **Output Format**  
A **Python list** of keys (e.g., ["1", "2", "3"]) corresponding to items that do **not** belong to long-term debt under the provided definition.
- The length of the list should be in the range of 0-{len(dict_paths)}.  
- **DO NOT include anything other than this list** — no explanations, extra text, numbers, or symbols.   

- **Exclusion Criteria**:  
  - Exclude items **unrelated to borrowing**, such as:  
    - General liabilities  
    - Accounts payable  
    - Operational expenses  
    - Deferred taxes  
    - Other non-debt obligations  
    - **Short-term debt or short-term borrowings** (except for the current portion of long-term debt, which should NOT be excluded)  
  - When uncertain, ask: *Does this item represent long-term debt in the context of company health (as per Peter Lynch)?*  
    - **If NO → Exclude it (include its key in the output list).**  
    - **If YES → Do NOT exclude it (do not include its key in the output list).**  

## **Examples**  

### **Example 1**

**Input:**  
{{
  "1": ["Current Liabilities", "Loans and notes payable"],
  "2": ["Current Liabilities", "Current maturities of long-term debt"],
  "3": ["Long-term debt"]
}}

**Expected Output:**
["1"]

### **Example 2**

**Input:**  
{{
  "1": ["Current Liabilities", "Term debt"],
  "2": ["Long-term debt"]
}}

**Expected Output:**
[]

## **Important Notes**

- Items such as "Loans and notes payable" under "Current Liabilities" is **not** considered part of long-term debt because it is typically short-term borrowing  → Exclude such items.
- Items such as "Current maturities of long-term debt" **should not** be excluded, because it still represents part of the company's LTD in terms of a company's health → **DO NOT** Exclude such items.

## **Strict Compliance Warning**

This task requires **precise attention to detail**. I will verify that:
- Your response contains only a Python list of keys (each key a **string representing a number**), or an empty list.
- The response strictly adheres to the exclusion criteria.
Failure to meet these requirements will result in serious consequences, including termination. Proceed with caution.
"""  

    get_supervisor_call_user = f"""Here is structured JSON data containing dictionary paths that represent rows in the **Liabilities** section of a **Balance Sheet table**. The entries provided are **suspected** to include some items that are **not** part of the company's **long-term debt**.
Your task is to extract and return a Python list of numbered keys corresponding to items that **definitely do not** belong to the **long-term debt**, as defined. Follow the instructions carefully and exclude only those items that meet the exlusion criteria: {dict_paths}"""

    return gpt_completion(model, get_supervisor_call_sys, get_supervisor_call_user, output_dtype=output_dtype, trials=trials) 
        

def get_ltd(log_path, table_path, form_name):
    """Main function for extracting Long-Term Debt (LTD) from the Balance Sheet table.

    Args:
        log_path (str): Path to the log JSON file for updating results and problems.
        table_path (str): Path to the input JSON file containing the Balance Sheet table.
        form_name (str): Identifier for the current filing (used in problem reporting).

    Returns:
        None
    """

    table_json = read_from_json(table_path)
    liabilities_key = find_key(table_json, "liabilit")
    if not liabilities_key:
        update_json(log_path, [('problems',)], ['LTD: liabilities not found in json table'])
        report_problems(form_name, log_path, ['LTD: liabilities not found in json table'])
        return None
    
    liabilities = table_json[liabilities_key]

    ltd_dict = {"key_paths": None, "path_sums" : None, "total_sum": None}
    model_dict = {MINI: {'votes': None, 'decision': None}, GPT_4O: {'votes': None, 'decision': None}, SUPERVISOR: {}}

    for i in range(2): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per model)

        model, trials, set_seed = (MINI, MAX_MINI_VOTES, True) if i == 0 else (GPT_4O, 1, False)

        #ask GPT model to identify LTD entries 
        votes = ask_ltd_dict_paths(liabilities, model=model, trials=trials, set_seed=set_seed)
        model_dict[model]['votes'] = votes
        model_dict[model]['decision'] = count_votes(votes)

        if ( 
            model_dict[model]['decision'] == 'null' 
            or model_dict[model]['decision'] == 'None' 
            or model_dict[model]['decision'] is None
        ):
            problems_list.append('LTD: key paths not found')
            continue #try with larger model

        if not isinstance(model_dict[model]['decision'], dict):
            problems_list.append('LTD: dict paths not in dict format')
            continue #try with larger model

        #check if dict paths are valid - if not, specify which are not
        invalid_dict_paths = check_dict_paths(model_dict[model]['decision'], liabilities)

        if invalid_dict_paths:
            for idx in invalid_dict_paths:
                problems_list.append(f'LTD: problematic dict path: index = {idx}')
            continue #try with larger model

        #non-fatal problems (log problems but accept large model decision)

        dict_paths = model_dict[model]['decision']

        # paths should NOT contain 'current' or 'short-term' unless they also contain a hint of long term
        if suspect_ltd_terms(dict_paths):            
            model_dict[SUPERVISOR][model] = {'votes': None, 'decision': None}
            supervisor_votes = ask_ltd_supervisor(dict_paths, trials=MAX_SUPERVISOR_VOTES)
            model_dict[SUPERVISOR][model]['votes'] = supervisor_votes
            model_dict[SUPERVISOR][model]['decision'] = count_votes(supervisor_votes)

            if model_dict[SUPERVISOR][model]['decision'] and isinstance(model_dict[SUPERVISOR][model]['decision'], list):
                dict_paths = {
                    key: value for key, value in dict_paths.items() 
                    if key not in model_dict[SUPERVISOR][model]['decision']
                    }
            else: #supervisor found no issues (or issues with supervisor), flag to check manually
                problems_list.append("LTD: suspicious key path(s) detected")

        #get values (sums) referenced by the last key in each path  
        path_sums = get_sums_per_key_paths(dict_paths, liabilities, log_path)
        for idx, path_sum in path_sums.items():
            if path_sum is None:
                problems_list.append(f'LTD: missing sum(s) detected: index = {idx}')  

        ltd_dict['key_paths'] = dict_paths
        ltd_dict['path_sums'] = path_sums
        ltd_dict['total_sum'] = sum(v for v in path_sums.values() if v is not None)
        
        if not problems_list: #valid result obtained 
            print("- Extracting LTD-related sums....")
            break #don't run again

    update_json(
        log_path, 
        [('long_term_debt',), ('long_term_debt', 'model'), ('problems',)],
        [ltd_dict, model_dict, problems_list]        
        )

    if problems_list: #after both models, problems remain        
        report_problems(form_name, log_path, problems_list) #print out detected problems

 
"""Functions for updating the SQL DB"""

def get_balance_problems(log_path):
    """Retrieve a list of problem IDs (from the Problems table) that match problems reported in the log file.

	Args:
		balance_log_path (str): Path to log JSON file for the Balance Sheet table.
	
	Returns:
		problem_ids (list): A list of problem IDs (ints) pointing to the types of problems detected (Problems.id).
	
	Globals:
		filings_db_path (str): Path to the SQL database containing the Problems table.
	"""

    balance_log_problems = read_from_json(log_path, ('problems', 'data')) #get problem descriptions

    problem_ids = [] 

    if balance_log_problems: #if any problems were logged
        with sqlite3.connect(filings_db_path) as conn:
            cur = conn.cursor()
            for problem in balance_log_problems:
                problem_trunc = re.sub(r'(^[^:]*:[^:]*):.*', r'\1', problem) #remove higher-level title from problem description if exists (for future use)
                cur.execute("SELECT id FROM Problems WHERE Description = ?", (problem_trunc, ))
                result = cur.fetchone()
                if result:
                    problem_ids.append(result[0])
                else:
                    raise ValueError(f"\n**** Mismatch between problem listed in JSON file and SQL 'Problems' table: ***\n{problem_trunc}\n")

    return problem_ids


def update_sql(form_id, log_path, problem_ids):
    """Updates the Tasks and FormProblems tables in the SQL database.
	
	This function reads data related to this program's tasks from the log file and updates the Tasks table in the SQL DB.
	If problems were encountered, they are stored in the FormProblems table. 
	If the database is locked, the function will prompt the user to resolve the issue before retrying.
	
	Args:
		form_id (int): Form ID, as appears in the Forms table.
		log_path (str): Path to log JSON file for the Balance Sheet table.
		problem_ids (list): A list of problem IDs to log in the FormProblems table (may be empty).
	
	Returns:
		None

	Raises:
		sqlite3.OperationalError: If an operational error occurs while accessing the SQLite database.
	
	Globals:
		filings_db_path (str): Path to the SQL database containing the Problems table.
	"""

    sum_divider = read_from_json(log_path, ("units", "data", "sum_divider"))
    vd_column = read_from_json(log_path, ("value_date_column", "data", "value_date_column"))
    ccp = read_from_json(log_path, ("current_cash_position", "data", "total_sum"))
    ltd = read_from_json(log_path, ("long_term_debt", "data", "total_sum"))
    if vd_column is None:
        ccp = ltd = None
    elif sum_divider != 1:            
        ccp = round(ccp / sum_divider, 3)
        ltd = round(ltd / sum_divider, 3)
    
    problem_str = ", logging problems in FormProblems table" if problem_ids else ""

    print(f"- Writing data to Tasks table{problem_str}....")

    while True:        
        try:
            with sqlite3.connect(filings_db_path) as conn:
                cur = conn.cursor()
               
                cur.execute("UPDATE Tasks SET (ValueColumn, CCP, LTD) = (?, ?, ?) WHERE Form_id = ?", (vd_column, ccp, ltd, form_id))

                if (not SKIP_EXISTING) or RETRY_LIST: #if overwriting, delete previously logged problems
                    cur.execute("DELETE FROM FormProblems WHERE Form_id = ?", (form_id, ))
                    
                for problem_id in problem_ids: #if problems were detected, log them in the FormProblems table
                    cur.execute("INSERT OR IGNORE INTO FormProblems VALUES (?, ?)", (form_id, problem_id))
                
            break #update successful, break out of while loop              

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                retry = input(
                    "Database is locked - please close the SQLite file and press ENTER to try again, or enter 'Q' to quit program.\t"
                ).strip()
                if retry.upper() == "Q":
                    print("\n*** Program terminated by user ***\n\n")
                    sys.exit()
                continue  # try again
            else:
                raise sqlite3.OperationalError(
                    f"\nError encountered, program terminated:\n{e}\n"
                    )


def report_done(forms_with_problems, start_time, form_cnt, previous_tasks_incomplete):
    """Outputs a summary report upon completion of data extraction from the specified batch.

	Args:
	    forms_with_problems (list): A list of strs (form_id + form_name) indicating the forms in which problems were encountered (if any).
	    start_time (float): The time when the extraction process started, used to calculate runtime.
	    form_cnt (int): Total count of forms processed in this batch.
        previous_tasks_incomplete (list): A list of forms which were skipped by this program because prerequisite tasks were not yet completed.

	Returns:
	    None

    Globals:
        NEW_TASKS (list): List of strs representing the tasks to be processed by this program (see Tasks table in SQL DB).
	"""

    if previous_tasks_incomplete:
        form_cnt -= len(previous_tasks_incomplete)
        skipped_previous_incomplete_text = f" ({len(previous_tasks_incomplete)} forms skipped due to incomplete previous tasks of missing JSON files)"
        previous_problem_text = f"probelms related to previous step(s) encountered for the following form IDs: {previous_tasks_incomplete}"
    else:
        skipped_previous_incomplete_text = ""
        previous_problem_text = ""

    if forms_with_problems:
        if skipped_previous_incomplete_text:
            problem_text = f"Problems encountered in {len(forms_with_problems)} filings:\n{forms_with_problems}\nIn addition, {previous_problem_text}" 
        else:
            problem_text = f"Problems encountered in {len(forms_with_problems)} filings:\n{forms_with_problems}"        
    else:
        if skipped_previous_incomplete_text:
            problem_text = f"No problems detected during this run. However, {previous_problem_text}"
        else:
            problem_text = "No problems detected."

    if start_time and form_cnt > 0:
        task_text = f"\nLast task executed: '{NEW_TASKS[-1]}'."
        runtime_text = f"\nRuntime {round((time.time() - start_time) / 60, 2)} minutes ({round((time.time() - start_time) / form_cnt, 2)} seconds per form, on average)."
    else:
        runtime_text = task_text = ""

    if form_cnt > 0:
        db_text = f"\nData stored to SQL DB: {filings_db_path}"
    else:
        db_text = ""

    print(f"""\n\n\n*********************************************************************************************************
Completed data extraction for batch of {form_cnt} 10-Q/10-K filings{skipped_previous_incomplete_text}.{task_text}{runtime_text}
{problem_text}{db_text}
*********************************************************************************************************\n
""")  
    
  

"""****************************************************************************************************************************"""
def main():
    """Program for extracting Current Cash Position (CCP) and Long-Term Debt (LTD) data from the Balance Sheet table in financial filings to the SEC (10-Q and 10-K forms).

    Should be run after running SEC_filing_reader_step2.py.
    Identifies dictionary paths associated with CCP and LTD, extracts and logs dollar sums from the identified paths.
    Model workflow: mini model is queried first, optionally followed by supervisor review; if problems persist, the large model is queried and may also be followed by supervisor review.
    In cases of unresolved issues, results are flagged for manual inspection by logging problems in the FormProblems table of the SQL database.
    """
    
    try:

        #initialize some vars
        i = 0 #default form count in case program is terminated early
        start_time = None #for runtime calculation
        forms_with_problems = [] #for storing the id+name of forms for which problems were encountered

        #check that user-defined variables are of the right types
        check_user_vars()

        #get basic form info from the SQL database
        forms_info = get_forms_info()

        #if user-defined vars lead to overwriting, confirm with user
        if (not SKIP_EXISTING) or RETRY_LIST:
            check_overwrite()                 

        if not forms_info:
            print(f"""\n\n**** All filings in the Forms table have already been processed for the task '{NEW_TASKS[-1]}', program terminated ****
Note: If you wish to overwrite existing data, set SKIP_EXISTING to False, or specify relevant form IDs in RETRY_LIST\n\n""")
            sys.exit()

        print(f"\n**** Processing {BATCH_SIZE} filings ****")

        start_time = time.time() 

        previous_tasks_incomplete = check_previous_tasks(forms_info)

        #for each form (filing)
        for i, (form_id, form_name) in enumerate(forms_info):
            
            print(f"\n\n.......... Processing filing #{form_id} ({i+1} / {BATCH_SIZE} in batch): '{form_name}' ........")   

            if form_id in previous_tasks_incomplete:
                print("** Skipping form: prerequisite task(s) were not completed for this form (check Tasks table and rerun previous steps if relevant) **")
                continue

            #fetch required paths to JSON files
            log_path = get_json_path(form_id, form_name, 'log')
            table_path = get_json_path(form_id, form_name, 'table')

            if not log_path or not table_path:
                previous_tasks_incomplete.append(form_id)
                continue
            
            #if overwriting, reset problems list in the log file
            if (not SKIP_EXISTING) or RETRY_LIST:
                reset_problems(log_path)

            #insert additional dicts to the log file, to be updated by this program
            init_new_log_entries(log_path)

            #identify value date column
            get_vd_column(log_path, table_path, form_name) 

            #get current cash position
            get_ccp(log_path, table_path, form_name)

            #get long-term debt
            get_ltd(log_path, table_path, form_name)        
                         
            #check if problems were encountered for this form, and get their ids (see Problems table in the SQL DB)            
            sql_problem_ids = get_balance_problems(log_path) 
            if sql_problem_ids:
                forms_with_problems.append(f'{form_id}_{form_name}')

            #update problems and results for this form in the SQL DB
            update_sql(form_id, log_path, sql_problem_ids)                            

    except KeyboardInterrupt:
        sys.exit("\n\n**** Program terminated by user (KeyboardInterrupt) ****\n\n")

    except Exception as e:
        if "[Errno 22] Invalid argument" in str(e):
            sync_issue = "\nIsuse may be related to automatic cloud syncing (e.g., Dropbox), pause syncing and try again."
        else:
            sync_issue = ""
        print(f"\n\n**** Program terminated, {type(e).__name__} encountered: ****\n{e}{sync_issue}\n\n")
        sys.exit(1)
    
    finally:
        #check why program was terminated and how many forms were processed during this run
        exc_type, _, _ = sys.exc_info()
        if exc_type:
            forms_examined = i
        else: 
            forms_examined = i+1

        #if forms were processed during this run, provide a summary
        if forms_examined > 0:
            report_done(forms_with_problems, start_time, forms_examined, previous_tasks_incomplete)


if __name__ == "__main__":
    main()
