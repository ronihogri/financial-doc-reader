"""
Roni Hogri, March 2025

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

REPORT_DB_FN = "filings_demo_step2.sqlite" #SQL file name 

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
import ast
import sys
from datetime import datetime
from jsonpath_ng import parse

#paths, etc.
curdir = os.path.dirname(os.path.abspath(__file__)) #path of this script
filings_db_path = os.path.join(curdir, REPORT_DB_FN) #path to SQL file
NEW_TASKS = ['SumDivider', 'JsonTable'] #tasks to be updated by this program in the SQL DB's Tasks table
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) or OpenAI(api_key=MY_API_KEY) #openai client

#third party interactions
PLAY_NICE = 1.0 #time (s) to wait before making a request to third party
GPT_ATTEMPTS = 3 #number of attempts to reach openai in case of failure 

#models:
MINI = "gpt-4o-mini-2024-07-18"
GPT_4O = "gpt-4o-2024-08-06"


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

    vote_counter = Counter(votes.values())

    if vote_counter.most_common(1)[0][1] == np.ceil(trials / 2): #a majority exists that can't be overturned by the remaining trials
        return True
    

def gpt_completion(model, system_content, user_content, trials=1, trial_counter=0): 
    """General function for querying GPT (completions mode).

    Args:
        model (str): The model to be used for generating completions.
        system_content (str): The content that sets the behavior of the assistant.
        user_content (str): The input content from the user for which a completion is requested.
        trials (int): The number of trials for querying the model, must be a positive integer; default is 1.
        trial_counter (int): Index of first trial upon function call; deafult value = 0.

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

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
                ]
                )
            
            gpt_output = completion.choices[0].message.content            
            votes[trial_counter] = gpt_output.replace("`", "").strip() #vote for this trial is the trimmed GPT output

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
        str or None: The value that received an absolute majority (>= 50%) of the votes, or None if no value received this majority.
    """

    vote_counter = Counter(votes.values())
    if vote_counter.most_common(1)[0][1] < np.ceil(len(votes)/2): #majority vote does not have 50% or higher
        return None #undecided
    else:
        return vote_counter.most_common(1)[0][0]    


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

    post_table_dict = {'data': None, 'model': None, 'timestamp': None}
    table_body_dict = {'data': None, 'timestamp': None}
    table_dict = {'data': None, 'model': None, 'timestamp': None}

    insert_into_json(log_path, post_table_dict, 'post_table_text')
    insert_into_json(log_path, table_body_dict, 'table_body') 
    insert_into_json(log_path, table_dict, 'table_json_created')


def get_table_text(log_path, text_path, form_name): 
    """Gets the text block containing the Balance Sheet table from the relevant JSON file.

	Args:
		log_path (str): Path to log JSON file for the Balance Sheet table.
		text_path (str): Path to file containing text blocks. 
        form_name (str): FormName, as appears in the Forms table.

	Returns:
		table_text(str) or None: Balance Sheet table text, or None if table index unknown or text does not exist
	"""

    #initial values
    problem = None 
    table_text = None 

    #get table index from log file
    table_index = read_from_json(log_path, ('text_blocks', 'data', 'table_index'))
     
    if table_index is not None: #table index found, try to get text
        text_block_list = read_from_json(text_path, ('balance', 'data'))
        if table_index >= 0 and table_index < len(text_block_list):            
            table_text = text_block_list[table_index]
        else:
            problem = 'balance sheet: table index out of range'

    if not table_text: #index not found, or no text in indexed location
        if not problem: #different problem not encountered, report default problem
            problem = 'balance sheet: no table found in text block list'
        update_json(log_path, [('problems', )], [problem])
        report_problems(form_name, log_path, [problem])
        return None
    
    return table_text 


"""Functions for extracting data from the Balance Sheet text (nested within get_table_data())"""

def get_pre_table_comments(text):
    """Returns text that precedes the Balance Sheet table body (keyword-based). 
    
    The output is used for: 
    i. Extracting sum units (typically reported above the table) for future normalization;
    ii. Cropping the table body (see crop_table()). 

    Args:
        text (str): The full Balance Sheet table text block. 
    
    Returns:
        str or None: The text preceding the first row of Balance Sheet table (including column headers), 
            or None if keyword not found.     
    """
    
    match_pre = re.search(r'assets', text, re.IGNORECASE) #look in beginning of text (until the first assets)
    if match_pre:
        return text[:match_pre.start()] #text before first table row
    else: return None       
    

def ask_sum_units(text, model=MINI, trials=1):
    """Ask GPT to identify the units in which dollar sums are reported in the Balance Sheet table. 

	Args:
		text (str): The text preceding the first row of Balance Sheet table.
		model (str, optional): The model to be used for identification; defaults to MINI.
		trials (int, optional): Maximum number times to ask GPT (maximum number of votes); defaults to 1.

	Returns:
		dict: keys: vote IDs (trial numbers), values: GPT output per vote.
	"""

    print(f"...Asking the '{model}' model to identify the dollar units in which sums are reported....")

    get_sum_units_sys = """You are an intern at a mutual fund. Your only job is to go over 10-Q or 10-K filings submitted by public companies to the SEC
    and extract very specific information from them.

    ## Context
    - You will be provided with a text containing general comments about the 'Consolidated Balance Sheets' section of a single filing.
    - Your goal is to identify the reporting units in which dollar sums are reported (e.g., 'millions', 'thousands', or absolute sum). 

    ## Constraints
    - You are to specifically extract reporting units for DOLLAR SUMS.
    - Ignore reporting units related to other amounts such as share counts, par value, per share data, etc...
    - If the text doesn't state to what the reporting units refer, you may assume (if supported by context) that the units refer to dollar sums.  
    - The default (in case no units stated) is absolute sums.
    - Only use information available to you in the provided text, and your knowledge regarding Balance Sheet data. 

    ## Output Format  
    - Return a SINGLE integer: 1, 1000, or 1000000 to represent absolute sum, thousands, or millions, respectively.
    - If you cannot identify the dollar sum reporting units based on the provided text, return `None`.
    - Apart from a single integer or `None`, do not add any explanation, text, numbers or symbols to your response.  

    ## Examples  
    ### Example 1: 
    **User Input:** '(Unaudited)(in thousands, except share and per share amounts)'
    **Expected Output:** '1000'
    ### Example 2: 
    **User Input:** 'ConocoPhillips  Millions of Dollars'
    **Expected Output:** '1000000'
    ### Example 3: 
    **User Input:** '(In millions, except number of shares which are reflected in thousands and par value)'
    **Expected Output:** '1000000'
"""

    get_sum_units_user = f"""Return a single integer as instructed, based on the comments regarding the units used in a 10-Q/10-K report's 
    'Consolidated Balance Sheets' section, contained in the following text:\n'{text}'"""

    return gpt_completion(model, get_sum_units_sys, get_sum_units_user, trials=trials) 


def get_sum_units(text, log_path, form_name):
    """Determine the the units in which dollar sums are reported in the Balance Sheet table.

	Args:
        log_path (str): Path to log JSON file for the Balance Sheet table.
		text (str): The text preceding the first row of Balance Sheet table.
		form_name (str): FormName, as appears in the Forms table.		

	Returns:
        None

	Globals:
		MINI (str): Mini model name; used first (with voting).
		GPT_4O (str): Large model name; used only if mini model fails.
    """

    unit_dict = {'sum_units': None, 'sum_divider': None}
    model_dict = {MINI: {'votes': None, 'decision': None}, GPT_4O: {'votes': None, 'decision': None}}

    for i in range(2): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per model)

        model, trials = (MINI, 3) if i == 0 else (GPT_4O, 1)

        #ask GPT model to identify the 
        votes = ask_sum_units(text, model, trials)
        model_dict[model]['votes'] = votes
        model_dict[model]['decision'] = count_votes(votes)

        if ( 
            model_dict[model]['decision'] == 'null' 
            or model_dict[model]['decision'] == 'None' 
            or model_dict[model]['decision'] is None
        ):
            problems_list.append('units: sum reporting units not found')
            continue #try with larger model

        units = convert_model_output(model_dict[model]['decision'], 'int')

        if not units:
            problems_list.append('units: sum reporting units not convertible to int')
            continue #try with larger model            

        if units not in [1, 1000, 10**6]:
            problems_list.append('units: sum reporting units out of range')
            continue #try with larger model

        if not problems_list: #valid result obtained
            unit_dict['sum_units'] = units
            sum_divider =  10**6 / units #used to normalize sums to millions
            unit_dict['sum_divider'] = sum_divider        
            break #don't run again
        
    update_json(log_path, 
                [('units', ), ('units', 'model'), ('problems', )], 
                [unit_dict, model_dict, problems_list]
                )

    if problems_list: #after both models, problems remain        
        report_problems(form_name, log_path, problems_list) #print out detected problems


def ask_post_table_text(text, model=MINI, trials=1, trial_counter=0):
    """Ask GPT to identify a str beginning directly after the end of the table body. 

	Args:
		text (str): The text block containing the Balance Sheet table, excluding the pre-table text.
		model (str, optional): The model to be used for identification; defaults to MINI.
		trials (int, optional): Maximum number times to ask GPT (maximum number of votes); defaults to 1.
        trial_counter (int): Index of first trial upon function call; deafult value = 0.

	Returns:
		dict: keys: trial numbers, values: GPT output per trial.
	"""

    get_post_table_text_sys = """You are an intern at a mutual fund. Your only job is to go over 10-Q or 10-K filings submitted by public companies to the SEC
    and extract very specific information from them.

    ## Context
    - You will be provided with a text block which contains the 'Consolidated Balance Sheets' section of a single filing.
    - The table structure may have been botched when the text was extracted - for example, sentences and table elements may be broken apart by spaces and tabs, the separation between 2 consecutive statements/rows may not be clear, etc...
    - Your goal is to identify the end of the Balance Sheet TABLE. 
    - You are to return the first 100 characters directly after the last character belonging to the Balance Sheet table.  
    - The text to-be-returned will most likely begin with a tab.

    ## Constraints
    - Only use information available to you in the provided text, and your knowledge regarding Balance Sheet data. 
    - Make sure that the text you return is identical to the post-table text; do not omit any characters including double spaces, special characters, tabs, etc.
    - If you cannot identify the end of the table, return `None`

    ## Output Format  
    - A single string composed of 100 characters starting directly after the table ends. If there are less than 100 characters remaining after table end, return as many characters as exist directly after the Balance Sheet table text - do NOT add any characters that do not exist in the post-table text!
    - Apart from the str containing the post-table text or `None`, do not add any explanation, text, numbers or symbols to your response.  

    ## Examples
    ### Example 1:  
    **User Input:** "ASSETS:\tCurrent assets:\tCash and cash equivalents\t$\t28,408\t$\t23,646\tMarketable securities\t34,074\t24,658\tAccounts receivable, net\t19,549\t28,184\tInventories\t7,351\t4,946\tVendor non-trade receivables\t19,637\t32,748\tOther current assets\t13,640\t21,223\tTotal current assets\t122,659\t135,405\tNon-current assets:\tMarketable securities\t104,061\t120,805\tProperty, plant and equipment, net\t43,550\t42,117\tOther non-current assets\t64,768\t54,428\tTotal non-current assets\t212,379\t217,350\tTotal assets\t$\t335,038\t$\t352,755\tLIABILITIES AND SHAREHOLDERS  EQUITY:\tCurrent liabilities:\tAccounts payable\t$\t46,699\t$\t64,115\tOther current liabilities\t58,897\t60,845\tDeferred revenue\t8,158\t7,912\tCommercial paper\t3,993\t9,982\tTerm debt\t7,216\t11,128\tTotal current liabilities\t124,963\t153,982\tNon-current liabilities:\tTerm debt\t98,071\t98,959\tOther non-current liabilities\t51,730\t49,142\tTotal non-current liabilities\t149,801\t148,101\tTotal liabilities\t274,764\t302,083\tCommitments and contingencies\tShareholders  equity:\tCommon stock and additional paid-in capital, $\t0.00001\tpar value:\t50,400,000\tshares authorized;\t15,647,868\tand\t15,943,425\tshares issued and outstanding, respectively\t70,667\t64,849\tRetained earnings/(Accumulated deficit)\t1,408\t(\t3,068\t)\tAccumulated other comprehensive income/(loss)\t(\t11,801\t)\t(\t11,109\t)\tTotal shareholders  equity\t60,274\t50,672\tTotal liabilities and shareholders  equity\t$\t335,038\t$\t352,755\tSee accompanying Notes to Condensed Consolidated Financial Statements.\tApple Inc. | Q3 2023 Form 10-Q | 3\tApple Inc.\tCONDENSED CONSOLIDATED STATEMENTS OF SHAREHOLDERS  EQUITY (Unaudited)\t(In millions, except per share amounts)\tThree Months Ended\tNine Months Ended\tJuly 1,\t2023\tJune 25,\t2022\tJuly 1,\t2023\tJune 25,\t2022\tTotal shareholders  equity, beginning balances\t$\t62,158\t$\t67,399\t$\t50,672\t$\t63,090\tCommon stock and additional paid-in capital:\tBeginning balances\t69,568\t61,181\t64,849\t57,365\tCommon stock issued\t \t \t690\t593\tCommon stock withheld related to net share settlement of equity awards\t(\t1,595\t)\t(\t1,371\t)\t(\t3,310\t)\t(\t2,783\t)\tShare-based compensation\t2,694\t2,305\t8,438\t6,940\tEnding balances\t70,667\t62,115\t70,667\t62,115\tRetained earnings/(Accumulated deficit):\tBeginning balances\t4,336\t12,712\t(\t3,068\t)\t5,562\tNet income\t19,881\t19,442\t74,039\t79,082\tDividends and dividend equivalents declared\t(\t3,811\t)\t(\t3,760\t)\t(\t11,207\t)\t(\t11,058\t)\tCommon stock withheld related to net share settlement of equity awards\t(\t858\t)\t(\t1,403\t)\t(\t1,988\t)\t(\t3,323\t)\tCommon stock repurchased\t(\t18,140\t)\t(\t21,702\t)\t(\t56,368\t)\t(\t64,974\t)\tEnding balances\t1,408\t5,289\t1,408\t5,289\tAccumulated other comprehensive income/(loss):\tBeginning balances\t(\t11,746\t)\t(\t6,494\t)\t(\t11,109\t)\t163\tOther comprehensive income/(loss)\t(\t55\t)\t(\t2,803\t)\t(\t692\t)\t(\t9,460\t)\tEnding balances\t(\t11,801\t)\t(\t9,297\t)\t(\t11,801\t)\t(\t9,297\t)\tTotal shareholders  equity, ending balances\t$\t60,274\t$\t58,10"
    **Expected Output:** '\tSee accompanying Notes to Condensed Consolidated Financial Statements.\tApple Inc. | Q3 2023 Form 1'
    ### Example 2:
    **User Input:** "CONSOLIDATED BALANCE SHEETS\t(in millions, except share data)\tDecember 30,\tDecember 31,\t2023\t2022\tASSETS\tCurrent assets:\tCash and cash equivalents\t$\t171\t$\t117\tAccounts receivable, net of allowance for credit losses of $\t83\tand $\t65\t(1)\t1,863\t1,442\tInventories, net\t1,815\t1,963\tPrepaid expenses and other\t639\t466\tTotal current assets\t4,488\t3,988\tProperty and equipment, net\t498\t383\tOperating lease right-of-use assets\t325\t284\tGoodwill\t3,875\t2,893\tOther intangibles, net\t916\t587\tInvestments and other\t471\t472\tTotal assets\t$\t10,573\t$\t8,607\tLIABILITIES, REDEEMABLE NONCONTROLLING INTERESTS AND\tSTOCKHOLDERS' EQUITY\tCurrent liabilities:\tAccounts payable\t$\t1,020\t$\t1,004\tBank credit lines\t264\t103\tCurrent maturities of long-term debt\t150\t6\tOperating lease liabilities\t80\t73\tAccrued expenses:\tPayroll and related\t332\t314\tTaxes\t137\t132\tOther\t700\t592\tTotal current liabilities\t2,683\t2,224\tLong-term debt (1)\t1,937\t1,040\tDeferred income taxes\t54\t36\tOperating lease liabilities\t310\t275\tOther liabilities\t436\t361\tTotal liabilities\t5,420\t3,936\tRedeemable noncontrolling interests\t864\t576\tCommitments and contingencies\t(nil)\t(nil)\tStockholders' equity:\tPreferred stock, $\t0.01\tpar value,\t1,000,000\tshares authorized,\tnone\toutstanding\t-\t-\tCommon stock, $\t0.01\tpar value,\t480,000,000\tshares authorized,\t129,247,765\toutstanding on December 30, 2023 and\t131,792,817\toutstanding on December 31, 2022\t1\t1\tAdditional paid-in capital\t-\t-\tRetained earnings\t3,860\t3,678\tAccumulated other comprehensive loss\t(\t206\t)\t(\t233\t)\tTotal Henry Schein, Inc. stockholders' equity\t3,655\t3,446\tNoncontrolling interests\t634\t649\tTotal stockholders' equity\t4,289\t4,095\tTotal liabilities, redeemable noncontrolling\tinterests and stockholders' equity\t$\t10,573\t$\t8,607\t(1)\tAmounts presented include balances held by our consolidated variable interest entity (\u201cVIE\u201d).\tAt December 30, 2023 and\tDecember 31, 2022, includes trade accounts receivable of $\t284\tmillion and $\t327\tmillion, respectively, and long-term debt of $\t210\tmillion and $\t255\tmillion, respectively.\tSee\tNote 1 \u2013 Basis of Presentation and Significant Accounting Policies\tfor further\tinformation.\tTable of Contents\tSee accompanying notes.\t66\tHENRY SCHEIN, INC.\tCONSOLIDATED STATEMENTS\tOF INCOME\t(in millions, except share and per share data)\tYears\tEnded\tDecember 30,\tDecember 31,\tDecember 25,\t2023\t2022\t2021\tNet sales\t$\t12,339\t$\t12,647\t$\t12,401\tCost of sales\t8,478\t8,816\t8,727\tGross profit\t3,861\t3,831\t3,674\tOperating expenses:\tSelling, general and administrative\t2,956\t2,771\t2,634\tDepreciation and amortization\t210\t182\t180\tRestructuring and integration costs\t80\t131\t8\tOperating income\t615\t747\t852\tOther income (expense):\tInterest income\t17\t8\t6\tInterest expense\t(\t87\t)\t(\t35\t)\t(\t27\t)\tOther, net\t(\t3\t)\t1\t-\tIncome before taxes, equity in\tearnings of affiliates and noncontrolling interests\t542\t721\t831\tIncome taxes\t(\t120\t)\t(\t170\t)\t(\t198\t)\tEquity in earnings of affiliates, net of tax\t14\t15\t20\tGain on sale of equity investment\t-\t-\t7\tNet income\t436\t566\t660\tLess: Net income attributab"
    **Expected Output:** '\t(1)\tAmounts presented include balances held by our consolidated variable interest entity (“VIE”).\tA'    
    ### Example 3:
    **User Input:** "CONSOLIDATED BALANCE SHEET\t(Unaudited)\tJune 27,\tDecember 31,\t(In millions except share and per share amounts)\t2020\t2019\tAssets\tCurrent Assets:\tCash and cash equivalents\t$\t5,818\t$\t2,399\tAccounts receivable, less allowances of $\t113\tand $\t102\t4,478\t4,349\tInventories\t3,648\t3,370\tContract assets, net\t686\t603\tOther current assets\t1,145\t1,172\tTotal current assets\t15,775\t11,893\tProperty, Plant and Equipment, Net\t4,887\t4,749\tAcquisition-related Intangible Assets, Net\t13,170\t14,014\tOther Assets\t2,061\t2,011\tGoodwill\t25,700\t25,714\tTotal Assets\t$\t61,593\t$\t58,381\tLiabilities and Shareholders' Equity\tCurrent Liabilities:\tShort-term obligations and current maturities of long-term obligations\t$\t675\t$\t676\tAccounts payable\t1,385\t1,920\tAccrued payroll and employee benefits\t1,184\t1,010\tContract liabilities\t975\t916\tOther accrued expenses\t1,794\t1,675\tTotal current liabilities\t6,013\t6,197\tDeferred Income Taxes\t1,750\t2,192\tOther Long-term Liabilities\t3,317\t3,241\tLong-term Obligations\t20,638\t17,076\tShareholders' Equity:\tPreferred stock, $\t100\tpar value,\t50,000\tshares authorized;\tnone\tissued\tCommon stock, $\t1\tpar value,\t1,200,000,000\tshares authorized;\t435,885,737\tand\t434,416,804\tshares issued\t436\t434\tCapital in excess of par value\t15,334\t15,064\tRetained earnings\t23,860\t22,092\tTreasury stock at cost,\t40,296,337\tand\t35,676,421\tshares\t(\t6,766\t)\t(\t5,236\t)\tAccumulated other comprehensive items\t(\t2,989\t)\t(\t2,679\t)\tTotal shareholders' equity\t29,875\t29,675\tTotal Liabilities and Shareholders' Equity\t$\t61,593\t$\t58,381\tThe accompanying notes are an integral part of these consolidated financial statements.\t3\tTHERMO FISHER SCIENTIFIC INC.\tCONSOLIDATED STATEMENT OF INCOME\t(Unaudited)\tThree Months Ended\tSix Months Ended\tJune 27,\tJune 29,\tJune 27,\tJune 29,\t(In millions except per share amounts)\t2020\t2019\t2020\t2019\tRevenues\tProduct revenues\t$\t5,250\t$\t4,827\t$\t9,880\t$\t9,547\tService revenues\t1,667\t1,489\t3,267\t2,894\tTotal revenues\t6,917\t6,316\t13,147\t12,441\tCosts and Operating Expenses:\tCost of product revenues\t2,391\t2,478\t4,731\t4,892\tCost of service revenues\t1,149\t1,015\t2,299\t2,019\tSelling, general and administrative expenses\t1,710\t1,565\t3,261\t3,093\tResearch and development expenses\t264\t246\t509\t494\tRestructuring and other costs (income), net\t12\t(\t484\t)\t50\t(\t473\t)\tTotal costs and operating expenses\t5,526\t4,820\t10,850\t10,025\tOperating Income\t1,391\t1,496\t2,297\t2,416\tInterest Income\t8\t60\t44\t127\tInterest Expense\t(\t137\t)\t(\t181\t)\t(\t263\t)\t(\t370\t)\tOther (Expense) Income, Net\t(\t9\t)\t18\t3\t37\tIncome Before Income Taxes\t1,253\t1,393\t2,081\t2,210\tProvision for Income Taxes\t(\t97\t)\t(\t274\t)\t(\t137\t)\t(\t276\t)\tNet Income\t$\t1,156\t$\t1,119\t$\t1,944\t$\t1,934\tEarnings per Share\tBasic\t$\t2.92\t$\t2.80\t$\t4.91\t$\t4.84\tDiluted\t$\t2.90\t$\t2.77\t$\t4.87\t$\t4.80\tWeighted Average Shares\tBasic\t395\t400\t396\t400\tDiluted\t398\t403\t399\t403\tThe accompanying notes are an integral part of these consolidated financial statements.\t4\tTHERMO FISHER SCIENTIFIC INC.\tCONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\t(Unaudited)\tThree Months Ended\tSix Months E"
    **Expected Output:** '\tThe accompanying notes are an integral part of these consolidated financial statements.\t3\tTHERMO FIS'
    """

    get_post_table_text_user = f"""The following text contains the 'Consolidated Balance Sheets' section of a single 10-Q/10-K filing:\n{text}
    Return a string of up to 100 characters as instructed, containing the first characters directly after the contents of the 'Consolidated Balance Sheets' table.
    """

    return gpt_completion(model, get_post_table_text_sys, get_post_table_text_user, trials=trials, trial_counter=trial_counter) 


def crop_table(table_text, pre_table_comments, post_table_text):
    """Crops Balance Sheet table text to return only the table body (excluding pre- or post-table comments/text).
    
    Args:
        table_text (str): The full Balance Sheet table text block. 
        pre_Table_comments (str): The text preceding the table body.
        post_table_text (str): The text trailing the table body.

    Returns:
        str or None: The cropped table text, or None if the post-table text not within the text block.        
    """

    min_end_table_ratio = 0.3 #a the minimal distance from start to end is min_end_table_ratio * len(table_text)

    post_table_index = table_text.find(post_table_text, int(len(table_text) * min_end_table_ratio))
  
    if post_table_index < 0:
        return None
    
    return table_text[len(pre_table_comments) : post_table_index]

    
def get_table_body(table_text, pre_table_comments, log_path, form_name):
    """Extract the 'pure' table body from the Balance Sheet text block
    
    Args:
        table_text (str): The full Balance Sheet table text block. 
        pre_Table_comments (str): The text preceding the table body.
		log_path (str): Path to log JSON file for the Balance Sheet table.
		form_name (str): FormName, as appears in the Forms table.		

    Returns:
        table_body (str or None): The cropped table text, or None if the post-table text not within the text block.

	Globals:
		MINI (str): Mini model name; used first (until a valid response is obtained, or trial limit reached).
		GPT_4O (str): Large model name; used only if mini model fails.   
    """

    #for this task, voting doesn't make sense - the model may return text with slightly different lengths.
    #instead, trials are executed until a valid decision is reached

    max_mini_trials = 3 #maximum number of trials using the mini model
    model_dict = {MINI: {'trials': {}, 'decision': None}, GPT_4O: {'trials': {}, 'decision': None}}

    for i in range(max_mini_trials+1): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per trial)

        model, trials, trial_counter = (MINI, i, i) if i < max_mini_trials else (GPT_4O, 1, 0)

        print(f"...Asking the '{model}' model to identify the end of the Balance Sheet table (trial {trial_counter})....")

        #ask GPT model to identify the post-table text
        response = ask_post_table_text(table_text[len(pre_table_comments):], model=model, trials=trials, trial_counter=trial_counter)
        post_table_text = list(response.values())[0]
        if (
            post_table_text == 'None' 
            or post_table_text == 'null' 
            or not post_table_text
        ):
            post_table_text = None

        model_dict[model]['trials'][trial_counter] = post_table_text
        model_dict[model]['decision'] = post_table_text #decision = last trial (either valid or reached max trials)

        try: #make sure that text wasn't missed due to extra quotations inserted by the model
            post_table_text = ast.literal_eval(post_table_text) 
        except:
            pass

        if not post_table_text:                                
            problems_list.append('post-table text: content not found')
            continue #retry

        if not isinstance(post_table_text, str):
            problems_list.append('post-table text: content not str')
            continue #retry

        if not problems_list:
            #get "pure" table text without pre or post text
            table_body = crop_table(table_text, pre_table_comments, post_table_text)

            if not table_body:
                problems_list.append('post-table text: no match to text block')
                continue #retry

            print("- Balance Sheet table text cropped....")      
            break #don't run again if response was valid

    if problems_list: #after both models, problems remain 
        table_body = None       
        report_problems(form_name, log_path, problems_list) #print out detected problems

    update_json(log_path, 
                [('post_table_text', ), ('post_table_text', 'model'), ('table_body', ), ('problems', )], 
                [post_table_text, model_dict, table_body, problems_list]
                )
    
    return table_body
    

def ask_table_json(table_body, model=MINI, trials=1, trial_counter=0):
    """Ask GPT to produce structured JSON data representing the Balance Sheet table. 

	Args:
		table_body (str): The cropped Balance Sheet table text.
		model (str, optional): The model to be used for identification; defaults to MINI.
		trials (int, optional): Maximum number times to ask GPT (maximum number of votes); defaults to 1.
        trial_counter (int): Index of first trial upon function call; deafult value = 0.

	Returns:
		dict: keys: trial numbers, values: GPT output per trial.
	"""

    get_table_json_sys = """You are a data engineer working for a mutual fund. 
    Your only job is to convert free text extracted from SEC filings (10-Q and 10-K forms) into JSON data with a hierarchical structure.

    ## Context
    - You will be provided with the text of the 'Consolidated Balance Sheets' table extracted from a single 10-Q or 10-K filing.
    - The table structure may have been botched when the text was extracted - for example, row headers may be broken apart by spaces and tabs, the separation between 2 consecutive rows may not be clear, etc...
    - Your goal is to produce a JSON object that preserves the original hierarchical structure of the table. Use your knowledge about Balance Sheet tables to help you determine the structure.
    - The text data you will receive does not contain any column headers. 
    - The text data contains row headers and their associated values (or nested headers). 
    - Based on your knowledge of balance sheet structure in 10-Q and 10-K filings, arrange the data such that each row header is a key pointing to either an array of 2 numerical values, or to a nested key.
    - If a certain value appears as a dash, represent this value as 0. However, if a value in a column is completely missing (empty space or tab), represent it as the JSON literal null.
    - Do NOT exclude any textual comments contained in row headers!
    - If the beginning and/or end of the provided text includes text that you think does not belong to the Balance Sheet table itself (e.g., leading/trailing comments), disregard this part of the text.
    
    ## Constraints
    - Only use information available to you in the provided text and your knowledge regarding Balance Sheet data. 
    - Under the liabilities section, there will usually be a row header referring to 'Common stock'. This row header often includes a long text which contains numerical values. Make sure that the key you create is an exact copy of the header - do not add or remove any text.
    - If you cannot decide regarding the hierarchical structure of the table, return `None`.
    
    ## Output Format  
    - Return a single JSON object that represents the Balance Sheet table, or `None` if you cannot determine its hierarchical structure.
    - The JSON object must follow the JSON specification exactly: use double quotes for all keys and string values.
    - The entire JSON must be output as a single line—no newline characters, extra spaces, markdown formatting, or escape sequences.
    - Each key should either point to another JSON object or to a JSON array containing exactly two integer values (or null for missing values).
    - Do not include any additional text, commentary, or formatting beyond the JSON object.

    ## Examples
    ### Example 1:  
    **User Input:** "ASSETS:\tCurrent assets:\tCash and cash equivalents\t$\t28,408\t$\t23,646\tMarketable securities\t34,074\t24,658\tAccounts receivable, net\t19,549\t28,184\tInventories\t7,351\t4,946\tVendor non-trade receivables\t19,637\t32,748\tOther current assets\t13,640\t21,223\tTotal current assets\t122,659\t135,405\tNon-current assets:\tMarketable securities\t104,061\t120,805\tProperty, plant and equipment, net\t43,550\t42,117\tOther non-current assets\t64,768\t54,428\tTotal non-current assets\t212,379\t217,350\tTotal assets\t$\t335,038\t$\t352,755\tLIABILITIES AND SHAREHOLDERS  EQUITY:\tCurrent liabilities:\tAccounts payable\t$\t46,699\t$\t64,115\tOther current liabilities\t58,897\t60,845\tDeferred revenue\t8,158\t7,912\tCommercial paper\t3,993\t9,982\tTerm debt\t7,216\t11,128\tTotal current liabilities\t124,963\t153,982\tNon-current liabilities:\tTerm debt\t98,071\t98,959\tOther non-current liabilities\t51,730\t49,142\tTotal non-current liabilities\t149,801\t148,101\tTotal liabilities\t274,764\t302,083\tCommitments and contingencies\tShareholders  equity:\tCommon stock and additional paid-in capital, $\t0.00001\tpar value:\t50,400,000\tshares authorized;\t15,647,868\tand\t15,943,425\tshares issued and outstanding, respectively\t70,667\t64,849\tRetained earnings/(Accumulated deficit)\t1,408\t(\t3,068\t)\tAccumulated other comprehensive income/(loss)\t(\t11,801\t)\t(\t11,109\t)\tTotal shareholders  equity\t60,274\t50,672\tTotal liabilities and shareholders  equity\t$\t335,038\t$\t352,755"
    **Expected Output:** {"ASSETS": {"Current assets": {"Cash and cash equivalents": [28408, 23646], "Marketable securities": [34074, 24658], "Accounts receivable, net": [19549, 28184], "Inventories": [7351, 4946], "Vendor non-trade receivables": [19637, 32748], "Other current assets": [13640, 21223], "Total current assets": [122659, 135405]}, "Non-current assets": {"Marketable securities": [104061, 120805], "Property, plant and equipment, net": [43550, 42117], "Other non-current assets": [64768, 54428], "Total non-current assets": [212379, 217350]}, "Total assets": [335038, 352755]}, "LIABILITIES AND SHAREHOLDERS EQUITY": {"Current liabilities": {"Accounts payable": [46699, 64115], "Other current liabilities": [58897, 60845], "Deferred revenue": [8158, 7912], "Commercial paper": [3993, 9982], "Term debt": [7216, 11128], "Total current liabilities": [124963, 153982]}, "Non-current liabilities": {"Term debt": [98071, 98959], "Other non-current liabilities": [51730, 49142], "Total non-current liabilities": [149801, 148101]}, "Total liabilities": [274764, 302083], "Commitments and contingencies": null, "Shareholders equity": {"Common stock and additional paid-in capital, $ 0.00001 par value: 50,400,000 shares authorized; 15,647,868 and 15,943,425 shares issued and outstanding, respectively": [70667, 64849], "Retained earnings/(Accumulated deficit)": [1408, -3068], "Accumulated other comprehensive income/(loss)": [-11801, -11109], "Total shareholders equity": [60274, 50672]}, "Total liabilities and shareholders equity": [335038, 352755]}}
    ### Example 2:
    **User Input:** "ASSETS\tCurrent assets:\tCash and cash equivalents\t................\t$\t88,115\t$\t56,885\tAccounts receivable, net of reserves of $52,205 and $53,121\t............\t1,193,054\t1,168,776\tInventories, net\t...................\t1,370,376\t1,415,512\tPrepaid expenses and other\t...................\t457,566\t451,033\tAssets of discontinued operations\t................\t-\t1,083,014\tTotal current assets\t................\t3,109,111\t4,175,220\tProperty and equipment, net\t................\t315,393\t314,221\tOperating lease right-of-use asset, net\t...............\t248,122\t-\tGoodwill\t......................\t2,413,566\t2,081,029\tOther intangibles, net\t..................\t654,668\t376,031\tInvestments and other\t....................\t404,004\t420,367\tAssets of discontinued operations\t..................\t-\t1,133,659\tTotal assets\t..................\t$\t7,144,864\t$\t8,500,527\tLIABILITIES AND STOCKHOLDERS' EQUITY\tCurrent liabilities:\tAccounts payable\t..................\t$\t695,204\t$\t785,756\tBank credit lines\t..................\t299,914\t951,458\tCurrent maturities of long-term debt\t.................\t9,117\t8,280\tOperating lease liabilities\t..................\t68,460\t-\tLiabilities of discontinued operations\t.................\t-\t577,607\tAccrued expenses:\tPayroll and related\t...................\t210,016\t242,876\tTaxes\t....................\t162,483\t154,613\tOther\t.....................\t433,582\t498,237\tTotal current liabilities\t..................\t1,878,776\t3,218,827\tLong-term debt\t.....................\t973,500\t980,344\tDeferred income taxes\t....................\t76,850\t27,218\tOperating lease liabilities\t....................\t187,308\t-\tOther liabilities\t..................\t327,057\t357,741\tLiabilities of discontinued operations\t...............\t-\t62,453\tTotal liabilities\t....................\t3,443,491\t4,646,583\tRedeemable noncontrolling interests\t...............\t286,700\t219,724\tRedeemable noncontrolling interests from discontinued operations\t...........\t-\t92,432\tCommitments and contingencies\t..................\tStockholders' equity:\tPreferred stock, $.01 par value, 1,000,000 shares authorized,\tnone outstanding\t.................\t-\t-\tCommon stock, $.01 par value, 480,000,000 shares authorized,\t148,996,092 outstanding on March 30, 2019 and\t151,401,668 outstanding on December 29, 2018\t...............\t1,490\t1,514\tAdditional paid-in capital\t................\t86,128\t-\tRetained earnings\t..................\t2,859,182\t3,208,589\tAccumulated other comprehensive loss\t...............\t(149,878)\t(248,771)\tTotal Henry Schein, Inc. stockholders' equity\t..............\t2,796,922\t2,961,332\tNoncontrolling interests\t..................\t617,751\t580,456\tTotal stockholders' equity\t.................\t3,414,673\t3,541,788\tTotal liabilities, redeemable noncontrolling interests and stockholders' equity\t............\t$\t7,144,864\t$\t8,500,527"
    **Expected Output:** {"ASSETS": {"Current assets": {"Cash and cash equivalents": [88115, 56885], "Accounts receivable, net of reserves of $52,205 and $53,121": [1193054, 1168776], "Inventories, net": [1370376, 1415512], "Prepaid expenses and other": [457566, 451033], "Assets of discontinued operations": [0, 1083014], "Total current assets": [3109111, 4175220]}, "Property and equipment, net": [315393, 314221], "Operating lease right-of-use asset, net": [248122, 0], "Goodwill": [2413566, 2081029], "Other intangibles, net": [654668, 376031], "Investments and other": [404004, 420367], "Assets of discontinued operations": [0, 1133659], "Total assets": [7144864, 8500527]}, "LIABILITIES AND STOCKHOLDERS" EQUITY": {"Current liabilities": {"Accounts payable": [695204, 785756], "Bank credit lines": [299914, 951458], "Current maturities of long-term debt": [9117, 8280], "Operating lease liabilities": [68460, 0], "Liabilities of discontinued operations": [0, 577607], "Accrued expenses": {"Payroll and related": [210016, 242876], "Taxes": [162483, 154613], "Other": [433582, 498237]}, "Total current liabilities": [1878776, 3218827]}, "Long-term debt": [973500, 980344], "Deferred income taxes": [76850, 27218], "Operating lease liabilities": [187308, 0], "Other liabilities": [327057, 357741], "Liabilities of discontinued operations": [0, 62453], "Total liabilities": [3443491, 4646583], "Redeemable noncontrolling interests": [286700, 219724], "Redeemable noncontrolling interests from discontinued operations": [0, 92432], "Commitments and contingencies": {}, "Stockholders" equity": {"Preferred stock, $.01 par value, 1,000,000 shares authorized, none outstanding": [0, 0], "Common stock, $.01 par value, 480,000,000 shares authorized, 148,996,092 outstanding on March 30, 2019 and 151,401,668 outstanding on December 29, 2018": [1490, 1514], "Additional paid-in capital": [86128, 0], "Retained earnings": [2859182, 3208589], "Accumulated other comprehensive loss": [-149878, -248771], "Total Henry Schein, Inc. stockholders" equity": [2796922, 2961332], "Noncontrolling interests": [617751, 580456], "Total stockholders" equity": [3414673, 3541788]}, "Total liabilities, redeemable noncontrolling interests and stockholders" equity": [7144864, 8500527]}}
    ### Example 3:
    **User Input:** "ASSETS\tCurrent assets:\tCash and cash equivalents\t$\t119,133\t$\t421,185\tAccounts receivable, net of reserves of $\t73,095\tand $\t88,030\t1,551,946\t1,424,787\tInventories, net\t1,784,050\t1,512,499\tPrepaid expenses and other\t457,232\t432,944\tTotal current assets\t3,912,361\t3,791,415\tProperty and equipment, net\t355,675\t342,004\tOperating lease right-of-use assets\t329,886\t288,847\tGoodwill\t2,779,234\t2,504,392\tOther intangibles, net\t645,832\t479,429\tInvestments and other\t397,764\t366,445\tTotal assets\t$\t8,420,752\t$\t7,772,532\tLIABILITIES AND STOCKHOLDERS' EQUITY\tCurrent liabilities:\tAccounts payable\t$\t1,057,127\t$\t1,005,655\tBank credit lines\t59,394\t73,366\tCurrent maturities of long-term debt\t9,638\t109,836\tOperating lease liabilities\t77,383\t64,716\tAccrued expenses:\tPayroll and related\t345,438\t295,329\tTaxes\t157,446\t138,671\tOther\t594,979\t595,529\tTotal current liabilities\t2,301,405\t2,283,102\tLong-term debt\t705,540\t515,773\tDeferred income taxes\t37,248\t30,065\tOperating lease liabilities\t270,152\t238,727\tOther liabilities\t388,211\t392,781\tTotal liabilities\t3,702,556\t3,460,448\tRedeemable noncontrolling interests\t612,582\t327,699\tCommitments and contingencies\tStockholders' equity:\tPreferred stock, $\t0.01\tpar value,\t1,000,000\tshares authorized,\tnone\toutstanding\t-\t-\tCommon stock, $\t0.01\tpar value,\t480,000,000\tshares authorized,\t139,129,543\toutstanding on September 25, 2021 and\t142,462,571\toutstanding on December 26, 2020\t1,391\t1,425\tAdditional paid-in capital\t-\t-\tRetained earnings\t3,594,238\t3,454,831\tAccumulated other comprehensive loss\t(\t137,640\t)\t(\t108,084\t)\tTotal Henry Schein, Inc. stockholders' equity\t3,457,989\t3,348,172\tNoncontrolling interests\t647,625\t636,213\tTotal stockholders' equity\t4,105,614\t3,984,385\tTotal liabilities, redeemable noncontrolling interests and stockholders' equity\t$\t8,420,752\t$\t7,772,532\t"
    **Expected Output:** {"ASSETS": {"Current assets": {"Cash and cash equivalents": [119133, 421185], "Accounts receivable, net of reserves of $ 73,095 and $ 88,030": [1551946, 1424787], "Inventories, net": [1784050, 1512499], "Prepaid expenses and other": [457232, 432944], "Total current assets": [3912361, 3791415]}, "Property and equipment, net": [355675, 342004], "Operating lease right-of-use assets": [329886, 288847], "Goodwill": [2779234, 2504392], "Other intangibles, net": [645832, 479429], "Investments and other": [397764, 366445], "Total assets": [8420752, 7772532]}, "LIABILITIES AND STOCKHOLDERS" EQUITY": {"Current liabilities": {"Accounts payable": [1057127, 1005655], "Bank credit lines": [59394, 73366], "Current maturities of long-term debt": [9638, 109836], "Operating lease liabilities": [77383, 64716], "Accrued expenses": {"Payroll and related": [345438, 295329], "Taxes": [157446, 138671], "Other": [594979, 595529]}, "Total current liabilities": [2301405, 2283102]}, "Long-term debt": [705540, 515773], "Deferred income taxes": [37248, 30065], "Operating lease liabilities": [270152, 238727], "Other liabilities": [388211, 392781], "Total liabilities": [3702556, 3460448], "Redeemable noncontrolling interests": [612582, 327699], "Commitments and contingencies": null, "Stockholders" equity": {"Preferred stock, $ 0.01 par value, 1,000,000 shares authorized, none outstanding": [0, 0], "Common stock, $ 0.01 par value, 480,000,000 shares authorized, 139,129,543 outstanding on September 25, 2021 and 142,462,571 outstanding on December 26, 2020": [1391, 1425], "Additional paid-in capital": [0, 0], "Retained earnings": [3594238, 3454831], "Accumulated other comprehensive loss": [-137640, -108084], "Total Henry Schein, Inc. stockholders" equity": [3457989, 3348172], "Noncontrolling interests": [647625, 636213], "Total stockholders" equity": [4105614, 3984385]}, "Total liabilities, redeemable noncontrolling interests and stockholders" equity": [8420752, 7772532]}}
    """

    get_table_json_user = f"""Return the following text as a valid single-line JSON object, as instructed: {table_body}"""    

    return gpt_completion(model, get_table_json_sys, get_table_json_user, trials=trials, trial_counter=trial_counter) 


def get_table_json(table_body, log_path, table_path, form_name):
    """Export the Balance Sheet table to a structured JSON file.
    
    Args:
        table_body (str): The cropped Balance Sheet table text.
		log_path (str): Path to log JSON file for the Balance Sheet table.
        table_path (str): Path to file where JSON version of the Balance Sheet table will be stored.
		form_name (str): FormName, as appears in the Forms table.

    Returns:
        None
    
	Globals:
		MINI (str): Mini model name; used first (until a valid response is obtained, or trial limit reached).
		GPT_4O (str): Large model name; used only if mini model fails.   
    """

    #for this task, voting doesn't make sense - the model may return json structures with slight variations. 
    #instead, trials are executed until a valid decision is reached

    max_mini_trials = 3 #maximum number of trials using the mini model
    table_json_created = False #default value - indicates that a JSON version of the table was not (yet) created

    model_dict = {MINI: {'trials': {}, 'decision': None}, GPT_4O: {'trials': {}, 'decision': None}}
    min_key_count = 20 #minimal number of keys expected in json data

    for i in range(max_mini_trials+1): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per trial)

        model, trials, trial_counter = (MINI, i, i) if i < max_mini_trials else (GPT_4O, 1, 0)

        print(f"...Asking the '{model}' model to convert Balance Sheet text into a JSON data file (trial {trial_counter})....")

        #ask GPT model to convert the table text into a JSON-like str
        response = ask_table_json(table_body, model=model, trials=trials, trial_counter=trial_counter)
        response = list(response.values())[0]
        model_dict[model]['trials'][trial_counter] = response

        if (
            response == 'None' 
            or response == 'null'
            or not response
        ):
            problems_list.append('json output: model failed to produce table JSON')
            continue #retry

        #a common issue is that the JSON-like str is invalid due to a missing closing '}'; I chose to fix this algorithmically 
        if response.count('{') > response.count('}'):
            response_for_dict = response + '}'
        else:
            response_for_dict = response
        
        try: #try to convert the JSON-like str into a dict
            table_json = json.loads(response_for_dict)
            if not isinstance(table_json, dict):
                problems_list.append('json output: table not in dict form')
                continue #retry
            model_dict[model]['decision'] = response #if successful, the last response is the model's decision
        except:
            table_json = None
            problems_list.append('json output: table not in valid JSON format')
            continue #retry

        #extracts all keys from all levels of table_json dict
        expr = parse("$..*") 
        keys = [match.path.fields[-1] for match in expr.find(table_json)] 
        
        if len(keys) < min_key_count: 
            problems_list.append('json output: not enough keys in balance sheet JSON')
            continue #retry

        if not problems_list: 
            with open(table_path, 'w') as f: #save json table to file
                json.dump(table_json, f, indent=4)
            table_json_created = True #note in log that json table created
            break #don't run again if response was valid

    if problems_list: #after both models, problems remain 
        report_problems(form_name, log_path, problems_list) #print out detected problems

    update_json(log_path, 
                [('table_json_created', ), ('table_json_created', 'model'), ('problems', )], 
                [table_json_created, model_dict, problems_list]
                )
   

def get_table_data(table_text, log_path, table_path, form_name):
    """Main function for exporting the Balance Sheet table to a structured JSON file.

    Crops the Balance Sheet text block to get the "pure" table body,
    asks GPT to export the table to a JSON data file preserving the table's stucture,
    checks and modifies the GPT output as needed.

    Args:
        table_text (str): The full Balance Sheet table text block. 
		log_path (str): Path to log JSON file for the Balance Sheet table.
        table_path (str): Path to file where JSON version of the Balance Sheet table will be stored.
		form_name (str): FormName, as appears in the Forms table.

    Returns:
        None    
    """

    min_pre_comment_len = 50 #minimal number of chars in a valid pre-table comments section

    pre_table_comments = get_pre_table_comments(table_text) #get comments at beginning before table text (unit info expected there)

    update_json(log_path, [('table_comments', )], [pre_table_comments])

    if not pre_table_comments:
        update_json(log_path, [('problems',)], ['pre-table comments: no comments found'])
        return
    
    if len(pre_table_comments) < min_pre_comment_len:
        update_json(log_path, [('problems',)], ['pre-table comments: extracted str too short'])
        return
    
    get_sum_units(pre_table_comments, log_path, form_name) 

    table_body = get_table_body(table_text, pre_table_comments, log_path, form_name)

    if not table_body: #if table could not be cropped, try to get the table JSON based on the entire text
        table_body = table_text

    get_table_json(table_body, log_path, table_path, form_name)

 
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
    table_json_created = 1 if read_from_json(log_path, ("table_json_created", "data")) else 0

    problem_str = ", logging problems in FormProblems table" if problem_ids else ""

    print(f"- Writing data to Tasks table{problem_str}....")

    while True:        
        try:
            with sqlite3.connect(filings_db_path) as conn:
                cur = conn.cursor()
               
                cur.execute("UPDATE Tasks SET (SumDivider, JsonTable) = (?, ?) WHERE Form_id = ?", (sum_divider, table_json_created, form_id))

                if (not SKIP_EXISTING) or RETRY_LIST: #if overwriting, delete previously logged problems
                    cur.execute("DELETE FROM FormProblems WHERE Form_id = ?", (form_id, ))
                    
                for problem_id in problem_ids: #if problems were detected, log them in the FormProblems table
                    cur.execute("INSERT INTO FormProblems VALUES (?, ?)", (form_id, problem_id))
                
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
        runtime_text = f"\nRuntime {np.round((time.time() - start_time) / 60, 2)} minutes ({np.round((time.time() - start_time) / form_cnt, 2)} seconds per form, on average)."
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
    """Program for extracting data from the Balance Sheet table in financial filings to the SEC (10-Q and 10-K forms). 
    
    Should be run after running SEC_filing_reader_step1.py.
    Uses ChatGPT to identify dollar sum units in pre-table text (model use workflow involves mini model voting, similarly to step1).
    Uses ChatGPT to identify the end of the Balance Sheet table within the text block (first valid response is used). 
    Uses ChatGPT to convert the table text into a structured JSON data file (first valid or semi-valid response is used and fixed algorithmically as needed).
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
            text_path = get_json_path(form_id, form_name, 'text')
            if not log_path or not text_path:
                previous_tasks_incomplete.append(form_id)
                continue
            table_path = get_json_path(form_id, form_name, 'table')

            #if overwriting, reset problems list in the log file
            if (not SKIP_EXISTING) or RETRY_LIST:
                reset_problems(log_path)

            #insert additional dicts to the log file, to be updated by this program
            init_new_log_entries(log_path)

            #get text block containing balance sheet table (as identified in step1)
            table_text = get_table_text(log_path, text_path, form_name) 

            #export table data to table JSON file  
            if table_text:
                get_table_data(table_text, log_path, table_path, form_name)

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
