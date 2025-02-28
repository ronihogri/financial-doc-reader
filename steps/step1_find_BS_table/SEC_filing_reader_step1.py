"""
Roni Hogri, February 2025

This demo program utilizes large language models (LLMs) to extract the Balance Sheet table from quarterly and yearly filings to the SEC (10-Q and 10-K, respectively).
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
REPORT_DB_FN = "filings_demo_step1.sqlite" #SQL file name 

MY_API_KEY = "insert_your_OpenAI_API_key_here_if_you_don't_want_to_set_it_as_an_environment_var"
CHECKPOINT_TASK = 'TextListLen' #task in SQL DB that determines which filings were already processed

"""End of user-defined variables"""

"""***************************************************************************************************************************************"""


"""Globals"""

# import required libraries for global use:
import os
import requests
from bs4 import BeautifulSoup
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


#paths, etc.
curdir = os.path.dirname(os.path.abspath(__file__)) #path of this script
filings_db_path = os.path.join(curdir, REPORT_DB_FN) #path to SQL file
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) or OpenAI(api_key=MY_API_KEY) #openai client

#input and data variabls
POSSIBLE_BS_TITLES = [
    'CONSOLIDATED BALANCE SHEET', 'COMBINED BALANCE SHEET', 'CONSOLIDATED STATEMENTS OF FINANCIAL POSITION', 
    'CONSOLIDATED STATEMENT OF FINANCIAL POSITION', 'CONSOLIDATED STATEMENTS OF FINANCIAL CONDITION']
MANDATORY_FIELDS_BS = ['asset', 'cash', 'liabilit', '$', 'total'] #keywords for correct identification of balance sheet table
MAX_DISTANCE_BS = 3000 #number of chars extracted starting from table title

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

    Raises:
        ValueError: If BATCH_SIZE is defined but not a positive integer.
        TypeError: If SKIP_EXISTING is not a boolean.
        Exception: If the path specified in filings_db_path does not exist.

    Globals:
        BATCH_SIZE (int or None): Size of the batch for processing, must be a positive integer or None (= all filings will be processed in one run).
        SKIP_EXISTING (bool): If set to False, completed filings will be re-processed and data re-written.
        filings_db_path (str): Path to SQL DB holding the Forms table.
    """

    if ((BATCH_SIZE is not None) and (not isinstance(BATCH_SIZE, int))) or ((isinstance(BATCH_SIZE, int)) and (BATCH_SIZE < 1)):
        raise ValueError("**** BATCH_SIZE incorrectly defined, must be a positive int or None ****\n\n")
    
    if not isinstance(SKIP_EXISTING, bool):
        raise TypeError("**** SKIP_EXISTING incorrectly defined, must be True/False ****\n\n")
    
    if (not SKIP_EXISTING) and ((not isinstance(FIRST_ROW_TO_OVERWRITE, int)) or (FIRST_ROW_TO_OVERWRITE < 1)):
        raise ValueError("**** FIRST_ROW_TO_OVERWRITE incorrectly defined, must be a positive int when SKIP_EXISTING set to False ****\n\n")
    
    if not os.path.exists(filings_db_path):
        raise Exception(f"**** Path to SQL DB incorrectly defined, no such path exists: ****\n{filings_db_path}\n\n")
    
   
def get_forms_info():
    """Gets the next filing to work on from the SQL DB's "Forms" table.

    Globals:
        SKIP_EXISTING (bool): If set to False, completed filings will be re-processed and data re-written. 
        BATCH_SIZE (int or None): If int, the number of filings to process in each run; 
            if set to None, the program will run through all incomplete filings remaining in the Forms tables.
        filings_db_path (str): Path to SQL DB holding the Forms table.        

    Returns:
        list: A list of tuples; each tuple contains the following elements extracted from the Forms table:
            i. int: id
            ii. str: FormName
            iii. str: FormURL
    """

    global BATCH_SIZE

    #connect to SQL DB and get identifiers for next filing to be processed
    with sqlite3.connect(filings_db_path) as conn:
        cur = conn.cursor() 

        if RETRY_LIST: #if list is populated, will only work on this list
            BATCH_SIZE = len(RETRY_LIST)
            placeholders = ",".join("?" for _ in range(len(RETRY_LIST)))
            cur.execute(f"SELECT id, FormName, FormURL FROM Forms WHERE id IN ({placeholders})", RETRY_LIST)
            return cur.fetchall()        
   
        #get all form ids
        cur.execute("SELECT id FROM Forms ORDER BY id")
        form_ids = [item[0] for item in cur.fetchall()]

        if not SKIP_EXISTING and FIRST_ROW_TO_OVERWRITE > len(form_ids):
            raise ValueError(f"**** FIRST_ROW_TO_OVERWRITE out of range: must be between 1 and {len(form_ids)} ****\n\n")
                
        #if not overwriting, return only info on files with NULL Tasks
        #get completed
        cur.execute(f"SELECT Form_id FROM Tasks WHERE {CHECKPOINT_TASK} NOT NULL")
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

        if not SKIP_EXISTING: #if completed filings should be overwritten, batch will start at first row to overwrite
            cur.execute("SELECT id, FormName, FormURL FROM Forms WHERE id >= ? AND id < ?", (FIRST_ROW_TO_OVERWRITE, FIRST_ROW_TO_OVERWRITE + BATCH_SIZE))
            return cur.fetchall()

        #make list of incomplete with len = len(batch_size)
        ids_to_get = [form for form in form_ids if form not in existing][:BATCH_SIZE]

        #get form info based on ids_to_get, sort by ids
        cur.execute("CREATE TEMPORARY TABLE temp_ids (id INTEGER PRIMARY KEY)")
        cur.executemany("INSERT INTO temp_ids VALUES (?)", [(id, ) for id in ids_to_get])
        cur.execute("SELECT f.id, f.FormName, f.FormURL FROM Forms f JOIN temp_ids t ON f.id = t.id ORDER BY f.id")
        return cur.fetchall()
    

def check_overwrite():
    """Check if the user wants to proceed with overwriting existing data. Called when RETRY_LIST or SKIP_EXISTING lead to overwriting.
    If the user chooses to terminate overwriting, the function exits the program.

    Globals:
        RETRY_LIST (list): List of form IDs that may be overwritten if user confirms.

    Returns:
        None 
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
    

def gpt_completion(model, system_content, user_content, trials=1): 
    """General function for querying GPT (completions mode).

    Args:
        model (str): The model to be used for generating completions.
        system_content (str): The content that sets the behavior of the assistant.
        user_content (str): The input content from the user for which a completion is requested.
        trials (int): The number of trials for querying the model, must be a positive integer; default is 1.

    Globals:
        PLAY_NICE (float): Sleep time between API calls (seconds).
        GPT_ATTEMPTS (int): Number of attempts to connect to OpenAI API before failing.

    Returns:
        votes (dict): A dictionary containing GPT outputs indexed by trial number.
    """

    fail_counter = 0
    trial_counter = 0
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

            if trial_counter == trials: #reached maximal number of votes
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
        votes (dict): keys: vote IDs, values: GPT output per vote.

    Returns:
        str or None: The value that received an absolute majority (>= 50%) of the votes, or None if no value received this majority.
    """

    vote_counter = Counter(votes.values())
    if vote_counter.most_common(1)[0][1] < np.ceil(len(votes)/2): #majority vote does not have 50% or higher
        return None #undecided
    else:
        return vote_counter.most_common(1)[0][0]


def read_from_json(file_path, key_path=()):
    """Read data from a JSON file and optionally retrieve nested values based on a given key path.

	Args:
		file_path (str): Path to the JSON file to read.
		key_path (tuple, optional): A sequence of keys to navigate through nested dictionaries. Default is an empty tuple, which returns the entire JSON object.

	Raises:
		KeyError: If any key in the key_path is not found in the JSON structure.

	Returns:
		json_dict (dict): The retrieved JSON structure, or a part of it as specified by key_path.
	"""    

    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    
    for key in key_path:
        try:
            json_dict = json_dict[key]
        except (KeyError, TypeError) as e:
            raise KeyError(f"Key path {key_path} is invalid at {key}: {e}")

    return json_dict  


def update_json(file_path, dict_path_list, value_list):
    """Update a JSON file at the specified path.

	This function modifies the JSON structure located at file_path by following the paths specified in dict_path_list. 
    With the exception of model info, each dict path culminates in a dictionary that holds a key called 'data', where the corresponding value from value_list will be stored.     
    The function also records a timestamp of when the data was updated.

	Args:
		file_path (str): The path to the JSON file to be updated.
		dict_path_list (list): A list of paths (each path a tuple) in the JSON structure where each value from value_list should be inserted.
		value_list (list): A list of values to be stored in the corresponding paths specified by dict_path_list.

	Raises:
		TypeError: If either dict_path_list or value_list is not a list.
		Exception: If the file at file_path does not exist or has not been initialized.

	Returns:
		None
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
            sub_dict['data'] = list(set(sub_dict['data'])) #don't store the same problem twice
        else: #key != 'problems'
            sub_dict['data'] = value

        sub_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')        
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def text_content_exists(form_name, text_path, section_name): 
    """Check if content exists for a specific text block extracted from the form.
	
	Args:
		form_name (str): Name of the form as appears in the Forms table.
		text_path (str): Path to the JSON file containing the text block data.
		section_name (str): Name of the section for which text blocks are expected.
	
	Returns:
		text_block_list (list): A list containing the text block data if found and non-empty;
		otherwise, prints a notification and returns None.
	"""

    if not os.path.exists(text_path):
        print(f"*** No file found for form '{form_name}' at expected path:\n{text_path}\n")
        print("***************************************************")
        return None   

    text_block_list = read_from_json(text_path, (section_name, 'data'))
    if text_block_list and isinstance(text_block_list, list):
        return text_block_list


"""Functions for extracting the content of relevant text blocks and saving them to a JSON file (nested within get_text_blocks())"""

def set_text_path(form_id, form_name): 
    """Set the path for the text block JSON file based on the given form ID and name.

	Args:
		form_id (int): Form ID, as appears in the Forms table.
		form_name (str): FormName, as appears in the Forms table.

	Returns:
		str: The full path to the JSON file where text blocks will be saved.
	"""

    path = os.path.join(curdir, 'extracted', 'text_blocks', f'{form_id}_{form_name}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True) #create necessary folders if they don't already exist

    return path


def get_form_content(form_url, previous_request_time_edgar): 
    """Retrieve the full content of a financial report form from EDGAR.

	Args:
		form_url (str): The URL of the form to retrieve from EDGAR.
		previous_request_time_edgar (float or None): The time of the previous request to EDGAR, used to control request rate.

	Globals:
		PLAY_NICE (float): Minimum time (seconds) that must pass between requests to EDGAR.

	Returns:
		tuple: A tuple containing:
			- bytes: The binary content of the retrieved form.
			- float: The time at which the request was made.
	"""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.sec.gov",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",  
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1"
    }

    current_request_time = time.time()
    if previous_request_time_edgar and ((current_request_time - previous_request_time_edgar) < PLAY_NICE):
        time.sleep(PLAY_NICE - (current_request_time - previous_request_time_edgar))

    try:
        response = requests.get(form_url, headers=headers)

    except requests.exceptions.ConnectionError:
        print("Error: No internet connection.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")

    if not response.ok:
        raise Exception (f"Could not reach EDGAR website, response code: {response.status_code}") from None
    
    return response.content, time.time()


def get_text_from_soup(form_content): 
    """Extract Balance Sheet text from the HTML content of a form.

	Args:
		form_content (bytes): The binary HTML content of the form, as returned by `get_form_content`.

	Globals:
		POSSIBLE_BS_TITLES (list): A list of strs representing potential titles for the Balance Sheet that may appear in the text.
		MANDATORY_FIELDS_BS (list): A list of strs representing mandatory fields that must be present in the Balance Sheet.
		MAX_DISTANCE_BS (int): The maximum distance (No. of chars) to search for Balance Sheet tables after title detection.

	Returns:
		balance_sheet_text (list): A list of strs potentially containing the Balance Sheet table.
	"""

    #parse the HTML form content and retrieve the text it contains
    soup = BeautifulSoup(form_content, 'html.parser') 
    all_text = soup.get_text(separator="\t", strip=True).replace('\xa0', ' ').replace('\u2019', ' ').replace('\u2014', ' ')
    all_text = all_text.replace("....", "..").replace(".....", ".") #for files with multiple dots which may lead to oversized table texts

    #find instances of 'balance sheet(s)' in the all-text version of soup:
    balance_sheets_indices = []
    for title in POSSIBLE_BS_TITLES:
        balance_sheets_indices.extend([at for at in range(len(all_text)) if all_text[at:at+len(title)].upper() == title])
    
    #look for mandatory fields after each time you encounter 'balance sheet' and store positive cases in list:
    balance_sheet_text = []
    for index in balance_sheets_indices:
        start_index = max(0, index)
        if not all_text[start_index].isupper(): continue #first letter of table title should be capitalized
        end_index = index + MAX_DISTANCE_BS
        if end_index > len(all_text): continue #table is expected relatively early in the report, definitely not at the very end
        if all(x.lower() in all_text[start_index:end_index].lower() for x in MANDATORY_FIELDS_BS): #check that all mandatory fields exist in the text
            balance_sheet_text.append(all_text[start_index:end_index].strip())

    return balance_sheet_text


def init_text_list_file(path, sections):
    """Initialize a JSON file to hold extracted text blocks for each selected form section (e.g., Balance Sheet table).

	Args:
		sections (list): List of section names (str) to be included in the JSON file.

	Globals:
		path (str): Path to the JSON file that will be created and initialized.

	Returns:
		None
	"""

    data = {}

    for section in sections: 
        data[section] = {'data': None, 'timestamp': None}

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_text_blocks(form_url, form_id, form_name, previous_request_time_edgar): 
    """Main function for text extraction workflow.

	Args:
		form_url (str): The URL of the form to retrieve from EDGAR.
		form_id (int): Form ID, as appears in the Forms table.
		form_name (str): FormName, as appears in the Forms table.
		previous_request_time_edgar (float): The previous request time, to space out repeated requests to EDGAR.

	Returns:
		text_path (str): The path of the JSON file where the extracted text blocks are stored.
		previous_request_time_edgar (float): The updated previous request time, as returned by time.time().
	"""

    print("- Collecting text blocks of relevant tables.......")

    text_path = set_text_path(form_id, form_name)
    form_content, previous_request_time_edgar = get_form_content(form_url, previous_request_time_edgar) #get form HTML content   

    balance_sheet_text = get_text_from_soup(form_content) #use bs4 to get text blocks to be examined based on keywords
    init_text_list_file(text_path, ['balance'])
    update_json(text_path, [('balance', )], [balance_sheet_text])

    return text_path, previous_request_time_edgar


"""Functions for extracting Balance Sheet table index from text blocks and logging it into a structured JSON file (nested within detect_balance_sheet())"""

def set_balance_log_path(form_id, form_name):
    """Set the path for the Balance Sheet's log JSON file. 

	Args:
		form_id (int): Form ID, as appears in the Forms table.
		form_name (str): FormName, as appears in the Forms table.

	Returns:
		path (str): Path of the balance log JSON file.
	"""

    path = os.path.join(curdir, 'extracted', 'logs', 'balance', f'{form_id}_{form_name}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True) #create necessary folders if they don't already exist

    return path


def init_balance_log_file(balance_log_path):
    """Initialize the balance log file with default structure.

	This function creates a JSON file at the specified path containing a 
	default structure to log information related to balances, text blocks, 
	table comments, units, and problems.

	Args:
		balance_log_path (str): Path to the file where the balance log 
			will be created. Must point to a writable location.

	Returns:
		None
	"""

    data = { 
            'text_blocks': {'data': None, 'model': None, 'timestamp': None}, 
            'table_comments': {'data': None, 'timestamp': None}, 
            'units': {'data': None, 'model': None, 'timestamp': None}, 
            'problems': {'data': None, 'timestamp': None}
            } 

    with open(balance_log_path, 'w') as f:
        json.dump(data, f, indent=4)


def ask_balance_table_index(text_list, model=MINI, trials=1):
    """Ask GPT to identify the index of the Balance Sheet table out of the list of text blocks retrieved by get_text_blocks().

	Args:
		text_list (list): The list of text blocks to be analyzed.
		model (str, optional): The model to be used for identification; defaults to MINI.
		trials (int, optional): Maximum number times to ask GPT (maximum number of votes); defaults to 1.

	Returns:
		dict: keys: vote IDs, values: GPT output per vote.
	"""

    print(f"...Asking the '{model}' model to identify the Balance Sheet table out of {len(text_list)} possibilities....")

    get_table_index_sys = """You are to help the user extract financial information from 10-Q or 10-K filings submitted by public companies to the SEC.
You will be provided with a list of text excerpts from such filings.
Your task is to identify the index of the entry (starting from 0) in the list that contains the full content of the Balance Sheet / Financial Position TABLE.

Notes:
1. The structure of the table was distorted during extraction (e.g., missing columns, misalignment), but you should still be able to identify the table by the existence of content that is usually contained is such tables, as well as the order of this content.
2. The correct item may include some text outside the table, but it must contain all rows and columns of the table.
3. If you find multiple items with a complete table, return the index of the first one.
4. If no item contains the complete table, return `None`.
"""

    get_table_index_user = f"""Return a single integer identifying the index of the entry in the following list that contains the Balance Sheet / Financial Position TABLE from a 10-Q/10-K filing: {text_list}.
The returned index must be between 0 and {len(text_list) - 1}. Return only the index int - do not add any additional text, symbols, or explanations"""

    return gpt_completion(model, get_table_index_sys, get_table_index_user, trials) 


def convert_model_decision(decision):
    """Convert the model's decision to an int-type index if possible.

	Args:
		decision (str): The model's decision (majority vote). 

	Returns:
		int or None: The table index (as identified by GPT), or None if the decision was not a number. 
	"""

    try:        
        table_index = ast.literal_eval(decision)
        if isinstance(table_index, (int, float)): #if not a number, will return None
            return int(table_index) #in case it was float
    except: 
        return None

 
def report_problems(form_name, path, problems): 
    """Print problems detected in the Balance Sheet log file.

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


def get_table_index(balance_sheet_text, form_name, balance_log_path):  
    """Determine the index of the text block that likely contains the Balance Sheet table.

	Args:
		balance_sheet_text (list): List of text blocks extracted from the form by get_text_blocks().
		form_name (str): FormName, as appears in the Forms table.
		balance_log_path (str): Path to the log JSON file holding the extracted Balance Sheet data.

	Globals:
		MINI (str): Identifier for the mini model to be used in identifying the table.
		GPT_4O (str): Identifier for the larger model to be used in identification if needed.
		MAX_DISTANCE_BS (int): Threshold value to determine if the text block is sufficiently long.

	Returns:
		table_index (int) or None: Index of the text block containing the table, or None if not found.
    """

    index_dict = {'block_count': len(balance_sheet_text), 'table_index': None}
    model_dict = {MINI: {'votes': None, 'decision': None}, GPT_4O: {'votes': None, 'decision': None}}

    for i in range(2): #first run with the mini model; if there are problems, repeat process with the large model

        problems_list = [] #for temporarily storing problems (per model)

        model, trials = (MINI, 5) if i == 0 else (GPT_4O, 1)

        #ask GPT model to identify the text block containing the Balance Sheet table and store model output(s):
        votes = ask_balance_table_index(balance_sheet_text, model, trials)
        model_dict[model]['votes'] = votes
        model_dict[model]['decision'] = count_votes(votes)
        table_index = convert_model_decision(model_dict[model]['decision'])
        index_dict['table_index'] = table_index

        if table_index is None: #could not identify table text, note problem and skip the rest            
            problems_list.append('balance sheet: no table found in text block list')
            continue #try with larger model

        if table_index < 0 or table_index > (len(balance_sheet_text) - 1):
            problems_list.append(f'balance sheet: table index out of range: {table_index}, when len(balance_sheet_text) = {len(balance_sheet_text)}')
            continue

        if not balance_sheet_text[table_index] or len(balance_sheet_text[table_index]) < (MAX_DISTANCE_BS * 0.9):
            problems_list.append(f'balance sheet: table text block too short: {len(balance_sheet_text[table_index])}')
            continue

        if not problems_list: break #don't run on larger model if mini model was sufficient

    update_json(balance_log_path, 
                [('text_blocks', ), ('text_blocks', 'model'), ('problems', )], 
                [index_dict, model_dict, problems_list]
                )

    if problems_list: #after both models, problems remain        
        report_problems(form_name, balance_log_path, problems_list) #print out detected problems

    return table_index

    
def detect_balance_sheet(form_name, text_path, balance_log_path): 
    """Main function for extracting Balance Sheet table index from text blocks and logging it into a structured JSON file.

	Args:
		form_name (str): FormName as it appears in the Forms table.
		text_path (str): Path to the text file containing report data.
		balance_log_path (str): Path to the JSON log file where balance sheet information will be stored.

	Returns:
		None
	"""

    print("- Creating Balance Sheet log file.......")

    balance_sheet_text = text_content_exists(form_name, text_path, 'balance') #retrieve balance table raw text from file, or None if no content in file

    if not balance_sheet_text:
        update_json(balance_log_path,
                    [('text_blocks', ), ('problems', )], 
                    [{'block_count': 0, 'table_index': None}, 'balance sheet: no text in file']
                    )
        report_problems(form_name, balance_log_path, ['balance sheet: no text in file']) #print out detected problems
        return 
    
    if len(balance_sheet_text) > 1: #text block list contains more than one possibility for the balance sheet table, ask GPT
        get_table_index(balance_sheet_text, form_name, balance_log_path)        
    else: #only one text block retrieved by keywords, don't involve GPT
        update_json(balance_log_path, [('text_blocks', )], [{'block_count': 1, 'table_index': 0}]) 
        

"""Functions for updating the SQL DB"""

def get_balance_problems(balance_log_path):
    """Retrieve a list of problem IDs (from the Problems table) that match problems reported in the JSON balance log file.
	
	Args:
		balance_log_path (str): Path to the JSON file containing balance log data.
	
	Globals:
		filings_db_path (str): Path to the SQL database containing the Problems table.
	
	Returns:
		problem_ids (list): A list of problem IDs (ints) pointing to the types of problems detected (Problems.id).
	"""

    balance_log_problems = read_from_json(balance_log_path, ('problems', 'data')) #get problem descriptions

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


def update_sql(form_id, json_path, problem_ids):
    """Updates the Tasks and FormProblems tables in the SQL database.
	
	This function reads form data from the specified JSON file and updates the Tasks table with the form ID, 
	the length of the text blocks, and the table index. If problem IDs are provided, it logs them in the FormProblems table. 
	If the database is locked, the function will prompt the user to resolve the issue before retrying.
	
	Args:
		form_id (int): Form ID, as appears in the Forms table.
		json_path (str): Path to the JSON file containing Balance Sheet log.
		problem_ids (list): A list of problem IDs to log in the FormProblems table (may be empty).
	
	Raises:
		sqlite3.OperationalError: If an operational error occurs while accessing the SQLite database.
	
	Returns:
		None
	"""

    problem_str = ", logging problems in FormProblems table" if problem_ids else ""

    print(f"- Writing data to Tasks table{problem_str}....")

    text_list_len = read_from_json(json_path, ("text_blocks", "data", "block_count"))
    table_index = read_from_json(json_path, ("text_blocks", "data", "table_index"))

    while True:
        
        try:
            with sqlite3.connect(filings_db_path) as conn:
                cur = conn.cursor() 
                if (not SKIP_EXISTING) or RETRY_LIST: #if data is overwritten
                    cur.execute("DELETE FROM Tasks WHERE Form_id = ?", (form_id, ))
                    cur.execute("DELETE FROM FormProblems WHERE Form_id = ?", (form_id, ))
                    
                cur.execute("INSERT INTO Tasks VALUES (?, ?, ?)", (form_id, text_list_len, table_index)) 

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



def report_done(total_problem_cnt, forms_with_problems, start_time, form_cnt):
    """Outputs a summary report upon completion of data extraction from specified batch.

	Args:
	    total_problem_cnt (int): Total count of problems encountered during the extraction process.
	    forms_with_problems (list): A list of strs (form_id + form_name) indicating the forms in which problems were encountered (if any).
	    start_time (float): The time when the extraction process started, used to calculate runtime.
	    form_cnt (int): Total count of forms processed in this batch.

	Returns:
	    None
	"""

    if total_problem_cnt:
        problem_text = f"Problems encountered in {total_problem_cnt} filings:\n{forms_with_problems}"        
    else:
        problem_text = "No problems detected."

    print(f"""\n\n\n*********************************************************************************************************
Completed data extraction for batch of {form_cnt} 10-Q/10-K filings. Last task executed: '{CHECKPOINT_TASK}'.
Runtime {np.round((time.time() - start_time) / 60, 2)} minutes ({np.round((time.time() - start_time) / form_cnt, 2)} seconds per form, on average).
{problem_text}
Data stored to SQL DB: 
{filings_db_path}
*********************************************************************************************************\n
""")
    


"""****************************************************************************************************************************"""
def main():
    """Program for identifying the Balance Sheet table in financial filings to the SEC (10-Q and 10-K forms).

    Uses keywords to identify text blocks that may contain the Balance Sheet table. 
    If more than one text block is found, ChatGPT (mini and larger versions of the 4o model) is used to identify the one that actually contains the table. 
    Detailed ata related to table extraction (including model outputs) is stored in JSON files, and final results are stored in a SQL DB. 
    """
    
    try:

        #initialize some vars
        i = 0 #default form count in case program is terminated early
        start_time = time.time() #for runtime calculation
        previous_request_time_edgar = None #default value for previous EDGAR request
        total_problem_cnt = 0 #for storing number of forms for which problems were encountered
        forms_with_problems = [] #for storing the id+name of forms for which problems were encountered

        #check that user-defined variables are of the right types
        check_user_vars()

        #get basic form info from the SQL database
        forms_info = get_forms_info()

        #if user-defined vars lead to overwriting, confirm with user
        if (not SKIP_EXISTING) or RETRY_LIST:
            check_overwrite()            

        if not forms_info:
            print(f"""\n\n**** All filings in the Forms table have already been processed for the task '{CHECKPOINT_TASK}', program terminated ****
Note: If you wish to overwrite existing data, set SKIP_EXISTING to False, or specify relevant form IDs in RETRY_LIST\n\n""")
            sys.exit()

        print(f"\n**** Processing {BATCH_SIZE} filings ****")

        #for each form (filing)
        for i, (form_id, form_name, form_url) in enumerate(forms_info):
              
            print(f"\n\n.......... Processing filing #{form_id} ({i+1} / {BATCH_SIZE} in batch): '{form_name}' ........")   

            #store relevant text blocks to designated json file
            text_path, previous_request_time_edgar = get_text_blocks(form_url, form_id, form_name, previous_request_time_edgar) 

            #set path of log JSON file to store info extracted from the Balance Sheet table, and initiate it
            balance_log_path = set_balance_log_path(form_id, form_name) 
            init_balance_log_file(balance_log_path) 

            #use LLM to identify text block containing balance sheet table; store model results in log file
            detect_balance_sheet(form_name, text_path, balance_log_path) 

            #check if problems were encountered for this form, and get their ids (see Problems table in the SQL DB)
            sql_problem_ids = get_balance_problems(balance_log_path) 
            if sql_problem_ids:
                forms_with_problems.append(f'{form_id}_{form_name}')
                total_problem_cnt += 1

            #update problems and results for this form in the SQL DB
            update_sql(form_id, balance_log_path, sql_problem_ids)
                             

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
            report_done(total_problem_cnt, forms_with_problems, start_time, forms_examined)


if __name__ == "__main__":
    main()
