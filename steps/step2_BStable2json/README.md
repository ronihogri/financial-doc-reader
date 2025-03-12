# Step 2: Export Balance Sheet Table to a Structured JSON File

**A general overview of the project can be found [here](https://github.com/ronihogri/financial-doc-reader/blob/main/README.md).**   
For the previous step, see [Step 1](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table). 


## The Problem: Transforming a Text Block into a Structured Data File
In Step 1, we identified the text block within the document that contains the Balance Sheet table. However, this text block is unstructured, and also contains text that is not part of the actual table.

## The Solution: Extract Table Body Text and Convert it to JSON

Using a combination of LLM-based and algorithmic approaches, the Step 2 program:

1. Extracts the table body text (row headers and related values).
2. Converts the table body text into a structured JSON data file. The JSON format maintains the original table’s hierarchy, allowing easy access to specific values. The JSON file can be very easily converted into a table that can be viewed and edited in your tool of choice (e.g., Excel). 
3. Extracts the dollar amount units used in the table (e.g., millions) from the text preceding the table ("pre-table text").

## Workflow

1. Make sure you have cloned the most recent version of the repository (see [installation instructions](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/README.md#installation)). The Step 2 folder contains all necessary files, so you can run it independently of [Step 1](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table). 
2. Navigate to `financial-doc-reader/steps/step2_BStable2json/`
3. *Optional*: Open `SEC_filing_reader_step2.py` in an editor and adjust user-definable variables as needed.
4. Run the Python script:

```console
$ python3 SEC_filing_reader_step2.py
```
<br>
The program processes text blocks from 252 financial documents produced by 12 selected companies (see the "Forms" and "Stocks" tables of the SQL database in <code>./filings_demo_step1.sqlite</code>). <a href="#figure-2-1" style="white-space: nowrap; font-weight: bold;">Fig. 2.1</a> illustrates the workflow for each document. As the program runs, it stores detailed data for each document in designated JSON files within the <code>./extracted/</code> folder. The main results are saved to the "Tasks" table of the SQL database. Example results are provided in the <code>./results_step2.zip</code> archive. 

<br><br>

### <a id="figure-2-1"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step2_BStable2json/images/SEC_step2_general_workflow.png)

**Figure 2.1: General workflow per document.** LLM-assisted processes are shown in pink. In the vast majority of cases, LLM-assisted tasks were successfully performed using the "mini" model <span style="white-space: nowrap;">(gpt-4o-mini-2024-07-18)</span>. The "large" model <span style="white-space: nowrap;">(gpt-4o-2024-08-06;</span> \~17x more expensive than the mini model) was only used in the rare cases where the mini model failed. The first row header of Balance Sheet tables typically begins with "Assets", and can therefore be identified using a keyword search (no LLM required). In contrast, the text following the end of the table ("post-table text") is highly variable between documents, and is therefore identified using an LLM. The table body, defined as the text between the pre- and post-table texts, is then exported into a structured JSON file. 

<br>

## Results
For testing purposes, the program was run three times. All tasks were successfully completed for all documents and runs. On average, the runtime per document was \~16 seconds, and the cost was less than $0.0015. The following sections provide additional details about the LLM-assisted processes (pink rectangles in <a href="#figure-2-1" style="white-space: nowrap; font-weight: bold;">Fig. 2.1</a>):

#### Post-Table Text Identification

This is a helper process, meant to facilitate the main process of converting the table text into structured data. In essence, when making the conversion, we don't want to feed the LLM any text that is not part of the actual table. 

To assist with the identification of the table end, the LLM was asked to return a string containing the first 100 characters directly following the end of the table. Unsurprisingly, the model's output varied slightly, particularly with respect to the actual string length (LLMs are notoriously bad at counting...). To evaluate each output, the program checked whether it matched a string within the text block. If a match was found, the process was deemed successful and was terminated. If not, the model was queried again; this process was repeated up to three times for the mini model, and an additional attempt was made by the large model if needed. On average, this task required <span style="white-space: nowrap;">1.04</span> mini model uses per document (error rate = <span style="white-space: nowrap;">4.3%</span>). The large model was required (and successful) for <span style="white-space: nowrap;">0.4%</span> of documents. 

#### Table Text to Structured JSON Conversion

The LLM was provided the table body text, and was asked to convert it into structured JSON data. An illustration of this process is shown in <a href="#figure-2-2" style="white-space: nowrap; font-weight: bold;">Fig. 2.2</a>*A*,*B*. Each LLM output (string) was checked to confirm that it could be converted into JSON data using Python's <span style="white-space: nowrap;"><code>json.loads()</code></span> function. If the conversion was successful, the process was deemed successful and was terminated; if not, the model was queried again. 

This task was performed under two <u>response type settings</u>: "text", and "json_object". Overall, the mini model performed this task well under both conditions, and results were comparable. In the text type condition, a common issue was that the returned JSON structure was unbalanced, as it lacked the closing "\}" character. When this issue was addressed algorithmically, the error rate was drastically reduced from <span style="white-space: nowrap;">15.3%</span> to <span style="white-space: nowrap;">0.13%</span> (a single error across three runs). The error rate for the json_object condition was also <span style="white-space: nowrap;">0.13%</span> (<a href="#figure-2-2" style="white-space: nowrap; font-weight: bold;">Fig. 2.2</a>*C*). Curiously, in this condition as well, the error was due to a missing closing "\}" character (no algorithmic correction was applied in this condition). The large model was never required for this task. 

### <a id="figure-2-2"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step2_BStable2json/images/json_table_results.png)

**Figure 2.2: JSON data extraction.** ***A***, An example of unstructured table body text. ***B***, Layout of the JSON structure produced by the mini LLM based on the text provided in *A* (dots represent nested key-value pairs, omitted from the figure for brevity). The JSON data preserved the original table’s structure and hierarchy. ***C***, When the model's response type was set to "text", its output often lacked the closing "\}" character, leading to a conversion error in 15.3% of cases. Applying algorithmic correction to text output ("text + algo"), or using the "json_object" response type, reduced the error rate by two orders of magnitude to 0.13%. 

<br>

#### Dollar Amount Unit Identification

Our end goal is to extract specific sums from the table, and use them for our analyses. For this, we need to know the amount units in which sums are reported (e.g., "millions of dollars"). This information is typically provided in the text preceding the table. 

The mini model was provided the pre-table text, and was asked to return a single integer representing the amount units (e.g., 1000 for "thousands"). To minimize errors, the "voting" method described in <span style="white-space: nowrap;"><a href="https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/README.md">Step 1</a></span> was used, with the maximum vote count set to 3 (i.e., a majority vote could be achieved within 2 attempts). However, this task proved to be very easy for the mini model, and no errors were detected. <a href="#figure-2-3" style="white-space: nowrap; font-weight: bold;">Fig. 2.3</a> illustrates the amount units used in reports filed by two companies &ndash; Apple Inc. ('AAPL') and Henry Schein ('HSIC'), throughout the analyzed period. 

**Note:** To be able to compare between documents, we eventually want to normalize all dollar values to millions of dollars (the most common reporting unit). Therefore, rather than storing the amount units in the SQL DB's "Tasks" table, the "SumDivider" variable is stored &ndash; this is the value by which dollar sums in a particular Balance Sheet table should be divided in order to be represented in millions of dollars (e.g., a SumDivider value of 1000 for "thousands of dollars").

### <a id="figure-2-3"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step2_BStable2json/images/reported_units_aapl_hsic.png)

**Figure 2.3: Examples of reported monetary units in Balance Sheet tables.** Apple Inc. (AAPL) reported dollar sums in millions throughout the analyzed period. In contrast, Henry Schein (HSIC) reported dollar sums in thousands until 2022, when it began reporting in millions. Value dates are shown as year-month.

<br>


## Conclusions

- Combining LLM-based and algorithmic approaches, the Python programs provided in Steps 1 and 2 fetch financial reports from the SEC's EDGAR system, identify the reported dollar amount units, and export the Balance Sheet table to a structured JSON data file. 
- Using this approach, the combined cost of LLM usage for Steps 1 & 2 is \~$0.002 per document. This cost is dramatically lower than that of processing entire documents with LLMs, even when employing efficient methods such as RAG (Retrieval-Augmented Generation).
- The structured JSON format now allows us to reproduce the structure of the Balance Sheet table, as it appeared in the original document (<a href="#figure-2-4" style="white-space: nowrap; font-weight: bold;">Fig. 2.4</a>). 

### <a id="figure-2-4"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step2_BStable2json/images/example_form10_webpageVScsv_ha.png)

**Figure 2.4: Table reproduction.** *Left*: An example of a Balance Sheet table from a filing by Apple Inc. (adapted from the SEC website). *Right*: The same table in a CSV file, generated from the structured JSON data. For additional examples, see `results_step2.zip/balance_csvs`.

<br>
 
## *Coming Soon*
In the next step, we will:
- Normalize dollar amounts where necessary.
- Identify and extract values of interest from the JSON version of the table. 
