# Step 3: Identify and Extract Values of Interest


**A general overview of the project can be found [here](https://github.com/ronihogri/financial-doc-reader/blob/main/README.md).**   
For the previous steps, see [Step 1](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table) and [Step 2](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step2_BStable2json). 


## The Problem: Grouping Values by Meaning, Not by Name

In Step 2, we converted each company’s Balance Sheet into a clean, structured table. The output is well-formatted — but row labels still vary between companies (and sometimes even within the same company over time), making comparisons across reports difficult.

Now, we want to extract a few standard financial concepts that are commonly used across the industry, such as **current cash position** and **long-term debt**. These are high-level categories — not always labeled directly, and not always organized in the same way.

One company might report *“Cash and Cash Equivalents”*; another might separate this into *“Cash”* and *“Short-term Investments”*. Similarly, what one filing calls *“Long-Term Debt”* might be broken up elsewhere into *“Senior Notes”*, *“Term Debt”*, or other debt-related rows. To extract these concepts, we first need to figure out which rows belong together — even when they’re worded differently.


## The Solution: Use Language Models to Group Rows by Meaning

To extract high-level concepts like **current cash position** or **long-term debt**, we use large language models (LLMs) to decide which rows in the table belong together.

This task is a good fit for LLMs: the labels vary, the patterns are inconsistent, and there are no fixed rules that work across all cases. Instead of relying on hard-coded logic, we use models that can understand context, interpret meaning, and make flexible decisions — much like a human would.

Using a combination of LLM-based and algorithmic approaches, the Step 3 program:

1. Identifies which rows in the table contribute to each target concept (current cash position, long-term debt), even when row labels differ across filings.
2. Applies checks to detect questionable rows before they are included in the group.
3. Aggregates the relevant values into *a single total for each concept*, producing one value per concept per report.


## Workflow

1. Make sure you have cloned the most recent version of the repository (see [installation instructions](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/README.md#installation)). The Step 3 folder contains all necessary files, so you can run it independently of previous steps. 
2. Navigate to `financial-doc-reader/steps/step3_extract_by_concept/`
3. *Optional*: Open `SEC_filing_reader_step3.py` in an editor and adjust user-definable variables as needed.
4. Run the Python script:

```console
$ python3 SEC_filing_reader_step3.py
```
<br>

### *...Under construction...*

