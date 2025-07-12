# Step 3: Identify and Extract Values of Interest


**A general overview of the project can be found [here](https://github.com/ronihogri/financial-doc-reader/blob/main/README.md).**   
For the previous steps, see [Step 1](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table) and [Step 2](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step2_BStable2json). 


## The Problem: Grouping Values by Meaning, Not by Name

In Step 2, we converted each company's Balance Sheet into a clean, structured table. The output is well-formatted — but row labels still vary between companies (and sometimes even within the same company over time), making comparisons across reports difficult.

Now, we want to extract a few standard financial concepts that are commonly used across the industry, such as **current cash position** and **long-term debt**. These are high-level categories — not always labeled directly, and not always organized in the same way.

One company might report *"Cash and Cash Equivalents"*; another might separate this into *"Cash"* and *"Short-term Investments"*. Similarly, what one filing calls *"Long-Term Debt"* might be broken up elsewhere into *"Senior Notes"*, *"Term Debt"*, or other debt-related rows. To extract these concepts, we first need to figure out which rows belong together — even when they're worded differently.


## The Solution: Use Language Models to Group Rows by Meaning

To extract high-level concepts like **current cash position** or **long-term debt**, we use large language models (LLMs) to decide which rows in the table belong together.

This task is a good fit for LLMs: the labels vary, the patterns are inconsistent, and there are no fixed rules that work across all cases. Instead of relying on hard-coded logic, we use models that can understand context, interpret meaning, and make flexible decisions — much like a human would.

Using a combination of LLM-based and algorithmic approaches, the Step 3 program:

1. Identifies which rows in the table contribute to each target concept (current cash position, long-term debt), even when row labels differ across documents.
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

The program processes the structured Balance Sheet tables (JSON files) produced in Step 2. For each of the 252 documents, the program identifies entries referring to a company's current cash position (CCP; total liquid assets immediately available) and long-term debt (LTD; financial obligations not due within the next year). 

To reduce the risk of errors, the program uses a layered "Swiss Cheese" approach: no single mechanism is perfect, but together they form a robust filter that prevents errors from reaching the final output (<a href="#figure-3-1" style="white-space: nowrap; font-weight: bold;">Fig. 3.1</a>). An illustration of the workflow applied for each document is shown in <a href="#figure-3-2" style="white-space: nowrap; font-weight: bold;">Fig. 3.2</a>.

As the program runs, it stores detailed results for each concept in designated JSON files within the <code>./extracted/logs/balance</code> folder. Values related to the same concept (CCP or LTD) are summed up, and the main results — one value per concept per report — are saved to the "Tasks" table of the SQL database (<code>./filings_demo_step3.sqlite</code>). Example outputs are provided in the <code>./results_step3.zip</code> archive.

<br><br>

### <a id="figure-3-1"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/swiss_cheese_model.png)

**Figure 3.1: Swiss cheese model for error prevention.** The program applies several processes in sequence (see <a href="#figure-3-2" style="white-space: nowrap; font-weight: bold;">Fig. 3.2</a>), each represented by a slice of Swiss cheese. Arrows represent potential errors. While any single process may allow some errors to pass through, their combination forms a layered defense that greatly reduces the risk of flawed values reaching the final output.

<br>

### <a id="figure-3-2"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/SEC_step3_flowchart.png)

**Figure 3.2: Workflow overview.** For each document, the "mini" model (gpt-4o-mini-2024-07-18) is used to identify Balance Sheet entries related to current cash position (CCP) and long-term debt (LTD). Voting is applied to reduce noise, similarly to [Step 1](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table). If a majority vote is reached, the selected entries are checked for suspicious terms — signs that a row may have been wrongly included. If suspicious terms are found, the "large" model (gpt-4o-2024-08-06) acts as a supervisor: it reviews the mini model's output and removes any entries that don't belong. If doubts remain after the supervisor's revision (or if no majority was reached by the mini model), the large model is asked to repeat the task independently — this time with no voting. If the result still contains suspicious items, the case is flagged for manual inspection.


<br>

## Results

### Effectiveness of the Layered "Swiss Cheese" Approach
The program extracted CCP and LTD values for all 252 documents. The average API usage cost per document was approximately $0.0023. <a href="#table-3-1" style="white-space: nowrap; font-weight: bold;">Table 3.1</a> shows the percentage of documents for which each process event occurred. <a href="#figure-3-3" style="white-space: nowrap; font-weight: bold;">Fig. 3.3</a> shows an example in which irrelevant items extracted by the mini model were removed by the supervisor process. <a href="#figure-3-4" style="white-space: nowrap; font-weight: bold;">Fig. 3.4</a> shows an example in which the supervisor failed to remove an irrelevant item extracted by mini; this issue was detected by the program, and consequently corrected by the large model. No issues were identified in the subset of documents flagged for manual review, suggesting that the layered safeguards were effective in preventing faulty entries. To further validate the results, the program was run a second time, and the results of the two runs were compared. Discrepancies were found in four document pairs (1.6% of documents); the program was then re-run on these four documents, and the results of this run were manually reviewed and validated. The dataset provided here in the `results.zip` archive contains the validated results.   
<br> 


| Process Event                       |   CCP |   LTD |
|:------------------------------------|------:|------:|
| No majority reached by mini model   |   0.8 |  11.1 |
| Supervisor checked mini output      |   8.3 |   6.7 |
| Supervisor revised mini output      |   0   |   6.7 |
| Large model re-performed extraction |   8.3 |   1.2 |
| Supervisor checked large output     |   8.3 |   0   |
| Supervisor revised large output     |   0   |   0   |
| Human inspection suggested          |   8.3 |   0   |
| Issues revealed by human inspection |   0   |   0   |

### <a id="table-3-1"></a> 
**Table 3.1: Events triggered in the Swiss Cheese workflow.** Values indicate the percentage of documents (out of 252) in which each event occurred, separately for CCP and LTD. 
<br>  
<br>

### <a id="figure-3-3"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/36_supervisor_corrects_MINI.png)

**Figure 3.3: Example of error correction by supervisor process.** In this document, the mini model determined that 4 items in the Balance Sheet are related to LTD. However, two of these items should not have been included (red arrows): 'Long-term deferred tax liabilities' and 'Long-term tax liabilities'. The supervisor successfully identified these errors (items "3" and "4") and removed them. 
<br>
<br>  

### <a id="figure-3-4"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/34_large_corrects_MINI+supervisor.png)

**Figure 3.4: Example of error correction by large model.** In this document, the mini model determined that 3 items in the Balance Sheet are related to LTD. However, one of these items should not have been included (top red arrow): 'Long-term tax liabilities'. The supervisor was called, but failed to identify this error (returned an empty list; bottom red arrow). Since the output following the supervisor contained a suspicious term ('tax'), the large model was called and extracted only the two relevant entries.  
<br>
<br>  

### Example Data Analysis
Now that we've extracted data from the documents, we can analyze it. By deducting LTD from CCP, we get a company's **net cash position**, a rough indicator of financial flexibility. Positive values suggest excess liquidity; negative values reflect reliance on long-term borrowing, which may signal either strategic investment or financial strain. <a href="#figure-3-5" style="white-space: nowrap; font-weight: bold;">Fig. 3.5</a> shows *normalized* net cash positions for all companies and value dates included in the current dataset. Note that for most companies and most time points, the net cash position is negative, indicating that long-term debt exceeds liquidity.  
<br><br>

### <a id="figure-3-5"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/normalized_net_cash_all.png)

**Figure 3.5: Normalized net cash positions for all companies in dataset.** For each quarter, the net cash position was normalized by dividing it by the company's market cap on the financial report's value date. The y-axis is on a symmetric logarithmic scale (symlog), using linear scaling within ±0.01 and logarithmic scaling beyond that. AAPL = Apple Inc., AMGN = Amgen, AMZN = Amazon, BBWI = Bath & Body Works, Inc., BMY = Bristol Myers Squibb, CNC = Centene Corporation, COP = ConocoPhillips, GILD = Gilead Sciences, HSIC = Henry Schein, KO = The Coca-Cola Company, TMO = Thermo Fisher Scientific, UPS = United Parcel Service.
<br>
<br>  

<a href="#figure-3-6" style="white-space: nowrap; font-weight: bold;">Fig. 3.6</a> and <a href="#figure-3-7" style="white-space: nowrap; font-weight: bold;">Fig. 3.7</a> show the net cash position and stock price for two representative companies (Amazon and Centene Corporation, respectively). These companies were chosen because their net cash position varied widely throughout the analyzed period, including bouts of positive and negative net cash.

Overall, no significant correlation was found between changes in net cash position and change in stock price on the next trading day after report publication (<a href="#figure-3-8" style="white-space: nowrap; font-weight: bold;">Fig. 3.8</a>), suggesting that net cash position is not a primary driver of short-term stock price changes.
<br><br>

### <a id="figure-3-6"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/price_vs_netcash_amzn.png)

**Figure 3.6: Amazon's stock price and net cash position for the analyzed period.** Stock price is shown in blue, net cash position in orange. Dashed orange line signifies 0 net cash. Net cash is shown by filing dates - i.e., the dates at which financial reports became publicly available. 
<br>
<br>  

### <a id="figure-3-7"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/price_vs_netcash_cnc.png)

**Figure 3.7: Centene Corporation's stock price and net cash position for the analyzed period.** Stock price is shown in blue, net cash position in orange. Dashed orange line signifies 0 net cash. Net cash is shown by filing dates - i.e., the dates at which financial reports became publicly available. 
<br>
<br>  

### <a id="figure-3-8"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step3_extract_by_concept/images/change_correlations.png)

**Figure 3.8: Correlations between net cash and stock price changes.** For each company, the correlation (r) between Δ net cash (absolute difference in net cash between consecutive reports) and % price change (between stock prices on the next trading day following report publications). Green and red bars signify positive and negative correlations, respectively. None of the correlations were found to be statistically significant (*p* < 0.05, Pearson's correlation with Holm's correction for multiple comparisons), suggesting that changes in published net cash do not usually drive changes in stock price. 
<br>
<br>  


## General Conclusions  

- This project demonstrates the power of combining LLM-based and algorithmic approaches to perform reliable automated data extraction. 
- Affordable access to LLM (and other) APIs democratizes automated data extraction. The **total cost** of API usage for the full pipeline ([from obtaining the documents to value extraction](https://github.com/ronihogri/financial-doc-reader/blob/main/README.md)) was **less than $0.005 per document**, enabling even small teams to process thousands of documents with minimal investment. 
- The ability to extract structured data from previously inaccessible sources can significantly enhance any enterprise's ability to integrate hard-to-access data into operational decisions.


