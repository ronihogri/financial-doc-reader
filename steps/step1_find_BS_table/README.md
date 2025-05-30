# Step 1: Extract Balance Sheet Table Content

**A general overview of the project can be found [here](https://github.com/ronihogri/financial-doc-reader/blob/main/README.md).**

## The Problem: Extracting the Right Data

We often need to extract data from semi-structured or semi-standardized sources. This can be quite difficult using "classic" programming methods, since we need to account for *a lot* of variability.

This project focuses on extracting data from the Balance Sheet table of quarterly and yearly financial reports filed with the SEC (10-Q and 10-K forms, respectively). While these reports follow general conventions, their wording and HTML structure can vary significantly. Therefore, **it is difficult to design a program that would successfully locate specific data within different financial reports.**

For example:

1. The Balance Sheet *table* isn't always placed in a predictable or clearly labeled table header in the HTML code.
2. We may use keywords to try to identify the Balance Sheet table *text*. However, we can't use too many keywords, since some tables might not contain them. On the other hand, if we only use a handful of keywords, we'll get multiple text blocks that may contain the Balance Sheet table; designing a program that would pick the right text block is not straightforward.

## The Solution: Integrating AI Into Python Programs

Large language models (LLMs) like ChatGPT have been trained on **very** large datasets (including many financial documents). As a result, LLMs are quite good at determining whether a certain block of text belongs to a specified category. Using the OpenAI API, we can easily integrate ChatGPT into our Python script, and consult it when appropriate.


## Workflow

1. Obtain your own [OpenAI API key](https://platform.openai.com/docs/quickstart?desktop-os=windows). Set this key as an environment variable in your OS. Alternatively, use an editor to open the Python script at `financial-doc-reader/steps/step1_find_BS_table/SEC_filing_reader_step1.py`, and insert your key as the string value of MY_API_KEY (under "User-definable variables").
2. *Optional*: Adjust user-definable variables as needed.
3. Navigate to `financial-doc-reader/steps/step1_find_BS_table/`, and run the Python script:

```console
$ python3 SEC_filing_reader_step1.py
```
<br>
The program processes 252 financial documents, filed by 12 selected companies (see the "Forms" and "Stocks" tables of the SQL database in <code>./filings_demo_step1.sqlite</code>). The workflow for each document is illustrated in <a href="#figure-1-1" style="white-space: nowrap; font-weight: bold;">Fig. 1.1</a>. As the program runs, it stores detailed data for each document in designated JSON files within the <code>./extracted/</code> folder. These JSON files enable tracing of the program's steps, and are particularly useful for offline evaluation of LLM performance (see example in <a href="#figure-1-2" style="white-space: nowrap; font-weight: bold;">Fig. 1.2</a>). In addition, the main results are saved to the "Tasks" table of the SQL database (<a href="#figure-1-3" style="white-space: nowrap; font-weight: bold;">Fig. 1.3</a>). Example results are provided in the <code>./results.zip</code> archive. 


<br><br>

### <a id="figure-1-1"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/flow_chart_JSON_reference.png)

**Figure 1.1: Workflow overview.** For each document, text blocks containing keywords commonly found in Balance Sheet tables are extracted. If multiple keyword-containing text blocks are found, the "mini" <span style="white-space: nowrap;">model (gpt-4o-mini-2024-07-18)</span> is asked to identify the text block containing the Balance Sheet table and to return the index of this text block (a single integer) &ndash; this is repeated up to five times for the purpose of collecting "votes" (see the [Results](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/README.md#results) section below). If three of the mini model's responses are identical, this is considered the model's "majority decision". If this decision "makes sense" &ndash; i.e., a single integer within the expected range, the response is considered valid and stored in the SQL database. Otherwise, the "large" <span style="white-space: nowrap;">model (gpt-4o-2024-08-06)</span> is asked the same question once, and its response is evaluated and stored. In practice, the program provided here performed the task with 100% accuracy relying exclusively on the mini model (see [Results](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/README.md#results)). Dark blue nodes indicate data being stored in designated JSON files.

<br>

### <a id="figure-1-2"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/JSON_example_clipped.png)<br>  
**Figure 1.2: JSON data.** Example data for one form, focusing on the text block identification task. In this case, three text blocks were extracted from the document based on keywords. Based on a majority decision, the mini model correctly identified text block #1 as the block containing the Balance Sheet table. 

<br>

### <a id="figure-1-3"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/SQL_tables_results.png)<br>  
**Figure 1.3: SQL data.** Example contents of SQL tables (*Left*: "Stocks", *Middle*: "Forms", *Right*: "Tasks"). After running the program, the "Tasks" table shows for each document the number of text blocks extracted ("TextListLen") and the block identified as the one containing the Balance Sheet table ("TableIndex").
<br><br>
<br>



## Results

<a href="#figure-1-4" style="font-weight: bold;">Fig. 1.4</a> shows the distribution of text block counts per document. Keyword-based extraction returned multiple <span style="white-space: nowrap;">(2&ndash;16)</span> text blocks for 53.6% of documents; in these cases, the LLM was used to identify the text block containing the Balance Sheet table. 


<br>

### <a id="figure-1-4"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/text_block_distribution.png)

**Figure 1.4: Distribution of text block counts.** *Left*: The proportion of documents where keyword-based extraction yielded a single vs. multiple text blocks. Only documents with multiple text blocks were further processed. *Right*: The distribution of text block counts for documents with multiple text blocks.
<br>  

The Python program provided here has been optimized. When run three times, it achieved <span style="white-space: nowrap; font-weight: bold;">100% accuracy</span> in picking the correct text block for the 135 multi-text documents (i.e., a total of 405 decisions) **using only the mini model**. The cost per run was \~$0.17, or **\~$0.001 per document**. Optimization focused on two main aspects:

1. **Prompt engineering**: The full prompt sent to ChatGPT can be seen in <a href="#figure-1-5" style="white-space: nowrap; font-weight: bold;">Fig. 1.5</a>. The "User Role" part of the prompt is color-coded to represent three versions (A, B, C) that were tested for their effect on response accuracy. All versions contained all necessary instructions. Nevertheless, the instructions in versions B and C were more exhaustive, and resulted in significantly smaller error rates compared to version A (7.8x reduction B vs A, 62.3x reduction C vs A; <a href="#figure-1-6" style="white-space: nowrap; font-weight: bold;">Fig. 1.6</a>).   
2. **Voting**: By definition, there is some randomness to the output of the LLM, which can be averaged out. To reduce error rates, the model performed the task up to five times ("votes") per document, stopping early if a majority decision (three identical votes) was reached. Voting reduced the error rate for prompt version A by more than 4x, and entirely eliminated errors for versions B and C (<a href="#figure-1-6" style="white-space: nowrap; font-weight: bold;">Fig. 1.6</a>).

The program provided here utilizes version C in combination with voting.

<br>

### <a id="figure-1-5"></a>
![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/prompt_versions.png)  

**Figure 1.5: Prompt versions.** To test the effects of prompt content on task performance, the program was run using different versions of 'User Role' instructions. Version A contained only the orange text; version B contained both the orange and blue text; version C contained all text. Expressions in curly brackets are refrences to variables in the Python script: 'text_list' refers to the list of keyword-retrieved text blocks, with possible index values ranging between 0 and len(text_list)-1. 

<br>

### <a id="figure-1-6"></a>![](https://github.com/ronihogri/financial-doc-reader/blob/main/steps/step1_find_BS_table/images/optimization_bar_graph_significance.png)
  
**Figure 1.6: Effects of prompt engineering and voting on accuracy.** Error rates for each prompt version (A, B, C; see <a href="#figure-1-5" style="white-space: nowrap; font-weight: bold;">Fig. 1.5</a>), under two voting conditions: i. "All Votes", with each vote counted as a standalone decision (smooth bars); ii. "Majority Decisions", with decisions based on 3&ndash;5 votes (striped bars, not seen for versions B and C due to error rates being 0%). The program was run three times for each prompt version, resulting in 1215&ndash;1272 votes and 405 majority decisions per version. A logistic regression analysis revealed significant effects of prompt version, voting, and their interaction, suggesting that the importance of voting decreased as the prompt version improved. <br><br>   


## Conclusions

- The Python program provided here identified the relevant text out of 2&ndash;16 options, with 100% accuracy, at a cost of ~$0.001 per document.
- The combination of prompt optimization and voting significantly contributed to decision accuracy.
- The low cost is due to the effectiveness of the "mini" model gpt-4o-mini-2024-07-18 (<span style="white-space: nowrap;">~17x</span> cheaper than the larger gpt-4o-2024-08-06 model) in this particular task. The cost could be further reduced by lowering the number of votes required for a majority decision.
 
## Next Step
[**Step 2**](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step2_BStable2json) involves converting the Balance Sheet table text into a structured data format (JSON), enabling more efficient and accurate access to values of interest. 