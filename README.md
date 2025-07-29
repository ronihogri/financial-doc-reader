# Financial Document Reader
Roni Hogri, January 2025.  

**Keywords:** AI-assisted automation, Data Extraction, ChatGPT, Python, OpenAI API, Prompt Engineering, BeautifulSoup, SQL, JSON, LLM Evaluation.

## Why This Project?

Large Language Models (LLMs) like ChatGPT can be highly effective for *certain* tasks, such as identifying relevant data based on context and semantics. However, they suffer from notable limitations, such as unexpected outputs and miscalculations.

In the emerging field of **"AI-Assisted Systems"**, LLMs are integrated with "old-school" code, so that developers and users can enjoy the best of both worlds.

This demo project is meant to provide a basic and simple example of implementing AI-Assisted software for the purpose of data extraction from financial documents.
Specifically, I will focus on filings to the SEC (10-Q and 10-K forms). These filings are standardized in some ways, but are highly variable in their exact wording and HTML structure. This variability makes it very difficult to automate data extraction based on properties such as table structure and header titles. Nowadays, this task can be assisted by LLMs, which can "understand" the context and semantics of data fields.

I will show how to easily integrate ChatGPT into Python scripts, and focus on some of the main challenges of such integrations:

1. **Identifying the right tool** for specific tasks and sub-tasks.
2. **Prompt engineering**: How to get ChatGPT to return values that can be used by the rest of the Python program.
3. **Evaluation and error minimization**. This is a critical point, since LLMs (and their mistakes) are inherently unpredictable. Importantly, <u>errors introduced by LLMs can propagate throughout the process and corrupt it</u>.

This project is meant to demonstrate how affordable and accessible off-the-shelf solutions can add value to anyone wishing to extract data from semi-structured sources. As such, it will *not* cover more advanced topics such as the workings of LLM technology, setting up local LLMs, or implementing Retrieval-Augmentend Generation (RAG) systems.

## Steps

The data extraction process will be explained and demonstrated step by step. Each step has a dedicated folder with files, documentation, and example results. Each step is self-contained, so you can run any step independently of the others.  

- [**Step 1**](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step1_find_BS_table) : Extract Balance Sheet table content
- [**Step 2**](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step2_BStable2json) : Export Balance Sheet table to a structured JSON file
- [**Step 3**](https://github.com/ronihogri/financial-doc-reader/tree/main/steps/step3_extract_by_concept) : Identify and extract values of interest, based on meaning rather than exact label matches 



## Installation

```console
# Clone this repository
$ git clone https://github.com/ronihogri/financial-doc-reader.git

# Go to the appropriate directory
$ cd financial-doc-reader

# Install requirements
$ python -m pip install -r requirements.txt
```

**Note:** If you would like to run the Python scripts yourself, you must possess a valid [**OpenAI API key**](https://platform.openai.com/docs/quickstart?desktop-os=windows).



