
# Pre-processing the data

Since one of the tasks in the experiment by Fan et al. (2017) involves control over named entities, the dataset used for training the model was entity-anonymized CNN/Daily Mail. While there are various versions of the non-anonymized dataset modified for PyTorch and TensorFlow, I have not found an available dataset or tutorial that satisfied the requirement of entity anonymization and byte pair encoding. Therefore, I proceeded with manually pre-processing the dataset; this is not only necessary for testing the benchmark but also for running the main experiment of my thesis.

First, it is important to note that the original CNN/Daily Mail dataset, available [here](https://cs.nyu.edu/~kcho/DMQA/), is originally a corpus both for question answering and for text summarization. For our task, the latter is relevant, and hence we make use of over 300000 article-summary pairs in this dataset. The authors provide the stories as raw text, with the summaries contained in the body of the story separated by "@highlight" marks. Moreover, the authors provide their tokenization mapping files for every story, therefore eliminating the need for tokenizers. The consistency of the tokenization process is important since the entities in the dataset are anonymized based on their token index. 

Entity-anonymization of the dataset refers to replacing all named entities with placeholders of the form "@entityX", where X is a natural number. The placeholders are not consistent across the dataset, e.g. if "@entity123" corresponds to Walt Disney in one article, "@entity123" in the next article is not necessarily a placeholder for Disney. This strips the entities of any semantic meaning but allows to both reduce the issue of rare vocabulary items and exert control over entity presence. The mappings for entity-anonymization are not present in the original dataset but are shared as additional content, available [here](https://storage.googleapis.com/deepmind-data/20150824/data.tar.gz).

Moreover, the dataset uses byte-pair encoding to deal with out-of-vocabulary of rare words. This is a theoretically sound approach that identifies most common substrings of tokens in the vocabulary and through using those substrings as a dictionary item allows to represent more unique words with a shorter dictionary (Sennrich et al., 2015). Therefore special UNICODE characters that often pose an issue for rare or out-of-vocabulary tokens is alleviated; however, those characters have proved to be inhibitng benchmark performance as the model would recreate certain errors present in the original dataset and inconsistently use symbols for hyphens and quotation marks. As a result, it was decided to encode the text in ASCII and decode UNICODE characters that appear in more than 50 words in the dataset with their ASCII equivalent, most of them punctuations signs or accents in foreign words used in English. 

As a result of byte-pair-encoding, however, the entity placeholders were split into substrings, resulting in encodings such as "@@@ enti@@ ty@@ 1@@ 2@@ 3", where a double @ sign indicates that the next token is part of the same word. Such encoding is undesirable since we want to condition the model on certain entity placeholders to exert control at inference time. Therefore, using regular expressions all entity placeholders were converted back to their pre-BPE encoding, e.g. "@entity123". This increases the size of the vocabulary but allows the model to control for entities. 

To allow for control over length, it is also necessary to divide the corpus into 10 categories based on the summary length. Therefore, we go over the dataset once to keep track of all lengths and divide the resulting list into ten groups, where len1 corresponds to the shortest and len10 to the longest summaries. While saving the entries, we assign a length code to each sample, therefore allowing in the future to exert control over summary length.

After preprocessing, the dataset is saved as a tabular file and fed to an iterator from PyTorch to handle tokenization, batching and padding. When saved, the stories are cut at 400 tokens, as suggested by Fan et al. (2017); this has to do with the inverted pyramide structure of news articles and the observation that ground-truth summaries rarely contain information taken after the 400 tokens threshold and model performance does not improve. Original length statistics are as follows:
* **Stories**:
* * Mean story length, in tokens: 766.33
* * Maximum length, in tokens: 2001
* **Summaries**:
* * Mean summary length, in tokens: 59.88
* * Maximum length, in tokens: 681.
After applying BPE and cutting summaries over threshold length, length statistics are as follows:
* **Stories**:
* * Mean story length, in tokens: 405.85
* * Maximum length, in tokens: 710
* **Summaries**:
* * Mean summary length, in tokens: 57.63
* * Maximum length, in tokens: 542.

# Pre-processing pipeline

1. Download original data and entity anonymization mapping files
2. Produce BPE codes for encoding over the original dataset
3. Go over the all strings in the dataset to compute the a dictionary of special UNICODE characters and their frequency of occurence in tokens
4. Go over every sample in the dataset: 
* 4a. Read the string and tokenize it based on the provided mapping file 
* 4b. Replace entities with the placeholders based on the token indices and provided entity file 
* 4c. Replace special UNICODE characters with the ASCII equivalent that appear more than 50 times; otherwise delete token
* 4d. Separate the tokens into input, i.e. news article, and target, i.e. summaries. 
* 4e. Compute the length of input and target in number of tokens 
* 4f. Cut the input at 400 tokens 
* 4g. Encode using BPE 
* 4h. Decode entity placeholders from BPE
* 4i. Write the input, target, style category (CNN or DM), entity mappings and length of summary to a tabular file
5. Divide summary lengths into 10 evenly sized categories, assigning a *lenY* token, where Y is a natural number such that 1<=Y<=10
6. Go over rows in the tabular file and append the length code to each sample based on the mapping obtained in step 4


Fan, A., Grangier, D., & Auli, M. (2017). Controllable abstractive summarization. arXiv preprint arXiv:1711.05217.
Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.
