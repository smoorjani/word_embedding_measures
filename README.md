# Word Embedding Measures

Python implementation of "How quantifying the shape of stories predicts their success" by Toubia et al. This repository is intended for academic abstracts but can easily be generalized to other domains, including those used in the paper (e.g. movies). **This is not my paper.** You can find it [here](https://www.pnas.org/content/118/26/e2011695118.short?rss=1).

## Installation

You should create a fresh environment and install all dependencies either using `pip install -r requirements.txt` or through the `environment.yml` for conda. Note that the `environment.yml` may contain extra packages which may not be necessary for this project.

You will also need to download some form of a word embedding model. In this repositoy, we use [FastText](https://fasttext.cc/) whose models can be downloaded [here](https://fasttext.cc/docs/en/english-vectors.html). Note that if you intend to use data with out of vocabulary words, you need to fine-tune the model and thus you will need to download the `.bin` version. If you want to use an out of box model, you can simply use the `.vec` version. You can specify the location of your model using the `--proj_dir` and `--model_name` arguments. To indicate model training, use the `--train_model` flag.

You will also need to specify the location of your data using the `--data_file` argument. If this argument is a directory, you will need to specify the type of file you want to read in using the `--data_file_type` argument. Note that all of this data should be stored in a directory which is specified by the `--proj_dir` argument. 

### Example Data/Model Structure

As an example, I store my data and models in a directory like this:

```
    .
    ├── .gitignore                   
    ├── utils/                   # Utils folder
    ├── saved/
    │   ├── data/
    │   │   └── dataset.json     # dataset
    │   ├── model/
    │   │   └── fasttext.vec     # fasttext model
    ├── requirements.txt         # requirements
    ├── environment.yml          # conda environment
    ├── main.py                  # main file to run
    └── README.md   
```

In this case, `--proj_dir saved/`, `--model_name model/fasttext.vec`, and `--data_file data/dataset.json`.

### Academic Abstracts

Since this repository is meant to analyze the values of speed, volume, and circuitousness for academic abstracts, our dataset had to contain academic abstracts and the number of citations each paper received. I obtained abstracts from [this Kaggle dataset](https://www.kaggle.com/kmader/aminer-academic-citation-dataset?select=dblp-ref-0.json). You can get this dataset by doing the following:

`wget https://www.kaggle.com/kmader/aminer-academic-citation-dataset/download/o0mFH8IcsQHZEJ2HX1E1%2Fversions%2FzOZutSMcvhpIpY7AXXtt%2Ffiles%2Fdblp-ref-0.json?datasetVersionNumber=2`

## Running the code

When running for the first time, you will have to generate the chunk embeddings of your entire dataset and potentially train the FastText model. You should use the `--train_model` flag and leave the `--chunk_embs_file` argument as empty. The `--chunk_embs_file` argument is meant to load in a set of existing embeddings to avoid the cost of training the model/creating the embeddings each run. 

The two other arguments to consider are the `--limit` and `--T` arguments. The former sets a cap on the amount of data used and the latter represents the number of chunks a document is broken into.

### Running for the first time
You should ideally use the following command (assuming the project directory, data, and model are the same as the default arguments):

`python main.py --train_model`

Note that you can change the values for `--limit` and `--T` based on your needs.

### Running on existing chunks
You should use the following command (assuming the project directory, data, and model are the same as the default arguments):

`python main.py --chunk_embs_file data/chunk_embs.txt`  

Note that you can change the values for `--limit` and `--T` based on your needs.

### Running on 16k Persuasive Pairs
As an example of running this repository with custom data, you can run the following command: 

`python main.py --limit 10000 --data_file_type xml --data_file ../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/ --model_name fasttext_model/cc.en.300.bin --train_model`

And if the chunk embeddings were already calculated, you could run the following:

`python main.py --limit 10000 --data_file_type xml --data_file ../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/ --chunk_embs_file data/16k_chunk_embs.txt`


## Citation

If you use this work in your experiments, please cite the original author's paper with the following BibTex:

```
@article{
doi:10.1073/pnas.2011695118,
author = {Olivier Toubia  and Jonah Berger  and Jehoshua Eliashberg },
title = {How quantifying the shape of stories predicts their success},
journal = {Proceedings of the National Academy of Sciences},
volume = {118},
number = {26},
pages = {e2011695118},
year = {2021},
doi = {10.1073/pnas.2011695118},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2011695118},
eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2011695118},
abstract = {Why are some narratives (e.g., movies) or other texts (e.g., academic papers) more successful than others? Narratives are often described as moving quickly, covering lots of ground, or going in circles, but little work has quantified such movements or tested whether they might explain success. We use natural language processing and machine learning to analyze the content of almost 50,000 texts, constructing a simple set of measures (i.e., speed, volume, and circuitousness) that quantify the semantic progression of discourse. While movies and TV shows that move faster are liked more, TV shows that cover more ground are liked less. Academic papers that move faster are cited less, and papers that cover more ground or are more circuitous are cited more. Narratives, and other forms of discourse, are powerful vehicles for informing, entertaining, and making sense of the world. But while everyday language often describes discourse as moving quickly or slowly, covering a lot of ground, or going in circles, little work has actually quantified such movements or examined whether they are beneficial. To fill this gap, we use several state-of-the-art natural language-processing and machine-learning techniques to represent texts as sequences of points in a latent, high-dimensional semantic space. We construct a simple set of measures to quantify features of this semantic path, apply them to thousands of texts from a variety of domains (i.e., movies, TV shows, and academic papers), and examine whether and how they are linked to success (e.g., the number of citations a paper receives). Our results highlight some important cross-domain differences and provide a general framework that can be applied to study many types of discourse. The findings shed light on why things become popular and how natural language processing can provide insight into cultural success.}}
```

You can also link this GitHub repository for those who may want to build upon this work.
