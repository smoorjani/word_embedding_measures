# word_embedding_measures

Python implementation of "How quantifying the shape of stories predicts their success" by Toubia et al.

This is not my paper. You can find it [here](https://www.pnas.org/content/118/26/e2011695118.short?rss=1).

## Academic Abstracts

I obtained abstracts from [this Kaggle dataset](https://www.kaggle.com/kmader/aminer-academic-citation-dataset?select=dblp-ref-0.json). Get research paper abstracts using:

`wget https://www.kaggle.com/kmader/aminer-academic-citation-dataset/download/o0mFH8IcsQHZEJ2HX1E1%2Fversions%2FzOZutSMcvhpIpY7AXXtt%2Ffiles%2Fdblp-ref-0.json?datasetVersionNumber=2`

## Running on existing chunks
`python main.py --limit 30000 --chunk_embs_file data/chunk_embs.txt`  

## Running on 16k Persuasive Pairs

`python main.py --limit 10000 --data_file_type xml --data_file ../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/ --model_name fasttext_model/cc.en.300.bin --chunk_embs_file data/16k_chunk_embs.txt`