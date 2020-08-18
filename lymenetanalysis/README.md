# Determining topic size using topic model evaluation methods

## Getting Started

These instructions will setup an environment on your local machine for development and analysis purposes.

### Prerequisites

Software to be installed:

* Anaconda Python 3.6
* MongoDB

Additionally, the following Python modules are required:

* numpy
* pandas
* matplotlib
* sklearn
* keras
* pymongo
* bson
* queue
* scipy.spatial.distance
* scipy.stats

Running the project:

macOS High Sierra or higher:
* Install Mongodb
* How to run mongodb if data/db is not created:
  * mkdir -p /System/Volumes/Data/data/db
  * mongod --dbpath /System/Volumes/Data/data/db
  
The data required for this project can be accessed [here](https://drive.google.com/drive/folders/1eIKizQOROuz-dF5icmGT8EmILLoKLGOg?usp=sharing)
  
 
Running scripts:
1. Create an empty directory called "graphsAndFigures"
2. Import JSON file into MongoDB
   ```
    mongoimport --db lymeDiseaseDB --collection medicalQuestionData --file medicalQuestionData.json
   ```
3. Extracting posts/documents from MongoDB
   ```
    python3 generate_documentsJSON.py
   ```
4. Run Topic Modeling - Generate graphs for Coherence, Perplexity, Kendall-Tau similarity
   ```
    python3 LDA_topicModeling.py
   ```

5. Topic Model Visualization using PyLDA
   ```
    python3 topic_model_visualization.py
   ```



#CODEBASE DESCRIPTION WORK SAMPLE
* I have documented the Topic Similarity Algorithm in the PDF [report](https://github.com/Aditya9517/Lymenet-Analysis/blob/master/LymeNetReport_AdityaJayanti.pdf), this algorithm is implemented in ```topic_similarity.py```
* Files to consider
```
LDATopicModeling.py
```

```
topicDistribution.py
```

```
topicModelVisualization.py
```

```
topicSimilarity.py
```
