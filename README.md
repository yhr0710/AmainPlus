# AmainPlus
An Efficient Code Clone Detection Method Based on Second-Order Markov Chains

AmainPlus is an efficient and accurate code clone detection method. It treats two consecutive nodes in the abstract syntax tree (AST) as the initial state of a Markov chain, and the subsequent node connected to them as the next state of the Markov chain, thus fully considering the semantic information in the AST. Another core innovation is that, to save on costs, we consider the column similarity of the constructed state transition matrix to extract features.

It includes the following core stages:
1. State Matrix Construction: After obtaining the AST, we derive the second-order Markov state transition matrix by partitioning the nodes.
2. Feature Extraction: We obtain features by calculating the column similarity between the matrices of two programs, which has the advantage of significantly saving time.
3. Classification:The purpose of this stage is to use machine learning classification algorithms to train and classify the extracted features. Of course, you can also try using emerging models such as Mamba.

### Project Structure  
  
```shell  
AmainPlus  
|-- Clone_type            // It includes the clone types from two common datasets in the code clone domain: Google Code Jam (GCJ) and BigCloneBench (BCB).
|-- Train                 // It includes the three files needed for training.
|   |-- get_matrix.py         // Used to obtain state matrices.
|   |-- get_distance.py       // Used to obtain distance feature vectors.
|   |-- classification.py     // Used for classification.
|-- train_system.py       // You can use this class to train your own code. You just need to provide the path to your Java files, the CSV file with clone and non-clone pairs, and of course, you can also customize the path where the matrices are generated.
```

### train_system.py
- Input: Your Java file path, and the CSV file representing clones and non-clones.
- Output:  A fully trained clone detection model.
```
python train_system.py
```


### Parameter details of our comparative tools
|Tool            |Parameters                     |
|----------------|-------------------------------|
|SourcererCC	|Min lines: 6, Similarity threshold: 0.7            |
|Deckard      |Min tokens: 100, Stride: 2, Similarity threshold: 0.9 |
|RtvNN       |RtNN phase: hidden layer size: 400, epoch: 25, $\lambda_1$ for L2 regularization: 0.005, Initial learning rate: 0.003, Clipping gradient range: (-5.0, 5.0), RvNN phase: hidden layer size: (400, 400)-400, epoch: 5, Initial learning rate: 0.005, $\lambda_1$ for L2 regularization: 0.005, Distance threshold: 2.56    |
|ASTNN      |symbols embedding size: 128, hidden dimension: 100, mini-batch: 64, epoch: 5, threshold: 0.5, learning rate of AdaMax: 0.002  |
|SCDetector      |distance measure: Cosine distance, dimension of token vector: 100, threshold: 0.5, learning rate: 0.0001 |
|DeepSim      |Layers size: 88-6, (128x6-256-64)-128-32, epoch: 4, Initial learning rate: 0.001, $\lambda$ for L2 regularization: 0.00003, Dropout: 0.75 |
|CDLH      |Code length 32 for learned binary hash codes, size of word embeddings: 100 |
|TBCNN      |Convolutional layer dim size: 300，dropout rate: 0.5, batch size: 10 |
|FCCA      |Size of hidden states: 128(Text), 128(AST), embedding size: 300(Text), 300(AST), 64(CFG) clipping gradient range: (-1.2，1.2), epoch: 50, initial learning rate: 0.0005, dropout:0.6, batchsize: 32|

## Experimental results
#### Detection effectiveness
On the entire Big Clone Bench (BCB) dataset, our model achieved an F1 score of 0.99205, precision of 0.99643, and recall of 0.98771. On the entire Google Code Jam (GCJ) dataset, our F1 score is 0.98936, precision is 0.99445, and recall is 0.98569. This demonstrates the strong clone detection capabilities of our model, surpassing the performance of some currently popular state-of-the-art (SOTA) models.
#### Time overhead.
For training and inference time, we used the same measurement methods as in DeepSim. The results show that training only requires 2216 ± 81 seconds, and inference takes only 29 ± 1 seconds.