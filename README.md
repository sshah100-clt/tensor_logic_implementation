Tensor Logic: Empirical Validation

This repository contains empirical implementations of the "Tensor Logic" framework proposed by Pedro Domingos (2025). The project validates the core premise that logical rules and Einstein summation (einsum) are mathematically equivalent operations.

The code demonstrates how logical inference can be performed using tensor contraction, both in discrete symbolic space (using NumPy) and learnable embedding space (using PyTorch).


The Concept:

Based on the paper Tensor Logic: The Language of AI, this project explores the idea that logical reasoning is geometrically equivalent to multiplying matrices.

Facts are Tensors: Information (e.g., "Bob is Alice's father") is stored as a connection in a sparse matrix.

Rules are Operations: Applying a logical rule (e.g., "A parent of a parent is a grandparent") is performed by multiplying those matrices together.

Inference is Traversal: Finding an answer to a query is equivalent to checking if a path exists between two points in the matrix or vector space.


Experiments:

1. Symbolic Reasoning: Transitive Closure
File: bible.ipynb

The Challenge: Use matrix multiplication to discover every ancestor in a genealogy graph starting only with a list of immediate parents.

Data: A graph of 1,972 people and 1,727 parent-child links derived from biblical text.

Method: The code treats the "Parent" list as an adjacency matrix. By repeatedly multiplying this matrix by itself (iterative tensor contraction), it ripples through the generations to find the "transitive closure".


Results:

The system successfully mapped the entire lineage, converging in 74 iterations (representing the generational depth).

It discovered 33,945 ancestor relationships purely through matrix operations.

Verification: Checks confirmed that the mathematical operations preserved logical consistency (e.g., ensuring no one is their own ancestor).


2. Neural Reasoning: Compositionality in Embedding Space

File: tensor_01.ipynb

The Challenge: Train a neural network to learn concepts like "Capital" and "Location" as geometric movements, allowing it to answer compositional questions it was never explicitly taught.

Data: A dataset of Cities, Capitals, Countries, and Regions.


Method:

Instead of hard-coded rules, the model learns Transformation Matrices for relations.

To answer a query like "What continent is Tokyo in?", the model starts at the vector for Tokyo, applies the Is Capital matrix, and then applies the Is Located In matrix.

The resulting vector lands near the vector for Asia in the embedding space.


Results:

Achieved 100% accuracy on Zero-Shot queries.

The model correctly inferred continents for cities despite never seeing a direct link between a city and a continent in the training data.


Usage:

Dependencies

Python 3.x

numpy (for symbolic matrix operations)

pandas (for data handling)

torch (for neural network training)


Running the Experiments:

Symbolic (Bible): Run bible.ipynb to download the dataset and visualize the convergence trace of the genealogy graph.

Neural (Countries): Run tensor_01.ipynb to train the embedding model and test zero-shot compositional queries.


Data Sources:

Bible Data: https://github.com/BradyStephenson/bible-data

Countries Data: https://github.com/mledoze/countries