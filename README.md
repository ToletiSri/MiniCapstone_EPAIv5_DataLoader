# Mini Capstone - Data loader

In this project, we build a flexible DataLoader class that can load, preprocess, and manage different types of datasets commonly used in AI and machine learning projects, including:

Image datasets: CIFAR-10, CIFAR-100, MNIST
Text datasets: Small text datasets
Structured data: CSV files
Unstructured data: Folders containing multiple files of various formats
Your DataLoader should:

Download datasets from online sources if they are not already present locally.
Handle different file formats and organize data appropriately.
Provide data in batches for efficient processing.
Support data augmentation and preprocessing steps.
Be extensible to accommodate new data types and sources.
This project gets a hands-on experience with data preprocessing, which is a critical step in AI projects, while applying the Python concepts we've learned so far.

### Project Breakdown by Concepts

#### Basics
Classes: Implement the DataLoader as a class. \
Functions: Define methods for downloading, loading, and preprocessing data. \
Loops and Conditionals: Iterate over files and data, check for file existence. 

#### Object Mutability and Interning
Manage mutable data structures like lists and dictionaries to store datasets. \
Understand how changes to objects affect data integrity.

#### Numeric Types I & II
Handle numerical computations during preprocessing (e.g., normalization). \
Use booleans and comparison operators for condition checks.

#### Functional Parameters
Create flexible methods that accept various parameters for data transformations. \
Use **kwargs to pass optional preprocessing functions.

#### First-Class Functions Part I & II
Use lambda functions for simple data transformations. \
Employ map and filter to process data iterables.

#### Scopes and Closures
Maintain state within data loading functions using closures if necessary. 

#### Decorators
Implement decorators to log the time taken for data loading and preprocessing.\
Use decorators for caching data to avoid redundant computations.

#### Tuples and NamedTuples
Use namedtuples to represent data samples with features and labels. 

#### Modules, Packages, and Namespaces
Organize code into modules for loaders, preprocessors, utils, etc.\
Use packages to separate different components logically.

#### f-Strings, Timing Functions, and Command Line Arguments
Use f-strings for informative print statements. \
Accept command-line arguments for configuration (e.g., dataset selection). 

#### Sequence Types I & II and Advanced List Comprehension
Manage collections of data samples. \
Use list comprehensions for efficient data processing. 

#### Iterables and Iterators
Implement an iterator protocol in the DataLoader to iterate over data batches.

#### Generators and Iteration Tools
Use generators to load data on-the-fly without consuming excessive memory.

#### Context Managers
Use context managers when opening files to ensure they are properly closed.

#### Exception Handling (Try/Except)
Handle exceptions during file I/O and data processing to prevent crashes.
