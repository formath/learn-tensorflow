# tensorflow models

## logistic regression for dense or sparse data
In this example, you will learn the basic flow of building model and training it. It supports two data type, namely **dense data** and **libsvm data**. 
On both data type, you will learn how to load training data using numpy and parse the data line by line. On dense data, line will be parsed into an label
 and a numpy array which is feeded to the model using **Tensor** in tensorflow. While on libsvm data, line will be parsed into an label and two numpy arrays
 where one holds the sparse id list and another holds the value list. The libsvm data will then be feeded to the model using **Sparse Tensor** in tensorflow.
 You will also learn how to carry out math operation using **Tensor** and **Sparse Tensor**. It should be emphasied the difference between those two types.
 For example, **Tensor** should use **embedding_lookup** method while **Sparse Tensor** should use **embedding_lookup_sparse** instead.
 [Detail](https://github.com/formath/learn-tensorflow/tree/master/logistic_regression)

## singular value decomposition
In this example, you will learn how to build a **svd** model and train it. In *logistic regression* part, the data is readed and parsed by yourself. While
in this part, you will learn the use of **tfrecords**. It is a serialized data format using **protobuf**.
The data should be transformed into **tfrecords** firstly and tensorflow has some inner designs to read and parse it. Those designs simplify the input process to tensorflow
via **Queue** and **Threads**. It also suppliments other convenience such as **batch read** and **shuffle**.
[Detail](https://github.com/formath/learn-tensorflow/tree/master/svd)

## deep and wide model
This example is similar to **singular value decomposition** in many parts except model building. And, this example use **libsvm** data. For sparse data,
the process of transforming data to **tfrecord** and reading from it is something different from the dense one.
[Detail](https://github.com/formath/learn-tensorflow/tree/master/wide_and_deep)
