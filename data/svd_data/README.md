# SVD data

## Format
> userid,itemid,rating,timestamp
[Detail](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)

## Example (each line should be the same size)
773,31,2.5,1260759144
484,1029,3.0,1260759179
56,1061,3.0,1260759182
25,1129,2.0,1260759185
451,1263,2.0,1260759151
2,1287,2.0,1260759187
98,1293,2.0,1260759148
78,1339,3.5,1260759125

## Feature
This data set has 671 userid and 163949 itemid.

## Convert to Tfrecords
```python
python convert_to_tfrecord.py
```

## Check Tfrecords
```python
python print_tfrecord.py
```