# logistic regression

## dense data

The data is stored in *data/dense_data*.

data example (each line should be the same size)
>1 20.57 17.77 132.9 1326 0.08474
>1 12.45 15.7 82.57 477.1 0.1278
>1 18.25 19.98 119.6 1040 0.09463

Run demo
```python
python binary_logistic_regression.py \
    --train_file ../data/dense.train \
    --test_file ../data/dense.test \
    --feature_num 30 \
    --dense True \
    --batch_num 100
```

## sparse data

The data is stored in *data/libsvm_data*.

data example (label id:value id:value ...)
> 0 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1
> 0 5:1 7:1 14:1 19:1 39:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 78:1 83:1
> 1 3:1 6:1 17:1 22:1 36:1 41:1 53:1 64:1 67:1 73:1 74:1 76:1 80:1 83:1
> 0 2:1 6:1 18:1 19:1 39:1 40:1 52:1 61:1 71:1 72:1 74:1 76:1 80:1 95:1
> 1 3:1 6:1 18:1 29:1 39:1 40:1 51:1 61:1 67:1 72:1 74:1 76:1 80:1 83:1

Run demo
```python
python binary_logistic_regression.py \
    --train_file ../data/sparse.train \
    --test_file ../data/sparse.test \
    --feature_num 124 \
    --dense False \
    --batch_num 100
```