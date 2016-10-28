# Sparse data

## Format
> label id:value id:value ...

## Example (each line should be the same size)
> 0 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1
> 0 5:1 7:1 14:1 19:1 39:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 78:1 83:1
> 1 3:1 6:1 17:1 22:1 36:1 41:1 53:1 64:1 67:1 73:1 74:1 76:1 80:1 83:1
> 0 2:1 6:1 18:1 19:1 39:1 40:1 52:1 61:1 71:1 72:1 74:1 76:1 80:1 95:1
> 1 3:1 6:1 18:1 29:1 39:1 40:1 51:1 61:1 67:1 72:1 74:1 76:1 80:1 83:1

## Feature
This data has 123 features at most.

## Convert to Tfrecords
```python
python convert_to_tfrecord.py
```

## Check Tfrecords
```python
python print_tfrecord.py
```