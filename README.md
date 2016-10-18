# learn-tensorflow
learning tensorflow

# dense logistic regression
python binary_logistic_regression.py \
    --train_file ../data/dense.train \
    --test_file ../data/dense.test \
    --feature_num 30 \
    --dense True \
    --batch_num 100

# sparse logistic regression
python binary_logistic_regression.py \
    --train_file ../data/sparse.train \
    --test_file ../data/sparse.test \
    --feature_num 124 \
    --dense False \
    --batch_num 100