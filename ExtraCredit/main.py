import argparse
import numpy as np
from decision_tree import Decision_Tree as DT
from sklearn import metrics

################# Helper Functions #################

def zscore(x, y):
    '''
    This function will zscore every column of both x and y based
    on the mean and standard deviation of the current x column and
    returned the zscored data.
    '''
    col_num = x.shape[1]
    
    train = []
    val = []
    
    for col in range(col_num):
        mean = np.mean(x[:, col])
        std = np.std(x[:, col])
        
        z_train = (x[:, col]-mean)/std
        z_val = (y[:, col]-mean)/std

        z_train = np.expand_dims(z_train, axis=1)
        z_val = np.expand_dims(z_val, axis=1)
        
        train.append(z_train)
        val.append(z_val)
        
    train = np.concatenate(train, axis=1)
    val = np.concatenate(val, axis=1)
    
    return train, val



def read_data(filename, skip_rows):
    '''
    Reads in the data file, splits into train and validation and zscores
    the data.
    '''
    X = np.genfromtxt(filename, delimiter=',')
    if skip_rows:
        X = X[skip_rows:, :]
        
    np.random.shuffle(X)
    Y = X[:, -1]
    X = X[:, :-1]
    Y = np.expand_dims(Y, axis=1)
    
    N = X.shape[0]

    split_val = int(np.ceil((N*2)/3))
    
    train_X = X[:split_val, :]
    val_X = X[split_val:, :]
    
    train_Y = Y[:split_val, :]
    val_Y = Y[split_val:, :]
    
    real_X = np.copy(val_X)
    train_X, val_X = zscore(train_X, val_X)
            
    return train_X, val_X, train_Y, val_Y, real_X


def make_cat(x):
    return np.where(x < np.mean(x), 0.0, 1.0)

####################################################
def evaluate():
    np.random.seed(0)
    
    train_X, val_X, train_Y, val_Y, _ = read_data('CTG.csv', 2)
    train_X = np.apply_along_axis(make_cat, 0, train_X)
    val_X = np.apply_along_axis(make_cat, 0, val_X)
    train_data = np.c_[train_X, train_Y]
    
    dt = DT(train_data)
    
    preds = np.apply_along_axis(dt.predict, axis=1, arr=val_X, tree=dt.tree)
    preds = np.expand_dims(preds, axis=1)
    
    # Print the accuracy, precision, and recall, among other metrics
    print(metrics.classification_report(val_Y, preds, digits=3))
    
    
def predict(row=None, values=None):
    np.random.seed(0)
    
    train_X, val_X, train_Y, val_Y, real_X = read_data('CTG.csv', 2)
    train_X = np.apply_along_axis(make_cat, 0, train_X)
    val_X = np.apply_along_axis(make_cat, 0, val_X)
    train_data = np.c_[train_X, train_Y]
    
    dt = DT(train_data)
    
    if row is not None and row > len(val_X):
        print('Row out of index')
        return
    elif row is not None:
        pred_values = real_X[row]
    elif values is not None:
        pred_values = values
        
    print(f'Input values: {pred_values}')    
    prediction = dt.predict(pred_values, tree=dt.tree)
    print(f'Prediction: {prediction}')
    

def main():
    parser = argparse.ArgumentParser(description='Produce homework answers.')
    parser.add_argument('func', metavar='f', type=str,
                        help='the function being called, either predict or evaluate.')
    parser.add_argument('-r', '--row', type=int, required=False,
                        help='the row to predict on from the test set.')
    parser.add_argument('-v', '--values', nargs='+', required=False,
                        help='values to be used when making a prediction')
    args = parser.parse_args()
        
    if args.func == 'evaluate':
        evaluate()
    elif args.func == 'predict':
        if args.row is not None:
            predict(args.row, None)
        elif args.values is not None:
            v = [float(x) for x in args.values]
            values = np.array(v)
            predict(None, values)
        else:
            print('A row number or values are required to make a prediction.')
    
        
    
main()