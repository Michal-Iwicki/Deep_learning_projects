import torch.optim as optim
import numpy as np
from loader import load_png_images
from implementation import CNNClassifier, train_model,evaluate
import pandas as pd
import csv
import os
import datetime

def save_to_csv(data, filename, column_names):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")
    filename = f'{filename}_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  
        writer.writerows(data)

def aggregate_csv_files(folder_path = "./results"):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []
    
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        group_col = df.columns[0]  
        aggregated_df = df.groupby(group_col).agg(['mean', 'std']).round(3)
        aggregated_df.columns = ["_".join(col).lower() for col in aggregated_df.columns]
        dataframes.append(aggregated_df)
    
    return dataframes

def test_batchsizes(times = 5, batchsizes = [8,16,32,64,128]):
    test_loader = load_png_images("data/test_sample", batch_size=1024, shuffle=False)[0] 
    val_loader = load_png_images("data/val_sample", batch_size=1024, shuffle=False)[0]  
    result = []
    column_names = ["batchsize","test_loss","test_accuracy"]
    print(*column_names)
    for batchsize in batchsizes:
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=batchsize)  
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            test_loss, test_acc = evaluate(model, test_loader)
            values = [batchsize,test_loss,test_acc]
            result.append(values)
            print(*values)
    save_to_csv(result,"results/batchsize_test.csv", column_names)

def test_lr(times = 5, learning_rates = [0.0001,0.0005,0.001,0.005,0.01]):
    test_loader = load_png_images("data/test_sample", batch_size=1024, shuffle=False)[0] 
    val_loader = load_png_images("data/val_sample", batch_size=1024, shuffle=False)[0] 
    result = []
    column_names = ["learning_rate","test_loss","test_accuracy"]
    print(*column_names) 
    for lr in learning_rates:
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=32)  
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            test_loss, test_acc = evaluate(model, test_loader)
            values = [lr,test_loss,test_acc]
            result.append(values)
            print(*values)
    save_to_csv(result,"results/lr_test.csv", column_names)

def test_dropout(times = 5, dropout_rates = [0,0.2,0.4,0.6,0.8]):
    test_loader = load_png_images("data/test_sample", batch_size=1024, shuffle=False)[0] 
    val_loader = load_png_images("data/val_sample", batch_size=1024, shuffle=False)[0] 
    result = []
    column_names = ["dropout_rate","test_loss","test_accuracy"]
    print(*column_names)
    for dr in dropout_rates:
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=32)  
            model = CNNClassifier(num_classes,dropout_rate=dr)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            test_loss, test_acc = evaluate(model, test_loader)
            values = [dr,test_loss,test_acc]
            result.append(values)
            print(*values)
    save_to_csv(result,"results/dropout_rate_test.csv", column_names)

def test_batchnorm(times = 5):
    test_loader = load_png_images("data/test_sample", batch_size=1024, shuffle=False)[0] 
    val_loader = load_png_images("data/val_sample", batch_size=1024, shuffle=False)[0]  
    result = []
    column_names = ["batch_norm_used","test_loss","test_accuracy"]
    print(*column_names)
    for use_bn in [True, False]:
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=32)  
            model = CNNClassifier(num_classes,use_bn=use_bn)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            test_loss, test_acc = evaluate(model, test_loader)
            values = [use_bn,test_loss,test_acc]
            result.append(values)
            print(*values)
    save_to_csv(result,"results/batch_norm_test.csv", column_names)
