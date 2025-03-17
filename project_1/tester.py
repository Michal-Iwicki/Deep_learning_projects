import torch.optim as optim
import numpy as np
from loader import load_png_images
from implementation import CNNClassifier, train_model,evaluate

def test_batchsizes(times = 5, batchsizes = [8,16,32,64,128]):
    test_loader,n = load_png_images("data/test_sample", batch_size=1024, shuffle=False) 
    val_loader,n = load_png_images("data/val_sample", batch_size=1024, shuffle=False)  
    losses, acc=[],[]
    for batchsize in batchsizes:
        result = np.zeros((times,2))
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=batchsize)  
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            result[i] = evaluate(model, test_loader)
        print("Batchsize", batchsize,result)
        losses.append(result[:,0])
        acc.append(result[:,1])
    return losses, acc 

def test_lr(times = 5, learning_rates = [0.0001,0.001,0.01,0.1,1]):
    test_loader,n = load_png_images("data/test_sample", batch_size=1024, shuffle=False) 
    val_loader,n = load_png_images("data/val_sample", batch_size=1024, shuffle=False)  
    losses, acc=[],[]
    for lr in learning_rates:
        result = np.zeros((times,2))
        for i in range(times):
            train_loader, num_classes = load_png_images("data/train_sample", batch_size=32)  
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, train_loader, val_loader, optimizer, epochs=10,printer = False)
            result[i] = evaluate(model, test_loader)
        print("Learning rate", lr,result)
        losses.append(result[:,0])
        acc.append(result[:,1])
    return losses, acc