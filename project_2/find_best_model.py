import torch.optim as optim

from tester import get_loader
from CNN_transfomers_implementation import Mel_transformer, RawAudioTransformer, train_transformer, evaluate_model
from EfficientNet_implementation import get_pretrained_model, train_model, test_model

def find_best_model(model_type, learning_rates=[0.1, 0.01, 0.001, 0.0001], weights_decays=[0.1, 0.01, 0.001, 0.0001], data_size="sample", denoised=False):
    # retrive respective functions to train and to test model based on model's type
    train_f = train_model if model_type == "EfficientNet" else train_transformer
    test_f = test_model if model_type == "EfficientNet" else evaluate_model
    
    # retrive respective validation loader based on model's type
    val_loader = get_loader(data_size, denoised, True if model_type in ["EfficientNet", "MelTransformer"] else False, "validation")
    
    # initiate default values
    best_val_loss = float("+inf")
    best_model = model.state_dict()
    best_learning_rate = 0.001
    best_weights_decay = 0
    
    # look for the best learning rate
    for learning_rate in learning_rates:
        # retrive respective train loader based on model's type
        train_loader = get_loader(data_size, denoised, True if model_type in ["EfficientNet", "MelTransformer"] else False, "train")
        
        # create respective model based on model's type and its optimizer
        model = get_pretrained_model() if model_type == "EfficientNet" else Mel_transformer if model_type == "MelTransformer" else RawAudioTransformer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # train model
        train_f(model, optimizer, train_loader, val_loader)
        
        # calculate validation loss
        val_loss, _ = test_f(model, val_loader)
        
        # update if it is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_learning_rate = learning_rate
            
    # look for the best weights decay rate
    for weights_decay in weights_decays:
        # retrive respective train loader based on model's type
        train_loader = get_loader(data_size, denoised, True if model_type in ["EfficientNet", "MelTransformer"] else False, "train")
        
        # create respective model based on model's type and its optimizer
        model = get_pretrained_model() if model_type == "EfficientNet" else Mel_transformer if model_type == "MelTransformer" else RawAudioTransformer
        optimizer = optim.AdamW(model.parameters(), lr=best_learning_rate, weights_decay=weights_decay)
        
        # train model
        train_f(model, optimizer, train_loader, val_loader)
        
        # calculate validation loss
        val_loss, _ = test_f(model, val_loader)
        
        # update if it is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_weights_decay = weights_decay
            
    # return best model configuration, best learning rate and weights decay rate
    return model.load_state_dict(best_model), best_learning_rate, best_weights_decay
            