import copy
import os
import sys
from time import time

import numpy as np
import toolz as tz 
import torch 

from tqdm.auto import tqdm

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

def make_mlp(input_size:int,output_size:int,hidden_layers=[20,20,20]):
    sizes = [input_size, *hidden_layers, output_size]
    layers = [torch.nn.Linear(a,b,bias=True) for (a,b) in tz.sliding_window(2,sizes)]
    n = len(layers)
    activations = [torch.nn.ReLU() for _ in range(n-1)]
    return torch.nn.Sequential(*tz.interleave([layers,activations]))

# prefiero hacer esto que hacer una subclase
def make_dataloader(X,y,batch_size):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X,y),
        batch_size=batch_size,
        shuffle=True
    )

def train_mlp_classifier(train_Xy,val_Xy,hidden_layers=[20]*3,acc_tol=1e-2,patience=15,
                         learning_rate=1e-3,batch_size=200,max_epochs=100,L2_lambda=1e-4,progress=True):
    n_in = val_Xy[0].shape[1]
    n_out = len(np.unique(train_Xy[1]))
    train_loader = make_dataloader(*train_Xy,batch_size=batch_size)
    val_loader = make_dataloader(*val_Xy,batch_size=len(val_Xy[1]))

    model = make_mlp(n_in,n_out,hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda, fused=True)

    best_acc = -np.inf
    best_state = {}
    no_improv = 0

    for epoch in tqdm(range(max_epochs),desc="Epoch",disable=not progress,mininterval=20,file=sys.stdout):
        # Train section
        model.train()
        for X,y in train_loader:
            pred = model(X)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # validate section
        model.eval()
        with torch.no_grad():
            for X,y in val_loader:
                y_pred = model(X).argmax(axis=1)
                val_acc = balanced_accuracy_score(y,y_pred)
        # check section
        if val_acc > best_acc + acc_tol:
            best_acc = val_acc
            no_improv = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improv += 1
        if no_improv > patience:
            break
    # print last epoch for the logs
    # either way, return
    model.load_state_dict(best_state)
    return model, best_acc, epoch


def train_and_save(internal_idx,external_idx,depth,width,batch_size,learning_rate=5e-3):
    hidden_layers = [width]*depth
    model, val_acc, last_epoch = train_mlp_classifier(
        train_Xy = data["train"],
        val_Xy = data["val"],
        hidden_layers = hidden_layers,
        batch_size = batch_size,
        learning_rate = learning_rate,
        max_epochs = 300,
        progress = True
    )
    # measure on test, finally and only once
    model.eval()
    with torch.no_grad():
        for X,y in test_loader:
            y_pred = model(X).argmax(axis=1)
            test_acc = balanced_accuracy_score(y,y_pred)
    # save the model
    torch.save(model.state_dict(),f"models/mlp_{internal_idx}_{external_idx}.pt")
    return val_acc, test_acc, last_epoch



data = {}
for name in ["train","test","val"]:
    loaded = np.load(f"data/{name}.npz")
    data[name] = (
        torch.tensor(loaded["X"],dtype=torch.float32),
        torch.tensor(loaded["y"]),
    )


test_loader = make_dataloader(*data["test"],batch_size=len(data["test"][1]))

depths = [1, 2, 3, 5, 6, 8][::-1]
widths = [10,20,30,40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400, 500, 750, 900, 1000][::-1]


external_grid = ParameterGrid({
    "depth":depths,
    "width":widths,
})

batch_sizes = [80, 100, 150, 200, 300] 
learning_rates = [3e-3,5e-3,8e-3]

internal_grid = ParameterGrid({
    "learning_rate":learning_rates,
    "batch_size":batch_sizes
})

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    external_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    depth_width = external_grid[external_idx]
    print(depth_width)

    out_path = f"stats/{external_idx}.csv"
    if not os.path.exists(out_path):
        with open(out_path,"w") as file:
            file.write(f"internal_idx,external_idx,val_acc,test_acc,last_epoch,training_time,depth,width,lr,bs\n")


    for (internal_idx, bs_lr) in enumerate(internal_grid):
        depth = depth_width["depth"]
        width = depth_width["width"]
        bs = bs_lr["batch_size"]
        lr = bs_lr["learning_rate"]
        try:
            if depth*width <= 1050: # do not train mlp with more than 1050 neurons
                start = time() # a simple spell, yet powerful
                val_acc, test_acc, last_epoch = train_and_save(internal_idx,external_idx,**depth_width,**bs_lr)
                training_time = time()-start

                with open(out_path,"a") as file:
                    file.write(f"{internal_idx},{external_idx},{val_acc},{test_acc},{last_epoch},{training_time},{depth},{width},{lr},{bs}\n")

        except Exception as e:
            print(e)


    
