
import torch
import torch.nn as nn
import matplotlib as plt
import os

device = "cuda"

#uncomplete
def read_anuexdata(data):
    return_data = []

    age = data["age"]       
    sex = data["sex"]
    if sex == "female" or sex == "woman":
        sex = [0,1]
    else:
        sex = [1,0]
    return_data.extend(age)
    return_data.extend(sex)

    return return_data


def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            images, labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            images = images.to(device)
            labels = labels.to(device)

            outputs = net.forward(images)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total


# run model
def run_model_get(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            
            outputs = modelnet.forward(inputs)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)




def show_graph(path,device):
    data = torch.load(path, map_location=device) # by doing map_location=device, you can use trained model on GPU --> to test on CPU
    statsrec = data["stats"]
    fig, ax1 = plt.subplots()
    plt.plot(statsrec[0], 'r', label = 'training loss', )
    plt.plot(statsrec[2], 'g', label = 'test loss' )
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and test loss, and test accuracy')
    ax2=ax1.twinx()
    ax2.plot(statsrec[1], 'm', label = 'training accuracy')
    ax2.plot(statsrec[3], 'b', label = 'test accuracy')
    ax2.set_ylabel('accuracy')
    plt.legend(loc='upper right')
    fig.savefig("roc.svg")
    plt.show()
    
