import time
import PIL
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import apex
from tqdm import tqdm
import multiprocessing
num_threads = 2
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001
AMP_LEVEL = "O1"
PATH = "~/scratch/demo/cifar_net.pth"

import os

print(" Configuration ")
print("===============")
print("   Batch size:", BATCH_SIZE)
print("  CPU threads:", num_threads)
print("Learning rate:", LEARNING_RATE)
print("    AMP level:", AMP_LEVEL)
print("       Epochs:", EPOCHS)
print("")
print("Model to be saved to", PATH)

print("\n\n")

time.sleep(1)

def return_cnn_model(num_class=2, pretrained=True):
    model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    return model

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
def main():
    st = time.time()
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomAffine(5, scale=(0.9,1.1), resample=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="~/data", train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=num_threads)

    testset = torchvision.datasets.CIFAR10(root="~/data", train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=num_threads)

    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = return_cnn_model(num_class=len(classes), pretrained=True).cuda()
    print("Number of parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=LEARNING_RATE)

    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=AMP_LEVEL)

    interval = 50
    train_steps = int(len(trainset)/BATCH_SIZE)
    img_per_epoch = train_steps*BATCH_SIZE
    
    fps_list = []

    for epoch in range(EPOCHS):
        print("\nEpoch: "+str(epoch+1)+"/"+str(EPOCHS))
        print("")

        running_loss = 0.0
        
        epoch_start = time.time()

        for i, data in tqdm(enumerate(trainloader), total=train_steps):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step = i+1
            if step % interval == 0:
                print(step, running_loss/interval)
                running_loss = 0.0
                
        duration = time.time() - epoch_start
        fps = int(img_per_epoch/duration)
        fps_list.append(fps)
        print("\nImages/sec:", fps, "\n")
        
    et = time.time()
    
    print("\n\n")
    print("Finished training!")
    print("==================")
    avg_fps = np.mean(np.asarray(fps_list))
    min_fps = min(fps_list)
    print("Average images/sec:", int(avg_fps))
    print("Min images/sec:", int(min_fps))
    torch.save(model.state_dict(), PATH)
    print("Model saved to", PATH)
    print("End-to-end time:", int(et-st))
    
    uid = str(int(time.time()))
    result_path = "~/scratch/demo/"+uid+"_result.txt"
    f = open(result_path,"w+")
    f.write("Avg images/sec:"+str(avg_fps))
    f.write("\nMin images/sec:"+str(min_fps))
    f.write("\nEnd-to-end time:"+str(et-st))
    f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
