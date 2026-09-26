import csv
import matplotlib.pyplot as plt

epoch = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []

with open('log_SGD.csv','r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        epoch.append(int(row[0]))
        train_loss.append(float(row[1]))
        train_acc.append(float(row[2]))
        test_loss.append(float(row[3]))
        test_acc.append(float(row[4]))

plt.figure(figsize=(12,5))
# 左图
plt.subplot(1,2,1)
plt.plot(epoch,train_loss,label='Train Loss')
plt.plot(epoch,test_loss,label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("loss curve")

# 右图 Acc
plt.subplot(1,2,2)
plt.plot(epoch,train_acc,label='Train Accuracy')
plt.plot(epoch,test_acc,label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("accuracy curve")

plt.tight_layout()
plt.show()