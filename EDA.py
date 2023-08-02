# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
#
#
# # 定义LSTM模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         out = self.fc(h_n[-1])
#         return out
#
#
# # 定义五折交叉验证
# def five_fold_cross_validation(X, y):
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     accuracies = []
#
#     for train_idx, test_idx in kfold.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         # 在每个fold上创建新的LSTM模型
#         input_size = X_train.shape[2]
#         hidden_size = 64
#         num_layers = 1
#         num_classes = 3
#         model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
#
#         # 定义损失函数和优化器
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#         # 训练模型
#         num_epochs = 10
#         batch_size = 32
#         dataset = TensorDataset(X_train, y_train)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         for epoch in range(num_epochs):
#             for inputs, labels in dataloader:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#         # 在测试集上评估模型性能
#         with torch.no_grad():
#             model.eval()
#             y_pred = model(X_test).argmax(dim=1).numpy()
#             accuracy = accuracy_score(y_test.numpy(), y_pred)
#             accuracies.append(accuracy)
#
#     return np.mean(accuracies)
#
#
# # 从pkl文件加载数据
# def load_data_from_pkl(pkl_filename):
#     with open(pkl_filename, 'rb') as f:
#         data = pickle.load(f)
#         print(data)
#     X, y = data['PPG']['dataX'], data['PPG']['dataY']
#     return X, y
#
#
# if __name__ == '__main__':
#     # 假设你的pkl文件名为'data.pkl'
#     pkl_filename = 'PPG30.pkl'
#     X, y = load_data_from_pkl(pkl_filename)
#
#     # 执行五折交叉验证并打印结果
#     mean_accuracy = five_fold_cross_validation(X, y)
#     print("五折交叉验证准确率：", mean_accuracy)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# # 定义注意力机制
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
#         nn.init.xavier_uniform_(self.attention_weights)
#
#     def forward(self, x):
#         attention_scores = torch.matmul(x, self.attention_weights)
#         attention_weights = torch.softmax(attention_scores, dim=1)
#         attended_x = torch.sum(x * attention_weights, dim=1)
#         return attended_x
# # 定义LSTM模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
#                             bidirectional=True)
#         self.attention = Attention(hidden_size * 2)  # hidden_size * 2 due to bidirectional LSTM
#         self.fc1 = nn.Linear(hidden_size * 2, 2048)  # Add more fully connected layers
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(2048)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(2048, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         attended_out = self.attention(lstm_out)
#         x = self.fc1(attended_out)
#         x = self.relu(x)
#         x = self.bn1(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = self.dropout2(x)
#         output = self.fc3(x)
#         return output
# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended_x = torch.sum(x * attention_weights, dim=1)
        return attended_x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)  # hidden_size * 2 due to bidirectional LSTM
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # hidden_size * 2 due to bidirectional LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended_out = self.attention(lstm_out)
        output = self.fc(attended_out)
        return output


from torch.optim.lr_scheduler import StepLR
# 定义五折交叉验证
def five_fold_cross_validation(X, y, log_filename):
    # 定义五折交叉验证的分割器
    saveacc = 0.0  # 保存最佳准确率
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # 用于保存每次训练的精度、损失等指标的列表
    train_losses, train_accuracies, test_accuracies = [], [], []

    # 使用tqdm显示五折交叉验证进度
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        device='cuda'
        X_train, X_test =X[train_idx].to(device), X[test_idx].to(device)
        y_train, y_test = y[train_idx].to(device), y[test_idx].to(device)

        # 在每个fold上创建新的LSTM模型
        input_size = X_train.shape[2]
        hidden_size = 1024
        num_layers = 4
        num_classes = 3
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
        if(fold>0):
            model = load_model_weights(model, f"./model/best_model{fold-1}.pth")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 训练模型
        num_epochs = 2000
        batch_size = 5120
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 用于保存每次epoch的训练损失和训练精度
        epoch_train_losses, epoch_train_accuracies = [], []

        for epoch in range(num_epochs):
            model.train()  # 设置模型为训练模式

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

            # 在训练集上计算损失和准确率，并保存到列表中
            train_loss = loss.item()
            y_train_pred = model(X_train).argmax(dim=1).cpu().numpy()
            train_accuracy = accuracy_score(y_train.cpu().numpy(), y_train_pred)
            epoch_train_losses.append(train_loss)
            epoch_train_accuracies.append(train_accuracy)

            # 在tqdm中显示当前精度和损失
            tqdm.write(
                f"Fold {fold + 1}/{kfold.n_splits}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # 在测试集上计算准确率，并保存到列表中
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():
                y_pred = model(X_test).argmax(dim=1).cpu().numpy()
                test_accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)

                tqdm.write(f"Fold {fold + 1}/{kfold.n_splits}, Test Accuracy: {test_accuracy:.4f}")
                if test_accuracy > saveacc:
                    saveacc = test_accuracy
                    torch.save(model.state_dict(), f"./model/best_model{fold}.pth")  # 保存模型
        # 保存当前fold的训练损失和训练精度到整体列表中
        train_losses.append(epoch_train_losses)
        train_accuracies.append(epoch_train_accuracies)
        model = load_model_weights(model, f"./model/best_model{fold}.pth")
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            y_pred = model(X_test).argmax(dim=1).cpu().numpy()
            test_accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)

        test_accuracies.append(test_accuracy)
        tqdm.write(f"Best {fold + 1}/{kfold.n_splits}, Test Accuracy: {test_accuracy:.4f}")
        saveacc=0
    # 将每次训练的精度、损失等指标保存到txt文件中
    with open(log_filename, 'w') as f:
        f.write("Fold\tEpoch\tTrain Loss\tTrain Accuracy\n")
        for fold in range(len(train_losses)):
            for epoch, train_loss, train_accuracy in zip(range(1, num_epochs + 1), train_losses[fold],
                                                         train_accuracies[fold]):
                f.write(f"{fold + 1}\t{epoch}\t{train_loss:.4f}\t{train_accuracy:.4f}\n")

    return np.mean(test_accuracies)


# 从pkl文件加载数据
def load_data_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
        # print(data)
    X, y = data['PPG']['dataX'], data['PPG']['dataY']
    return X, y

# 读取已保存的模型权重
def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model
if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision
    # 假设你的pkl文件名为'data.pkl'
    pkl_filename = 'DICT_EDG300.pkl'
    X, y = load_data_from_pkl(pkl_filename)
    # 随机生成数据，大小为 [5000, 1, 1920]
    # num_samples = 5000
    # sequence_length = 1
    # num_features = 1920
    # X = torch.tensor(np.random.rand(num_samples, sequence_length, num_features),dtype=torch.float32)
    #
    # # 随机生成y数据，假设标签取值为0、1、2
    # y = torch.tensor(np.random.randint(0, 3, size=num_samples))

    # 设置日志文件路径
    log_filename = './training_log.txt'

    # 执行五折交叉验证并打印结果
    mean_accuracy = five_fold_cross_validation(X, y, log_filename)

    print("五折交叉验证准确率：", mean_accuracy)

    # 绘制训练过程中的曲线图并保存为图片
    data = np.loadtxt(log_filename, skiprows=1)
    epochs, train_losses, train_accuracies = data[:, 1], data[:, 2], data[:, 3]

    plt.figure(figsize=(10, 5))
    for fold in range(1, 6):
        fold_data = data[data[:, 0] == fold]
        epochs, train_losses = fold_data[:, 1], fold_data[:, 2]
        plt.plot(epochs, train_losses, label=f'Fold {fold} Train Loss')


    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./training_curve12.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    for fold in range(1, 6):
        fold_data = data[data[:, 0] == fold]
        epochs, train_accuracies = fold_data[:, 1], fold_data[:, 3]
        plt.plot(epochs, train_accuracies, label=f'Fold {fold} Train Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./training_curve.png')
    plt.show()







