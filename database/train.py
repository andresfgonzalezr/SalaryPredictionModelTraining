import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from database.database_ import df_final1
from database_ import df_final1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.init as init


def treat_data():
    # taking out the variable that we want to predict
    data_y = df_final1[["annual_salary"]]

    # removing from the dataset variables that doesn´t help us to predict
    data_x = df_final1.drop(columns=['timestamp', 'job_context', 'annual_salary', 'additional_compensation', 'currency_other', 'income_context','id', "job_title"])

    # Applying one hot encoding to the dataframe in order to the neural network works
    data_x = pd.get_dummies(data_x)

    scaler = StandardScaler()

    data_y_normalized = scaler.fit_transform(data_y)
    data_y_normalized = pd.DataFrame(data_y_normalized, columns=data_y.columns)

    # split the database into a train set and a test set, using sklearn for training using 80% and for test using 20%
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_normalized, test_size=0.2, random_state=21)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    n_entries = X_train.shape[1]

    tensor_X_train = torch.tensor(X_train.values, dtype=torch.float32).to('cpu')
    tensor_X_test = torch.tensor(X_test.values, dtype=torch.float32).to('cpu')
    tensor_y_train = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')
    tensor_y_test = torch.tensor(y_test.values, dtype=torch.float32).to('cpu')
    tensor_y_train = tensor_y_train[:,None]
    tensor_y_test = tensor_y_test[:,None]

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train) #new code
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test) #new code

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, data_x, n_entries, tensor_X_test, tensor_y_test


def train_model():
    train_loader, test_loader, scaler, data_x, n_entries, tensor_X_test, tensor_y_test = treat_data() # main
    class NeuralSalary(nn.Module):
        def __init__(self, n_entries):
            super(NeuralSalary, self).__init__()
            self.Linear1 = nn.Linear(n_entries, 128)
            self.Linear2 = nn.Linear(128, 128)
            self.Linear3 = nn.Linear(128, 128)
            self.Linear4 = nn.Linear(128, 1)
            self.init_weights() #new code

        def init_weights(self):
            init.xavier_uniform_(self.Linear1.weight)
            init.xavier_uniform_(self.Linear2.weight)
            init.xavier_uniform_(self.Linear3.weight)
            init.xavier_uniform_(self.Linear4.weight)

        def forward(self, inputs):
            prediction1 = F.relu(input=self.Linear1(inputs))
            prediction2 = F.relu(input=self.Linear2(prediction1))
            prediction3 = F.relu(input=self.Linear3(prediction2))
            prediction_f = self.Linear4(prediction3)

            return prediction_f

    lr = 0.0001
    n_epochs = 50

    model = NeuralSalary(n_entries)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        print(f'epoch[{epoch + 1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_X_test)

        mean_salary = scaler.mean_
        std_salary = scaler.scale_

        outputs_desnormalized = outputs * std_salary + mean_salary
        outputs_desnormalized = outputs_desnormalized.cpu().numpy()

        tensor_outputs_desnormalized = torch.tensor(outputs_desnormalized, dtype=torch.float32)

        test_loss = criterion(tensor_outputs_desnormalized.squeeze(), tensor_y_test.squeeze())

    # print(f'Test Loss: {test_loss.item():.4f}')

    mse_criterion = nn.MSELoss()
    mse = mse_criterion(tensor_outputs_desnormalized.squeeze(), tensor_y_test.squeeze())
    print(f'MSE: {mse.item():.4f}')

    mae = torch.mean(torch.abs(tensor_outputs_desnormalized.squeeze() - tensor_y_test.squeeze()))
    print(f'MAE: {mae.item():.4f}')

    torch.save(model.state_dict(), '../../Neural_Salary_Model.pth')

    model = NeuralSalary(n_entries)
    # model.load_state_dict(torch.load('../../Neural_Salary_Model.pth'))
    # model.eval()

    return model


def predict_salary(new_data):
    train_loader, test_loader, scaler, data_x, n_entries, tensor_X_test, tensor_y_test = treat_data()
    model = train_model()
    new_data = pd.DataFrame([new_data])
    new_data = pd.get_dummies(new_data)
    missing_cols = list(set(data_x.columns) - set(new_data.columns))
    new_cols = pd.DataFrame(0, index=new_data.index, columns=missing_cols)

    new_data = pd.concat([new_data, new_cols], axis=1)

    new_data = new_data[data_x.columns]
    new_data = new_data.astype(float)

    new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)
    with torch.no_grad():
        predicted_outputs = model(new_data_tensor)

    predicted_outputs = predicted_outputs.cpu().numpy()

    mean_salary = scaler.mean_[0]
    std_salary = scaler.scale_[0]

    predicted_outputs_desnormalized = predicted_outputs * std_salary + mean_salary

    print(f'Predicted outputs: {predicted_outputs}')
    print(f'Predicted outputs desnormalized: {predicted_outputs_desnormalized}')

    return predicted_outputs_desnormalized

new_data = {
    'age': '25-34',
    'industry': 'computing or tech',
    'currency': 'USD',
    'country': 'united states',
    'us_state': 'None',
    'city': 'boston',
    'years_experience_overall': '8-10 years',
    'years_experience_field': '5-7 years',
    'education_level': 'College degree',
    'gender': 'Woman',
    'race': 'White'
}

if __name__ == "__main__":
    treat_data()
    train_model()
    predict_salary(new_data)

