# 전처리 
# 생체 데이터 이상값 벗어날 경우 결측값 처리
data['age'] = data['age'].apply(lambda x: np.NaN if x > 102 else (np.NaN if x < 20 else x))
data['Ht'] = data['Ht'].apply(lambda x: np.NaN if x < 0 else x)
data['Wt'] = data['Wt'].apply(lambda x: np.NaN if x < 0 else x)
data['BMI'] = data['BMI'].apply(lambda x: np.NaN if x > 50 else (np.NaN if x < 10 else x))
data['SBP'] = data['SBP'].apply(lambda x: np.NaN if x > 250 else (np.NaN if x < 0 else x))
data['DBP'] = data['DBP'].apply(lambda x: np.NaN if x > 175 else (np.NaN if x < 4 else x))
data['PR'] = data['PR'].apply(lambda x: np.NaN if x > 200 else (np.NaN if x < 20 else x))
data['Cr'] = data['Cr'].apply(lambda x: np.NaN if x < 0 else x)
data['AST'] = data['AST'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
data['ALT'] = data['ALT'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
data['GGT'] = data['GGT'].apply(lambda x: np.NaN if x < 0 else x)
data['ALP'] = data['ALP'].apply(lambda x: np.NaN if x < 0 else x)
data['BUN'] = data['BUN'].apply(lambda x: np.NaN if x < 0 else x)
data['Alb'] = data['Alb'].apply(lambda x: np.NaN if x < 0 else x)
data['TG'] = data['TG'].apply(lambda x: np.NaN if x < 0 else x)
data['CrCl'] = data['CrCl'].apply(lambda x: np.NaN if x < 0 else x)
data['FBG'] = data['FBG'].apply(lambda x: np.NaN if x < 0 else x)
data['HbA1c'] = data['HbA1c'].apply(lambda x: np.NaN if x > 15 else (np.NaN if x < 0 else x))
data['LDL'] = data['LDL'].apply(lambda x: np.NaN if x < 0 else x)
data['HDL'] = data['HDL'].apply(lambda x: np.NaN if x < 0 else x)
data['FBG_level'] = data['FBG'].apply(lambda x: 2 if x >= 110 else (1 if x >= 100 else 0))
data['HbA1c_level'] = data['HbA1c'].apply(lambda x: 2 if x >= 6.1 else (1 if x >= 5.7 else 0))
data['Hb_FBG_div'] = data['FBG'] / np.square(data['HbA1c'])
# 날짜 datetime으로 변환
data.loc[:, 'date'] = pd.to_datetime(data['date'], format='%Y.%m.%d')
data.loc[:, 'date_E'] = pd.to_datetime(data['date_E'], format='%Y.%m.%d')
data['delta_date'] = data['date_E'] - data['date']
data['delta_date'] = data['delta_date'].astype(str)
data['delta_date'] = data['delta_date'].apply(lambda x: int(re.sub('[a-z]+', '', x)))
data['delta_date'] = data['delta_date'].apply(lambda x: 4 if x > 4000 else (3 if x > 3000 else (2 if x > 2000 else (1 if x > 1000 else 0))))
        
#모델

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output
      
  model = Feedforward(config.input_size, 32)#.to(device)
  loss_fn = nn.BCEWithLogitsLoss() 
  optimizer = optim.Adam(model.parameters(), lr=config.lr)
  
  
