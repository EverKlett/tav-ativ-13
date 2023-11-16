import pandas as pd
import numpy as np
import datetime as dt

print('Leitura dos dados originais...')
data = pd.read_excel('../origem/SS.xlsx')
print('--> Preparação dos Dados...')
print('Eliminando colunas...')
data = data.drop(columns=['Row ID', 'Postal Code'])
print('Ajuste de datas...')
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Period'] = ((data['Year'] - data['Year'].min()) * 12) + data['Month']
data['Order Date Month'] = data['Order Date'].apply(lambda x : x.strftime("%Y-%m-01"))
data['Order Date Month']= pd.to_datetime(data['Order Date Month'])
print('Novas medidas... Delivery e Price')
data['Delivery'] = (data['Ship Date'] - data['Order Date']).dt.days
data['Price'] = round((data['Sales'] / data['Quantity']),2)
print('Variável Dependente: Benefit...')
data['Benefit'] = data['Profit'].apply(lambda x : 1 if x > 0 else 0)
print('--> Gravando dados preparados...')
data.to_feather('../dados/ss.feather')