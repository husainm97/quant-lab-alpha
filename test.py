# test.py

'''
This is a temporary file to test the fetcher, references to local files may now be replaced using data.fetch(), any information on key-url-datatype may be noted in registry.py 
'''

import data

df_smb = data.fetch("FF/SMB")
df_hml = data.fetch("FF/HML")

print(df_smb)
print(df_hml)

print('now yahoo')
sp500 = data.fetch("Yahoo/SP500")
print(sp500)