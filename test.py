import data

df_smb = data.fetch("FF/SMB")
df_hml = data.fetch("FF/HML")
sp500 = data.fetch("Yahoo/SP500")

print(df_smb)
print(sp500)