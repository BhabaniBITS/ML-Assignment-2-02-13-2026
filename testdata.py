import pandas as pd

# Load the full dataset (already in your project root)
data = pd.read_csv("bank-additional-full.csv", sep=';')
# Cut out the first 3000 rows
test_data = data.iloc[:3000]   # these will go into the test file
train_data = data.iloc[3000:]  # remaining rows stay for training
# Save the 3000 records as a test file
test_data.to_csv("test_bank.csv", sep=';', index=False)

# (Optional) Save the reduced training file if you want
train_data.to_csv("bank-additional-full.csv", sep=';', index=False)


import pandas as pd

# Load the test file (3000 records)
test_data = pd.read_csv("test_bank.csv", sep=';')
print("Rows in test_bank.csv:", len(test_data))

# Load the reduced training file (if you saved it separately)
train_data = pd.read_csv("bank-additional-train.csv", sep=';')
print("Rows in bank-additional-train.csv:", len(train_data))

# Or, if you kept the original full file unchanged:
full_data = pd.read_csv("bank-additional-full.csv", sep=';')
print("Rows in bank-additional-full.csv:", len(full_data))
