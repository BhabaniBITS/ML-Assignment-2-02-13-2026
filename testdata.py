import pandas as pd




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
