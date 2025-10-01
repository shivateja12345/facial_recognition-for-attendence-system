import os
import pandas as pd

def create_csv(data_dir, csv_name):
    files = os.listdir(data_dir)
    data = []
    for f in files:
        if f.endswith(('.jpg','.png','.jpeg')):
            # Extract person name from filename (everything before first number or underscore)
            name = ''.join([c for c in f if c.isalpha()])
            data.append([f, name])
    df = pd.DataFrame(data, columns=['filename','label'])
    df.to_csv(csv_name, index=False)

create_csv('train', 'train_labels.csv')
create_csv('valid', 'valid_labels.csv')
create_csv('test', 'test_labels.csv')

print("CSV files created successfully!")
