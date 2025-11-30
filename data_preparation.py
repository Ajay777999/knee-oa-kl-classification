import os
import pandas as pd
from sklearn.model_selection import train_test_split

metadata = pd.DataFrame(columns=["Name", "Path", "KL"])
main_dir = r"data/OSAIL_KL_Dataset/Labeled"

"""for garde in range(5):
    folder_path = os.path.join(main_dir,str(garde))

    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)

        metadata = metadata._append({
            "Name": image,
            "Path": image_path,
            "KL": garde
        }, ignore_index=True )"""

#metadata.to_csv("metadata.csv", index=False)
metadata = pd.read_csv("metadata.csv")

'data/CSVs/fold_{}_train.csv'
'data/CSVs/fold_{}_test.csv'

train_val_data, test_data = train_test_split(metadata, test_size=0.2, stratify=metadata["KL"])
