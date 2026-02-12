import pandas as pd    #nickname to "pd", less typing, cleaner code
from openai import OpenAI

client = OpenAI()

ds = pd.read_csv(r"C:\Users\shock\CodingProjects\PythonProjects\SQLiV3.csv") #Named "ds" for DataSet


#print(ds.head())   #Looks at the first 5 rows of the dataset
#print(ds.columns)  #Shows all column names in the dataset


# print(ds[["Label", "Unnamed: 2", "Unnamed: 3"]].head(10)) # After running this code and looking into the 
# csv file it shows that there are areas in the unnamed columns where labels are kept. 


# We need the labels to apear in the "Labels" column only.
ds["clean_label"] = (
    ds["Label"]
    .fillna(ds["Unnamed: 2"])
    .fillna(ds["Unnamed: 3"])  #Starts with "Label" if slot is empty go to Unnamed: 2,
                               # if still empty, go to Unnmaed: 3. This will get us one clean "Label" column
)

#Verify our fix
#print(ds[["Sentence", "clean_label"]].head(10))
#print(ds["clean_label"].value_counts())  #After printing I found out that there are 
#                                         values in the Labels column that aren't only 1 & 0


#
ds = ds[["Sentence", "clean_label"]]
ds = ds.rename(columns={"clean_label": "Label"}) #Rename clean label to Label


#Fix unnecessary text in Labels column
ds["Label"] = pd.to_numeric(ds["Label"], errors="coerce")  #Keep only valid labels (0 or 1)
# If a value can become a number it becomes a number, if not becomes NaN (a missing value)


ds = ds.dropna(subset=["Label"])      #Deletes rows where Label is NaN.
ds["Label"] = ds["Label"].astype(int)  #Converts labels from floats to integers because of .to_numeric()

#Verify again
print(ds["Label"].value_counts())  
print(len(ds))       #Length is 30864





