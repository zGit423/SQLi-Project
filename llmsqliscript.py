import pandas as pd    #nickname to "pd", less typing, cleaner code
from openai import OpenAI
import json


client = OpenAI()

ds = pd.read_csv(r"C:\Users\shock\CodingProjects\SQLi Project\SQLiV3.csv") #Named "ds" for DataSet


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
# print(ds["Label"].value_counts())  
# print(len(ds))       #Length is 30864

#-------------------------Promting LLM-------------------------------------------------------------------------

def ask_llm(sentence):
    prompt = f"""
You are a cybersecurity classifier.

Determine whether the following input is a SQL injection attack.

Reply with ONLY:
1 if it is SQL injection
0 if it is benign

Input:
{sentence}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0 # Setting the temperature to 0 makes the model choose the highest probability answer. 
        #                  Forced binary output; rejecting bad answers instead of trusting them
    )

    answer = response.choices[0].message.content.strip()

    if answer not in {"0", "1"}:
        return None  # model failed to follow instructions

    return int(answer)


#testing on a small sub-set

test_ds = ds.sample(10, random_state=42) #Randomly selects 10 rows from dataset. random_state ensure the same random rows


#Cost Guard for API usage
if len(test_ds) > 20:
    raise ValueError("Too many rows for testing.")

results = []

for _, row in test_ds.iterrows():
    sentence = row["Sentence"]
    true_label = row["Label"]

    pred = ask_llm(sentence)

    results.append({
        "Sentence": sentence,
        "True_Label": true_label,
        "LLM_Prediction": pred
    })


#Turn results into DataFrame
results_df = pd.DataFrame(results)
# print(results_df)

#Compute the Accuracy of the LLM
valid = results_df.dropna()

accuracy = (valid["True_Label"] == valid["LLM_Prediction"]).mean()
# print(f"Accuracy: {accuracy:.2f}")

#-----------------LLM Generated Dataset----------------------------------------------------------------------------

def generate_samples(n=40):
    prompt = f"""
Generate {n} user input strings.

Half must be SQL injection attacks.
Half must be benign inputs.

Return ONLY JSON list where each item has:
- sentence
- label (1 = SQLi, 0 = benign)
- attack_type (tautology, union, error-based, time-based, stacked, comment, blind, none)

Return ONLY valid JSON.
Do NOT include explanation.
Do NOT include markdown.

Format EXACTLY like this:

[
  {{"sentence":"example","label":1,"attack_type":"tautology"}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    text = response.choices[0].message.content.strip()
    data = json.loads(text)

    return pd.DataFrame(data)


gen_df = generate_samples(40)

print(gen_df.head())

#Number of payloads generated
attack_counts = gen_df["label"].value_counts()
print("Attack counts:")
print(attack_counts)


# Self-critique evaluation
def judge_quality(sentence):
    prompt = f"""
Is the following a realistic SQL injection payload used in real attacks?

Reply ONLY:
valid
nonsense

Input:
{sentence}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    if answer not in {"valid", "nonsense"}:
        return None

    return answer

gen_df["quality"] = gen_df["sentence"].apply(judge_quality)


#Compute rate; giving us the percentage of meaningful vs nonsense attacks
print("Quality distribution:")
print(gen_df["quality"].value_counts(normalize=True))

#Types of attacks generated
print("Attack types:")
print(gen_df["attack_type"].value_counts())