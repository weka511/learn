# snarfed from Ivan Ega Pratama's Notebook: COVID19 initial exploration tool
# https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import json

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
    
root_path = r'C:\Users\Simon\learn\CORD-19-research-challenge\2020-03-13'
metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
print(meta_df.head())

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
print(len(all_json))

with open(all_json[0]) as file:
    first_entry = json.load(file)
    print(json.dumps(first_entry, indent=4))
    
first_row = FileReader(all_json[0])
print(first_row)

dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx>1000: break
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
print (df_covid.head())
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))
df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
print (df_covid.head())
df_covid.drop_duplicates(['abstract'], inplace=True)
df_covid[['abstract_word_count', 'body_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))
plt.show()
