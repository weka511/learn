import glob
import json
import os
import pandas as pd

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
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data
    
def read_json_files(root_path,start=None,end=None):
   
    metadata_path = f'{root_path}\\metadata.csv'
    meta_df = pd.read_csv(metadata_path, dtype={
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str, 
        'doi': str
    })
    meta_df.head()    
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
        content = FileReader(entry)
        
        # get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        # no metadata, skip this paper
        if len(meta_data) == 0:
            continue
        
        dict_['paper_id'].append(content.paper_id)
        dict_['abstract'].append(content.abstract)
        dict_['body_text'].append(content.body_text)
        
        # also create a column for the summary of abstract to be used in a plot
        if len(content.abstract) == 0: 
            # no abstract provided
            dict_['abstract_summary'].append("Not provided.")
        elif len(content.abstract.split(' ')) > 100:
            # abstract provided is too long for plot, take first 300 words append with ...
            info = content.abstract.split(' ')[:100]
            summary = get_breaks(' '.join(info), 40)
            dict_['abstract_summary'].append(summary + "...")
        else:
            # abstract is short enough
            summary = get_breaks(content.abstract, 40)
            dict_['abstract_summary'].append(summary)
            
        # get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        
        try:
            # if more than one author
            authors = meta_data['authors'].values[0].split(';')
            if len(authors) > 2:
                # more than 2 authors, may be problem when plotting, so take first 2 append with ...
                dict_['authors'].append(". ".join(authors[:2]) + "...")
            else:
                # authors will fit in plot
                dict_['authors'].append(". ".join(authors))
        except Exception as e:
            # if only one author - or Null valie
            dict_['authors'].append(meta_data['authors'].values[0])
        
        # add the title information, add breaks when needed
        try:
            title = get_breaks(meta_data['title'].values[0], 40)
            dict_['title'].append(title)
        # if title was not provided
        except Exception as e:
            dict_['title'].append(meta_data['title'].values[0])
        
        # add the journal information
        dict_['journal'].append(meta_data['journal'].values[0])
        
    df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
    df_covid.head()
    
    #index = 0
    #for path_name in all_json:
        #index+=1
        #if (start==None or index>= start):
            #with open(path_name) as f:
                #json_data = json.load(f)
                #print (path_name)
                #print(json.dumps(json_data, indent = 4, sort_keys=True))
        #if end!=None and index>= end:
            #break
 
            
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read JSON files')
    parser.add_argument('--root', metavar='N',  default=r'C:\Users\Simon\learn\CORD-19-research-challenge\2020-03-13',
                        help='Path to data)')
    parser.add_argument('--start',type=int,default=None)
    parser.add_argument('--end',type=int,default=None)
    args = parser.parse_args()
    read_json_files(root_path=args.root,start=args.start,end=args.end)
