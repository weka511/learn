import json
import os

def read_json_files(file_path,start=None,end=None):
    full_path_names = [os.path.join(dirname, filename)
                       for dirname, _, filenames in os.walk(file_path)
                       for filename in filenames
                       if filename.endswith('json')]
    index = 0
    for path_name in full_path_names:
        index+=1
        if (start==None or index>= start):
            with open(path_name) as f:
                json_data = json.load(f)
                print (path_name)
                print(json.dumps(json_data, indent = 4, sort_keys=True))
        if end!=None and index>= end:
            break
 
            
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read JSON files')
    parser.add_argument('--files', metavar='N',  default=r'C:\Users\Simon\learn\CORD-19-research-challenge\2020-03-13',
                        help='Path to data)')
    parser.add_argument('--start',type=int,default=None)
    parser.add_argument('--end',type=int,default=None)
    args = parser.parse_args()
    read_json_files(file_path=args.files,start=args.start,end=args.end)
