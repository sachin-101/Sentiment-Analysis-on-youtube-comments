import json
import os
import pandas as pd 


lang = "EN"
sentube_path = f'../data/SenTube/'
root_dir = os.path.join(sentube_path, f'automobiles_{lang}')

#####################
# We will extract only the comments
# related to product and save them
# in a dataframe along with their polarity
#####################

dataset = []
positive, negative, neutral = 0, 0, 0

for video in os.listdir(root_dir):
    video_path = os.path.join(root_dir, video)
    with open(video_path, 'r') as f:
        contents = json.loads(f.read())
    # Do I need to close it explicitly ?
    for comment in contents['comments']:
        text = comment['text']
        try:
            annotation = comment['annotation']
            if "product-related" in annotation.keys():
                if "positive-product" in annotation.keys():
                    polarity = 1
                    positive += 1
                elif "negative-product" in annotation.keys():
                    polarity = -1
                    negative += 1
                else:
                    polarity = 0
                    neutral += 1  
                dataset.append((text, polarity))
        except KeyError:
            pass

# save the dataset as .csv
df = pd.DataFrame(dataset)
df.to_csv('../data/Extracted_Comments/sentube.csv')

print('Total number of comments', len(dataset))
print('Total positive coments', positive)
print('Total negative comments', negative)
print('Total neutral comments', neutral)