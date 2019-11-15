import os
import pickle
import pandas as pd
import google.oauth2.credentials

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

CLIENT_SECRET_FILE = "api_key/client_secret.json"

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_autenticated_service():
    credentials = None
    # if file exists
    if os.path.exists('api_key/token.pickle'):
        with open('api_key/token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    # Else if it is invalid or does not exist
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            credentials = flow.run_console()
    
        # save the credentials
        with open('api_key/token.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def get_videos(service, max_pages, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()
    
    i = 0
    while results and i<max_pages:
        final_results.extend(results['items'])

        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']  # update current page token, and add it to kwargs
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            break
    
    return final_results

def extract_comments_by_video_keyword(service, max_pages, **kwargs):
    results = get_videos(service, max_pages, **kwargs)
    final_results = [] 
    for item in results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        comments = get_video_comments(service, part='snippet', videoId=video_id, textFormat='plainText')
        final_results.extend([(video_id, title, comment) for comment in comments])
    return final_results

def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']  # update current page token, and add it to kwargs
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break
        
    return comments


if __name__=='__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_autenticated_service()
    keyword = input('Enter a keyword: ')
    max_pages = int(input('Enter max_pages: '))
    all_comments = extract_comments_by_video_keyword(service, max_pages, q=keyword, part='id,snippet', eventType='completed', type='video')
    data = pd.DataFrame(all_comments)
    data.to_csv(f'../data/{keyword}_comments.csv')
