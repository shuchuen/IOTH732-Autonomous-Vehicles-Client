# Import packages
import os
import sys
import time
import requests
import argparse
from datetime import datetime, timedelta
    

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', help="Where to save processed imges from pi.",
                    required=True)
parser.add_argument('--api_path', help="The api that the log send to.",
                    required=True)                                          

args = parser.parse_args()
api_path = args.api_path
input_path = args.input_path

# Get path to current working directory
CWD_PATH = os.getcwd()

def send_log_to_server(api_path, data, image):
    isExist = os.path.exists(image)
    if isExist:
        files = {'media': open(image, 'rb')}
        r = requests.post(api_path, json = data, files=files)
    else:
        r = requests.post(api_path, json = data)
    
    return (r is not None and r.status_code == 200)
        


try:
    print("Progam started")
    print(input_path)
    
    while True:
        list_subfolders_with_paths = [f.path for f in os.scandir(input_path) if f.is_dir()]
        for folder in list_subfolders_with_paths:
            files = os.listdir(folder)
            if len(files) == 0:
                ti_c = os.path.getctime(folder)
                dt_c = datetime.fromtimestamp(ti_c)
                
                diff = datetime.now() - dt_c
                if(diff.days > 1):
                    os.rmdir(folder)
                    
                continue
            
            for file in files:
                if file.endswith(".log"):
                    log_path = str(folder) + "/" + file
                    image_path = str(folder) + "/" + str(os.path.splitext(file)[0]) + ".jpg"
                    with open(log_path) as log:
                        data = log.readline()
                        result = send_log_to_server(api_path, data, image_path)
                        
                        if result:
                            os.remove(log_path)
                            os.remove(image_path)
                        else:
                          print("Cannot connect with the API")  
                                 
                    
except Exception as e:
    e = sys.exc_info()
    error_Msg = f"{type(e)}:{e[1]}"
    print(error_Msg)

finally:
    pass
