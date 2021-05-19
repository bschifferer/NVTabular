import os

from nvtabular.utils import download_file
    
def gen_data(source='original', url=None):
    if source=='original':
        download_file(os.environ.get("DATA_URL", ""), 'data.zip')

if __name__ == "__main__":
    gen_data()
        
