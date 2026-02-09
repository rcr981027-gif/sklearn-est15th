import zipfile
import os

zip_path = r'c:\Users\alsld\Downloads\seoulBike.zip'
dest_path = 'data'

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dest_path)

print(f'Extracted to {os.path.abspath(dest_path)}')
print(os.listdir(dest_path))
