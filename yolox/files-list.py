import os
import glob

directory_path = '/mnt/beegfs/home/bergcoll/DjangoResearch/CV/ITMC/Hourly_Intersection_Recordings/8006'

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory_path, '*'))

# Sort the list by modification time
files = sorted(files, key=os.path.getmtime)

# Print the sorted list
print(files)

