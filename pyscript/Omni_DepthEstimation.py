from colmaplib import create_workspace

create_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='exr', work_dir = '../data', 
                 is_depth=True)

#create_workspace(image_dir='../../../dataset/omnidirectional/1024_512/', file_suffix='png', work_dir = '../data')
