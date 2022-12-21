import ast

with open('imagenet1000_clsidx_to_labels.txt', "r") as f:
	indx_to_word_dict = ast.literal_eval(f.read())

with open('imageNet_val_files.txt', 'r') as f:
	folder_list = f.read().splitlines()

# req_words = [plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
with open('imageNet_val_folderName_to_words.txt','w') as f:
	for (_fn, _l) in zip(folder_list, indx_to_word_dict.values()):
		f.write(f'{_fn} {_l} \n')

