'''
import json
import cv2
import numpy as np
import os
#data_dir='./hom'
#data_catal={0:,1:,}
json_file_path='/home/ekta/Downloads/export-2020-11-26T08_00_02.265Z.json'
#image_root_path='/home/ekta/Documents/vdot/inlets_folder/New Images_Patrick'
val_list=['202010221219370356.jpg','202010221221200914.jpg','202010221221350739.jpg' ,'202010221255500592.jpg','202010221256290726.jpg','202010221257410790.jpg','202010271410380820.jpg','202010271419260978.jpg','202010271419530136.jpg','202010271421500108.jpg',
'202010271422060801.jpg', '202010271436340604.jpg', '202010271436510210.jpg', '202010271459040930.jpg', '202010271459230349.jpg','202010291052590698.jpg','202010291053190890.jpg' , '202010291059020914.jpg','202010291105010345.jpg','202011110657510322.jpg','202011110658010100.jpg','202011110658500474.jpg','202011110659480957.jpg','202011110700040338.jpg','202011110701020011.jpg','202011110701150944.jpg', '202011110702210830.jpg','202011110703380836.jpg','202011110703540298.jpg','202011110704580228.jpg','202011110705090039.jpg','202011110706340827.jpg']


with open(json_file_path, 'r') as f:
    label_dict = json.load(f)

annotations={}
img_num=20
#for collection in os.listdir(data_dir):
for ann in label_dict:
	annotation={}
	filename=ann['External ID']
	if filename in val_list:

	objects=ann['Label']['objects']
	annotation['filename']=filename
	assets={}
	for obj in objects:
		#cv2.rectangle(img,(obj['bbox']['left'],obj['bbox']['top']),(obj['bbox']['left']+obj['bbox']['width'],obj['bbox']['top']+obj['bbox']['height']),(255,0,0),2)
		if obj['value'] == 'drop_inlet':
			#cv2.putText(img,'drop_inlet',(obj['bbox']['left'],obj['bbox']['top']),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
			#annotations.update( {'drop_inlet' : [obj['bbox']['left'],obj['bbox']['top'],obj['bbox']['left']+obj['bbox']['width'],obj['bbox']['top']+obj['bbox']['height']]})
			assets['drop_inlet']=[obj['bbox']['left'],obj['bbox']['top'],obj['bbox']['left']+obj['bbox']['width'],obj['bbox']['top']+obj['bbox']['height']]
		#if obj['value'] == 'drop_inlet':
			#annotations.update( {'storm_drain' : [obj['bbox']['left'],obj['bbox']['top'],obj['bbox']['left']+obj['bbox']['width'],obj['bbox']['top']+obj['bbox']['height']]})
			#assets['storm_drain']=[obj['bbox']['left'],obj['bbox']['top'],obj['bbox']['left']+obj['bbox']['width'],obj['bbox']['top']+obj['bbox']['height']]
		    #cv2.putText(img,'storm_drain',(obj['bbox']['left'],obj['bbox']['top']),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
			#cv2.imwrite(output_path+'img_'+filename,img)
	annotation['assets']=assets
	annotations[img_num]=annotation
	img_num+=1
    #print(annotations)
    #break
#print(annotations)
with open('unseen.json', 'w') as fp:
    json.dump(annotations, fp)'''






import json
import os
import numpy as np

image_path='/home/ekta/AI_current/vdot/vdot/Just_400/others_set'
val_list=os.listdir(image_path)
annotations={}
#annotation={}
img_num=0
for i in (val_list):
	annotation={}
	annotation['filename']=i
	annotations[img_num]=annotation
	img_num+=1
with open('others.json','w') as fp:
	json.dump(annotations, fp)
import json
data_json=json.load(open('./others.json'))
for k,v in data_json.items():
	print(v['filename'])
