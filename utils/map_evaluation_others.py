#Author : Ekta Bhojwani
#MSECE

'''Highway Asset detection predictions visualization script'''

from collections import namedtuple
import numpy as np
import cv2
import json
import os
import statistics
# define the `Detection` object

data_json=json.load(open('/home/ekta/AI_current/vdot/vdot/Just_400/others_mix.json'))
data_json_dt=json.load(open('./../results.json'))
image_dir= '/home/ekta/AI_current/vdot/vdot/Just_400/others_mix_set'

filenames=[]
for k, v in data_json.items():
    filenames.append(v['filename'])
'''bbox_dt=[]
scores=[]
img_ids=[]
for j in data_json_dt:
    img_ids.append(j['image_id'])
    bbox_dt.append(j['bbox'])
    scores.append(j['score'])
bbox_dt_scaled=[]
print(len(img_ids))
for i in  range(len(bbox_dt)):
    temp_dt=[]
    for bb in bbox_dt[i]:
        temp_dt.append(int(bb))
    bbox_dt_scaled.append(temp_dt)
files=[]
for j in img_ids:
    files.append(filenames[j])


cpt=0
for i in range(len(files)):
    img= cv2.imread(os.path.join(image_dir,files[i]))
    cv2.rectangle(img, (bbox_dt_scaled[i][0], bbox_dt_scaled[i][1]), (bbox_dt_scaled[i][2], bbox_dt_scaled[i][3]), (0,0,255), 3)
    cv2.putText(img, "score: {:.4f}".format(scores[i]), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    filename_d = "/home/ekta/AI_current/vdot/vdot/saved_images/multi-object/file_gt%d.jpg"%i
    cv2.imwrite(filename_d ,img)
    cpt+=1'''

iter=0
label_map=['drop_inlet', 'bus', 'car', 'person', 'train', 'truck']
bbox_dt=[]
bbox_dt_selected=[]
#cnt=img_ids[iter]
cnt=9
iter+=1
img_ids=[]
cats=[]
scores=[]
img_name = cv2.imread(os.path.join(image_dir,filenames[cnt]))
category=[]
for j in data_json_dt:
    if j['image_id']==cnt:
        bbox_dt_selected.append(j['bbox'])
        category.append(j['category_id'])
        scores.append(j['score'])
for cat in category:
    cats.append(label_map[cat-1])

for i in  range(len(bbox_dt_selected)):
    temp_dt=[]
    for bb in bbox_dt_selected[i]:
        temp_dt.append(int(bb))
    bbox_dt.append(temp_dt)
for i in range(len(bbox_dt)):
    #img_name=cv2.imread(os.path.join(image_dir,filenames[cnt]))
    cv2.rectangle(img_name, (bbox_dt[i][0], bbox_dt[i][1]), (bbox_dt[i][2], bbox_dt[i][3]), (0,0,255), 2)
    #x,y,x2,y2=cv2.boundingRect(bbox_dt[i])
    cv2.putText(img_name, "{:s}:{:.3f}".format(cats[i],scores[i]), (bbox_dt[i][0]+3, bbox_dt[i][1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

file='./../final_qual/gamma_2_others%d.jpg'%cnt
cv2.imwrite(file, img_name)


