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
#file='/home/ekta/AI_current/vdot/vdot/saved_images/files/file%d.jpg'%cnt
#file='/home/ekta/AI_current/vdot/vdot/saved_images/files/file_mult_%d.jpg'%cnt
file='./../final_qual/gamma_2_others%d.jpg'%cnt
cv2.imwrite(file, img_name)


'''label_=[

item {
  name: "/m/01g317"
  id: 1
  display_name: "person"
}
item {
  name: "/m/0199g"
  id: 2
  display_name: "bicycle"
}
item {
  name: "/m/0k4j"
  id: 3
  display_name: "car"
}
item {
  name: "/m/04_sv"
  id: 4
  display_name: "motorcycle"
}
item {
  name: "/m/05czz6l"
  id: 5
  display_name: "airplane"
}
item {
  name: "/m/01bjv"
  id: 6
  display_name: "bus"
}
item {
  name: "/m/07jdr"
  id: 7
  display_name: "train"
}
item {
  name: "/m/07r04"
  id: 8
  display_name: "truck"
}
item {
  name: "/m/019jd"
  id: 9
  display_name: "boat"
}
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
item {
  name: "/m/01pns0"
  id: 11
  display_name: "fire hydrant"
}
item {
  name: "/m/02pv19"
  id: 13
  display_name: "stop sign"
}
item {
  name: "/m/015qbp"
  id: 14
  display_name: "parking meter"
}
item {
  name: "/m/0cvnqh"
  id: 15
  display_name: "bench"
}
item {
  name: "/m/015p6"
  id: 16
  display_name: "bird"
}
item {
  name: "/m/01yrx"
  id: 17
  display_name: "cat"
}
item {
  name: "/m/0bt9lr"
  id: 18
  display_name: "dog"
}
item {
  name: "/m/03k3r"
  id: 19
  display_name: "horse"
}
item {
  name: "/m/07bgp"
  id: 20
  display_name: "sheep"
}
item {
  name: "/m/01xq0k1"
  id: 21
  display_name: "cow"
}
item {
  name: "/m/0bwd_0j"
  id: 22
  display_name: "elephant"
}
item {
  name: "/m/01dws"
  id: 23
  display_name: "bear"
}
item {
  name: "/m/0898b"
  id: 24
  display_name: "zebra"
}
item {
  name: "/m/03bk1"
  id: 25
  display_name: "giraffe"
}
item {
  name: "/m/01940j"
  id: 27
  display_name: "backpack"
}
item {
  name: "/m/0hnnb"
  id: 28
  display_name: "umbrella"
}
item {
  name: "/m/080hkjn"
  id: 31
  display_name: "handbag"
}
item {
  name: "/m/01rkbr"
  id: 32
  display_name: "tie"
}
item {
  name: "/m/01s55n"
  id: 33
  display_name: "suitcase"
}
item {
  name: "/m/02wmf"
  id: 34
  display_name: "frisbee"
}
item {
  name: "/m/071p9"
  id: 35
  display_name: "skis"
}
item {
  name: "/m/06__v"
  id: 36
  display_name: "snowboard"
}
item {
  name: "/m/018xm"
  id: 37
  display_name: "sports ball"
}
item {
  name: "/m/02zt3"
  id: 38
  display_name: "kite"
}
item {
  name: "/m/03g8mr"
  id: 39
  display_name: "baseball bat"
}
item {
  name: "/m/03grzl"
  id: 40
  display_name: "baseball glove"
}
item {
  name: "/m/06_fw"
  id: 41
  display_name: "skateboard"
}
item {
  name: "/m/019w40"
  id: 42
  display_name: "surfboard"
}
item {
  name: "/m/0dv9c"
  id: 43
  display_name: "tennis racket"
}
item {
  name: "/m/04dr76w"
  id: 44
  display_name: "bottle"
}
item {
  name: "/m/09tvcd"
  id: 46
  display_name: "wine glass"
}
item {
  name: "/m/08gqpm"
  id: 47
  display_name: "cup"
}
item {
  name: "/m/0dt3t"
  id: 48
  display_name: "fork"
}
item {
  name: "/m/04ctx"
  id: 49
  display_name: "knife"
}
item {
  name: "/m/0cmx8"
  id: 50
  display_name: "spoon"
}
item {
  name: "/m/04kkgm"
  id: 51
  display_name: "bowl"
}
item {
  name: "/m/09qck"
  id: 52
  display_name: "banana"
}
item {
  name: "/m/014j1m"
  id: 53
  display_name: "apple"
}
item {
  name: "/m/0l515"
  id: 54
  display_name: "sandwich"
}
item {
  name: "/m/0cyhj_"
  id: 55
  display_name: "orange"
}
item {
  name: "/m/0hkxq"
  id: 56
  display_name: "broccoli"
}
item {
  name: "/m/0fj52s"
  id: 57
  display_name: "carrot"
}
item {
  name: "/m/01b9xk"
  id: 58
  display_name: "hot dog"
}
item {
  name: "/m/0663v"
  id: 59
  display_name: "pizza"
}
item {
  name: "/m/0jy4k"
  id: 60
  display_name: "donut"
}
item {
  name: "/m/0fszt"
  id: 61
  display_name: "cake"
}
item {
  name: "/m/01mzpv"
  id: 62
  display_name: "chair"
}
item {
  name: "/m/02crq1"
  id: 63
  display_name: "couch"
}
item {
  name: "/m/03fp41"
  id: 64
  display_name: "potted plant"
}
item {
  name: "/m/03ssj5"
  id: 65
  display_name: "bed"
}
item {
  name: "/m/04bcr3"
  id: 67
  display_name: "dining table"
}
item {
  name: "/m/09g1w"
  id: 70
  display_name: "toilet"
}
item {
  name: "/m/07c52"
  id: 72
  display_name: "tv"
}
item {
  name: "/m/01c648"
  id: 73
  display_name: "laptop"
}
item {
  name: "/m/020lf"
  id: 74
  display_name: "mouse"
}
item {
  name: "/m/0qjjc"
  id: 75
  display_name: "remote"
}
item {
  name: "/m/01m2v"
  id: 76
  display_name: "keyboard"
}
item {
  name: "/m/050k8"
  id: 77
  display_name: "cell phone"
}
item {
  name: "/m/0fx9l"
  id: 78
  display_name: "microwave"
}
item {
  name: "/m/029bxz"
  id: 79
  display_name: "oven"
}
item {
  name: "/m/01k6s3"
  id: 80
  display_name: "toaster"
}
item {
  name: "/m/0130jx"
  id: 81
  display_name: "sink"
}
item {
  name: "/m/040b_t"
  id: 82
  display_name: "refrigerator"
}
item {
  name: "/m/0bt_c3"
  id: 84
  display_name: "book"
}
item {
  name: "/m/01x3z"
  id: 85
  display_name: "clock"
}
item {
  name: "/m/02s195"
  id: 86
  display_name: "vase"
}
item {
  name: "/m/01lsmm"
  id: 87
  display_name: "scissors"
}
item {
  name: "/m/0kmg4"
  id: 88
  display_name: "teddy bear"
}
item {
  name: "/m/03wvsk"
  id: 89
  display_name: "hair drier"
}
item {
  name: "/m/012xff"
  id: 90
  display_name: "toothbrush"
}]'''