from pycocotools.coco import COCO
import requests
total_imgIds=[]
total_images=[]

'''##Add the COCO json file from which to filter images'''
'''The pycocotools takes care of the filtering and loading of appropriate images'''

coco = COCO('merged_91_train/results/merged/annotations/merged.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))
#print(type(coco.getCatIds))
catIds = coco.getCatIds(catNms=['clock'])
for catids in catIds:
    imgIds = coco.getImgIds(catIds=catids)
    total_imgIds.append(imgIds)
for i in (total_imgIds):
    images = coco.loadImgs(i)
    total_images.append(images)
print(total_images)

'''###Storing the filtered images in the directory'''

for images in total_images:
    for im in images:
    #print("im: ", im)
        img_data = requests.get(im['coco_url']).content
        with open('/home/ekta/AI_current/vdot/vdot/Just_400/merged_6_train/images/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)


