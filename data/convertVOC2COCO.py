import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = [ "bicycle", "car", "motorcycle", "tree", "street_lamp", "well_cover"]
 
 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_id):
    print("2-open annotations")
    in_file = open('/home/jessy104/around_the_garden_front1/annotationXML/%s.xml'%(image_id))
    print("3-convert to txt")
    out_file = open('/home/jessy104/around_the_garden_front1/annotationTXT/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        print("write ok")
def generate_image_list(image_id):
    print("Generating Image List")
    # out_file = open('/home/jessy104/around_the_garden_front1/annotationTXT/%s.txt'%(image_id), 'w')
    out_file = open('/home/jessy104/around_the_garden_front1/image_list.txt', 'a')
    out_file.write('../coco/images/Infra1Front/' + image_id + '.png' + '\n')
    print("write ok")
def generate_shape(image_id):
    print("Generating Image List")
    # out_file = open('/home/jessy104/around_the_garden_front1/annotationTXT/%s.txt'%(image_id), 'w')
    out_file = open('/home/jessy104/around_the_garden_front1/image_list.shapes', 'a')
    out_file.write('640 480' + '\n')
    print("write ok")
#wd = getcwd()
directory = "/home/jessy104/around_the_garden_front1/annotationXML/"
 
# for year, image_set in sets:
#     if not os.path.exists('COCO_%s/labels/'%(year)):
#         os.makedirs('COCO_%s/labels/'%(year))
#     image_ids = open('%s.txt'%(image_set)).read().strip().split()
#     print("start ")
#     list_file = open('%s_%s.txt'%(year,image_set), 'w')

for image_id in os.listdir(directory):
    name=os.path.splitext(image_id)[0]
    convert_annotation(name)
    generate_image_list(name)
    generate_shape(name)

