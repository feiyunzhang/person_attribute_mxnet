import os 
import random
import argparse
def loadupdress():
    idx = 0
    dict ={}
    #with open(os.path.join(os.getcwd(),'prepare_data','updress_list.csv')) as f:
    with open(os.path.join(root,'updress_list.csv')) as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
    
    return dict

def loaddowndress():
    idx = 0
    dict ={}
    with open(os.path.join(root,'downdress_list.csv')) as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
        
    
    return dict
def loadshoes():
    idx = 0
    dict = {}
    with open(os.path.join(root,'shoes_list.csv')) as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
    return dict

def loadumbrella():
    idx = 0
    dict ={}
    with open(os.path.join(root,'umbrella_list.csv')) as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
    return dict

def loadhat():
    idx = 0
    dict ={}
    with open(os.path.join(root,'hat_list.csv')) as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
    return dict
def get_rec_lines(args):      
    rec_lines =[]
#    fout= open("attribute_duke_M_train.lst",'w')
    with open(os.path.join(root,args.json)) as f:
        idx = 0
        updress_dic = loadupdress()
        downdress_dic = loaddowndress()   
        hat_dic = loadhat()
        umbrella_dic=loadumbrella()
        shoes_dic=loadshoes() 
        for line in f.readlines():
            import json
            dict = json.loads(line)
    #        if dict['image_path'][0:4] == 'Mark':
    #            continue
            img_path = dict['image_path']
            rec_line = '%d\t' % idx
            if 'gender' in dict:
                if 'male' == dict['gender']:
                    rec_line +='%f\t' % 0
                else:
                    rec_line +='%f\t' % 1
                    
            if 'hat' in dict:
                if 'no' == dict['hat']:
                    rec_line +='%f\t' % 0
                else:
                    rec_line +='%f\t' % 1 
            if 'umbrella' in dict:
                if 'no' == dict['umbrella']:
                    rec_line +='%f\t' % 0
                elif 'yes' == dict['umbrella']:
                    rec_line +='%f\t' % 1 
                else:
	 	    rec_line +='%f\t' % -1			
            
            if 'bag' in dict:
                if 'no' == dict['bag']:
                    rec_line +='%f\t' % 0
                else:
                    rec_line +='%f\t' % 1 
                    
            if 'handbag' in dict:
                if 'no' == dict['handbag']:
                    rec_line +='%f\t' % 0
                else:
                    rec_line +='%f\t' % 1  
                    
            if 'backpack' in dict:
                if 'no' == dict['backpack']:
                    rec_line +='%f\t' % 0
                else:
                    rec_line +='%f\t' % 1  
                    
            if 'updress' in dict:
            #    print(dict['updress'])
            #    print(updress_dic)
                if dict['updress'] in updress_dic:
                    rec_line +='%f\t' % updress_dic[dict['updress']]
                else:
                    rec_line +='%f\t' % updress_dic['upunknown']
            else:
                rec_line +='%f\t' % updress_dic['upunknown']
                
            if 'downdress' in dict:
                #print(dict['downdress'])
                #print(downdress_dic)
                if dict['downdress'] in downdress_dic:
                    rec_line +='%f\t' % downdress_dic[dict['downdress']]
                else:
                    rec_line +='%f\t' % downdress_dic['downunknown']
            else:
                rec_line +='%f\t' % downdress_dic['downunknown']
        

            if 'hatcolor' in dict:
             #   print(dict['hatcolor'])
             #   print(hat_dic)
                if dict['hatcolor'] in hat_dic:
                    rec_line +='%f\t' % hat_dic[dict['hatcolor']]
                else:
                    rec_line +='%f\t' % -1 #hat_dic['hatunknown']
            else:
                rec_line +='%f\t' % -1



            if 'umbrellacolor' in dict:
##                print("umbrella",dict['umbrella'])
#                #print(umbrella_dic)
                if dict['umbrellacolor'] in umbrella_dic:
                    rec_line +='%f\t' % umbrella_dic[dict['umbrellacolor']]
#                  #  if umbrella_dic[dict['umbrella']] == 0:
#                        #print(idx)
##                    print(umbrella_dic[dict['umbrella']])
                else:
                    rec_line +='%f\t' % -1 #umbrella_dic['umbrellaunknown']
            else:
                rec_line +='%f\t' % -1

            if 'shoes' in dict:
#                print(dict['shoes'])
#                print(shoes_dic)
                if dict['shoes'] in shoes_dic:
                    rec_line +='%f\t' % shoes_dic[dict['shoes']]
                   # print(shoes_dic[dict['shoes']])
                else:
                    rec_line +='%f\t' % -1 #shoes_dic['shoesunknown']
            else:
                rec_line +='%f\t' % -1



            idx+=1
            rec_line += '%s\n' % img_path
            rec_lines.append(rec_line)
        return rec_lines, idx

'''           
    for i in range(80000):
        rec_line = '%d\t' % idx2
        rec_line +='%f\t' % anno['train_label'][i][0]#female
        rec_line +='%f\t' % anno['train_label'][i][7]#hat
        rec_line +='%f\t' % anno['train_label'][i][10]#PA100K shoulderbag~marketduke bag
        rec_line +='%f\t' % anno['train_label'][i][9]#handbag
        rec_line +='%f\t' % anno['train_label'][i][11]#backpack
        rec_line +='%f\t' % -1 #updress
        rec_line +='%f\t' % -1 #downdress
        img_path = 'PA100K/release_data/release_data/'+ str(anno['train_images_name'][i][0])[3:-2]
        #img_path = 'PA100K/release_data/release_data/'+ str(anno['train_images_name'][i][0])[2:-2]
        rec_line += '%s\n' % img_path
        idx2+=1
        rec_lines.append(rec_line)
    return rec_lines, idx2 
'''

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    
    parser.add_argument('--dataset',type = str, help='train or test.',default='train')
    parser.add_argument('--json',type = str, help='train or test.',default='train.json')
    
    args = parser.parse_args()
    return args
            
        
if __name__ == '__main__':
    args = parse_args()
    print(args) 
    root = '/workspace/mnt/group/video-det/zhangfeiyun/projects/person/vss/peron_attribute/prepare_data/datav3' 
    rec_lines,idx = get_rec_lines(args)
    import random
    random.shuffle(rec_lines)
    out_file = 'attribute_0308_datav3_2cls9_'+args.dataset+'.lst'
    fout =  open(out_file,'w')   
            
    for rec_line in rec_lines:        
        fout.write(rec_line)
        #print(rec_line)
    fout.close()
    print(idx)

