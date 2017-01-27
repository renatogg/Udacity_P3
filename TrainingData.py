import cv2, numpy as np

def processImg(img):
    yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)  
   
    channels= cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    channels[0] = clahe.apply(channels[0])

    
   
    img=cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB)  
    s = np.std(img)
    img = img/s
    mean = np.mean(img)    
    img = img - mean
    return img


def getData(data,angle,input_size,batch_size):
    '''    Generator for Batch loading the dataset. Receives the images path, their relative angle output, and how many images should be loaded each time
    '''
    #Creates index to be sampled based on size of the datased
    index = np.arange(len(data))
    while 1:
        #Randomly select images up to batch size
        for i in range(batch_size):
            #Allocates memory space for the images to be loaded
            batch_train = np.zeros([batch_size]+ list(input_size), dtype = np.float32)
            batch_angle = np.zeros((batch_size,), dtype = np.float32)   
            try:
                #select one random index at a time, without replacement
                random = int(np.random.choice(index,1,replace = False))
                index = np.delete(index,index==random)
            except :
                #If there are no indexes left, reset index array
                index = np.arange(len(data))    
                #cut batch size to selected items
                batch_train = batch_train[:i,:,:]
                batch_angle = batch_angle[:i]
                break                        
            batch_train[i] = processImg(cv2.imread(data[random]))
            batch_angle[i] = angle[random]
        yield (batch_train, batch_angle)
