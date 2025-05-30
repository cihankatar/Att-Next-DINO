
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch

def testsample_all(images, labels, prediction, prediction1, prediction2, prediction3, prediction4,prediction5, n):
    rand_idx = n  # Use specified index

    def process_prediction(pred):
        processed = pred[rand_idx]
        processed = np.squeeze(processed)  # Remove single-dimensional entries
        processed = processed > 0.5
        return np.array(processed, dtype=np.uint8).T  # Convert to uint8 and transpose

    # Process each prediction
    prediction = process_prediction(prediction)
    prediction1 = process_prediction(prediction1)
    prediction2 = process_prediction(prediction2)
    prediction3 = process_prediction(prediction3)
    prediction4 = process_prediction(prediction4)
    prediction5 = process_prediction(prediction5)

    # Process image and label
    im_test = np.array(images[rand_idx] * 255, dtype=int)
    im_test = np.transpose(im_test, (2, 1, 0))  # Reorder dimensions
    im_test = np.squeeze(im_test)  # Remove single-dimensional entries

    label_test = np.array(labels[rand_idx] * 255, dtype=int)
    label_test = label_test.T  # Transpose
    label_test = np.squeeze(label_test)  # Remove single-dimensional entries

    # Plotting all the images, predictions, and masks in one row
    plt.figure(figsize=(20, 2))
    plt.subplot(1, 8, 1)
    plt.imshow(im_test)
    plt.axis('off')

    plt.subplot(1, 8, 2)
    plt.imshow(prediction1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 3)
    plt.imshow(prediction2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 4)
    plt.imshow(prediction3, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 5)
    plt.imshow(prediction4, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 6)
    plt.imshow(prediction5, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 7)
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 8, 8)
    plt.imshow(label_test, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def testsample(images, labels, prediction,n):

    rand_idx = n #np.random.randint(batch_size)
    prediction    = prediction[rand_idx]        ## (1, 512, 512)
    prediction    = np.squeeze(prediction)     ## (512, 512)
    prediction    = prediction > 0.5
    prediction    = np.array(prediction, dtype=np.uint8)
    prediction    = np.transpose(prediction)

    im_test       = np.array(images[rand_idx]*255,dtype=int)
    im_test       = np.transpose(im_test, (2,1,0))
    im_test       = np.squeeze(im_test)     ## (512, 512)

    label_test    = np.array(labels[rand_idx]*255,dtype=int)
    label_test    = np.transpose(label_test)
    label_test    = np.squeeze(label_test)     ## (512, 512)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im_test)
    plt.title("Image") 
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(prediction,cmap='gray')
    plt.title("Model Prediction") 
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(label_test,cmap='gray')
    plt.title("Mask") 
    plt.axis('off')

def testsample(images, labels, prediction,prediction1,n):

    rand_idx = n #np.random.randint(batch_size)
    prediction    = prediction[rand_idx]        ## (1, 512, 512)
    prediction    = np.squeeze(prediction)     ## (512, 512)
    prediction    = prediction > 0.5
    prediction    = np.array(prediction, dtype=np.uint8)
    prediction    = np.transpose(prediction)
    
    prediction1    = prediction1[rand_idx]        ## (1, 512, 512)
    prediction1    = np.squeeze(prediction1)     ## (512, 512)
    prediction1    = prediction1 > 0.5
    prediction1    = np.array(prediction1, dtype=np.uint8)
    prediction1    = np.transpose(prediction1)
    
    im_test       = np.array(images[rand_idx]*255,dtype=int)
    im_test       = np.transpose(im_test, (2,1,0))
    im_test       = np.squeeze(im_test)     ## (512, 512)

    label_test    = np.array(labels[rand_idx]*255,dtype=int)
    label_test    = np.transpose(label_test)
    label_test    = np.squeeze(label_test)     ## (512, 512)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im_test)
    plt.title("Image") 
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(prediction,cmap='gray')
    plt.title("Model Prediction") 
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(prediction1,cmap='gray')
    plt.title("Mask") 
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(label_test,cmap='gray')
    plt.title("Mask") 
    plt.axis('off')


def trainsample(images_copy, images, labels, n):
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    sigm = nn.Sigmoid() 

    image = images[n].permute(2, 1, 0)
    label = labels[n].permute(2, 1, 0)
    image_copy = images_copy[n].permute(2, 1, 0)

    # prediction = model_output[n].permute(2, 1, 0)
    # prediction = sigm(prediction).detach().numpy()

    plt.figure(figsize=(6, 4))

    # Subplot (a)
    plt.subplot(1, 3, 1)
    plt.imshow(image_copy)
    plt.axis('off')
    # Subplot (b)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.axis('off')

    # Subplot (c)
    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def peristent_diag(edges_mask_n,quantized_mask,pi_mask,edges_pred_n,quantized_pred,pi_pred,topo_loss):
   
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("Mask Image",fontsize = 8)
    plt.imshow(edges_mask_n.detach().numpy())
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title("Model Output",fontsize = 8)
    plt.imshow(quantized_pred.detach().numpy())
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('off')


    if torch.count_nonzero(pi_mask[1][1]) > 0:
        if pi_mask[1][1].max()>pi_pred[1][1].max():
            x = [0, pi_mask[1][1].max()]
        else:
            x = [0, pi_pred[1][1].max()]

    elif torch.count_nonzero(pi_pred[1][1]) > 0:
        if pi_mask[1][1].max()>pi_pred[1][1].max():
            x = [0, pi_mask[1][1].max()]
        else:
            x = [0, pi_pred[1][1].max()]


    mi     = pi_mask[1][1].cpu().detach().numpy()
    pi     = pi_pred[1][1].cpu().detach().numpy()

    mask_d0     = pi_mask[0][1].cpu().detach().numpy()
    predict_d0  = pi_pred[0][1].cpu().detach().numpy()

    plt.subplot(2,2,3)
    if torch.count_nonzero(pi_mask[1][1]) > 0:
        h1=plt.scatter(mi[:,0],mi[:,1] ,marker='o',color='blue',s=10)
    if torch.count_nonzero(pi_mask[0][1]) > 0:
        h0=plt.scatter(mask_d0[:,0],mask_d0[:,1] ,marker='o',color='red',s=10)

    plt.plot(x, x, label='Diagonal',color='black')
    plt.title('Persistence Diagram MI',fontsize = 8)
    plt.xlabel('Birth Scale',fontsize = 8)
    plt.ylabel('Death Scale',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')
    plt.legend((h0,h1),('H0', 'H1',),scatterpoints=1,loc='lower right',ncol=1,fontsize=8)


    plt.subplot(2,2,4)
    if torch.count_nonzero(pi_pred[1][1]) > 0:
        h1=plt.scatter(pi[:,0], pi[:,1],marker='o',color='blue',s=10)
    if torch.count_nonzero(pi_pred[0][1]) > 0:
        h0=plt.scatter(predict_d0[:,0],predict_d0[:,1] ,marker='o',color='red',s=10)
    plt.plot(x, x, label='Diagonal',color='black')
    plt.title('Persistence Diagram MP',fontsize = 8)
    plt.xlabel('Birth Scale',fontsize = 8)
    plt.ylabel('Death Scale',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')
    plt.legend((h0,h1),('H0', 'H1',),scatterpoints=1,loc='lower right',ncol=1,fontsize=8)
    plt.tight_layout()

def barcod(mask,maskh,points_p,predict,predicth,points_m,topo_loss):
   
    plt.figure()

    plt.subplot(3,2,1)
    plt.title("Mask Image",fontsize = 8)
    plt.imshow(np.rot90(mask.detach().numpy()))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)

    plt.subplot(3,2,2)
    plt.title("Edges of MP",fontsize = 8)
    plt.imshow(np.rot90(predict.detach().numpy()))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90) 

    plt.subplot(3,2,4)
    plt.scatter(points_m[:,0],points_m[:,1] ,s=1)
    plt.title('Selected Points MI',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    plt.subplot(3,2,3)
    plt.scatter(points_p[:,0],points_p[:,1] ,s=1)
    plt.title('Selected Points MP ',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    if maskh[1][1].max()>predicth[1][1].max():
        x = [0, maskh[1][1].max().detach().numpy()]
    else:
        x = [0, predicth[1][1].max().detach().numpy()]

    mi     = maskh[1][1].cpu().detach().numpy()
    pi  = predicth[1][1].cpu().detach().numpy()

    mask_d0     = maskh[0][1].cpu().detach().numpy()
    predict_d0  = predicth[0][1].cpu().detach().numpy()

    plt.subplot(3,2,5)
    h0=plt.scatter(mi[:,0],mi[:,1] ,marker='o',color='blue',s=10)
    h1=plt.scatter(mask_d0[:,0],mask_d0[:,1] ,marker='o',color='red',s=10)
    plt.plot(x, x, label='Diagonal',color='black')
    plt.title('Persistence Diagram MI',fontsize = 8)
    plt.xlabel('Birth Time',fontsize = 8)
    plt.ylabel('Death Time',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.legend((h0,h1),('H0', 'H1',),scatterpoints=1,loc='lower right',ncol=1,fontsize=8)
    plt.axis('scaled')

    plt.subplot(3,2,6)
    h0=plt.scatter(pi[:,0], pi[:,1],marker='o',color='blue',s=10)
    h1=plt.scatter(predict_d0[:,0],predict_d0[:,1] ,marker='o',color='red',s=10)
    plt.plot(x, x, label='Diagonal',color='black')
    plt.title('Persistence Diagram MP, loss :{}'.format(np.round(topo_loss.detach().numpy())),fontsize = 8)
    plt.xlabel('Birth Time',fontsize = 8)
    plt.ylabel('Death Time',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.legend((h0,h1),('H0', 'H1',),scatterpoints=1,loc='lower right',ncol=1,fontsize=8)
    plt.axis('scaled')
    plt.tight_layout()

def figures (model_output,sobel_predictions,bins_pred,point_p,labels,sobel_masks,bins_mask,point_m,i,topo_loss):
            
            plt.figure()
            plt.subplot(4,2,1)
            plt.title("Mask Image",fontsize = 8)
            plt.imshow(np.rot90(labels[i][0].detach().numpy()))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

            plt.subplot(4,2,3)
            plt.title("Edges of MI",fontsize = 8)
            plt.imshow(np.rot90(sobel_masks[i][0].detach().numpy()))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

            plt.subplot(4,2,5)
            plt.title("Edges Points ",fontsize = 8)
            plt.scatter(bins_mask[:,0],bins_mask[:,1],s=1)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

            plt.subplot(4,2,7)
            plt.scatter(point_m[:,0],point_m[:,1] ,s=1)
            plt.title('Selected Points MI',fontsize = 8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

            plt.subplot(4,2,2)
            plt.title("Model Prediction",fontsize = 8)
            plt.imshow(np.rot90(model_output[i][0].detach().numpy()))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

            plt.subplot(4,2,4)
            plt.title("Edges of MP",fontsize = 8)
            plt.imshow(np.rot90(sobel_predictions[i][0].detach().numpy()))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90) 
            plt.axis('scaled')

            plt.subplot(4,2,6)    
            plt.title("Edges Points",fontsize = 8)
            plt.scatter(bins_pred[:,0],bins_pred[:,1],s=1)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90) 
            plt.axis('scaled')

            plt.subplot(4,2,8)
            plt.scatter(point_p[:,0],point_p[:,1] ,s=1)
            plt.title('Selected Points MP ',fontsize = 8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8,rotation=90)
            plt.axis('scaled')

def simclr(images,labels):
    import matplotlib.pyplot as plt
#   which_batch=np.random.randint(2)
#   which_image=np.random.randint(7)

    which_batch0=0
    which_batch1=1
    image1=images[which_batch0][1].permute(2,1,0)
    image2=images[which_batch0][2].permute(2,1,0)
    image3=images[which_batch0][3].permute(2,1,0)
    image4=images[which_batch0][4].permute(2,1,0)

    label1=labels[which_batch0][1].permute(2,1,0)
    label2=labels[which_batch0][2].permute(2,1,0)
    label3=labels[which_batch0][3].permute(2,1,0)
    label4=labels[which_batch0][4].permute(2,1,0)

    image5=images[which_batch1][1].permute(2,1,0)
    image6=images[which_batch1][2].permute(2,1,0)
    image7=images[which_batch1][3].permute(2,1,0)
    image8=images[which_batch1][4].permute(2,1,0)

    label5=labels[which_batch1][1].permute(2,1,0)
    label6=labels[which_batch1][2].permute(2,1,0)
    label7=labels[which_batch1][3].permute(2,1,0)
    label8=labels[which_batch1][4].permute(2,1,0)

    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(image1)
    plt.subplot(2,4,2)
    plt.imshow(image2)
    plt.subplot(2,4,3)
    plt.imshow(image3)
    plt.subplot(2,4,4)
    plt.imshow(image4)
    
    plt.subplot(2,4,5)
    plt.imshow(label1)
    plt.subplot(2,4,6)
    plt.imshow(label2)
    plt.subplot(2,4,7)
    plt.imshow(label3)
    plt.subplot(2,4,8)
    plt.imshow(label4)

    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(image5)
    plt.subplot(2,4,2)
    plt.imshow(image6)
    plt.subplot(2,4,3)
    plt.imshow(image7)
    plt.subplot(2,4,4)
    plt.imshow(image8)
    
    plt.subplot(2,4,5)
    plt.imshow(label5)
    plt.subplot(2,4,6)
    plt.imshow(label6)
    plt.subplot(2,4,7)
    plt.imshow(label7)
    plt.subplot(2,4,8)
    plt.imshow(label8)
