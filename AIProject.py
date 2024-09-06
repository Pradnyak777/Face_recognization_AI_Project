import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os, cv2


def plot_gallary(images, titles, h,w, n_row=3 , n_col=4):
    """Helper function as to plot a gallery of portraits """
    plt.figure(figsize=(1.8*n_col, 2.4 * n_row))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspaces=.35)

    for i in range(n_row * n_col):
        plt.imshow(images[i].reshape((h ,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



        dir_name="dataset/faces/"
y=[]; x=[];target_names=[]
person_id=0;h=w=300
n_samples=0
class_names=[]
for person_name in os.listdir("dataset/faces/"):
    # print(person_name)
    dir_path = dir_name +person_name+ " Amir "
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        #formulate the image path
        image_path = dir_path+ image_name
        #read the input image
img = cv2.imgread(image_path)
# Convert into grayscale
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# resize image to 300*300 dimention
resized_image= cv2.resize(gray,(h,w))
#convert matrix to vector
v=resized_image.flatten()
x.append(v)
# increase the number of smaples
n_samples= n_samples+1
#Addinng th categorical label
y.append(person_id)

# adding the person name
target_names.append(person_name)
#Increase the person id by 1 
person_id= person_id+1


#############################################################transform list to numpy array

y=np.array(y)
x=np.array(x)
target_names = np.array(target_names)
n_features = x.shape[1]
print(y.shape, x.shape, target_names.shape)
print("Number of samples:", n_samples)

n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_feature: %d" % n_features)
print("n_classes: %d" % n_classes)



#***********************************************************
# splite into a trainning set and test set using a stratified k  fold
# splite into a training and testing
x_train, x_test, y_train, y_test= train_test_split(x, y , test_size=0.25, random_state=42)


# ####################################################

#compute a PCA (eigenfaces)on the face dataset (treated as unlabeles 
# dataset):unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, x_train.shape[0]))

#Applyng PCA
pca = PCA(n_components = n_components, svd_solver='randomized' , white= True).fit(x_train)

#Generating eigenfaces
eigenfaces = pca.components_.reshape((n_components , h ,w))
# Plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallary(eigenfaces , eigenface_titles, h,w)
plt.show()

print("Projecting in the input data on the eigenfaces orthonormal basis")
x_train_pca =pca.transform(x_train)

x_test_pca= pca.transform(x_test)
print(x_train_pca.shape, x_test_pca.shape)
#%% compute fisherfaces

lda = LinearDiscriminantAnalysis()

# compute the LDA of reduced data
lda.fit(x_train_pca , y_train)

x_train_lda = lda.transform(x_train_pca)
x_test_lda=lda.transform(x_test_pca)
print("Project done..!!!!!!!!!!!!!!")


# Trainning with multilayer perception
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10,10),max_iter=1000,verbose=True).fit(x_train_lda, y_train)
print("Model Weight: ")
model_info =[coef.shape for coef in clf.coefs_]
print(model_info)

y_pred = []; y_prob = []
for test_face in x_test_lda:
    prob = clf.predict_proba([test_face])[0]
    # print(prob,np,max(prob))
class_id = np.where(prob == np.max (prob))[0][0]
#print(class_index)
# find the label of the mathed face
y_pred.append(class_id)
y_prob.append(np.max(prob))

# transform the data
y_pred = np.array(y_pred)

prediction_titles=[]
true_position = 0
for i in range(y_pred.shape[0]):


   true_name = class_names[y_test[i]]
   pred_name = class_names[y_pred[i]]
   result = 'pred : %s, pr: %s \n true: %s' % (pred_name, str(y_prob[i])[0:3], true_name) 
# result= 'prediction %s \ntrue:    %s' % (pred_name, true_name)
prediction_titles.append(result)

if true_name==pred_name:
    true_positive=true_position
    print("Accurancy: ", true_positive*100/y_pred.shape[0])

# plot result 
plot_gallary(x_test, prediction_titles, h ,w)
plt.show()





