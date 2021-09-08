from classes import *

folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    
# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.tight_layout()
plt.savefig("../../images/classification/classes.png")
plt.show()

# Visualizing 25 random images from test data

#test = pd.read_csv(data_dir + '/Test.csv')
#imgs = test["Path"].values

meta_list = (data_dir + "/Meta")
imgs = os.listdir(meta_list)

plt.figure(figsize=(25,25))

for i in range(1,41):
    ax3=plt.subplot(6,6,i)
    random_img_path = data_dir + '/' + imgs[i]
    rand_img = imread(random_img_path)
    ax3.set_yticks([])
    ax3.set_xticks([])
    plt.savefig("../../images/classification/classes_image.png")
    plt.imshow(rand_img)
    plt.grid(b=None)
