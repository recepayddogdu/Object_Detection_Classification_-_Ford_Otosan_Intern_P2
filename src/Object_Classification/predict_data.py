from classes import *

imgs = os.listdir(test_path_ford)
labels_path='../../data/classification_data/classes.json'
json_file = open(labels_path, 'r')#file reading process
json_dict=json.load(json_file)#Contents of json file converted to dict data type

data =[]

for img in tqdm.tqdm(imgs):
    image = cv2.imread(test_path_ford + '/' +img)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_fromarray = Image.fromarray(imageRGB, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    data.append(np.array(resize_image))

X_test = np.array(data)
X_test = X_test/255
model=models.load_model('../../models/sign_classification.h5')
y = model.predict(X_test)
pred=np.argmax(y,axis=1)


#Predictions on Test Data
plt.figure(figsize = (25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    prediction = str(pred[start_index + i])
    col = 'g'
    plt.xlabel('Pred={}'.format( json_dict[prediction]), color = col)
    plt.tight_layout(h_pad=3)
    plt.imshow(X_test[start_index + i])

plt.show()