from classes import *

image_data = []
image_labels = []

for i in tqdm.tqdm(range(NUM_CATEGORIES)):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print("Images: ",image_data.shape, "\nLabels: ",image_labels.shape)

shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255

print("\nX_train.shape: ", X_train.shape)
print("X_valid.shape: ", X_val.shape)
print("y_train.shape: ", y_train.shape)
print("y_valid.shape: ", y_val.shape)

## One hot encoding the labels
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print("\none_hot_y_train.shape: ",y_train.shape)
print("one_hot_y_valid.shape: ",y_val.shape)