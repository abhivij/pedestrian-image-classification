import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure


def show_image(path, file_name):
    img = cv2.imread(path + file_name)
    print(img.shape)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_file_names(dir_name):
    file_names = []
    for x in os.listdir(dir_name):
        path = os.path.join(dir_name, x)
        if os.path.isdir(path):
            file_names = file_names + get_file_names(path)
        else:
            file_names.append(path)
    return file_names


def get_custom_format_image(file_name, grayscale = False, is_flattened = False):
    if grayscale:
        color_flag = 0
    else:
        color_flag = 1
    img = cv2.imread(file_name, color_flag)
    if is_flattened:
        img = img.flatten()
    return img


def run_svm(train_data, train_label, test_data, test_label):
    svm_model = svm.SVC(gamma = 'scale')
    start_time = time.process_time()
    svm_model.fit(train_data, train_label)
    total_time = time.process_time() - start_time
    print('SVM Trained in', total_time, 'seconds')

    start_time = time.process_time()
    predicted_label = svm_model.predict(test_data)
    accuracy = accuracy_score(test_label, predicted_label)

    decision_function = svm_model.decision_function(test_data)
    auc = roc_auc_score(test_label, decision_function)
    conf_mat = confusion_matrix(test_label, predicted_label)

    total_time = time.process_time() - start_time

    print('SVM Prediction time :', total_time, 'seconds')
    print('Accuracy :', accuracy)
    print('AUC :', auc)
    print('Confusion Matrix :', conf_mat)


def run_knn(train_data, train_label, test_data, test_label):
    knn_model = KNeighborsClassifier()
    start_time = time.process_time()
    knn_model.fit(train_data, train_label)
    total_time = time.process_time() - start_time
    print('KNN Trained in', total_time, 'seconds')

    start_time = time.process_time()
    predicted_label = knn_model.predict(test_data)
    accuracy = accuracy_score(test_label, predicted_label)

    predict_proba = np.array([val[1] for val in knn_model.predict_proba(test_data)])
    auc = roc_auc_score(test_label, predict_proba)

    total_time = time.process_time() - start_time

    print('KNN Prediction time :', total_time, 'seconds')
    print('Accuracy :', accuracy)
    print('AUC :', auc)


def hog_features(img, resized_img_size = (64, 128), orientations = 9, ppc = (8, 8), cpb = (2, 2)):
    resized_img = cv2.resize(img, resized_img_size)
    fd, hog_image = hog(resized_img, orientations = orientations, pixels_per_cell = ppc, 
                    cells_per_block = cpb, visualize = True, multichannel = True)    
    return fd 



def main():    

    trainA_file_names = get_file_names('Individual_Component/train/train_positive_A/')
    trainB_file_names = get_file_names('Individual_Component/train/train_positive_B/')
    trainC_file_names = get_file_names('Individual_Component/train/train_positive_C/')
    train_neg_file_names = get_file_names('Individual_Component/train/train_negative_A/')
    
    test_pos_file_names = get_file_names('Individual_Component/test/test_positive/')
    test_neg_file_names = get_file_names('Individual_Component/test/test_negative/')

    train_file_names = trainA_file_names + trainB_file_names + trainC_file_names + train_neg_file_names
    test_file_names = test_pos_file_names + test_neg_file_names


    #test with grayscale image flattened

    print('----Test with flattened grayscale images----')
    #generate train data
    train_data = np.array([get_custom_format_image(file_name, grayscale = True, is_flattened = True) for file_name in train_file_names])
    train_label = [1] * len(trainA_file_names + trainB_file_names + trainC_file_names)
    train_label = np.array(train_label + ([0] * len(train_neg_file_names)))

    print(train_data.shape)
    print(train_label.shape)

    #generate test data
    test_data = np.array([get_custom_format_image(file_name, grayscale = True, is_flattened = True) for file_name in test_file_names])
    test_label = np.array([1] * len(test_pos_file_names) + [0] * len(test_neg_file_names))

    print(test_data.shape)
    print(test_label.shape)

    #generate subsets
    train_subset_indices = np.array(random.sample(range(train_data.shape[0]), 8000))
    test_subset_indices = np.array(random.sample(range(test_data.shape[0]), 2000))

    train_data_subset = train_data[train_subset_indices, :]
    train_label_subset = train_label[train_subset_indices]

    test_data_subset = test_data[test_subset_indices, :]
    test_label_subset = test_label[test_subset_indices]

    print(train_data_subset.shape)
    print(train_label_subset.shape)
    print(test_data_subset.shape)
    print(test_label_subset.shape)
    print(np.sum(train_label_subset))
    print(np.sum(test_label_subset))

    run_svm(train_data_subset, train_label_subset, test_data_subset, test_label_subset)
    run_knn(train_data_subset, train_label_subset, test_data_subset, test_label_subset)




    #test with hog features 


    print('----Test with HOG features (2, 2) cells per block----')

    #generate train data
    train_data = np.array([get_custom_format_image(file_name) for file_name in train_file_names])
    train_label = [1] * len(trainA_file_names + trainB_file_names + trainC_file_names)
    train_label = np.array(train_label + ([0] * len(train_neg_file_names)))

    print(train_data.shape)
    print(train_label.shape)

    #generate test data
    test_file_names = test_pos_file_names + test_neg_file_names
    test_data = np.array([get_custom_format_image(file_name) for file_name in test_file_names])
    test_label = np.array([1] * len(test_pos_file_names) + [0] * len(test_neg_file_names))

    print(test_data.shape)
    print(test_label.shape)


    #generate subsets
    train_data_subset = np.array([hog_features(img) for img in train_data[train_subset_indices, :]])
    train_label_subset = train_label[train_subset_indices]
    
    test_data_subset = np.array([hog_features(img) for img in test_data[test_subset_indices, :]])
    test_label_subset = test_label[test_subset_indices]
    
    print(train_data_subset.shape)
    print(train_label_subset.shape)
    print(test_data_subset.shape)
    print(test_label_subset.shape)
    print(np.sum(train_label_subset))
    print(np.sum(test_label_subset))
    
    run_svm(train_data_subset, train_label_subset, test_data_subset, test_label_subset)
    run_knn(train_data_subset, train_label_subset, test_data_subset, test_label_subset)




    print('----Test with HOG features (1, 1) cells per block----')

    train_data_subset = np.array([hog_features(img, cpb = (1, 1)) for img in train_data[train_subset_indices, :]])
    train_label_subset = train_label[train_subset_indices]
    
    test_data_subset = np.array([hog_features(img, cpb = (1, 1)) for img in test_data[test_subset_indices, :]])
    test_label_subset = test_label[test_subset_indices]
    
    print(train_data_subset.shape)
    print(train_label_subset.shape)
    print(test_data_subset.shape)
    print(test_label_subset.shape)
    print(np.sum(train_label_subset))
    print(np.sum(test_label_subset))
    
    run_svm(train_data_subset, train_label_subset, test_data_subset, test_label_subset)
    run_knn(train_data_subset, train_label_subset, test_data_subset, test_label_subset)



    #run on full data

    print('----Test with HOG features (1, 1) cells per block on full data----')

    start_time = time.process_time()
    train_data_full = np.array([hog_features(img, cpb = (1, 1)) for img in train_data])
    train_label_full = train_label
    total_time = time.process_time() - start_time
    print('Processed train data in', total_time, 'seconds')
        
    start_time = time.process_time()
    test_data_full = np.array([hog_features(img, cpb = (1, 1)) for img in test_data])
    test_label_full = test_label
    total_time = time.process_time() - start_time
    print('Processed test data in', total_time, 'seconds')
    
    print(train_data_full.shape)
    print(train_label_full.shape)
    print(test_data_full.shape)
    print(test_label_full.shape)
    
    print(np.sum(train_label_full))
    print(np.sum(test_label_full))
    
    run_svm(train_data_full, train_label_full, test_data_full, test_label_full)


    #run on larger train subset and full test data

    print('----Test with HOG features (1, 1) cells per block on larger train subset and full test data----')

    #obtain train:test ratio = 80:20
    train_data_subset_size = test_data.shape[0] * 4
    train_subset_indices = np.array(random.sample(range(train_data.shape[0]), train_data_subset_size))

    start_time = time.process_time()
    train_data_subset = np.array([hog_features(img, cpb = (1, 1)) for img in train_data[train_subset_indices, :]])
    train_label_subset = train_label[train_subset_indices]
    total_time = time.process_time() - start_time
    print('Processed train data in', total_time, 'seconds')
    
    start_time = time.process_time()
    test_data_full = np.array([hog_features(img, cpb = (1, 1)) for img in test_data])
    test_label_full = test_label
    total_time = time.process_time() - start_time
    print('Processed test data in', total_time, 'seconds')
    
    print(train_data_subset.shape)
    print(train_label_subset.shape)
    print(test_data_full.shape)
    print(test_label_full.shape)
    print(np.sum(train_label_subset))
    print(np.sum(test_label_full))
    
    run_svm(train_data_subset, train_label_subset, test_data_full, test_label_full)


if __name__ == "__main__":
    main()