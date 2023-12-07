import torch
import numpy as np

def get_prototypes_for_session(trainset, transform, model, args, session, use_vector_variance = False, views=1):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    if session == 0:
        num_classes = args.base_class
    else:
        num_classes = args.way

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)

    trainloader.dataset.transform = transform

    embedding_list = []
    label_list = []
    old_mode = model.module.mode
    with torch.no_grad():
        # Better modelling using multiple augmentation
        for v in range(views):
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                # embedding = torch.nn.functional.normalize(model(data), dim = 1)
                embedding = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
            
    embedding_list = np.concatenate(embedding_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    # Protolist contains the prototypes
    numInstancesPerClass = int(label_list.shape[0]/num_classes)                                     # Number of instances each class has, for example for minet 500 in base session and 5 in incremental sessions
    prototypes = np.zeros(shape=(num_classes, embedding_list.shape[1]), dtype=np.float32)           # Means of feature embeddings per class. 100x512
    exmplesPerClass = np.zeros(num_classes)                                                         # List maintaining the number of examples
    prototypeLabels = np.zeros(num_classes)                                                         # Label association for each prototype
    examplesInClass = np.zeros(shape=(num_classes, numInstancesPerClass, embedding_list.shape[1]))  # contains the embeddings here
    smallestClassIndex = np.min(label_list)
    classVariances = np.zeros(shape=(num_classes, embedding_list.shape[1]))
    
    for k in range(embedding_list.shape[0]):
        rowIndex = int(label_list[k] - smallestClassIndex)  # finding the right label 
        prototypes[rowIndex,:] += embedding_list[k,:]   # Making the prototypes here
        examplesInClass[rowIndex,int(exmplesPerClass[rowIndex]),:] = embedding_list[k,:]        # Storing the embeddings here
        exmplesPerClass[rowIndex] += 1

    if use_vector_variance:
        for k in range(num_classes):
            classVariances[k,:] = np.var(examplesInClass[k,:,:], axis=0)
    else:
        for k in range(num_classes):
            classVariances[k,:] = np.var(examplesInClass[k,:,:]) * np.ones(embedding_list.shape[1])
    
    for k in range(num_classes):
        prototypes[k] /= exmplesPerClass[k]
        prototypeLabels[k] = int(k + smallestClassIndex)
    
    model.module.mode = old_mode

    return prototypes, prototypeLabels, classVariances, numInstancesPerClass

def generateMutivariateData(prototypes, prototypeLabels, classVariances, numInstancesPerClass):
    numClasses = prototypes.shape[0]
    numFeatures = prototypes.shape[1]
    generatedData = np.zeros(shape=(numInstancesPerClass*numClasses,numFeatures))
    generatedLabels = np.zeros(numInstancesPerClass*numClasses)


    for k in range(numClasses):
        x = np.random.multivariate_normal(prototypes[k,:],  np.eye(numFeatures)*classVariances[k,:].T, numInstancesPerClass)
        generatedLabels[k*numInstancesPerClass:(k+1)*numInstancesPerClass] = prototypeLabels[k]*np.ones(numInstancesPerClass)
        generatedData[k*numInstancesPerClass:(k+1)*numInstancesPerClass,:] = x

    return generatedData, generatedLabels

def augmentMultivariateData(sub_feature_matrix, sub_labels, feature_matrix, labels):
    # For each curr label find a label in all_labels
    new_sub_feature_matrix = np.zeros_like(sub_feature_matrix)

    for i in range(len(sub_feature_matrix)):
        sub_label = sub_labels[i]
        indices_same_label = np.where(labels == sub_label)[0]
        random_index = np.random.choice(indices_same_label)
        new_row = (sub_feature_matrix[i] + feature_matrix[random_index]) / 2.0
        new_sub_feature_matrix[i] = new_row

    return new_sub_feature_matrix

def augmentSingleMultivariateData(sub_feature_matrix, sub_label, feature_matrix, labels):
    # For each curr label find a label in all_labels
    indices_same_label = np.where(labels == sub_label)[0]
    random_index = np.random.choice(indices_same_label)
    new_sub_feature_matrix = (sub_feature_matrix + feature_matrix[random_index]) / 2.0
    return new_sub_feature_matrix

def compute_aug_feature_matrix(feature_matrix, labels):
    unique_labels = np.unique(labels)
    new_feature_matrix = np.zeros_like(feature_matrix)

    for label in unique_labels:
        indices_same_label = np.where(labels == label)[0]

        for i in indices_same_label:
            # Exclude the current index from the random choice
            indices_to_choose_from = np.delete(indices_same_label, np.where(indices_same_label == i))
            random_index = np.random.choice(indices_to_choose_from)

            # Calculate the average of the two rows
            new_row = (feature_matrix[i] + feature_matrix[random_index]) / 2
            new_feature_matrix[i] = new_row

    return new_feature_matrix

from torch.utils.data import Dataset
class GaussianWrapperDataSet(Dataset):
    # The class wraps the dataset object and the gaussian gener
    def __init__(self, trainset, gaussian_data, gaussian_labels, augmented_data=None, augmented_label=None): # Extra parameters
        # Combine the train set and the gaussian data. 
        # In a list store the trainset i.e. the current novel class and the gaussian data
        self.trainset_len = len(trainset)   # This will the indicator in our 
        self.trainset = trainset

        self.data = trainset.data + gaussian_data.tolist()
        self.targets = trainset.targets + gaussian_labels.astype(np.int).tolist()
        
        img, _ = trainset.__getitem__(0)
        self.place_holder_image = img

        self.gaussian_data = torch.tensor(gaussian_data, dtype=torch.float32)
        self.place_holder_gaussian = self.gaussian_data[0]

        self.augmented_data = torch.tensor(augmented_data, dtype=torch.float32)
        self.augmented_label = augmented_label

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, i):
        # Now the train loader might have an issue. 
        # To bypass this maintain a place holder zero_like tensor 
        # for either the image data or the embedding data 
        # i.e. we return image, targets, embedding, ei_flag
        # Where the ei_flag determines whether this sample should be considered as embeddings or image
        if i >= self.trainset_len:
            # Sampling the gaussian data
            # Augment gaussian data
            gaus_i = i - self.trainset_len

            if self.augmented_data is not None:
                augmented_encoding = augmentSingleMultivariateData(self.gaussian_data[gaus_i], self.targets[i], self.augmented_data, self.augmented_label)
                return self.place_holder_image, augmented_encoding, self.targets[i], 0

            return self.place_holder_image, self.gaussian_data[gaus_i], self.targets[i], 0
        else:
            img, target = self.trainset.__getitem__(i)
            return img, self.place_holder_gaussian, target, 1
            