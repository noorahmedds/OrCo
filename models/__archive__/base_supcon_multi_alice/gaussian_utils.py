import torch
import numpy as np

def get_prototypes_for_session(trainset, transform, model, args, session, use_vector_variance = False):
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
    numInstancesPerClass = int(label_list.shape[0]/num_classes)
    prototypes = np.zeros(shape=(num_classes, embedding_list.shape[1]), dtype=np.float32)
    exmplesPerClass = np.zeros(num_classes)
    prototypeLabels = np.zeros(num_classes)
    examplesInClass = np.zeros(shape=(num_classes, numInstancesPerClass, embedding_list.shape[1]))
    smallestClassIndex = np.min(label_list)
    classVariances = np.zeros(shape=(num_classes, embedding_list.shape[1]))
    
    for k in range(embedding_list.shape[0]):
        rowIndex = int(label_list[k] - smallestClassIndex)  # finding the right label 
        prototypes[rowIndex,:] += embedding_list[k,:]   # Making the prototypes here
        examplesInClass[rowIndex,int(exmplesPerClass[rowIndex]),:] = embedding_list[k,:]
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