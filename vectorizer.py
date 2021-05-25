import numpy as np
import torch

def getTrainingAndValidationDataAsTorchTuples(data):
    training_dataset = []
    validation_dataset = []
    for i in range(0, 850):
        training_dataset.append((convert_to_input_vector(data["collector"][i]), convert_ground_truth_to_vector(data["collector"][i])))
    for i in range(850, 900):
        validation_dataset.append((convert_to_input_vector(data["collector"][i]), convert_ground_truth_to_vector(data["collector"][i])))
    return training_dataset, validation_dataset

def convert_to_input_vector(data):
    captionVector = convert_caption_to_vector(data)
    authorVector = convert_author_to_vector(data["authorMeta"])
    audioVector = convert_audio_to_vector(data["musicMeta"])
    timeOfDayVector = np.array([((data["createTime"] % 86400) // (14400))]) # Split day into 6 parts

    vec = torch.from_numpy(np.concatenate([captionVector, authorVector, audioVector, timeOfDayVector]))
    return vec.type(torch.FloatTensor)
    
def convert_caption_to_vector(data):
    captionWords = get_number_of_words(data["text"])
    hashTagCount = len(data["hashtags"])
    return np.array([captionWords, hashTagCount])

def convert_author_to_vector(author):
    authorName = get_number_of_characters(author["nickName"])
    authorBio = get_number_of_words(author["signature"])
    authorVerified = 1 if author["verified"] else 0
    return np.array([authorName, authorBio, authorVerified])

def convert_audio_to_vector(audio):
    audioName = get_number_of_words(audio["musicName"])
    audioAuthor = get_number_of_words(audio["musicAuthor"])
    audioOriginality = 1 if audio["musicOriginal"] else 0
    return np.array([audioName, audioAuthor, audioOriginality])

def convert_ground_truth_to_vector(data):
    vec = torch.from_numpy(np.array([data["diggCount"], data["playCount"], data["shareCount"], data["commentCount"]]))
    return vec.type(torch.FloatTensor)
    
def get_number_of_characters(text):
    return len(text)

def get_number_of_words(text):
    return len(text.strip("\n").split(" "))
