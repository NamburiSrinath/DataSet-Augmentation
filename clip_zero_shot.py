import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torchvision import datasets
import torchvision.transforms as transforms

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
batch_size = 1
test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set_testing/'
testset = datasets.ImageFolder(root=test_folder, transform=transform)

test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
          
# print(test_dataloader)
# for data, labels in test_dataloader:
#     print(data, labels)
#     print("---------")

# Prepare the inputs
# image, class_id = cifar100[3637]
# print(image)
# print("-----")
# print(class_id)
# image_input = preprocess(image).unsqueeze(0).to(device)
# print(image_input.shape)
# print("-----")
# print(cifar100.classes)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
# print(text_inputs)
# print("--------")
correct, total = 0, 0
for image_input, labels in test_dataloader:
# Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input.to(device))
        text_features = model.encode_text(text_inputs)
        

# Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
    # print(f"Actual class: {classes[labels]}")
    if classes[indices] == classes[labels]:
        correct += 1
    total += 1
print(f"Accuracy is {correct/total}")