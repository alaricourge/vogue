import requests
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#TEXTE

def decoupage_lien_designer_saison(lien):
    return lien.split("/")[-1],lien.split("/")[-2]

def find_date_in_string(string):
    import re
    return re.findall(r'\d{4}', string)




#BASIQUE IMAGE

def telecharger_jpg(list_picture):

    dossier_nom = os.path.join("best_designer")
    if not os.path.exists(dossier_nom):
        os.makedirs(dossier_nom)

    for i, url in enumerate(list_picture):
        # Télécharger et enregistrer l'image dans le dossier spécifique
        with open(os.path.join("best_designer/"+str(i)+".jpg"), "wb") as f:
            response = requests.get(url)
            f.write(response.content)

def open_image(location):
    image = cv2.imread(location,cv2.IMREAD_UNCHANGED )
    return image

def show_image(image):
    if image is not None:
        if image.shape[2] == 4:
            image=cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(3, 3))  
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        print("Image is empty.")
    return image

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)



#CALCUL IMAGE 

def remove_transparent_pixels(image):
    # Check if image has an alpha channel
    if image.shape[2] == 4:
        non_transparent_pixels = image[image[:, :, 3] > 0]
        return non_transparent_pixels[:, :3]  # Discard the alpha channel
    else:
        return image
    
    
def extract_dominant_color(image, k=5):
    pixels = remove_transparent_pixels(image)
    pixels = pixels.reshape(-1, 3)

    # Apply k-means to find clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Find the most frequent cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    
    color_freq_pairs = list(zip(kmeans.cluster_centers_, counts))
    color_freq_pairs.sort(key=lambda x: x[1], reverse=True)
    sorted_colors = [color for color, freq in color_freq_pairs]
    dominant_color= [[int(i) for i in t]for t in sorted_colors] 
    
    return dominant_color

def numbers_dominant_color(image):
    pixels = remove_transparent_pixels(image)
    pixels = pixels.reshape(-1, 3)

    variance = []
    # Apply k-means to find clusters
    for k in range(1, 6):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        variance.append(kmeans.inertia_)
        
    plt.plot(range(1, 6), variance, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Variance')
    plt.show()

def show_dominant_color(color):
    # Create a small image filled with the dominant color
    color_square = np.ones((100, 100, 3), dtype=np.uint8) * color.astype(np.uint8)
    
    # Display the color square
    plt.figure(figsize=(2, 2))
    plt.imshow(cv2.cvtColor(color_square,cv2.COLOR_BGRA2RGBA))
    plt.axis('off')
    plt.show()

def person_detection(results):
    max_size=0
    person=None
    for obj in results[0].boxes:
        size=int(obj.xyxy[0][2]-obj.xyxy[0][0])*int(obj.xyxy[0][3]-obj.xyxy[0][1])
        if obj.cls[0].tolist()==0 and size>max_size and obj.conf>0.5:
            max_size=size
            person=obj
    return person


def get_handbag(results):
    handbag=None
    for obj in results[0].boxes:
        if obj.cls[0].tolist()==28 and obj.conf>0.3:
            handbag=obj
        if obj.cls[0].tolist()==26 and obj.conf>0.3:
            handbag=obj
    return handbag 




def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)# -1 means all the rest in the dimension
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def inside_mask(mask, image):
    mask = np.stack([mask, mask, mask], axis=2)
    mask_image =mask*image
    return mask_image   