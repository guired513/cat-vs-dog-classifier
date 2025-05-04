import matplotlib.pyplot as plt
import os
from PIL import Image

cat_dir = "data/training_set/cats"
dog_dir = "data/training_set/dogs"

cat_sample = os.path.join(cat_dir, os.listdir(cat_dir)[0])
dog_sample = os.path.join(dog_dir, os.listdir(dog_dir)[0])

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(Image.open(cat_sample))
axs[0].set_title("Sample Cat")
axs[1].imshow(Image.open(dog_sample))
axs[1].set_title("Sample Dog")
plt.show()