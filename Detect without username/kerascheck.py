import time
import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model('kerasFaceHash')


start_time = time.time()
print(model.predict(np.array([[5.74817618968, 2.78047185593, 3.64338878977, 5.30215932554, 6.25623406457, 6.63546084672, 8.25728836014, 8.90828390808, 1.0, 3.98996990559, 4.14730012975, 4.03618344427, 1.68730537362, 3.23670006881, 2.48855973959, 3.76408782268, 1.69563679774, 5.43784904142, 5.29101793332, 5.66541122962, 6.53131476292, 6.56471608808, 6.5759556029, 5.95557981513, 5.76617895934]])))
print("--- Detection time: %s seconds ---" % (time.time() - start_time))