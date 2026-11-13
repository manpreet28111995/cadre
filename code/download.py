import gdown
from config import DATA
gdown.download_folder('https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l', output=DATA, quiet=False)
