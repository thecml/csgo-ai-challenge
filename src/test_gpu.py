import tensorflow as tf

def get_available_gpus():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    get_available_gpus()