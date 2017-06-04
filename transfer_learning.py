import tensorflow as tf
from squeezenet import SqueezeNet

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def load_squeezenet(session):
    SAVE_PATH = 'squeezenet/squeezenet.ckpt'
    model = SqueezeNet(save_path=SAVE_PATH, sess=session)
    return model



# tf.reset_default_graph() # remove all existing variables in the graph
# sess = get_session() # start a new Session
# model = load_squeezenet(sess)
# print("DONE LOADING MODEL")
