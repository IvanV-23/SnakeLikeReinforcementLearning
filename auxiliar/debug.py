import matplotlib.pyplot as plt
from lib.enviroment.navigation import NavigationEnv

env = NavigationEnv(training=True)
obs = env.reset_cnn()
plt.imshow(obs.transpose(1,2,0))
plt.show()