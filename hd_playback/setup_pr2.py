import openravepy as rave
env = rave.Environment()
env.Load('data/pr2test1.env.xml')
pr2 = env.GetRobots()[0]

