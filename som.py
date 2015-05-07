from math import *
from random import random

class Som:
    def __init__(self,_num_input, _num_featureX, _num_featureY, total_iterations):
        self.num_input = _num_input
        self.num_featureX = _num_featureX
        self.num_featureY = _num_featureY
        self.wts_input_map = self.initialise()
        self.bmu = {}

        self.i_learning = 0.5
        self.i_width = max(self.num_featureX, self.num_featureY)/2.0
        self.total_iterations = total_iterations

    def set_total_iterations(self, x):
        self.total_iterations = x

    def initialise(self):
        #dic (X,Y): [..]

        dic = {}
        for i in range(self.num_featureX):
            for j in range(self.num_featureY):
                dic[(i,j)] = []
                for k in range(self.num_input):
                    dic[(i,j)].append(random())
                #print(dic[(i,j)])
        return dic

    def get_bmu_pos(self, list_input):
        best = self.euclidian_distance(list_input, self.wts_input_map[(0,0)])
        bmu = [0,0]
        temp = 0.0
        for i in range(1, self.num_featureX):
            for j in range(1, self.num_featureY):
                temp = self.euclidian_distance(list_input, self.wts_input_map[(i,j)])
                if temp < best:
                    best = temp
                    bmu = [i,j]
        return bmu

    def euclidian_distance(self, l1, l2):
        distance = 0.0
        for i in range(len(l1)):
            distance = distance + pow((l1[i] - l2[i]), 2)

        return sqrt(distance)

    def learning_rate(self, t):
        return self.i_learning*exp(-float(t)/float(self.total_iterations))

    def neighbourhood_width(self, t):
        lam = float(self.total_iterations)/self.i_width
        return self.i_width*exp(-float(t)/lam)

    def neighbourhood_function(self, t, x, y, bmu):
        learning_rate = self.learning_rate(t)
        neighbourhood_width = self.neighbourhood_width(t)

        temp = [x,y]
        distance_from_best = self.euclidian_distance(temp, bmu)
        comp_2 = (-pow(distance_from_best, 2))/(2.0*pow(neighbourhood_width,2))

        return learning_rate*exp(comp_2)
    
    def step(self, t, list_input):
        bmu = self.get_bmu_pos(list_input)
        self.bmu[t] = self.wts_input_map[(bmu[0],bmu[1])]

        neighbourhood_radius = self.neighbourhood_width(t)
        rad_sqr = pow(neighbourhood_radius, 2)

        for i in range(self.num_featureX):
            for j in range(self.num_featureY):

                dist_bmu = pow(self.euclidian_distance([i,j], bmu), 2)
                
                if(dist_bmu < rad_sqr):
                    for k in range(self.num_input):
                        wt = self.wts_input_map[(i,j)][k]
                        h = self.neighbourhood_function(t, i, j, bmu)
                        self.wts_input_map[(i,j)][k] = wt + h*(list_input[k] - wt)

