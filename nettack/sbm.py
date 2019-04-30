# -*- coding: utf-8 -*-
import logging
import random

import numpy as np


class SBM(object):

    def __init__(self, num_vertices, communities, vertex_labels, p_matrix, p_matrix_2):
        logging.info('Initializing SBM Model ...')
        self.num_vertices = num_vertices
        self.communities = communities
        self.vertex_labels = vertex_labels
        self.p_matrix = p_matrix
        self.p_matrix_2 = p_matrix_2
        self.block_matrix, self.block_matrix_2 = self.generate()
        

    def generate(self):
        logging.info('Generating SBM  ...')
        v_label_shape = (1, self.num_vertices)
        p_matrix_shape = (self.communities, self.communities)
        block_matrix_shape = (self.num_vertices, self.num_vertices)
        block_matrix = np.zeros(block_matrix_shape, dtype=int)
        block_matrix_2 = np.zeros(block_matrix_shape, dtype=int)

        for row, _row in enumerate(block_matrix):
            for col, _col in enumerate(block_matrix[row]):
                if row>col:
                    community_a = self.vertex_labels[row]
                    community_b = self.vertex_labels[col]

                    p = random.random()
               
                    val = self.p_matrix[community_a][community_b]

                    if p < val:
                        block_matrix[row][col] = 1
                        block_matrix[col][row] = 1

                    val = self.p_matrix_2[community_a][community_b]

                    if p < val:
                        block_matrix_2[row][col] = 1
                        block_matrix_2[col][row] = 1

        return block_matrix, block_matrix_2


pass