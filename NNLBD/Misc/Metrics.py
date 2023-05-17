#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    05/02/2022                                                                   #
#    Revised: 05/02/2022                                                                   #
#                                                                                          #
#    Metric Class For The NNLBD Package.                                                   #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Modules
import numpy as np

############################################################################################
#                                                                                          #
#    Utils Model Class                                                                     #
#                                                                                          #
############################################################################################

class Metrics:
    def __init__( self ):
        pass

    def __del__( self ):
        pass

    """
        Computes Cosine Similarity Between Two Embeddings/Vectors
    """
    def Cosine_Similarity( self, x_instance, y_instance ):
        dot_product_value  = np.dot( x_instance, y_instance )
        x_instance_l2_norm = np.linalg.norm( x_instance, ord = 2 )
        y_instance_l2_norm = np.linalg.norm( y_instance, ord = 2 )
        cross_product      = x_instance_l2_norm * y_instance_l2_norm
        return dot_product_value / cross_product


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    exit()