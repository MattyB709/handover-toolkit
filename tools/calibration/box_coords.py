import numpy as np

# mm
BOX_SIZE_X = 190 
BOX_SIZE_Y = 270
BOX_SIZE_Z = 64
TAG_SIZE = 44 # size of just the black box of the tag
TAG_WHITE_SIZE = 7 # size of the whites, the total tag length is TAG_SIZE + 2 * TAG_WHITE_SIZE
NUM_TAGS = 18

# 6 DOF pose of the box in the camera frame

# TODO make absolutely sure I've got the corner ordering correct
tag_0_coords = np.array([[(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, 0], 
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, 0],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, 0],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, 0]
                         ])

tag_1_coords = np.array([[(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, 0],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, 0],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, 0],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, 0]
                         ])

tag_2_coords = np.array([[(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + (TAG_WHITE_SIZE + TAG_SIZE), 0],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + (TAG_WHITE_SIZE + TAG_SIZE), 0],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE ,0],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE , 0]
                         ])

tag_3_coords = np.array([[(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + (TAG_WHITE_SIZE + TAG_SIZE), 0],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + (TAG_WHITE_SIZE + TAG_SIZE), 0],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, 0],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, 0]
                         ])

tag_4_coords = np.array([[(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_5_coords = np.array([[(BOX_SIZE_X / 4) + (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), BOX_SIZE_Y, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])
                
tag_6_coords = np.array([[0, (BOX_SIZE_Y * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])
                    
tag_7_coords = np.array([[0, (BOX_SIZE_Y / 2) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 2) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 2) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 2) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_8_coords = np.array([[0, (BOX_SIZE_Y / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [0, (BOX_SIZE_Y / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_9_coords = np.array([[(BOX_SIZE_X / 4) - (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_10_coords = np.array([[(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), 0, (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_11_coords = np.array([[BOX_SIZE_X, (BOX_SIZE_Y / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_12_coords = np.array([[BOX_SIZE_X, (BOX_SIZE_Y / 2) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 2) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 2) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y / 2) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])

tag_13_coords = np.array([[BOX_SIZE_X, (BOX_SIZE_Y * 3/ 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) - (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)],
                         [BOX_SIZE_X, (BOX_SIZE_Y * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Z / 2) + (TAG_SIZE / 2)]
                         ])
                        
tag_14_coords = np.array([[(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE + TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE + TAG_SIZE, BOX_SIZE_Z]
                         ])

tag_15_coords = np.array([[(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE + TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y / 4) + TAG_WHITE_SIZE + TAG_SIZE, BOX_SIZE_Z],
                         ])

tag_16_coords = np.array([[(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, BOX_SIZE_Z],
                         ])

tag_17_coords = np.array([[(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE - TAG_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) + (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, BOX_SIZE_Z],
                         [(BOX_SIZE_X * 3 / 4) - (TAG_SIZE / 2), (BOX_SIZE_Y * 3 / 4) - TAG_WHITE_SIZE, BOX_SIZE_Z],
                         ])

ALL_TAG_COORDS = np.concatenate([tag_0_coords, tag_1_coords, tag_2_coords, tag_3_coords, tag_4_coords,
                                  tag_5_coords, tag_6_coords, tag_7_coords, tag_8_coords, tag_9_coords,
                                  tag_10_coords, tag_11_coords, tag_12_coords, tag_13_coords, tag_14_coords,
                                  tag_15_coords, tag_16_coords, tag_17_coords])

ALL_TAG_COORDS = np.concatenate([ALL_TAG_COORDS, np.ones((ALL_TAG_COORDS.shape[0], 1))], axis=1).reshape(NUM_TAGS, 4, 4) # add homogeneous coordinate