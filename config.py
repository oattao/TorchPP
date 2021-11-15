label_class_dict = {'NoFault': 0, 
                    'Inner7': 1, 'Inner14': 2, 'Inner21': 3, 
                    'Ball7': 4, 'Ball14': 5, 'Ball21': 6,
                    'Outer7': 7, 'Outer14': 8, 'Outer21': 9}

datapath = './data/transformed_images/hp_{}/noise_{}_db/'
horse_powers = [0, 1, 2, 3]
noise_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RDS = 911