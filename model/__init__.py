from .hdmapnet import BiMapper


def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):


    if method == 'BiMapper':
        model = BiMapper(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred,
                         direction_dim=angle_class, lidar=False)
    else:
        print('no model find')

    return model
