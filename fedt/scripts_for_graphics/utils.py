import numpy as np

def remove_outliers_from_list(group_values, remove_outliers_strategy):
    if not remove_outliers_strategy or len(group_values) < 4:
        return list(group_values)

    q1, q3 = np.percentile(group_values, [25, 75])
    interquartile_range = q3 - q1

    lim_inf_mod = q1 - 1.5 * interquartile_range
    lim_sup_mod = q3 + 1.5 * interquartile_range
    lim_inf_ext = q1 - 3.0 * interquartile_range
    lim_sup_ext = q3 + 3.0 * interquartile_range

    clean_data = []
    for value in group_values:
        is_extremo = value < lim_inf_ext or value > lim_sup_ext
        is_moderado = (value < lim_inf_mod or value > lim_sup_mod) and not is_extremo
        
        if remove_outliers_strategy == 'ambos' and (is_extremo or is_moderado): 
            continue
        if remove_outliers_strategy == 'extremos' and is_extremo: 
            continue
        if remove_outliers_strategy == 'moderados' and is_moderado: 
            continue
        
        clean_data.append(value)
        
    return clean_data