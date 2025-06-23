from rl_common.class_op import ClassOp
import os

def load_shared_data(sys_name, reward_v):
    if sys_name == "daisy" or sys_name == "elevator" or sys_name == "email__spl" or sys_name == "notepad__spl":
        classes_file = f'/home/ps/jy_exp/input_/{sys_name}/classes.txt'
        attributes_file = f'/home/ps/jy_exp/input_/{sys_name}/attribute.txt'
        methods_file = f'/home/ps/jy_exp/input_/{sys_name}/method.txt'
    else:
        classes_file = f'/home/ps/jy_exp/{input}/{sys_name}/classes'
        attributes_file = f'/home/ps/jy_exp/{input}/{sys_name}/attribute'
        methods_file = f'/home/ps/jy_exp/{input}/{sys_name}/method'
    
    output_dir = f'/home/ps/jy_exp/output/{sys_name}/{reward_v}'
    os.makedirs(output_dir, exist_ok=True)

    classes = ClassOp.load_classes(classes_file)        # 下标从1开始
    #input的num_classes
    num_classes = max(map(int, classes.keys()))
    attributes = ClassOp.load_data(attributes_file, num_classes)
    methods = ClassOp.load_data(methods_file, num_classes)
    method_counts, attr_counts = ClassOp.count_features(methods, attributes)

    NOF_dir = f'/home/ps/jy_exp/input_/{sys_name}/NOF.txt'
    if not os.path.exists(NOF_dir):
        NOF_dir = ""
    NOF = ClassOp.get_NOF(NOF_dir)
    
    GNN_class = ClassOp.load_GNNclass(sys_name)

    return {
        "classes": classes,
        "methods": methods,
        "attributes": attributes,
        "method_counts": method_counts,
        "attr_counts": attr_counts,
        "output_dir": output_dir,
        "NOF": NOF, 
        "GNN_class": GNN_class
    }