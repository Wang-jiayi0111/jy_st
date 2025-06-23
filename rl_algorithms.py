def generate_elevator_file(classes_file, method_file, attribute_file, output_file):
    # Step 1: Read classes.txt to create vertices
    classes = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_id = parts[0]
                class_name = parts[1]
                classes[class_id] = class_name

    # Step 2: Initialize edges dictionary to store weights
    edges = {}

    # Step 3: Read method.txt and add weights to edges
    with open(method_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                source = parts[0]
                target = parts[1]
                weight = int(parts[2])
                key = (source, target)
                edges[key] = edges.get(key, 0) + weight

    # Step 4: Read attribute.txt and add weights to edges
    with open(attribute_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                source = parts[0]
                target = parts[1]
                weight = int(parts[2])
                key = (source, target)
                edges[key] = edges.get(key, 0) + weight

    # Step 5: Sort edges by source and then target (both as integers)
    sorted_edges = sorted(edges.items(), key=lambda x: (int(x[0][0]), int(x[0][1])))

    # Step 6: Write to output file (elevator.txt)
    with open(output_file, 'w') as f:
        # Write vertices
        f.write("*Vertices {}\n".format(len(classes)))
        for class_id in sorted(classes.keys(), key=int):
            f.write('{} "{}"\n'.format(class_id, classes[class_id]))

        # Write edges (sorted)
        f.write("*Arcs\n")
        for (source, target), weight in sorted_edges:
            f.write("{} {} {:.1f}\n".format(source, target, float(weight)))



generate_elevator_file(
    classes_file='/home/ps/jy_exp/input/input_SPM/classes',
    method_file='/home/ps/jy_exp/input/input_SPM/method',
    attribute_file='/home/ps/jy_exp/input/input_SPM/attribute',
    output_file='/home/ps/jy_exp/input_GNN/SPM_MCN/combined_SPM_GN.net'
)