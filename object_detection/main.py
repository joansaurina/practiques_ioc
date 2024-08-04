import os
import sys

def read_labels(label_dir):
    """Reads all label files in the given directory and returns a list of sets of detected objects."""
    label_sets = []
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(label_dir, filename), 'r') as f:
                labels = {line.split()[0] for line in f}  # Extract only the object class (first column)
                label_sets.append(labels)
    return label_sets

def find_common_objects(label_sets):
    """Finds and returns the set of objects that are present in all label sets."""
    if not label_sets:
        return set()
    common_objects = label_sets[0]
    for label_set in label_sets[1:]:
        common_objects &= label_set
    return common_objects

def write_summary(common_objects, output_dir):
    """Writes the common objects to a summary file."""
    summary_file = os.path.join(output_dir, "label_summary.txt")
    with open(summary_file, 'w') as f:
        for obj in common_objects:
            f.write(f"{obj}\n")
    print(f"Summary saved to {summary_file}")

def main(args):
    if len(args) != 6:
        print("Usage: main.py <img_size> <conf> <device> <weights> <source_folder> <project>")
        sys.exit(1)
    
    img_size = args[0]
    conf = args[1]
    device = args[2]
    weights = args[3]
    source_folder = args[4]
    project = args[5]

    command = (f"python ../../yolov9/detect.py --img {img_size} --conf {conf} --device {device} "
               f"--weights {weights} --source ../../test_images/{source_folder} --save-txt "
               f"--save-conf --project {project}")
    
    print(f"Executing command: {command}")
    os.system(command)
    
    # Read the generated label files
    output_dir = os.path.join(project)
    label_sets = read_labels(os.path.join(output_dir, "labels"))
    
    # Find common objects
    common_objects = find_common_objects(label_sets)
    
    # Write summary file
    write_summary(common_objects, output_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
