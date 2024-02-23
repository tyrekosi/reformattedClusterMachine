import os

# Formats with ClusterNumber: for easier PCAing
def toDatasheet(clusters_dir, output_dir, hash):
    combined_lines = []
    for filename in os.listdir(clusters_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(clusters_dir, filename), 'r') as file:
                tag = filename.split('_')[1].split('.')[0] + ":"
                for line in file:
                    combined_lines.append(f"{tag}{line.rstrip()}\n")
    
    output_path = os.path.join(output_dir, f"PCA_{hash}.txt")
    
    with open(output_path, 'w') as file:
        file.writelines(combined_lines)