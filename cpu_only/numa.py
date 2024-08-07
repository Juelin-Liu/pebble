import os

# Function to read the contents of a file
def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()

# Function to get NUMA nodes and their memory
def get_numa_nodes_info():
    numa_nodes = []
    numa_nodes_dir = '/sys/devices/system/node/'

    if os.path.exists(numa_nodes_dir):
        for node in os.listdir(numa_nodes_dir):
            if node.startswith('node'):
                meminfo_file = os.path.join(numa_nodes_dir, node, 'meminfo')
                if os.path.exists(meminfo_file):
                    meminfo = read_file(meminfo_file)
                    numa_nodes.append({
                        'node': node,
                        'meminfo': meminfo
                    })

    return numa_nodes

# Get NUMA nodes information
numa_info = get_numa_nodes_info()
for node_info in numa_info:
    print(f"Node: {node_info['node']}")
    print(f"Meminfo: {node_info['meminfo']}")

