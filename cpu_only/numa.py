import os

# Function to read the contents of a file
def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()

def parse_cpulist(line:str):
    ret = []
    cpu_id_list = line.split(",")
    for cpu_id in cpu_id_list:
        start, end = cpu_id.split("-")
        start = int(start)
        end = int(end)
        for i in range(start, end + 1):
            ret.append(i)
    return ret

# Function to get NUMA nodes and their memory
def get_numa_nodes_info():
    numa_nodes = {}
    numa_nodes_dir = '/sys/devices/system/node/'

    if os.path.exists(numa_nodes_dir):
        for dirname in os.listdir(numa_nodes_dir):
            if dirname.startswith('node'):
                node_id = int(dirname[4:])
                cpulist_file = os.path.join(numa_nodes_dir, dirname, 'cpulist')
                if os.path.exists(cpulist_file):
                    cpulist = read_file(cpulist_file)
                    numa_nodes[node_id] = parse_cpulist(cpulist)
    else:
        numa_nodes[0] = [i for i in range(os.cpu_count())]
    return numa_nodes

numa_info = get_numa_nodes_info()

# Get NUMA nodes information
if __name__ == "__main__":
    print(f"numa: {numa_info}")