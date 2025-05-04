from collections import defaultdict, Counter

# Sample transactions
transactions = [
    ['I1', 'I2', 'I5'],
    ['I2', 'I4'],
    ['I2', 'I3'],
    ['I1', 'I2', 'I4'],
    ['I1', 'I3'],
    ['I2', 'I3'],
    ['I1', 'I3'],
    ['I1', 'I2', 'I3', 'I5'],
    ['I1', 'I2', 'I3'],
]

min_support = 2

# Step 1: Count item frequency
item_counter = Counter()
for t in transactions:
    for item in t:
        item_counter[item] += 1

# Step 2: Filter infrequent items
freq_items = {item for item, count in item_counter.items() if count >= min_support}

# Step 3: Sort items in transactions by frequency
def clean_sort(t):
    return sorted([i for i in t if i in freq_items], key=lambda x: item_counter[x], reverse=True)

transactions = [clean_sort(t) for t in transactions if clean_sort(t)]

# Step 4: Define FP-Tree Node
class TreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = defaultdict(lambda: None)

    def increment(self, count):
        self.count += count

# Step 5: Build the FP-Tree
def build_tree(transactions):
    root = TreeNode('root', 1, None)
    for t in transactions:
        current_node = root
        for item in t:
            if item in current_node.children:
                current_node.children[item].increment(1)
            else:
                current_node.children[item] = TreeNode(item, 1, current_node)
            current_node = current_node.children[item]
    return root

# Step 6: Display tree (optional, for understanding)
def display_tree(node, indent=0):
    print('  ' * indent + f"{node.name} ({node.count})")
    for child in node.children.values():
        if child:
            display_tree(child, indent + 1)

# Build and show the FP-tree
fp_tree = build_tree(transactions)
display_tree(fp_tree)
