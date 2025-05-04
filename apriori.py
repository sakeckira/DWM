from itertools import combinations

# Transactions
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

min_support_count = 2  # Updated minimum support

# Function to calculate support count
def get_support_count(itemset):
    return sum(1 for t in transactions if set(itemset).issubset(t))

# Function to generate candidate itemsets of size k from previous frequent itemsets
def generate_candidates(prev_frequent, k):
    candidates = set()
    for i in range(len(prev_frequent)):
        for j in range(i + 1, len(prev_frequent)):
            candidate = sorted(set(prev_frequent[i]) | set(prev_frequent[j]))
            if len(candidate) == k:
                candidates.add(tuple(candidate))
    return [list(c) for c in candidates]
    
# Apriori algorithm (modified to return only last L)
def apriori_last_L(transactions, min_support_count):
    items = sorted(set(item for t in transactions for item in t))
    L = [[item] for item in items if get_support_count([item]) >= min_support_count]
    
    k = 2
    last_L = L
    while L:
        Ck = generate_candidates(L, k)
        L = [c for c in Ck if get_support_count(c) >= min_support_count]
        if L:
            last_L = L  # Update last frequent itemsets
        k += 1
    return last_L

# Run and print only the last frequent itemsets
last_frequent_itemsets = apriori_last_L(transactions, min_support_count)
print("Last L (largest frequent itemsets):")
for itemset in last_frequent_itemsets:
    print(itemset)
