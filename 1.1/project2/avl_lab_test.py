# If the notebook code is not importable as a module, we include a minimal AVL here for direct testing.
# For reliability, create a small AVL implementation duplicated here to run in script form.

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTreeScript:
    def __init__(self):
        self.root = None
    def height_of(self, node):
        return node.height if node else 0
    def update_height(self, node):
        node.height = 1 + max(self.height_of(node.left), self.height_of(node.right))
    def get_balance(self, node):
        return self.height_of(node.left) - self.height_of(node.right) if node else 0
    def _right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        self.update_height(z)
        self.update_height(y)
        return y
    def _left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        self.update_height(z)
        self.update_height(y)
        return y
    def _insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        self.update_height(node)
        balance = self.get_balance(node)
        if balance > 1 and key < node.left.key:
            return self._right_rotate(node)
        if balance < -1 and key > node.right.key:
            return self._left_rotate(node)
        if balance > 1 and key > node.left.key:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)
        if balance < -1 and key < node.right.key:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)
        return node
    def insert(self, key):
        self.root = self._insert(self.root, key)
    def _preorder(self, node, res):
        if not node: return
        res.append(node.key)
        self._preorder(node.left, res)
        self._preorder(node.right, res)
    def preorder(self):
        res = []
        self._preorder(self.root, res)
        return res
    def _inorder_collect(self, node, res):
        if not node: return
        self._inorder_collect(node.left, res)
        res.append((node.key, self.get_balance(node)))
        self._inorder_collect(node.right, res)
    def collect_nodes(self):
        res = []
        self._inorder_collect(self.root, res)
        return res
    def height(self):
        return self.height_of(self.root)

if __name__ == '__main__':
    keys_to_insert = [30, 40, 50, 20, 10, 5, 35, 25]
    avl = AVLTreeScript()
    for k in keys_to_insert:
        avl.insert(k)
        print(f'After inserting {k}:')
        print('  Preorder:', avl.preorder())
        print('  Balance factors (inorder):', avl.collect_nodes())
        print('-'*40)
    print('AVL height after insertions:', avl.height())
