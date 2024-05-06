import numpy as np
import javalang
from javalang.ast import Node
from anytree import AnyNode
import os
import time
import json


class JavaSyntaxMatrixGenerator:
    def __init__(self, java_path, npy_path, json_path='type.json'):
        self.java_path = java_path
        self.npy_path = npy_path
        self.nodetypedict, self.tokendict, self.node2groups = self.load_dictionaries_from_json(json_path)

    def load_dictionaries_from_json(self, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data['nodetypedict'], data['tokendict'], data['node2groups']

    def listdir(self, path):
        """
        Recursively lists all files in the specified directory and subdirectories.

        Args:
        path (str): The directory path to list files from.

        Returns:
        list: A list of all file paths accumulated.
           """
        javalist = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                javalist.extend(self.listdir(file_path))
            else:
                javalist.append(file_path)
        return javalist

    def get_ast(self, path):
        """
            Read a Java source code file, tokenize it, parse it to create an AST, and print the AST.

            Args:
            path (str): The path to the Java file to be parsed.

            Returns:
            programast: The AST of the parsed Java member declaration.
            """
        programfile = open(path, encoding='utf-8')
        programtext = programfile.read()
        programfile.close()

        # Perform lexical analysis on the read text
        programtokens = javalang.tokenizer.tokenize(programtext)
        token_list = list(programtokens)

        # Parse tokens to generate AST
        parser = javalang.parse.Parser(token_list)
        programast = parser.parse_member_declaration()

        return programast, token_list

    def get_token(self, node):
        """
            Extracts a token from a given AST node, which represents the type or characteristic of the node.

            Args:
            node (Node|str|set): The node from which the token will be extracted. This node can be an
                                 instance of a Node class, a string, or a set.

            Returns:
            str: A token representing the type or characteristic of the node.
            """
        token = ''
        # print(isinstance(node, Node))
        # print(type(node))
        if isinstance(node, str):  # Directly use the string as a token
            token = node
        elif isinstance(node, set):  # Use a generic token for a set of modifiers
            token = 'Modifier'
        elif isinstance(node, Node):  # Use the class name of the node for more specific nodes
            token = node.__class__.__name__
        return token

    # Get the list of child nodes for the node
    def get_child(self, root):
        """
            Extracts and returns all child nodes from a given AST node, handling different types
            of node structures and expanding any nested lists.

            Args:
            root (Node|set|other): The AST node from which children are to be extracted. This can be an
                                   instance of a Node class, a set, or other possible structures that
                                   can contain child nodes.

            Returns:
            list: A flat list of all child nodes extracted from the root
            """
        # print(root)
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        # Expand any nested child nodes within the list
        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        # print(sub_item)
                        yield sub_item
                elif item:
                    # print(item)
                    yield item

        return list(expand(children))


    def create_tree(self, root, node, nodelist, parent=None):
        """
            Recursively creates a tree structure from an AST node using the AnyNode class. Each node in the
            created tree corresponds to an AST node and is added to a tree with parent-child relationships.

            Args:
            root (AnyNode): The root of the tree being constructed. This should be an AnyNode object.
            node (Node|any): The current AST node being processed.
            nodelist (list): A list that tracks all nodes that have been processed. Used to generate unique IDs.
            parent (AnyNode, optional): The parent node under which the current node should be placed. Defaults to None.

            Returns:
            None: The function modifies the tree structure in place and does not return a value.
            """
        id = len(nodelist)
        # print(id)
        # Retrieve the token and list of child nodes for the current node pair
        token, children = self.get_token(node), self.get_child(node)
        # If it is the first node in the list, set it as the root node's token and data
        if id == 0:
            root.token = token
            root.data = node
        else:
            newnode = AnyNode(id=id, token=token, data=node, parent=parent)
        nodelist.append(node)
        for child in children:
            if id == 0:
                self.create_tree(root, child, nodelist, parent=root)
            else:
                self.create_tree(root, child, nodelist, parent=newnode)

    def traverse(self, node, typedict, triads, path=None):
        """
           Recursively traverses a tree, collecting triads of tokens from paths and modifying node tokens
           based on a provided dictionary.

           Args:
           node (AnyNode): The current node being processed in the tree.
           typedict (dict): A dictionary mapping original tokens to desired token strings.
           triads (list): A list where each element is a triad (three consecutive tokens) from the path in the tree.
           path (list, optional): A list that tracks the current path of tokens as the recursion goes deeper.

           Yields:
           list: The complete token path for each leaf node encountered during the traversal.
           """
        if path is None:
            path = []
        if len(node.children) == 0:
            try:
                node.token = typedict[node.token]
            except KeyError:
                if node.token != 'ReturnStatement':
                    node.token = 'Null'
            path.append(node.token)
            if len(path) >= 3:
                triad = [path[-3], path[-2], path[-1]]
                triads.append(triad)
            yield path
            path.pop()
        else:
            path.append(node.token)
            if len(path) >= 3:
                triad = [path[-3], path[-2], path[-1]]
                triads.append(triad)
            for child in node.children:
                yield from self.traverse(child, typedict, triads, path)
            path.pop()

    # Generate a second-order Markov matrix.
    def second_order_matrix(self, path, npy_path):
        """
           Generates a matrix representation of the syntactic and structural patterns in a Java source file.

           Args:
           path (str): The file path to the Java source file.

           Returns:
           np.ndarray: A matrix where each entry represents normalized counts of specific syntactic patterns.
        """
        if not os.path.exists(npy_path):
            os.makedirs(npy_path)
            print(f"Created directory {npy_path}")

        # ast generation
        tree, tokens = self.get_ast(path)

        # create tree
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        self.create_tree(newtree, tree, nodelist)

        # token type dictionary
        # Create a dictionary mapping token values to their types
        typedict = {}
        for token in tokens:
            token_type = str(type(token))[:-2].split(".")[-1]
            token_value = token.value
            if token_value not in typedict:
                typedict[token_value] = token_type
            else:
                if typedict[token_value] != token_type:
                    print('!!!!!!!!')

        # # Traverse the tree to collect triads
        triads = []
        list(self.traverse(newtree, typedict, triads, path=None))

        # Initialize a matrix of zeros with dimensions 493x72
        matrix = [[0 for _ in range(72)] for row in range(493)]

        # Obtain the state transition matrix
        for i in range(len(triads)):
            m = self.node2groups[triads[i][0] + '2' + triads[i][1]]
            name = triads[i][2]
            try:
                n = self.nodetypedict[name]
            except KeyError:
                try:
                    n = self.tokendict[typedict[name]]
                except KeyError:
                    n = 62
            matrix[m][n] += 1

        # Obtain the state transition probability matrix
        for k in range(493):
            total = 0
            for l in range(72):
                total += matrix[k][l]
            if total != 0:
                for p in range(72):
                    matrix[k][p] = (matrix[k][p]) / total

        # Serialize and save the matrix to a file
        matrix = np.array(matrix)
        # Extract the filename from the file path, remove the .java extension, and obtain the filename.
        filename = os.path.basename(path)
        npypath = npy_path + filename
        # print(npypath)
        np.save(npypath, matrix)
        return matrix

    def allmain(self):
        """
            Main method to read all Java files from a folder and generate matrices for each file.
        """
        # Read all java files from a folder
        j = 0
        javalist = self.listdir(self.java_path)
        for javafile in javalist:
            try:
                self.second_order_matrix(javafile, self.npy_path)
            except (UnicodeDecodeError, javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
                print(javafile)
            j += 1
            print(f"Number of Java files converted to a matrix: {j}")


if __name__ == '__main__':
    # Example usage:
    java_path = './BCB_datasets_samples'
    npy_path = './npy_BCB/'
    syntax_matrix_generator = JavaSyntaxMatrixGenerator(java_path, npy_path)
    a = time.time()
    syntax_matrix_generator.allmain()
    b = time.time()
    print(b-a)
