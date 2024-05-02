import numpy as np
import javalang
from javalang.ast import Node
from anytree import AnyNode
import time
import os
from multiprocessing import Pool
from functools import partial


# 非叶子节点代码语法类型
nodetypedict = {'MethodDeclaration': 0, 'Modifier': 1, 'FormalParameter': 2, 'ReferenceType': 3, 'BasicType': 4,
     'LocalVariableDeclaration': 5, 'VariableDeclarator': 6, 'MemberReference': 7, 'ArraySelector': 8, 'Literal': 9,
     'BinaryOperation': 10, 'TernaryExpression': 11, 'IfStatement': 12, 'BlockStatement': 13, 'StatementExpression': 14,
     'Assignment': 15, 'MethodInvocation': 16, 'Cast': 17, 'ForStatement': 18, 'ForControl': 19,
     'VariableDeclaration': 20, 'TryStatement': 21, 'ClassCreator': 22, 'CatchClause': 23, 'CatchClauseParameter': 24,
     'ThrowStatement': 25, 'WhileStatement': 26, 'ArrayInitializer': 27, 'ReturnStatement': 28, 'Annotation': 29,
     'SwitchStatement': 30, 'SwitchStatementCase': 31, 'ArrayCreator': 32, 'This': 33, 'ConstructorDeclaration': 34,
     'TypeArgument': 35, 'EnhancedForControl': 36, 'SuperMethodInvocation': 37, 'SynchronizedStatement': 38,
     'DoStatement': 39, 'InnerClassCreator': 40, 'ExplicitConstructorInvocation': 41, 'BreakStatement': 42,
     'ClassReference': 43, 'SuperConstructorInvocation': 44, 'ElementValuePair': 45, 'AssertStatement': 46,
     'ElementArrayValue': 47, 'TypeParameter': 48, 'FieldDeclaration': 49, 'SuperMemberReference': 50,
     'ContinueStatement': 51, 'ClassDeclaration': 52, 'TryResource': 53, 'MethodReference': 54,
     'LambdaExpression': 55, 'InferredFormalParameter': 56}

# 叶子节点token类型
tokendict = {'DecimalInteger': 57, 'HexInteger': 58, 'Identifier': 59, 'Keyword': 60, 'Modifier': 61, 'Null': 62,
              'OctalInteger': 63, 'Operator': 64, 'Separator': 65, 'String': 66, 'Annotation': 67, 'BasicType': 68,
              'Boolean': 69, 'DecimalFloatingPoint': 70, 'HexFloatingPoint': 71}

# 两元组字典
node2groups = {'FieldDeclaration2Modifier': 0, 'FieldDeclaration2ReferenceType': 1,'FieldDeclaration2VariableDeclarator': 2, 'VariableDeclarator2Literal': 3,
          'ClassDeclaration2Modifier': 4, 'ClassDeclaration2MethodDeclaration': 5,'MethodDeclaration2Modifier': 6,
          'MethodDeclaration2FormalParameter': 7, 'FormalParameter2ReferenceType': 8, 'MethodDeclaration2LocalVariableDeclaration': 9, 'LocalVariableDeclaration2ReferenceType': 10,
          'LocalVariableDeclaration2VariableDeclarator': 11, 'VariableDeclarator2ClassCreator': 12,
          'ClassCreator2ReferenceType': 13, 'ClassCreator2MemberReference': 14, 'LocalVariableDeclaration2BasicType': 15,
         'VariableDeclarator2MethodInvocation': 16, 'MethodDeclaration2ForStatement': 17, 'ForStatement2ForControl': 18,
         'ForControl2VariableDeclaration': 19, 'VariableDeclaration2BasicType': 20,
         'VariableDeclaration2VariableDeclarator': 21, 'ForControl2BinaryOperation': 22,
         'BinaryOperation2MemberReference': 23, 'ForControl2MemberReference': 24, 'ForStatement2BlockStatement': 25,
         'BlockStatement2LocalVariableDeclaration': 26, 'VariableDeclarator2ArrayCreator': 27, 'ArrayCreator2BasicType': 28,
         'ArrayCreator2MemberReference': 29, 'BlockStatement2ForStatement': 30, 'ForStatement2StatementExpression': 31,
         'StatementExpression2Assignment': 32, 'Assignment2MemberReference': 33, 'MemberReference2ArraySelector': 34,
         'ArraySelector2MemberReference': 35, 'Assignment2MethodInvocation': 36, 'BinaryOperation2BinaryOperation': 37,
         'BinaryOperation2Literal': 38, 'BlockStatement2StatementExpression': 39, 'MethodInvocation2Literal': 40,
         'MethodInvocation2BinaryOperation': 41, 'ArraySelector2BinaryOperation': 42, 'MethodInvocation2MemberReference': 43,
         'StatementExpression2MethodInvocation': 44, 'ClassDeclaration2FieldDeclaration': 45, 'MethodDeclaration2ReferenceType': 46,
         'MethodDeclaration2WhileStatement': 47, 'WhileStatement2BinaryOperation': 48, 'BinaryOperation2MethodInvocation': 49,
         'WhileStatement2StatementExpression': 50, 'Assignment2ClassCreator': 51, 'ClassCreator2MethodInvocation': 52, 'MethodDeclaration2ReturnStatement': 53,
         'ReturnStatement2MethodInvocation': 54, 'MethodDeclaration2TryStatement': 55, 'TryStatement2StatementExpression': 56,
         'ClassCreator2ClassCreator': 57, 'ClassCreator2Literal': 58, 'TryStatement2LocalVariableDeclaration': 59,
         'MethodInvocation2MethodInvocation': 60, 'TryStatement2ForStatement': 61, 'BlockStatement2IfStatement': 62,
         'IfStatement2BinaryOperation': 63, 'IfStatement2BlockStatement': 64, 'Assignment2BinaryOperation': 65,
         'TryStatement2CatchClause': 66, 'CatchClause2CatchClauseParameter': 67, 'CatchClause2StatementExpression': 68,
         'ReferenceType2TypeArgument': 69, 'TypeArgument2ReferenceType': 70, 'ForStatement2EnhancedForControl': 71,
         'EnhancedForControl2VariableDeclaration': 72, 'VariableDeclaration2ReferenceType': 73, 'EnhancedForControl2MemberReference': 74,
         'IfStatement2StatementExpression': 75, 'MethodDeclaration2IfStatement': 76, 'IfStatement2MethodInvocation': 77, 'ClassCreator2BinaryOperation': 78, 'MethodDeclaration2StatementExpression': 79,
         'VariableDeclarator2MemberReference': 80, 'VariableDeclarator2BinaryOperation': 81, 'WhileStatement2MethodInvocation': 82,
         'ClassDeclaration2ConstructorDeclaration': 83, 'ConstructorDeclaration2Modifier': 84, 'ConstructorDeclaration2FormalParameter': 85,
         'ConstructorDeclaration2StatementExpression': 86, 'Assignment2This': 87, 'This2MemberReference': 88, 'FormalParameter2BasicType': 89, 'ReferenceType2ReferenceType': 90,
         'ArraySelector2Literal': 91, 'ArrayCreator2Literal': 92, 'ForStatement2IfStatement': 93, 'BlockStatement2WhileStatement': 94,
         'WhileStatement2BlockStatement': 95, 'Literal2MethodInvocation': 96, 'MethodInvocation2ClassCreator': 97, 'FieldDeclaration2BasicType': 98, 'ArrayCreator2MethodInvocation': 99,
         'StatementExpression2MemberReference': 100, 'Assignment2ArrayCreator': 101, 'BinaryOperation2TernaryExpression': 102, 'TernaryExpression2MemberReference': 103,
         'TernaryExpression2Literal': 104, 'LocalVariableDeclaration2Annotation': 105, 'Annotation2Literal': 106, 'BlockStatement2BlockStatement': 107, 'VariableDeclarator2TernaryExpression': 108,
         'TernaryExpression2BinaryOperation': 109, 'WhileStatement2Literal': 110, 'Assignment2Literal': 111, 'IfStatement2MemberReference': 112, 'ForControl2Assignment': 113, 'ArrayCreator2ReferenceType': 114,
         'BinaryOperation2Assignment': 115, 'FormalParameter2Modifier': 116, 'LocalVariableDeclaration2Modifier': 117, 'MethodInvocation2ClassReference': 118, 'ClassReference2MethodInvocation': 119,
         'ClassReference2ReferenceType': 120, 'BlockStatement2TryStatement': 121, 'ClassDeclaration2LocalVariableDeclaration': 122, 'ClassDeclaration2IfStatement': 123, 'EnhancedForControl2MethodInvocation': 124,
         'ConstructorDeclaration2ForStatement': 125, 'Assignment2TernaryExpression': 126, 'MemberReference2MethodInvocation': 127, 'BlockStatement2SwitchStatement': 128, 'SwitchStatement2MemberReference': 129,
         'SwitchStatement2SwitchStatementCase': 130, 'SwitchStatementCase2Literal': 131, 'SwitchStatementCase2StatementExpression': 132, 'TryStatement2IfStatement': 133, 'IfStatement2IfStatement': 134, 'Assignment2Assignment': 135,
         'TryStatement2TryStatement': 136, 'BinaryOperation2Cast': 137, 'Cast2BasicType': 138, 'Cast2MemberReference': 139,
         'Assignment2Cast': 140, 'Cast2BinaryOperation': 141, 'BinaryOperation2ClassReference': 142, 'ArrayCreator2BinaryOperation': 143, 'VariableDeclarator2Cast': 144,
         'Cast2ReferenceType': 145, 'BlockStatement2ThrowStatement': 146, 'ThrowStatement2ClassCreator': 147, 'VariableDeclarator2ClassReference': 148, 'TernaryExpression2ClassCreator': 149,
         'TryStatement2TryResource': 150, 'TryResource2ReferenceType': 151, 'TryResource2MethodInvocation': 152, 'MethodInvocation2MethodReference': 153, 'MethodReference2MemberReference': 154,
         'FieldDeclaration2Annotation': 155, 'Cast2Literal': 156, 'IfStatement2ClassCreator': 157, 'BinaryOperation2ClassCreator': 158, 'ArrayCreator2ArrayInitializer': 159, 'ArrayInitializer2MemberReference': 160,
         'TryStatement2WhileStatement': 161, 'MethodInvocation2LambdaExpression': 162, 'LambdaExpression2MemberReference': 163, 'LambdaExpression2MethodInvocation': 164, 'MethodInvocation2Assignment': 165,
         'TryResource2ClassCreator': 166, 'Cast2MethodInvocation': 167, 'VariableDeclarator2ArrayInitializer': 168, 'ArrayInitializer2Literal': 169, 'ClassDeclaration2TryStatement': 170, 'IfStatement2ForStatement': 171,
         'VariableDeclarator2Assignment': 172, 'ClassCreator2ClassReference': 173, 'ConstructorDeclaration2LocalVariableDeclaration': 174, 'ConstructorDeclaration2TryStatement': 175, 'CatchClause2ThrowStatement': 176,
         'ThrowStatement2MemberReference': 177, 'MethodInvocation2Cast': 178, 'ClassDeclaration2Annotation': 179, 'ArraySelector2Cast': 180, 'WhileStatement2ClassCreator': 181, 'TernaryExpression2TernaryExpression': 182,
         'TernaryExpression2MethodInvocation': 183, 'ClassCreator2TernaryExpression': 184, 'MethodInvocation2TernaryExpression': 185, 'TypeArgument2BasicType': 186, 'MethodInvocation2ArraySelector': 187, 'MethodDeclaration2BasicType': 188,
         'ReturnStatement2TernaryExpression': 189, 'IfStatement2ReturnStatement': 190, 'ReturnStatement2BinaryOperation': 191, 'ReturnStatement2Literal': 192, 'ForStatement2TryStatement': 193, 'IfStatement2ContinueStatement': 194, 'ForStatement2ForStatement': 195,
         'ArrayInitializer2ArrayInitializer': 196, 'ArrayCreator2ArraySelector': 197, 'WhileStatement2MemberReference': 198, 'TryResource2Modifier': 199, 'MemberReference2MemberReference': 200, 'ClassCreator2Assignment': 201, 'ClassCreator2MethodDeclaration': 202,
         'MethodDeclaration2Annotation': 203, 'BlockStatement2SynchronizedStatement': 204, 'SynchronizedStatement2ClassReference': 205, 'SynchronizedStatement2IfStatement': 206, 'SynchronizedStatement2StatementExpression': 207, 'ClassDeclaration2StatementExpression': 208,
         'ClassDeclaration2ForStatement': 209, 'ArraySelector2MethodInvocation': 210, 'ClassDeclaration2WhileStatement': 211, 'ConstructorDeclaration2WhileStatement': 212, 'StatementExpression2ClassCreator': 213, 'ReturnStatement2MemberReference': 214, 'ClassCreator2Cast': 215,
         'ReturnStatement2ClassCreator': 216, 'BinaryOperation2This': 217, 'ConstructorDeclaration2IfStatement': 218, 'TryResource2TernaryExpression': 219, 'ReturnStatement2Cast': 220, 'MethodDeclaration2BlockStatement': 221, 'ArrayInitializer2BinaryOperation': 222,
         'MethodInvocation2ArrayCreator': 223, 'IfStatement2ThrowStatement': 224, 'CatchClause2IfStatement': 225, 'BlockStatement2ContinueStatement': 226, 'CatchClause2TryStatement': 227, 'SwitchStatementCase2SwitchStatement': 228, 'BlockStatement2BreakStatement': 229,
         'SwitchStatementCase2IfStatement': 230, 'SwitchStatementCase2LocalVariableDeclaration': 231, 'SwitchStatementCase2ForStatement': 232, 'SwitchStatement2MethodInvocation': 233, 'ForStatement2SwitchStatement': 234, 'IfStatement2Literal': 235,
         'CatchClause2LocalVariableDeclaration': 236, 'ArrayInitializer2MethodInvocation': 237, 'LambdaExpression2BinaryOperation': 238, 'LambdaExpression2InferredFormalParameter': 239, 'LambdaExpression2TernaryExpression': 240, 'Literal2ArraySelector': 241,
         'BlockStatement2DoStatement': 242, 'DoStatement2BinaryOperation': 243, 'DoStatement2BlockStatement': 244, 'LambdaExpression2IfStatement': 245, 'LambdaExpression2StatementExpression': 246, 'This2MethodInvocation': 247, 'ForControl2MethodInvocation': 248,
         'EnhancedForControl2ArrayCreator': 249, 'ClassDeclaration2ReferenceType': 250, 'VariableDeclaration2Modifier': 251, 'TryStatement2ReturnStatement': 252, 'CatchClause2ReturnStatement': 253, 'StatementExpression2SuperConstructorInvocation': 254,
         'SuperConstructorInvocation2MemberReference': 255, 'IfStatement2WhileStatement': 256, 'BlockStatement2AssertStatement': 257, 'AssertStatement2BinaryOperation': 258, 'DoStatement2Literal': 259, 'TernaryExpression2Cast': 260, 'MethodDeclaration2DoStatement': 261,
         'ForStatement2WhileStatement': 262, 'ArrayCreator2Cast': 263, 'DoStatement2MemberReference': 264, 'TernaryExpression2ArrayCreator': 265, 'ArraySelector2Assignment': 266, 'Cast2Cast': 267, 'MethodDeclaration2SwitchStatement': 268, 'SwitchStatementCase2TryStatement': 269,
         'BinaryOperation2ReferenceType': 270, 'StatementExpression2Cast': 271, 'StatementExpression2This': 272, 'MethodInvocation2This': 273, 'ClassCreator2This': 274, 'ClassCreator2ArrayCreator': 275, 'StatementExpression2SuperMethodInvocation': 276, 'SuperMethodInvocation2MemberReference': 277,
         'BlockStatement2ReturnStatement': 278, 'MethodDeclaration2SynchronizedStatement': 279, 'SynchronizedStatement2MemberReference': 280, 'SynchronizedStatement2LocalVariableDeclaration': 281, 'SynchronizedStatement2ForStatement': 282, 'SynchronizedStatement2WhileStatement': 283, 'SynchronizedStatement2ReturnStatement': 284,
         'DoStatement2MethodInvocation': 285, 'SynchronizedStatement2TryStatement': 286, 'Annotation2BinaryOperation': 287, 'FormalParameter2Annotation': 288, 'MethodInvocation2InnerClassCreator': 289, 'InnerClassCreator2ReferenceType': 290, 'InnerClassCreator2MemberReference': 291, 'InnerClassCreator2Literal': 292,
         'InnerClassCreator2MethodInvocation': 293, 'ReturnStatement2This': 294, 'StatementExpression2ExplicitConstructorInvocation': 295, 'ExplicitConstructorInvocation2MemberReference': 296, 'ThrowStatement2MethodInvocation': 297, 'ExplicitConstructorInvocation2TernaryExpression': 298, 'IfStatement2BreakStatement': 299,
         'MethodDeclaration2ThrowStatement': 300, 'SuperConstructorInvocation2BinaryOperation': 301, 'Annotation2ElementValuePair': 302, 'ElementValuePair2ClassReference': 303, 'VariableDeclarator2This': 304, 'IfStatement2TryStatement': 305, 'TryStatement2ThrowStatement': 306, 'ArrayInitializer2ClassReference': 307,
         'SwitchStatementCase2BlockStatement': 308, 'SwitchStatementCase2ThrowStatement': 309, 'ReturnStatement2ArrayCreator': 310, 'MethodDeclaration2AssertStatement': 311, 'Annotation2ElementArrayValue': 312, 'ElementArrayValue2Annotation': 313, 'ElementValuePair2MemberReference': 314, 'ElementValuePair2Literal': 315,
         'ElementValuePair2ElementArrayValue': 316, 'ElementArrayValue2ClassReference': 317, 'SwitchStatementCase2ReturnStatement': 318, 'TryStatement2SwitchStatement': 319, 'SuperConstructorInvocation2Literal': 320, 'SwitchStatementCase2MemberReference': 321, 'Cast2TernaryExpression': 322, 'VariableDeclarator2SuperMethodInvocation': 323,
         'ClassReference2BasicType': 324, 'Cast2This': 325, 'TernaryExpression2This': 326, 'IfStatement2This': 327, 'ElementArrayValue2Literal': 328, 'ExplicitConstructorInvocation2MethodInvocation': 329, 'TryStatement2DoStatement': 330, 'MethodDeclaration2TypeParameter': 331, 'CatchClause2ForStatement': 332, 'TryStatement2BlockStatement': 333,
         'SuperMethodInvocation2MethodInvocation': 334, 'ThrowStatement2Cast': 335, 'Cast2ClassCreator': 336, 'ClassCreator2FieldDeclaration': 337, 'ReturnStatement2SuperMethodInvocation': 338, 'ExplicitConstructorInvocation2ClassCreator': 339, 'Assignment2SuperMemberReference': 340, 'StatementExpression2SuperMemberReference': 341,
         'ClassCreator2StatementExpression': 342, 'Assignment2ClassReference': 343, 'IfStatement2Cast': 344, 'IfStatement2Assignment': 345, 'TryStatement2SynchronizedStatement': 346, 'AssertStatement2Literal': 347, 'ExplicitConstructorInvocation2Literal': 348, 'ArrayInitializer2ClassCreator': 349, 'SuperConstructorInvocation2ClassCreator': 350,
         'Cast2SuperMethodInvocation': 351, 'SwitchStatement2BinaryOperation': 352, 'SwitchStatementCase2WhileStatement': 353, 'ArrayCreator2This': 354, 'SuperConstructorInvocation2MethodInvocation': 355, 'IfStatement2SuperMethodInvocation': 356, 'ArrayCreator2TernaryExpression': 357, 'ReturnStatement2Assignment': 358,
         'MethodInvocation2SuperMemberReference': 359, 'SynchronizedStatement2This': 360, 'This2ArraySelector': 361, 'SwitchStatement2This': 362, 'ArraySelector2This': 363, 'SynchronizedStatement2AssertStatement': 364, 'AssertStatement2MemberReference': 365, 'SwitchStatementCase2DoStatement': 366, 'IfStatement2ClassReference': 367,
         'MethodDeclaration2ClassDeclaration': 368, 'ConstructorDeclaration2BlockStatement': 369, 'ConstructorDeclaration2ClassDeclaration': 370, 'ConstructorDeclaration2Annotation': 371, 'EnhancedForControl2This': 372, 'CatchClause2BreakStatement': 373, 'ArrayInitializer2Cast': 374, 'SwitchStatementCase2BreakStatement': 375,
         'ConstructorDeclaration2SynchronizedStatement': 376, 'TryStatement2AssertStatement': 377, 'AssertStatement2MethodInvocation': 378, 'SwitchStatementCase2SynchronizedStatement': 379, 'SynchronizedStatement2MethodInvocation': 380, 'IfStatement2SwitchStatement': 381, 'IfStatement2TernaryExpression': 382, 'TypeParameter2ReferenceType': 383,
         'SwitchStatement2Cast': 384, 'CatchClause2DoStatement': 385, 'SynchronizedStatement2ThrowStatement': 386, 'Assignment2SuperMethodInvocation': 387, 'StatementExpression2TernaryExpression': 388, 'MethodInvocation2TypeArgument': 389, 'ArrayInitializer2This': 390, 'CatchClause2SynchronizedStatement': 391, 'BinaryOperation2SuperMethodInvocation': 392,
         'CatchClause2WhileStatement': 393, 'SuperMethodInvocation2BinaryOperation': 394, 'IfStatement2SynchronizedStatement': 395, 'WhileStatement2This': 396, 'VariableDeclarator2InnerClassCreator': 397, 'ExplicitConstructorInvocation2Cast': 398, 'ConstructorDeclaration2SwitchStatement': 399, 'EnhancedForControl2ClassCreator': 400, 'IfStatement2AssertStatement': 401,
         'Cast2Assignment': 402, 'SwitchStatementCase2AssertStatement': 403, 'ArrayInitializer2TernaryExpression': 404, 'Assignment2InnerClassCreator': 405, 'WhileStatement2IfStatement': 406, 'ArrayInitializer2ArrayCreator': 407, 'Cast2ClassReference': 408, 'SuperConstructorInvocation2ArrayCreator': 409, 'CatchClause2SwitchStatement': 410, 'SuperConstructorInvocation2Cast': 411,
         'SuperMethodInvocation2ClassCreator': 412, 'TryStatement2ClassDeclaration': 413, 'ForControl2Literal': 414, 'SynchronizedStatement2SynchronizedStatement': 415, 'IfStatement2DoStatement': 416, 'BinaryOperation2BasicType': 417, 'ArrayCreator2ClassCreator': 418, 'MethodInvocation2SuperMethodInvocation': 419, 'ConstructorDeclaration2DoStatement': 420,
         'SuperMethodInvocation2Literal': 421, 'ReturnStatement2ClassReference': 422, 'TernaryExpression2ClassReference': 423, 'SwitchStatement2Assignment': 424, 'EnhancedForControl2TernaryExpression': 425, 'EnhancedForControl2Cast': 426, 'BinaryOperation2SuperMemberReference': 427, 'SuperMemberReference2MethodInvocation': 428, 'ElementArrayValue2MemberReference': 429,
         'ClassCreator2SuperMethodInvocation': 430, 'InnerClassCreator2BinaryOperation': 431, 'Cast2ArrayCreator': 432, 'Cast2SuperMemberReference': 433, 'CatchClause2AssertStatement': 434, 'SwitchStatement2ClassCreator': 435, 'SwitchStatementCase2ContinueStatement': 436, 'DoStatement2StatementExpression': 437, 'ConstructorDeclaration2AssertStatement': 438,
         'Annotation2MemberReference': 439, 'BlockStatement2ClassDeclaration': 440, 'VariableDeclarator2SuperMemberReference': 441, 'SuperMethodInvocation2ArrayCreator': 442, 'ArraySelector2ClassCreator': 443, 'ExplicitConstructorInvocation2BinaryOperation': 444, 'AssertStatement2ClassReference': 445, 'SynchronizedStatement2SwitchStatement': 446,
         'ExplicitConstructorInvocation2ClassReference': 447, 'SuperMethodInvocation2ClassReference': 448, 'StatementExpression2ClassReference': 449, 'AssertStatement2ClassCreator': 450, 'SuperMethodInvocation2ArraySelector': 451, 'ForControl2TernaryExpression': 452, 'ClassCreator2ArraySelector': 453, 'ElementValuePair2BinaryOperation': 454,
         'ArraySelector2TernaryExpression': 455, 'CatchClause2ContinueStatement': 456, 'SuperMethodInvocation2This': 457, 'SwitchStatementCase2BinaryOperation': 458, 'SuperMemberReference2ArraySelector': 459, 'WhileStatement2TryStatement': 460, 'ClassCreator2InnerClassCreator': 461, 'MemberReference2SuperMethodInvocation': 462, 'TernaryExpression2Assignment': 463,
         'SuperConstructorInvocation2TernaryExpression': 464, 'SynchronizedStatement2DoStatement': 465, 'WhileStatement2Assignment': 466, 'EnhancedForControl2ClassReference': 467, 'VariableDeclaration2Annotation': 468, 'InnerClassCreator2TernaryExpression': 469, 'ClassCreator2LocalVariableDeclaration': 470, 'ClassCreator2TryStatement': 471, 'SwitchStatement2TernaryExpression': 472,
         'ThrowStatement2This': 473, 'SwitchStatementCase2Cast': 474, 'ConstructorDeclaration2ThrowStatement': 475, 'WhileStatement2ForStatement': 476, 'ClassReference2MemberReference': 477, 'SuperConstructorInvocation2This': 478, 'StatementExpression2Literal': 479, 'ForControl2This': 480, 'ClassDeclaration2ClassDeclaration': 481, 'StatementExpression2InnerClassCreator': 482,
         'SynchronizedStatement2Literal': 483, 'ArrayCreator2SuperMethodInvocation': 484, 'ClassCreator2SuperMemberReference': 485, 'SuperConstructorInvocation2Assignment': 486, 'ThrowStatement2Literal': 487, 'EnhancedForControl2Literal': 488, 'AssertStatement2This': 489, 'InnerClassCreator2Cast': 490, 'CatchClause2BlockStatement': 491, 'ClassReference2ArraySelector': 492}


class JavaSyntaxMatrixGenerator:
    def __init__(self, javapath, npy_path):
        self.javapath = javapath
        self.npy_path = npy_path

    def listdir(self, path, list_name):
        """
            Recursively lists all files in the specified directory and subdirectories.

            Args:
            path (str): The directory path to list files from.
            list_name (list): The list where all file paths are accumulated.

            Returns:
            None: This function modifies the list_name in-place and does not return anything.
            """
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path, list_name)
            else:
                list_name.append(file_path)

    def getast(self, path):
        """
            Read a Java source code file, tokenize it, parse it to create an AST, and print the AST.

            Args:
            path (str): The path to the Java file to be parsed.

            Returns:
            programast: The AST of the parsed Java member declaration.
            """
        # 打开文件并读取文本
        programfile = open(path, encoding='utf-8')
        print(programfile)
        programtext = programfile.read()
        print(programtext)
        programfile.close()

        # 对读取的文本进行词法分析
        programtokens = javalang.tokenizer.tokenize(programtext)
        token_list = list(programtokens)
        print(token_list)

        # 解析tokens生成AST
        parser = javalang.parse.Parser(token_list)  # 注意这里要传入列表
        programast = parser.parse_member_declaration()
        print(programast)

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
        # print(node.__class__.__name__,str(node))
        # print(node.__class__.__name__, node)
        return token

    # 得到节点的子节点列表
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

        # 展开列表中任何嵌套的子节点
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

    # 创建树结构
    # root:根节点
    # node:当前节点
    # nodelist:节点列表
    # parent:父节点
    # 每个节点(AnyNode)都是用唯一的标识符(id)、标记(token)、数据(data)和父节点(parent)
    def createtree(self, root, node, nodelist, parent=None):
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
        # 获取当前节点对的标记和子节点列表
        token, children = self.get_token(node), self.get_child(node)
        # 如果是列表中第一个节点，则将其作为根节点的token和data
        if id == 0:
            root.token = token
            root.data = node
        else:
            newnode = AnyNode(id=id, token=token, data=node, parent=parent)
        nodelist.append(node)
        for child in children:
            if id == 0:
                self.createtree(root, child, nodelist, parent=root)
            else:
                self.createtree(root, child, nodelist, parent=newnode)

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

    def one_matrix(self, path, npy_path):
        """
           Generates a matrix representation of the syntactic and structural patterns in a Java source file.

           Args:
           path (str): The file path to the Java source file.

           Returns:
           np.ndarray: A matrix where each entry represents normalized counts of specific syntactic patterns.
        """
        # ast generation
        tree, tokens = self.getast(path)

        # create tree
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        self.createtree(newtree, tree, nodelist)

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
        print(typedict)

        # # Traverse the tree to collect triads
        triads = []
        list(self.traverse(newtree, typedict, triads))
        print(triads)

        # Initialize a matrix of zeros with dimensions 493x72
        matrix = [[0 for col in range(72)] for row in range(493)]

        for i in range(len(triads)):
            m = node2groups[triads[i][0] + '2' + triads[i][1]]
            name = triads[i][2]
            try:
                n = nodetypedict[name]
            except KeyError:
                try:
                    n = tokendict[typedict[name]]
                except KeyError:
                    n = 62
            matrix[m][n] += 1

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
        filename = path.split('/')[-1].split('.java')[0]
        npypath = npy_path + filename
        np.save(npypath, matrix)
        return matrix

    def allmain(self, javapath, npy_path):
        """
            Main method to read all Java files from a folder and generate matrices for each file.
        """
        # Read all java files from a folder
        j = 0
        javalist = []
        self.listdir(javapath, javalist)
        for javafile in javalist:
            try:
                self.one_matrix(javafile, npy_path)
            except (UnicodeDecodeError, javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
                print(javafile)
            j += 1
            print(j)



if __name__ == '__main__':
    # Example usage:
    javapath = '/home/data4T/wym/fsl/markovchain/GCJdataset/'
    npy_path = './npy/'
    syntax_matrix_generator = JavaSyntaxMatrixGenerator(javapath, npy_path)
    syntax_matrix_generator.allmain()




