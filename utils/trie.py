import collections


# 前缀树，由词嵌入表生成。
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    # 插入前缀树，对于一个词，递归遍历每个字，加深树的深度，进行插入。
    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    # 给定单词，获取前缀树匹配到的词，一开始为最大长度，后来递减为1，例如中华人民，中华人，中华，中。
    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’
        matched = []

        while len(word) > 0:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

