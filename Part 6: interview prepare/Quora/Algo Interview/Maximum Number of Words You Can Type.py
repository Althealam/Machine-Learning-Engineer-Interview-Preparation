class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        words = text.split()
        broken_letters = list(brokenLetters)
        cnt = 0
        for word in words:
            for i in range(len(word)):
                if word[i] in broken_letters:
                    cnt+=1
                    break
        return len(words)-cnt

class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        broken = set(brokenLetters)
        can_type = True
        res = 0
        for ch in text+ ' ': # iterate all the charcter in text
            if ch == ' ': # means that we have already iterate one word
                if can_type == True:
                    res+=1
                can_type = True # start with a new word
            else:
                if ch in broken:
                    can_type = False
        return res