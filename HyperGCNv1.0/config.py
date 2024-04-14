class Config(object):
    def __init__(self):
        self.gpu_id = 3
        self.path = './data/Weeplaces/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = self.path + 'groupMember.txt'
        self.lr = [0.0001, 0.00005, 0.00002]
        self.drop_ratio = 0.1
        self.topK = [1, 10, 100]


if __name__ == '__main__':
    
    print('-'*89)