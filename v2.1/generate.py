import numpy as np
import argparse

def generate(dataset,choice,rate,args):
    file_path = './data/'+dataset+'/'+choice+'.txt'
    data_list = []
    item_list = []
    user_list = []

    with open(file_path, 'r') as file:
        first = 0
        i = 0
        num = 0
        line = ''
        for line in file:
            line_data = line.strip().split('\t')
            if int(line_data[0])!=first:
                user_list = list(set(user_list))
                for j in range(num):
                    data_list.append([first, item_list, user_list])
                num = 1
                first = int(line_data[0])
                item_list = [int(line_data[1])]
                user_list = [int(user)for user in line_data[2:]]
            elif int(line_data[0])==first:
                num += 1
                item_list.append(int(line_data[1]))
                user_list += [int(user)for user in line_data[2:]]

        last_line = line.strip().split('\t')
        item_list = [int(last_line[1])]
        user_list = [int(last_line[2])]
        data_list.append([int(last_line[0]),item_list, user_list])

    data = np.loadtxt(file_path, delimiter='\t', usecols=(0, 1, 2), dtype=np.int64)
    i_data = data[:,1:2]
    p_data = data[:,2:3]
    data_np = []
    for line in data_list:
        temp = np.random.randint(0, args.item_num, size=rate)
        for i in range(temp.shape[0]):
            while int(temp[i]) in line[1]:
                temp[i] = np.random.randint(0, args.user_num)
        data_np.append(temp)
        
    i_data_np = np.hstack((i_data, np.array(data_np)))
    print(i_data_np.shape)
    print(i_data_np)
    np.savetxt('data/'+dataset+'/'+choice+'_item_sampling.txt', i_data_np, fmt='%d', delimiter='\t')

    data_np = []
    for line in data_list:
        temp = np.random.randint(0, args.user_num, size=rate)
        for i in range(temp.shape[0]):
            while int(temp[i]) in line[2]:
                temp[i] = np.random.randint(0, args.user_num)
        data_np.append(temp)
        
    p_data_np = np.hstack((p_data, np.array(data_np)))
    print(p_data_np.shape)
    print(p_data_np)
    np.savetxt('data/'+dataset+'/'+choice+'_user_sampling.txt', p_data_np, fmt='%d', delimiter='\t')


if __name__ =='__main__':
    dataset = 'beibei'
    rate = 99
    parser = argparse.ArgumentParser(description='HyperGCN-MTL - A HyperGraph-based MTL model for Group Recommendation.')
    args = parser.parse_args()
    args.user_num = 125012
    args.item_num = 30516
    generate(dataset,'train',rate,args)
    generate(dataset,'tune',rate,args)
    generate(dataset,'test',rate,args)