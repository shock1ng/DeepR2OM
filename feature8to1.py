import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

def accumulated_nucleotide_frequency(sequence):
    # 初始化频率字典
    nucleotide_frequency = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
    # 遍历序列并计算频次
    for nucleotide in sequence:
        nucleotide_frequency[nucleotide] += 1
    encoded_list = [1/nucleotide_frequency[char] for char in sequence]
    return encoded_list

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized = [(val - min_val) / (max_val - min_val) for val in lst]
    return normalized

def NCP(sequence):
    # 定义编码字典
    encoding_dict = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'U': [0, 0, 1]}
    # 将RNA序列编码为列表
    encoded_list = []
    for char in sequence:
        encoded_list = encoded_list + encoding_dict[char]
    return encoded_list

def z_score(data):
    # 将列表转换为NumPy数组
    data_array = np.array(data)

    # 计算均值和标准差
    mean = np.mean(data_array)
    std = np.std(data_array)

    # Z-score归一化
    normalized_data = (data_array - mean) / std

    return normalized_data

def get_k_alpha(k):
    k_alpha = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars)**k
    for i in range(end):
        n = i
        chn = chars[int(n%base)]
        for j in range(1, k):
            n = n/base
            chn = chn + chars[int(n%base)]
        k_alpha.append(chn)
    return k_alpha

def get_kmer(seq, k, k_alpha):
    # k_alpha = get_k_alpha(k)
    kmer_fnum = []
    seq_len = len(seq)
    k_dict = {}
    for p in k_alpha:
        k_dict[p] = 0
    i = 0
    while i + k <= seq_len:
        mer = seq[i:i + k]
        i = i + 1
        k_dict[mer] = k_dict[mer] + 1
    kmer_fnum = [x/float(seq_len-k+1)  for x in list(k_dict.values())]

    # kmer_fnum = np.matrix(kmer_fnum)
    return kmer_fnum

def calculate_pseknc(sequence, k, lamada, w):
    # 定义核苷酸映射
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    # 初始化 PseKNC 向量
    pseknc_vector = [0] * (4 ** k)

    # 遍历序列计算 k-mer 出现频率
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if 'N' not in kmer:  # 忽略包含未知核苷酸的 k-mer
            index = sum([nucleotide_map[nuc] * 4 ** j for j, nuc in enumerate(kmer)])
            pseknc_vector[index] += 1

    # 计算 PseKNC 特征
    pseknc_feature = []
    for i in range(w):
        temp = 0
        for j in range((4 ** k)):
            temp += pseknc_vector[j] / (1 + lamada) ** (i + 1)
        pseknc_feature.append(temp)

    # 返回 PseKNC 特征
    return pseknc_feature

def extract_pseknc_features(sequences, k, lamada, w):
    # 初始化特征矩阵
    features = []

    # 提取每个序列的 PseKNC 特征
    for index, line in enumerate(sequences):
        if index > -1:  # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:  # 拿取奇数行的数据，因为偶数行是id
                line = line.strip()  # 拿走每一行最后的换行符'\n'
                pseknc_feature = calculate_pseknc(line, k, lamada, w)
                features.append(pseknc_feature)

    # 标准化特征矩阵
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features

def calculate_enac(sequence, d):
    # 定义氨基酸映射
    amino_acid_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    # 初始化 ENAC 字典
    enac_dict = {}
    for i in range(len(sequence) - d):
        dipeptide = sequence[i:i + d + 1]
        if 'N' not in dipeptide:  # 忽略包含未知氨基酸的二肽
            if dipeptide not in enac_dict:
                enac_dict[dipeptide] = 1
            else:
                enac_dict[dipeptide] += 1

    # 计算 ENAC 特征
    enac_feature = np.zeros(4 ** (d + 1))
    for dipeptide, count in enac_dict.items():
        index = sum([amino_acid_map[aa] * 4 ** i for i, aa in enumerate(dipeptide)])
        enac_feature[index] = count

    return enac_feature

def extract_enac_features(sequences, d):
    # 初始化特征矩阵
    features = []

    # 提取每个序列的 ENAC 特征
    for index, line in enumerate(sequences):
        if index > -1:  # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:  # 拿取奇数行的数据，因为偶数行是id
                line = line.strip()  # 拿走每一行最后的换行符'\n'
                enac_feature = calculate_enac(line, d)
                features.append(enac_feature)

    # 标准化特征矩阵
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features

def calculate_PS2_feature(sequence):
    ps2_feature = {}
    bases = ['A', 'C', 'G', 'U']
    for base1 in bases:
        for base2 in bases:
            dipeptide = base1 + base2
            count = sequence.count(dipeptide)
            ps2_feature[dipeptide] = count
    return ps2_feature

def calculate_DPCP_2(rna_sequence):
    # 定义一个字典，用于存储RNA碱基的索引
    base_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    # 初始化DPCP_2特征向量
    DPCP_2_vector = [0] * 16

    # 将RNA序列转换为大写字母
    rna_sequence = rna_sequence.upper()

    # 计算RNA序列长度
    seq_length = len(rna_sequence)

    # 遍历RNA序列中的每个相邻碱基对
    for i in range(seq_length - 1):
        pair = rna_sequence[i:i + 2]  # 获取相邻的两个碱基对
        if pair[0] in base_index and pair[1] in base_index:
            idx = base_index[pair[0]] * 4 + base_index[pair[1]]
            DPCP_2_vector[idx] += 1

    # 归一化特征向量
    total_pairs = sum(DPCP_2_vector)
    if total_pairs > 0:
        DPCP_2_vector = [count / total_pairs for count in DPCP_2_vector]

    return DPCP_2_vector

def get_pse_ellp_features(rna_sequence, window_size=5, d=2):
    # 定义一个字典，用于存储RNA碱基的索引
    base_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    # 将RNA序列转换为大写字母
    rna_sequence = rna_sequence.upper()

    # 计算RNA序列长度
    seq_length = len(rna_sequence)

    # 初始化特征向量
    features = []

    # 遍历RNA序列中的每个滑动窗口
    for i in range(seq_length - window_size + 1):
        # 初始化窗口内的特征向量
        window_features = np.zeros(4 ** window_size)

        # 获取当前窗口的子序列
        window_sequence = rna_sequence[i:i + window_size]

        # 遍历窗口内的每个碱基组合
        for j in range(window_size):
            base = window_sequence[j]
            base_idx = base_index[base]
            window_features[j * 4 + base_idx] = 1

        # 将窗口特征向量添加到总特征列表中
        features.append(window_features)

    # 计算PseEllp特征
    pse_ellp_features = np.zeros(d)
    for k in range(d):
        for i in range(len(features) - k - 1):
            pse_ellp_features[k] += np.dot(features[i], features[i + k + 1])

    return pse_ellp_features


def feature_com(lines):
    print(lines)
    # 读取seq文件
    # with open(file_lines, 'r') as file:
    #     # 逐行读取文件内容并存储到列表
    #     lines = file.readlines()      # 生成列表，每个元素就是每一行的文本，但是最后都有换行符

    # 提取ANF特征
    ANF_Embedding = []
    # 打印或处理列表中的每一行
    for index, line in enumerate(lines):
        if index > -1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                print(line)
                output = accumulated_nucleotide_frequency(line)
                output = normalize_list(output)    # 自己选择要不要给特征做归一化
                ANF_Embedding.append(output)
    ANF_Embedding_df = pd.DataFrame(ANF_Embedding)
    print('ANF搞定！')

    # 提取NCP特征
    NCP_Embedding = []
    # 打印或处理列表中的每一行
    for index, line in enumerate(lines):
        if index > -1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                output = NCP(line)
                # output = normalize_list(output)    # 自己选择要不要给特征做归一化
                NCP_Embedding.append(output)
    NCP_Embedding_df = pd.DataFrame(NCP_Embedding)  # [2176 rows x 123 columns]

    # 提取kmer特征
    k3Embedding = []
    k4Embedding = []
    k5Embedding = []
    for k in range(3,6):
        max_len = 4**k
        k_alpha = get_k_alpha(k)
        # 打印或处理列表中的每一行
        for index, line in enumerate(lines):
            if index % 2 != 0:  # 拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                output = get_kmer(line, k, k_alpha)
                output = z_score(output)
                if len(output) < max_len :
                    output.extend([0] * (max_len-len(output)))
                if k == 3:
                    k3Embedding.append(output)
                if k == 4:
                    k4Embedding.append(output)
                if k == 5:
                    k5Embedding.append(output)

    k3_Embedding_df = pd.DataFrame(k3Embedding)
    k4_Embedding_df = pd.DataFrame(k4Embedding)
    k5_Embedding_df = pd.DataFrame(k5Embedding)
    print('k345mer搞定！')


    # 提取 PseKNC 特征
    pseknc_features = extract_pseknc_features(lines, k=3, lamada=0.5, w=66)
    PseKNC_Embedding_df = pd.DataFrame(pseknc_features)  # [2176 rows x 123 columns]
    print('PseKNC搞定！')

    # 提取 ENAC 特征
    enac_features = extract_enac_features(lines, d=2)
    ENAC_Embedding_df = pd.DataFrame(enac_features)
    print('ENAC搞定！')

    # 提取PS2 特征
    PS2_Embedding = []
    # 打印或处理列表中的每一行
    for index, line in enumerate(lines):
        if index > -1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                output = calculate_PS2_feature(line)
                # output = normalize_list(output)    # 自己选择要不要给特征做归一化
                PS2_Embedding.append(output)

    PS2_Embedding_df = pd.DataFrame(PS2_Embedding)
    print('PS2搞定！')

    # 提取DPCP_2 特征
    DPCP_2_Embedding = []
    # 打印或处理列表中的每一行
    for index, line in enumerate(lines):
        if index > -1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                output = calculate_DPCP_2(line)
                # output = normalize_list(output)    # 自己选择要不要给特征做归一化
                DPCP_2_Embedding.append(output)


    DPCP_2_Embedding_df = pd.DataFrame(DPCP_2_Embedding)
    print('DPCP_2搞定！')

    # 提取PseEIIP 特征
    PseEIIP_Embedding = []
    id_list = []
    # 打印或处理列表中的每一行
    for index, line in enumerate(lines):
        if index > -1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
            if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
                line = line.strip()   # 拿走每一行最后的换行符'\n'
                output = get_pse_ellp_features(line, window_size=5, d=36)
                output = normalize_list(output)    # 自己选择要不要给特征做归一化
                PseEIIP_Embedding.append(output)
            if index % 2 == 0:  # 拿取偶数行的id
                line = line.strip()
                id_list.append(line)


    PseEIIP_Embedding_df = pd.DataFrame(PseEIIP_Embedding)
    print('PseEIIP搞定！')

    X = pd.concat([ANF_Embedding_df, k3_Embedding_df, k4_Embedding_df, k5_Embedding_df, NCP_Embedding_df, PseKNC_Embedding_df, ENAC_Embedding_df, PseEIIP_Embedding_df, PS2_Embedding_df, DPCP_2_Embedding_df], axis=1)
    return X, id_list