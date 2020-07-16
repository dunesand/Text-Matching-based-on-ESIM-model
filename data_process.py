# 加载并处理数据，对输入的文本进行编码处理，词转化成id，并对长度进行了对齐。
def load_esim_data_and_labels(data_path,char_to_id,q_max_len=22,t_max_len=22):
    f = open(data_path, 'r')
    y = []
    t_feat_index = []
    q_feat_index = []
    for line in f:
        line = line.strip().split(",")
        y.append([int(line[4])])
        query = []
        for i in line[1].split():
            if i in char_to_id:
                query.append(char_to_id[i])
            else:
                query.append(0)
        if len(query) < q_max_len:
            query = query + [0] * (q_max_len - len(query))
        else:
            query = query[:q_max_len]
        q_feat_index.append(query)
        title = []
        for i in line[3].split():
            if i in char_to_id:
                title.append(char_to_id[i])
            else:
                title.append(0)
        if len(title) < t_max_len:
            title = title + [0] * (t_max_len - len(title))
        else:
            title = title[:t_max_len]
        t_feat_index.append(title)
    f.close()
    return {"q_feat_index": q_feat_index,"t_feat_index": t_feat_index,"label": y}


# 一批一批加载并处理数据，对输入的文本进行编码处理，词转化成id，并对长度进行了对齐。
def yield_esim_data_and_labels(data_path,char_to_id,batch_size,q_max_len=22,t_max_len=22):
    f = open(data_path, 'r')
    y = []
    t_feat_index = []
    q_feat_index = []
    temp_b=0
    for line in f:
        line = line.strip().split(",")
        y.append([int(line[4])])
        query = []
        for i in line[1].split():
            if i in char_to_id:
                query.append(char_to_id[i])
            else:
                query.append(0)
        if len(query) < q_max_len:
            query = query + [0] * (q_max_len - len(query))
        else:
            query = query[:q_max_len]
        q_feat_index.append(query)
        title = []
        for i in line[3].split():
            if i in char_to_id:
                title.append(char_to_id[i])
            else:
                title.append(0)
        if len(title) < t_max_len:
            title = title + [0] * (t_max_len - len(title))
        else:
            title = title[:t_max_len]
        t_feat_index.append(title)
        temp_b += 1
        if temp_b == batch_size:
            yield {"q_feat_index": q_feat_index, "t_feat_index": t_feat_index, "label": y}
            y = []
            t_feat_index = []
            q_feat_index = []
            temp_b = 0
    if temp_b!=0:
        yield {"q_feat_index": q_feat_index,"t_feat_index": t_feat_index,"label": y}
    f.close()
